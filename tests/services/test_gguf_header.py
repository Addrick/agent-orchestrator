"""gguf header reader deployed to the pve node (DP-265).

The node-side installer folds this script's output into an install job's status
so a VRAM budget can be *evaluated* rather than recited. These tests build gguf
headers byte by byte, because the whole risk here is misreading a binary format:
a wrong `n_kv_head` produces a confident, wrong cache estimate, which is worse
than no estimate at all.
"""

from __future__ import annotations

import importlib.util
import struct
from pathlib import Path

_MOD_PATH = Path(__file__).resolve().parents[2] / "services" / "pve" / "gguf_header.py"
_spec = importlib.util.spec_from_file_location("gguf_header", _MOD_PATH)
assert _spec and _spec.loader
gguf_header = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(gguf_header)


# -- header construction helpers --------------------------------------------

_STRING, _ARRAY, _UINT32 = 8, 9, 4


def _gstr(text: str) -> bytes:
    raw = text.encode("utf-8")
    return struct.pack("<Q", len(raw)) + raw


def _kv_string(key: str, value: str) -> bytes:
    return _gstr(key) + struct.pack("<I", _STRING) + _gstr(value)


def _kv_u32(key: str, value: int) -> bytes:
    return _gstr(key) + struct.pack("<I", _UINT32) + struct.pack("<I", value)


def _kv_u32_array(key: str, values: list) -> bytes:
    """A per-layer integer array — how gemma4 publishes head_count_kv."""
    body = struct.pack("<I", _UINT32) + struct.pack("<Q", len(values))
    body += b"".join(struct.pack("<I", v) for v in values)
    return _gstr(key) + struct.pack("<I", _ARRAY) + body


def _kv_string_array(key: str, values: list) -> bytes:
    body = struct.pack("<I", _STRING) + struct.pack("<Q", len(values))
    body += b"".join(_gstr(v) for v in values)
    return _gstr(key) + struct.pack("<I", _ARRAY) + body


def _tensor_info(name: str, dims=(8, 8)) -> bytes:
    """One tensor-index entry: name, rank, dims, ggml type, offset."""
    blob = _gstr(name) + struct.pack("<I", len(dims))
    blob += b"".join(struct.pack("<Q", d) for d in dims)
    return blob + struct.pack("<I", 0) + struct.pack("<Q", 0)


def _blocks(count: int, *suffixes: str, start: int = 0) -> list:
    """``count`` consecutive blocks, each carrying every named suffix."""
    return [
        _tensor_info(f"blk.{i}.{suffix}")
        for i in range(start, start + count)
        for suffix in suffixes
    ]


def _write_gguf(path: Path, pairs: list, tensors: list = None) -> Path:
    tensors = tensors or []
    blob = b"GGUF" + struct.pack("<I", 3) + struct.pack("<Q", len(tensors))
    blob += struct.pack("<Q", len(pairs)) + b"".join(pairs)
    blob += b"".join(tensors)
    # Tensor data would follow; the reader must never need it.
    blob += b"\x00" * 4096
    path.write_bytes(blob)
    return path


# -- happy paths -------------------------------------------------------------

def test_reads_the_three_numbers_a_kv_budget_needs(tmp_path):
    path = _write_gguf(tmp_path / "m.gguf", [
        _kv_string("general.architecture", "qwen3"),
        _kv_u32("qwen3.block_count", 48),
        _kv_u32("qwen3.attention.head_count_kv", 8),
        _kv_u32("qwen3.attention.key_length", 128),
    ])
    assert gguf_header.fragment(str(path)) == ',"n_layer":48,"n_kv_head":8,"head_dim":128'


def test_published_key_length_wins_over_the_derived_one(tmp_path):
    """Several architectures use a head dim that is NOT embedding/heads. Deriving
    it there understates the KV cache on exactly the models that break the
    assumption — the direction that picks a context too large."""
    path = _write_gguf(tmp_path / "m.gguf", [
        _kv_string("general.architecture", "gemma4"),
        _kv_u32("gemma4.block_count", 62),
        _kv_u32("gemma4.attention.head_count_kv", 4),
        _kv_u32("gemma4.attention.head_count", 32),
        _kv_u32("gemma4.embedding_length", 4096),   # would derive 128
        _kv_u32("gemma4.attention.key_length", 256),
    ])
    assert '"head_dim":256' in gguf_header.fragment(str(path))


def test_head_dim_is_derived_when_the_model_publishes_no_key_length(tmp_path):
    path = _write_gguf(tmp_path / "m.gguf", [
        _kv_string("general.architecture", "llama"),
        _kv_u32("llama.block_count", 32),
        _kv_u32("llama.attention.head_count_kv", 8),
        _kv_u32("llama.attention.head_count", 32),
        _kv_u32("llama.embedding_length", 4096),
    ])
    assert gguf_header.fragment(str(path)) == ',"n_layer":32,"n_kv_head":8,"head_dim":128'


def test_head_count_kv_falls_back_to_head_count_for_non_gqa_models(tmp_path):
    path = _write_gguf(tmp_path / "m.gguf", [
        _kv_string("general.architecture", "llama"),
        _kv_u32("llama.block_count", 32),
        _kv_u32("llama.attention.head_count", 32),
        _kv_u32("llama.embedding_length", 4096),
    ])
    assert '"n_kv_head":32' in gguf_header.fragment(str(path))


def test_a_tokenizer_array_is_skipped_not_materialised(tmp_path):
    """Arrays are consumed for position and discarded. A 150k-entry vocabulary
    is the one thing that could make a header read expensive."""
    path = _write_gguf(tmp_path / "m.gguf", [
        _kv_string("general.architecture", "qwen3"),
        _kv_string_array("tokenizer.ggml.tokens", [f"tok{i}" for i in range(500)]),
        _kv_u32("qwen3.block_count", 48),
        _kv_u32("qwen3.attention.head_count_kv", 8),
        _kv_u32("qwen3.attention.key_length", 128),
    ])
    meta = gguf_header.read_metadata(str(path))
    # DP-344: the key is now RECORDED, as a marker carrying only its length.
    # It used to read back as None and be dropped, which made "an array lives
    # here" indistinguishable from "absent" — the ambiguity that let
    # `head_count_kv or head_count` silently substitute gemma4's *query* head
    # count for its per-layer KV head counts. What must still not happen is
    # materialising the 500 elements.
    marker = meta["tokenizer.ggml.tokens"]
    assert isinstance(marker, gguf_header.ArrayValue)
    assert marker.length == 500
    assert gguf_header.kv_shape(meta) == {"n_layer": 48, "n_kv_head": 8, "head_dim": 128}


# -- refusals: silence, never a wrong number or a failed install -------------

def test_a_non_gguf_file_produces_no_output(tmp_path):
    path = tmp_path / "not.gguf"
    path.write_bytes(b"this is not a model")
    assert gguf_header.fragment(str(path)) == ""


def test_a_truncated_header_produces_no_output(tmp_path):
    path = tmp_path / "trunc.gguf"
    path.write_bytes(b"GGUF" + struct.pack("<I", 3) + b"\x00\x00")
    assert gguf_header.fragment(str(path)) == ""


def test_a_missing_file_produces_no_output(tmp_path):
    assert gguf_header.fragment(str(tmp_path / "absent.gguf")) == ""


def test_an_implausible_kv_count_is_refused_rather_than_allocated(tmp_path):
    path = tmp_path / "huge.gguf"
    path.write_bytes(
        b"GGUF" + struct.pack("<I", 3) + struct.pack("<Q", 0)
        + struct.pack("<Q", 2 ** 60)
    )
    assert gguf_header.fragment(str(path)) == ""


def test_missing_architecture_key_yields_nothing(tmp_path):
    path = _write_gguf(tmp_path / "m.gguf", [_kv_u32("qwen3.block_count", 48)])
    assert gguf_header.fragment(str(path)) == ""


def test_partial_shape_yields_nothing_rather_than_a_guess(tmp_path):
    """Two of three numbers is not two-thirds of an answer — a cache estimate
    built from a defaulted head_dim is a confident wrong number."""
    path = _write_gguf(tmp_path / "m.gguf", [
        _kv_string("general.architecture", "qwen3"),
        _kv_u32("qwen3.block_count", 48),
    ])
    assert gguf_header.fragment(str(path)) == ""


# -- DP-344: n_layer is the CACHED layer count, not the block count ----------
#
# Every fixture below is the shape of a model this host actually serves, and the
# two qwen35 cases are pinned to koboldcpp's own `llama_kv_cache: KV buffer
# size` line rather than to a re-derivation of the same formula under test.

def test_hybrid_layer_count_comes_from_the_tensor_index_not_block_count(tmp_path):
    """Qwen3.6-40B (Deckard): 96 blocks, of which 24 attend and 72 are SSM.

    Using block_count reports 96 and overstates the cache 4x. Measured on the
    R9700: 828.75 MiB of KV at n_ctx 16640 = 52224 B/token, which is
    2 x 24 x 4 x 256 x 34/32 -- i.e. exactly 24 cached layers, not 96.
    """
    path = _write_gguf(
        tmp_path / "deckard.gguf",
        [
            _kv_string("general.architecture", "qwen35"),
            _kv_u32("qwen35.block_count", 96),
            _kv_u32("qwen35.attention.head_count_kv", 4),
            _kv_u32("qwen35.attention.key_length", 256),
        ],
        # 24 attention blocks, then 72 SSM blocks carrying the *fused* qkv
        # tensor. Preferring attn_qkv would count all 96 straight back.
        _blocks(24, "attn_k.weight", "attn_v.weight")
        + _blocks(72, "attn_qkv.weight", "ssm_conv1d.weight", start=24),
    )
    assert gguf_header.fragment(str(path)) == (
        ',"n_layer":24,"n_kv_head":4,"head_dim":256'
    )


def test_the_mtp_draft_block_is_not_counted(tmp_path):
    """Qwen3.8-27B: 65 blocks, 17 with a K projection, and kcpp caches 16.

    Block 64 is the `nextn` multi-token-prediction draft layer. It has a real
    attn_k and is still never cached, because MTP needs --usemtp (ROCm-only)
    and no unit here passes it. Measured: 8712.50 MiB at n_ctx 262400 =
    34816 B/token = 2 x 16 x 4 x 256 x 34/32.

    This one earns its own test because the error hid: 17/16 is exactly 1.0625,
    so counting 17 layers at "1 byte per element" produced the *right* total for
    this model and the wrong one for every other.
    """
    path = _write_gguf(
        tmp_path / "qwen38.gguf",
        [
            _kv_string("general.architecture", "qwen35"),
            _kv_u32("qwen35.block_count", 65),
            _kv_u32("qwen35.attention.head_count_kv", 4),
            _kv_u32("qwen35.attention.key_length", 256),
        ],
        _blocks(16, "attn_k.weight")
        + _blocks(48, "attn_qkv.weight", start=16)
        + [
            _tensor_info("blk.64.attn_k.weight"),
            _tensor_info("blk.64.nextn.eh_proj.weight"),
        ],
    )
    assert gguf_header.fragment(str(path)) == (
        ',"n_layer":16,"n_kv_head":4,"head_dim":256'
    )


def test_fused_qkv_models_count_their_qkv_blocks(tmp_path):
    """No separate attn_k anywhere: K lives inside attn_qkv, so those blocks are
    the cached ones. The qkv set is a fallback and never a preference -- on the
    hybrid archs above it is what the *uncached* blocks carry."""
    path = _write_gguf(
        tmp_path / "fused.gguf",
        [
            _kv_string("general.architecture", "phi3"),
            _kv_u32("phi3.block_count", 32),
            _kv_u32("phi3.attention.head_count_kv", 8),
            _kv_u32("phi3.attention.key_length", 128),
        ],
        _blocks(32, "attn_qkv.weight"),
    )
    assert '"n_layer":32' in gguf_header.fragment(str(path))


def test_unfamiliar_tensor_naming_falls_back_to_block_count(tmp_path):
    """An index that names no attention tensor we recognise falls back to
    block_count. That OVERSTATES the cache on a hybrid, which under-sizes the
    context -- the safe direction. Understating it overcommits VRAM."""
    path = _write_gguf(
        tmp_path / "odd.gguf",
        [
            _kv_string("general.architecture", "novel"),
            _kv_u32("novel.block_count", 40),
            _kv_u32("novel.attention.head_count_kv", 8),
            _kv_u32("novel.attention.key_length", 128),
        ],
        _blocks(40, "attn_wqkv.weight"),
    )
    assert '"n_layer":40' in gguf_header.fragment(str(path))


# -- DP-344: models the linear formula does not describe are REFUSED ---------

def test_per_layer_kv_head_counts_are_refused_with_a_reason(tmp_path):
    """gemma4 publishes attention.head_count_kv as a LIST -- 16 on its
    sliding-window layers, 4 on its full-attention ones. There is no single
    n_kv_head to put in the formula.

    The regression this pins: the array read back as None, `or` fell through to
    attention.head_count, and the *query* head count (32) was reported as the KV
    head count -- 8x over, on top of counting all 60 layers as full-attention.
    That produced ~2 MB/token for a model whose windowed layers stop growing at
    1024 tokens.
    """
    path = _write_gguf(tmp_path / "gemma.gguf", [
        _kv_string("general.architecture", "gemma4"),
        _kv_u32("gemma4.block_count", 60),
        _kv_u32_array("gemma4.attention.head_count_kv", [16] * 50 + [4] * 10),
        _kv_u32("gemma4.attention.head_count", 32),
        _kv_u32("gemma4.attention.key_length", 512),
        _kv_u32("gemma4.attention.sliding_window", 1024),
    ], _blocks(60, "attn_k.weight"))
    out = gguf_header.fragment(str(path))
    assert '"kv_shape_note"' in out
    assert "per layer" in out
    # The specific wrong answer must not come back by any route.
    assert '"n_kv_head"' not in out
    assert "32" not in out


def test_sliding_window_attention_is_refused_even_with_uniform_heads(tmp_path):
    """A uniform head count is not enough: if the cache stops growing at the
    window, it is not a linear function of contextsize and a per-token figure
    would overstate every context past the window."""
    path = _write_gguf(tmp_path / "swa.gguf", [
        _kv_string("general.architecture", "swamodel"),
        _kv_u32("swamodel.block_count", 32),
        _kv_u32("swamodel.attention.head_count_kv", 8),
        _kv_u32("swamodel.attention.key_length", 128),
        _kv_u32("swamodel.attention.sliding_window", 4096),
    ], _blocks(32, "attn_k.weight"))
    out = gguf_header.fragment(str(path))
    assert '"kv_shape_note"' in out
    assert "sliding-window" in out
    assert '"n_layer"' not in out


def test_a_refusal_note_is_valid_json_and_carries_no_raw_quotes(tmp_path):
    """The fragment is appended straight into a JSON object a bash script is
    printf-ing, with no encoder anywhere in the path. A stray quote makes the
    whole job record unparseable for the caller."""
    import json

    path = _write_gguf(tmp_path / "gemma.gguf", [
        _kv_string("general.architecture", "gemma4"),
        _kv_u32("gemma4.block_count", 60),
        _kv_u32_array("gemma4.attention.head_count_kv", [16, 4]),
        _kv_u32("gemma4.attention.head_count", 32),
        _kv_u32("gemma4.attention.key_length", 512),
    ], _blocks(60, "attn_k.weight"))
    assert json.loads('{"a":1' + gguf_header.fragment(str(path)) + "}")


def test_an_unreadable_tensor_index_still_yields_the_other_two_numbers(tmp_path):
    """The index is best-effort *within* a good file: a header whose metadata
    parsed keeps its n_kv_head/head_dim and falls back for the layer count,
    rather than losing the whole estimate to a truncated tail."""
    blob = b"GGUF" + struct.pack("<I", 3) + struct.pack("<Q", 4)
    pairs = [
        _kv_string("general.architecture", "llama"),
        _kv_u32("llama.block_count", 32),
        _kv_u32("llama.attention.head_count_kv", 8),
        _kv_u32("llama.attention.key_length", 128),
    ]
    blob += struct.pack("<Q", len(pairs)) + b"".join(pairs)
    blob += b"\x00" * 3  # a tensor index that stops mid-entry
    path = tmp_path / "trunc_index.gguf"
    path.write_bytes(blob)
    assert gguf_header.fragment(str(path)) == (
        ',"n_layer":32,"n_kv_head":8,"head_dim":128'
    )
