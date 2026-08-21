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


def _kv_string_array(key: str, values: list) -> bytes:
    body = struct.pack("<I", _STRING) + struct.pack("<Q", len(values))
    body += b"".join(_gstr(v) for v in values)
    return _gstr(key) + struct.pack("<I", _ARRAY) + body


def _write_gguf(path: Path, pairs: list) -> Path:
    blob = b"GGUF" + struct.pack("<I", 3) + struct.pack("<Q", 0)
    blob += struct.pack("<Q", len(pairs)) + b"".join(pairs)
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
    assert "tokenizer.ggml.tokens" not in meta
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
