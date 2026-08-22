#!/usr/bin/env python3
"""Read the numbers a KV-cache budget needs out of a gguf header (DP-265).

Deployed to the Proxmox node beside ``derpr-model-install``, which calls it after
a download verifies and folds the output into the job status. It exists so hypr
can *evaluate*

    KV_bytes_per_token = 2 · n_layer · n_kv_head · head_dim · bytes_per_elem

rather than only recite it. Without these numbers a context size is still picked
by hand with nothing to check it against.

⚠️ **``n_layer`` is the count of layers that actually cache K/V, which is NOT
``block_count`` on every architecture** (DP-344). Getting that wrong is worse
than having no number, because the formula's output is quoted to a human as
arithmetic. Two ways the header lies to a naive reader, both measured on models
this node serves:

- **Hybrid (Gated DeltaNet) archs** — ``qwen35``, ``qwen35moe`` — interleave
  attention blocks with SSM blocks that hold a *fixed* recurrent state and cache
  nothing per token. Qwen3.8-27B is 65 blocks of which **17** attend; using 65
  overstates the cache 3.8× and cost a live answer of "32k is the maximum
  context" for a model that fits 262144.
- **Per-layer attention shapes** — ``gemma4`` publishes
  ``attention.head_count_kv`` as a **list** (16 on sliding-window layers, 4 on
  full-attention ones) and a 1024-token ``attention.sliding_window``. There is no
  single ``n_kv_head``, and 50 of its 60 layers stop growing at the window, so a
  linear formula does not describe this model at any head count. Such a model is
  **refused** with a reason rather than estimated.
- **MTP draft blocks** carry a real ``attn_k`` and are still not cached, because
  koboldcpp only runs them under ``--usemtp`` (ROCm-only, and no unit here passes
  it). Qwen3.8-27B has 17 blocks with a K projection and kcpp allocates KV for
  **16**; block 64 is its ``nextn`` block.

So the layer count comes from the **tensor index** — the blocks carrying a K
projection, less the draft blocks — and not from a metadata integer that means
something else. Validated to the byte against koboldcpp's own
``llama_kv_cache: KV buffer size`` on two models: Qwen3.8-27B (16 layers,
34816 B/token at 262400 ctx) and Deckard-40B (24 layers, 52224 B/token at
16640 ctx).

Stdlib only, and it reads the header and the tensor index, never the tensor
*data* — a 30 GB file costs a few hundred KB of I/O here.

Output is a **JSON fragment**, not a document::

    ,"n_layer":48,"n_kv_head":8,"head_dim":128

or, when the model's cache cannot be described by the formula::

    ,"kv_shape_note":"per-layer attention.head_count_kv ..."

That shape is deliberate: the caller is a bash script on a node with no ``jq``,
and it appends this straight into the status object it is already printf-ing.
A file it cannot parse produces **no output and exit 0** — a header quirk must
never fail an install whose bytes verified.
"""

from __future__ import annotations

import struct
import sys
from typing import Any, BinaryIO, Dict, NamedTuple, Optional, Set

_MAGIC = b"GGUF"

# GGUF metadata value type ids → struct format for the scalar ones.
_SCALAR = {
    0: "<B",   # uint8
    1: "<b",   # int8
    2: "<H",   # uint16
    3: "<h",   # int16
    4: "<I",   # uint32
    5: "<i",   # int32
    6: "<f",   # float32
    7: "<?",   # bool
    10: "<Q",  # uint64
    11: "<q",  # int64
    12: "<d",  # float64
}
_STRING = 8
_ARRAY = 9

#: Refuse absurd lengths rather than trying to allocate them. A corrupt or
#: hostile header is the case this guards; the largest legitimate gguf string
#: (a chat template) is comfortably under this.
_MAX_LEN = 64 * 1024 * 1024
_MAX_KV = 100_000
#: Same guard for the tensor index. The largest model here has 866 tensors.
_MAX_TENSORS = 1_000_000

#: The tensor that proves a block caches K. A block without one contributes
#: nothing per token, whatever `block_count` says it is.
_K_TENSOR = "attn_k.weight"
#: Fused-QKV architectures publish no separate `attn_k`; K lives inside this.
#: Checked only when no block has a separate K projection — on the hybrid archs
#: `attn_qkv.weight` is what the *SSM* blocks carry, so preferring it there
#: would reintroduce exactly the overcount this module exists to avoid.
_QKV_TENSOR = "attn_qkv.weight"
#: Prefix of the multi-token-prediction draft block's tensors. Such a block has
#: a real K projection and is still **not cached**: koboldcpp only runs MTP with
#: ``--usemtp``, which needs ROCm, and no unit ``install_model`` writes passes
#: it. Measured — Qwen3.8-27B has 17 blocks with `attn_k` and kcpp allocates KV
#: for 16 of them; block 64 is the nextn block.
#: ⚠️ If a unit here ever enables MTP, the draft block caches too and this
#: exclusion under-counts. Under-counting is the direction that overcommits
#: VRAM, so revisit this the moment `--usemtp` becomes reachable.
_NEXTN_PREFIX = "nextn."


class _Bad(Exception):
    """The file is not a gguf we can read. Always handled, never propagated."""


class ArrayValue(NamedTuple):
    """Marker for a metadata key whose value is an array.

    The elements are consumed and thrown away — materialising a 150k-entry
    tokenizer vocabulary is the one way a header read could become expensive —
    but the *key* must still be recorded. An array that read back as ``None``
    was indistinguishable from an absent key, and that is precisely what let
    ``head_count_kv or head_count`` silently substitute gemma4's **query** head
    count (32) for its per-layer KV head counts (16/4).
    """

    # NOT `count`: NamedTuple fields shadow tuple methods, and `tuple.count`
    # is one. mypy rejects it outright, which is the good outcome — a silent
    # shadow would break `.count()` for any caller that expected a tuple.
    length: int
    elem_type: int


class Header(NamedTuple):
    """What one pass over a gguf's header yields."""

    #: Every metadata key. Array values are ``ArrayValue`` markers.
    meta: Dict[str, Any]
    #: Blocks carrying a K projection, or None if the tensor index was
    #: unreadable. ``0`` means the index read fine and named no such tensor.
    attn_layers: Optional[int]


def _read(fh: BinaryIO, size: int) -> bytes:
    if size < 0 or size > _MAX_LEN:
        raise _Bad(f"implausible length {size}")
    data = fh.read(size)
    if len(data) != size:
        raise _Bad("truncated header")
    return data


def _scalar(fh: BinaryIO, type_id: int) -> Any:
    fmt = _SCALAR.get(type_id)
    if fmt is None:
        raise _Bad(f"unknown value type {type_id}")
    return struct.unpack(fmt, _read(fh, struct.calcsize(fmt)))[0]


def _string(fh: BinaryIO) -> str:
    (length,) = struct.unpack("<Q", _read(fh, 8))
    return _read(fh, length).decode("utf-8", "replace")


def _value(fh: BinaryIO, type_id: int) -> Any:
    """One metadata value. Arrays are consumed and reduced to a marker."""
    if type_id == _STRING:
        return _string(fh)
    if type_id == _ARRAY:
        (elem_type,) = struct.unpack("<I", _read(fh, 4))
        (count,) = struct.unpack("<Q", _read(fh, 8))
        if count > _MAX_LEN:
            raise _Bad(f"implausible array count {count}")
        for _ in range(count):
            _value(fh, elem_type)
        return ArrayValue(length=count, elem_type=elem_type)
    return _scalar(fh, type_id)


def _attention_layers(fh: BinaryIO, tensor_count: int) -> Optional[int]:
    """Blocks that cache K/V, or None if the index could not be walked.

    Counting distinct block indices rather than tensors is deliberate: a block
    is either cached or it is not, and a future arch that splits K across two
    tensors must not count twice.
    """
    if tensor_count > _MAX_TENSORS:
        raise _Bad(f"implausible tensor count {tensor_count}")
    k_blocks: Set[str] = set()
    qkv_blocks: Set[str] = set()
    nextn_blocks: Set[str] = set()
    for _ in range(tensor_count):
        name = _string(fh)
        (n_dims,) = struct.unpack("<I", _read(fh, 4))
        if n_dims > 8:
            raise _Bad(f"implausible tensor rank {n_dims}")
        _read(fh, 8 * n_dims)          # dims
        _read(fh, 4)                   # ggml type
        _read(fh, 8)                   # offset
        if not name.startswith("blk."):
            continue
        parts = name.split(".", 2)
        if len(parts) != 3:
            continue
        index, suffix = parts[1], parts[2]
        if suffix == _K_TENSOR:
            k_blocks.add(index)
        elif suffix == _QKV_TENSOR:
            qkv_blocks.add(index)
        elif suffix.startswith(_NEXTN_PREFIX):
            nextn_blocks.add(index)
    if k_blocks:
        return len(k_blocks - nextn_blocks)
    return len(qkv_blocks - nextn_blocks)


def read_header(path: str) -> Header:
    """Metadata plus the attention-layer count. Raises ``_Bad`` on garbage.

    The tensor index is best-effort *within* an otherwise good file: a header
    whose metadata parsed but whose index did not still yields a usable
    ``n_kv_head``/``head_dim``, and ``kv_shape`` will fall back for the layer
    count rather than lose the whole estimate.
    """
    with open(path, "rb") as fh:
        if _read(fh, 4) != _MAGIC:
            raise _Bad("not a gguf file")
        struct.unpack("<I", _read(fh, 4))  # version
        (tensor_count,) = struct.unpack("<Q", _read(fh, 8))
        (kv_count,) = struct.unpack("<Q", _read(fh, 8))
        if kv_count > _MAX_KV:
            raise _Bad(f"implausible metadata count {kv_count}")
        meta: Dict[str, Any] = {}
        for _ in range(kv_count):
            key = _string(fh)
            (type_id,) = struct.unpack("<I", _read(fh, 4))
            value = _value(fh, type_id)
            if value is not None:
                meta[key] = value
        try:
            attn_layers: Optional[int] = _attention_layers(fh, tensor_count)
        except (_Bad, struct.error, UnicodeDecodeError):
            attn_layers = None
    return Header(meta=meta, attn_layers=attn_layers)


def read_metadata(path: str) -> Dict[str, Any]:
    """Every metadata key in the gguf header. Raises ``_Bad`` on garbage."""
    return read_header(path).meta


def _arch_get(meta: Dict[str, Any], suffix: str) -> Any:
    arch = meta.get("general.architecture")
    if not isinstance(arch, str) or not arch:
        return None
    return meta.get(f"{arch}.{suffix}")


def _num(meta: Dict[str, Any], suffix: str) -> Optional[int]:
    value = _arch_get(meta, suffix)
    # bool is an int subclass and would sail through; no count is ever a bool.
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return int(value)


def kv_shape_note(meta: Dict[str, Any]) -> Optional[str]:
    """Why this model's KV cache is not a linear function of context, or None.

    Returned instead of numbers, never beside them. A refusal with a reason is
    strictly better than a confident wrong estimate: the caller's fallback is
    "measure it", which is correct for exactly these models.
    """
    if not isinstance(meta.get("general.architecture"), str):
        return None
    if isinstance(_arch_get(meta, "attention.head_count_kv"), ArrayValue):
        return (
            "this model publishes attention.head_count_kv per layer, so it has "
            "no single KV head count and its cache is not a linear function of "
            "context; size the contextsize by measurement instead"
        )
    if _num(meta, "attention.sliding_window"):
        return (
            "this model uses sliding-window attention, so its cache stops "
            "growing at the window rather than scaling with contextsize; size "
            "the contextsize by measurement instead"
        )
    return None


def kv_shape(
    meta: Dict[str, Any], attn_layers: Optional[int] = None
) -> Optional[Dict[str, int]]:
    """``n_layer`` / ``n_kv_head`` / ``head_dim``, or None if not applicable.

    ``n_layer`` is the number of layers that **cache K/V**. It comes from
    ``attn_layers`` (counted off the tensor index) when that is available and
    non-zero, and falls back to ``block_count`` only when the index named no
    attention tensor at all — an unfamiliar naming convention, where
    overstating the cache is the safe direction because it under-sizes the
    context rather than overcommitting VRAM.

    All metadata keys are namespaced by the architecture (``qwen3.block_count``),
    so the arch is read first. ``head_dim`` is taken from
    ``attention.key_length`` when the model publishes it — several architectures
    use a head dim that is *not* ``embedding_length / head_count``, and deriving
    it there would understate the KV cache on exactly those models.
    """
    arch = meta.get("general.architecture")
    if not isinstance(arch, str) or not arch:
        return None
    if kv_shape_note(meta) is not None:
        return None

    if attn_layers:
        n_layer: Optional[int] = attn_layers
    else:
        n_layer = _num(meta, "block_count")
    # Only fall back to the query head count when head_count_kv is genuinely
    # ABSENT (an MHA model, where they are equal). A present-but-non-scalar one
    # was refused above and must never reach this `or`.
    n_kv_head = (
        _num(meta, "attention.head_count_kv")
        or _num(meta, "attention.head_count")
    )
    head_dim = _num(meta, "attention.key_length")
    if head_dim is None:
        embedding = _num(meta, "embedding_length")
        heads = _num(meta, "attention.head_count")
        if embedding and heads:
            head_dim = embedding // heads
    if not (n_layer and n_kv_head and head_dim):
        return None
    return {"n_layer": n_layer, "n_kv_head": n_kv_head, "head_dim": head_dim}


def _json_str(text: str) -> str:
    """Minimal JSON string escaping — these notes are ours, not user input."""
    return text.replace("\\", "\\\\").replace('"', '\\"')


def fragment(path: str) -> str:
    """The JSON fragment for ``path``, or an empty string if unreadable."""
    try:
        header = read_header(path)
    except (_Bad, OSError, struct.error, UnicodeDecodeError):
        return ""
    shape = kv_shape(header.meta, header.attn_layers)
    if shape is None:
        note = kv_shape_note(header.meta)
        if note is None:
            return ""
        return f',"kv_shape_note":"{_json_str(note)}"'
    return "".join(f',"{k}":{v}' for k, v in shape.items())


def main() -> int:
    if len(sys.argv) != 2:
        return 0
    text = fragment(sys.argv[1])
    if text:
        sys.stdout.write(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
