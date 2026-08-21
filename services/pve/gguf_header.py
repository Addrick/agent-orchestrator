#!/usr/bin/env python3
"""Read the three numbers a KV-cache budget needs out of a gguf header (DP-265).

Deployed to the Proxmox node beside ``derpr-model-install``, which calls it after
a download verifies and folds the output into the job status. It exists so hypr
can *evaluate*

    KV_bytes_per_token = 2 · n_layer · n_kv_head · head_dim · bytes_per_elem

rather than only recite it. Without these numbers a context size is still picked
by hand with nothing to check it against.

Stdlib only, and it reads the header, never the tensor data — a 30 GB file costs
a few KB of I/O here.

Output is a **JSON fragment**, not a document::

    ,"n_layer":48,"n_kv_head":8,"head_dim":128

That shape is deliberate: the caller is a bash script on a node with no ``jq``,
and it appends this straight into the status object it is already printf-ing.
A file it cannot parse produces **no output and exit 0** — a header quirk must
never fail an install whose bytes verified.
"""

from __future__ import annotations

import struct
import sys
from typing import Any, BinaryIO, Dict, Optional

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


class _Bad(Exception):
    """The file is not a gguf we can read. Always handled, never propagated."""


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
    """One metadata value. Arrays are consumed but returned as None.

    Nothing this script wants is an array, and materialising a 150k-entry
    tokenizer vocabulary to throw it away is the one way a header read could
    become expensive.
    """
    if type_id == _STRING:
        return _string(fh)
    if type_id == _ARRAY:
        (elem_type,) = struct.unpack("<I", _read(fh, 4))
        (count,) = struct.unpack("<Q", _read(fh, 8))
        if count > _MAX_LEN:
            raise _Bad(f"implausible array count {count}")
        for _ in range(count):
            _value(fh, elem_type)
        return None
    return _scalar(fh, type_id)


def read_metadata(path: str) -> Dict[str, Any]:
    """Every scalar/string key in the gguf header. Raises ``_Bad`` on garbage."""
    with open(path, "rb") as fh:
        if _read(fh, 4) != _MAGIC:
            raise _Bad("not a gguf file")
        struct.unpack("<I", _read(fh, 4))  # version
        struct.unpack("<Q", _read(fh, 8))  # tensor count
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
        return meta


def kv_shape(meta: Dict[str, Any]) -> Optional[Dict[str, int]]:
    """``n_layer`` / ``n_kv_head`` / ``head_dim``, or None if any is missing.

    All three keys are namespaced by the architecture (``qwen3.block_count``),
    so the arch is read first. ``head_dim`` is taken from
    ``attention.key_length`` when the model publishes it — several architectures
    use a head dim that is *not* ``embedding_length / head_count``, and deriving
    it there would understate the KV cache on exactly those models.
    """
    arch = meta.get("general.architecture")
    if not isinstance(arch, str) or not arch:
        return None

    def num(suffix: str) -> Optional[int]:
        value = meta.get(f"{arch}.{suffix}")
        return int(value) if isinstance(value, (int, float)) else None

    n_layer = num("block_count")
    n_kv_head = num("attention.head_count_kv") or num("attention.head_count")
    head_dim = num("attention.key_length")
    if head_dim is None:
        embedding = num("embedding_length")
        heads = num("attention.head_count")
        if embedding and heads:
            head_dim = embedding // heads
    if not (n_layer and n_kv_head and head_dim):
        return None
    return {"n_layer": n_layer, "n_kv_head": n_kv_head, "head_dim": head_dim}


def fragment(path: str) -> str:
    """The JSON fragment for ``path``, or an empty string if unreadable."""
    try:
        shape = kv_shape(read_metadata(path))
    except (_Bad, OSError, struct.error, UnicodeDecodeError):
        return ""
    if shape is None:
        return ""
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
