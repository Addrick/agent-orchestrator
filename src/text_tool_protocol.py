# src/text_tool_protocol.py
"""Shared `<tool_call>` text-protocol primitives.

Models without a native tool-calling API are driven over a plain-text
convention: tool definitions are rendered into the prompt, and the model
signals a call by emitting `<tool_call>{json}</tool_call>`. Two call sites use
this convention with legitimately different *shapes*:

  - `engine.py` (agy CLI path) parses a COMPLETE text response in one shot.
  - `stream_engine.py` (local kobold path) parses INCREMENTALLY as tokens
    arrive, holding lookahead so a partially-arrived tag never leaks.

This module owns only the genuinely-common core so the two paths cannot
drift on the wire format:

  - the literal open/close tags (`TOOL_CALL_OPEN` / `TOOL_CALL_CLOSE`),
  - extracting the complete `<tool_call>…</tool_call>` blocks from text,
  - JSON-decoding a block's inner payload into a dict.

The differing parser *machinery* (the streaming buffer/lookahead vs. the
single-shot regex sweep) and each caller's id-minting / field-validation
policy intentionally stay at their respective call sites — forcing them into
one parser would distort the streaming path without removing real
duplication.
"""

import json
import re
from typing import Any, Dict, List, Optional

# The literal tags the model emits to delimit a tool call. Both the agy
# complete-response path and the local streaming path key off these, so they
# live here to guarantee the two never disagree on the wire format.
TOOL_CALL_OPEN = "<tool_call>"
TOOL_CALL_CLOSE = "</tool_call>"

# One-line description of the protocol the model must follow, plus the exact
# syntax of a single call. Used to build each path's tool-instruction block.
TOOL_CALL_SYNTAX = (
    TOOL_CALL_OPEN
    + '{"name": "TOOL_NAME", "arguments": {"arg1": "value", ...}}'
    + TOOL_CALL_CLOSE
)

_TOOL_CALL_BLOCK_RE = re.compile(
    re.escape(TOOL_CALL_OPEN) + r"(.*?)" + re.escape(TOOL_CALL_CLOSE),
    flags=re.DOTALL,
)


def extract_tool_call_blocks(text: str) -> List[str]:
    """Return the inner (stripped) payload of EVERY complete
    `<tool_call>…</tool_call>` block in `text`, in emission order.

    Used by the complete-response path. The streaming path locates blocks
    incrementally instead, but decodes each block's payload via
    `decode_tool_call_payload`, so both share the same JSON semantics.

    All of them, not the first (DP-338): a model asked to batch independent
    reads answers with one block per call, and the first-match version of this
    function silently dropped every block after block 1 — no execution, no
    result, no error. The model then re-emitted the same batch to chase the
    answers it never got, and since the batch's first block does not change,
    that is a fixed point that spins until the turn's call budget trips. The
    streaming parser has always returned every block; this is what made the two
    paths agree.
    """
    if not text:
        return []
    return [m.group(1).strip() for m in _TOOL_CALL_BLOCK_RE.finditer(text)]


def extract_first_tool_call_block(text: str) -> Optional[str]:
    """The first block only, or None. Kept for callers that genuinely want one
    call (and for the `strip_tool_call_blocks` docstring's contrast); prefer
    `extract_tool_call_blocks` on any path that executes what it parses.
    """
    blocks = extract_tool_call_blocks(text)
    return blocks[0] if blocks else None


def strip_tool_call_blocks(text: str) -> str:
    """`text` with every complete `<tool_call>…</tool_call>` block removed.

    The prose a model writes beside its calls is the plan behind them, and the
    complete-response path has to separate the two by hand — the streaming
    parser gets `visible_text` for free because it splits the stream as it
    arrives. Returned stripped; collapsing the blank run the removed blocks
    leave behind keeps a multi-call response from rendering as a paragraph
    followed by four empty lines.
    """
    if not text:
        return ""
    without = _TOOL_CALL_BLOCK_RE.sub("", text)
    return re.sub(r"\n{3,}", "\n\n", without).strip()


def decode_tool_call_payload(raw_json: str) -> Optional[Dict[str, Any]]:
    """Parse a `<tool_call>` block's inner JSON into a dict.

    Returns the decoded dict on success, or None if the payload is not valid
    JSON or does not decode to a JSON object. Field-level validation (which
    keys are required) and id-minting are left to each caller, because the
    agy and streaming paths have different policies there.
    """
    try:
        parsed = json.loads(raw_json.strip())
    except json.JSONDecodeError:
        return None
    if not isinstance(parsed, dict):
        return None
    return parsed


def render_tool_descriptions(tools: List[Dict[str, Any]]) -> List[str]:
    """Render a flat `name/description/parameters` line per tool.

    This is the compact form the agy complete-response prompt uses. Each tool
    dict is the OpenAI-style `{"function": {...}}` envelope; the bare-function
    shape is tolerated as a fallback.
    """
    lines: List[str] = []
    for t in tools:
        func = t.get("function", {})
        name = func.get("name", "")
        description = func.get("description", "")
        parameters = func.get("parameters", {})
        lines.append(
            f"name: {name}, description: {description}, parameters: {parameters}"
        )
    return lines
