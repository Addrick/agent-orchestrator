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
  - extracting the first complete `<tool_call>…</tool_call>` block from text,
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


def extract_first_tool_call_block(text: str) -> Optional[str]:
    """Return the inner (stripped) payload of the first complete
    `<tool_call>…</tool_call>` block in `text`, or None if there is none.

    Used by the complete-response path. The streaming path locates blocks
    incrementally instead, but decodes each block's payload via
    `decode_tool_call_payload`, so both share the same JSON semantics.
    """
    if not text:
        return None
    match = _TOOL_CALL_BLOCK_RE.search(text)
    if not match:
        return None
    return match.group(1).strip()


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
