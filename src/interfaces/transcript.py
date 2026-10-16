# src/interfaces/transcript.py

"""DERPR message_history → history-contract transcript.

`build_transcript` is the DP-130 history-contract projection: an ordered list of
chunks, each addressed by a server-authored `interaction_id` (or flagged
`ephemeral` for a not-yet-persisted parked confirmation). The `/derpr` portal
renders from it — no consumer ever shadows the story positionally.

See memory/project/decisions/2026-06-02-portal-history-contract.md (C1–C5).
"""

import json
import logging
from typing import Any, Dict, List, Optional, Set

logger = logging.getLogger(__name__)


def _is_renderable(role: Optional[str], content: str, reasoning: str) -> bool:
    """A DB row becomes a visible story chunk iff it is a user/assistant turn
    with some content or reasoning. System rows, empty rows, and tool-call-only
    assistant rows (no content, no reasoning) are skipped — they have no chunk.
    """
    if role not in ("user", "assistant"):
        return False
    return bool(content or reasoning)


def _parse_tool_context(raw: Any) -> Optional[Any]:
    """tool_context is stored as a JSON string (or None). Return the parsed
    structure for the transcript, or None when absent/unparseable.
    Transforms raw OpenAI message dicts into the frontend ToolContext shape."""
    if not raw:
        return None
    try:
        if isinstance(raw, str):
            msgs = json.loads(raw)
        else:
            msgs = raw
            
        if not isinstance(msgs, list):
            return msgs
            
        contexts = {}
        for msg in msgs:
            if not isinstance(msg, dict):
                continue
            if msg.get("role") == "assistant" and "tool_calls" in msg:
                for call in msg.get("tool_calls", []):
                    if not isinstance(call, dict):
                        continue
                    call_id = call.get("id")
                    if call_id:
                        args_val = call.get("arguments", {})
                        if isinstance(args_val, str):
                            try:
                                args_val = json.loads(args_val)
                            except json.JSONDecodeError:
                                pass
                                
                        tool_name = call.get("name")
                        if not tool_name and "function" in call:
                            tool_name = call["function"].get("name")
                            
                        contexts[call_id] = {
                            "call_id": call_id,
                            "group_id": call.get("group_id"),
                            "tool_name": tool_name,
                            "arguments": args_val if isinstance(args_val, dict) else {},
                            "result": None,
                            "error": None,
                        }
            elif msg.get("role") == "tool":
                call_id = msg.get("tool_call_id")
                if call_id and call_id in contexts:
                    content_str = msg.get("content", "")
                    contexts[call_id]["result"] = content_str
                    try:
                        parsed = json.loads(content_str)
                        if isinstance(parsed, dict) and "error" in parsed:
                            contexts[call_id]["error"] = str(parsed["error"])
                    except Exception:
                        pass
        
        if contexts:
            return list(contexts.values())
        # Nothing resolved. Distinguish raw OpenAI messages (carry a `role`
        # key) that failed to resolve — orphaned/truncated call_ids — from an
        # already-structured ToolContext[] passed straight through. Returning
        # the raw {role, content} list would feed the ToolCard objects lacking
        # call_id/tool_name/arguments → garbled panel (DP-143).
        if any(isinstance(m, dict) and "role" in m for m in msgs):
            return None
        return msgs
    except (TypeError, ValueError, AttributeError):
        return None


def build_transcript(
    raw_history: List[Dict[str, Any]],
    *,
    ids_with_versions: Optional[Set[int]] = None,
    pending: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Project DERPR history rows into the DP-130 transcript contract.

    Returns `{"chunks": [...]}`. Each chunk is one rendered story turn:

        {
          "interaction_id": <int|null>,   # server-authored identity
          "role": "user|assistant",
          "content": "...",               # the reply alone; reasoning is separate
          "ephemeral": <bool>,            # true => not yet persisted
          "reasoning": "<str|null>",
          "tool_context": [...]|null,
          "has_versions": <bool>,         # regen/edit archives exist
        }

    **Invariant C1:** every chunk has exactly one `interaction_id` OR
    `ephemeral=true` (never both, never neither).

    `ids_with_versions` marks which interaction ids carry edit/regen archives
    (drives the chevron affordance). `pending` is the live gated writes for
    this session — each appended as a trailing ephemeral chunk
    (`ephemeral=true`, `interaction_id=null`) carrying its
    `ephemeral_chunk_id`, so a fresh load can render every awaiting-approval
    affordance without a DB row (invariant C3 on the projection side).

    DP-297 made `pending` a list. It was a single optional dict back when a
    conversation could hold only one parked write; rendering just one of
    several now would leave the rest unanswerable after a reload.
    """
    versions = ids_with_versions or set()
    chunks: List[Dict[str, Any]] = []

    for msg in raw_history:
        role = msg.get("author_role")
        content = (msg.get("content") or "").strip()
        reasoning = (msg.get("reasoning_content") or "").strip()

        if not _is_renderable(role, content, reasoning):
            continue

        iid = msg.get("interaction_id")
        iid = iid if isinstance(iid, int) else None
        chunks.append({
            "interaction_id": iid,
            "role": role,
            "content": content,
            "ephemeral": False,
            "reasoning": reasoning or None,
            "tool_context": _parse_tool_context(msg.get("tool_context")),
            "has_versions": iid in versions if iid is not None else False,
        })

    for park in (pending or []):
        chunks.append({
            "interaction_id": None,
            "ephemeral_chunk_id": park.get("ephemeral_chunk_id"),
            "role": "assistant",
            "content": park.get("content") or "",
            "ephemeral": True,
            "reasoning": None,
            "tool_context": park.get("tool_context"),
            "has_versions": False,
        })

    return {"chunks": chunks}
