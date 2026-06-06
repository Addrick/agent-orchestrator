# src/interfaces/kobold_export.py

"""DERPR message_history → kobold-lite savefile JSON + history-contract transcript.

`build_kobold_savefile` builds the v1 'oldui' savefile shape kobold-lite ingests
via its public load_file path. We wrap user turns with kobold's instruct
placeholder tags (`{{[INPUT]}}` / `{{[OUTPUT]}}`) so kobold's own template
renderer expands them at submit time — DERPR never picks the actual instruct tags.

`build_transcript` is the DP-130 history-contract projection: an ordered list of
chunks, each addressed by a server-authored `interaction_id` (or flagged
`ephemeral` for a not-yet-persisted parked confirmation). It is the single
projection source both the Lite re-sync (DP-131) and the bespoke UI (DP-132+)
render from — no consumer ever shadows the story positionally.

See memory/project/decisions/2026-06-02-portal-history-contract.md (C1–C5) and
memory/project/decisions/2026-04-19-portal-phase2-approach.md.
"""

import json
import logging
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)

# Mirror kobold-lite's instructstartplaceholder / instructendplaceholder
# constants from portal.html. These are stable cross-version tokens; kobold
# substitutes them with the active instruct_starttag/endtag at render time.
_INPUT_PLACEHOLDER = "\n{{[INPUT]}}\n"
_OUTPUT_PLACEHOLDER = "\n{{[OUTPUT]}}\n"


def _is_renderable(role: Optional[str], content: str, reasoning: str) -> bool:
    """A DB row becomes a visible story chunk iff it is a user/assistant turn
    with some content or reasoning. System rows, empty rows, and tool-call-only
    assistant rows (no content, no reasoning) are skipped — they have no chunk.
    """
    if role not in ("user", "assistant"):
        return False
    return bool(content or reasoning)


def _merge_reasoning(role: Optional[str], content: str, reasoning: str) -> str:
    """Fold an assistant row's reasoning into its rendered content as a
    <think> block (kobold/Lite convention; matches list_interaction_versions)."""
    if reasoning and role == "assistant":
        return f"<think>\n{reasoning}\n</think>\n{content}"
    return content


def build_kobold_savefile(
    raw_history: List[Dict[str, Any]],
) -> Tuple[Dict[str, Any], int]:
    """Translate DERPR User_Interactions rows into a kobold-lite savefile dict.

    Returns (savefile_dict, skipped_count). Skipped count covers system rows,
    empty-content rows, assistant rows whose only payload is a tool-call
    (tool_context present, content empty), and unrecognised roles.

    **Invariant C2 (DP-130) — gametext alignment:** `interaction_ids` carries
    exactly one entry per *visible story chunk*, in order, including the opening
    `prompt`. So `len(interaction_ids) == len(actions) + 1` whenever there is a
    prompt (and equals kobold-lite's `gametext_arr` length, which is
    `[prompt, *actions]`). This is the invariant that prevents drift: the portal
    keys `derpr_interaction_ids[modified_turn]` by the gametext index — index 0
    is the prompt — and the SSE id-frame path likewise pushes one id per visible
    chunk. The id may be `None` for an unaddressable renderable row (the portal
    guards `if (interactionId)`), but the *slot* is always present so the array
    can never desync from the story. The earlier `actions=12 ids=13` shape was
    in fact correct gametext alignment — the real defect was a conditional
    id-append that could drop an id without dropping its chunk; fixed here by
    appending a slot for every renderable row and never popping the id array.

    Tool-call / tool-result rendering is explicitly out of scope for Phase 2.1
    — see web_ui_roadmap backlog. Persona system prompt is pushed into
    kobold-lite's `instruct_sysprompt` setting by the UI, not into savefile
    `memory`, so the memory block stays free for future use.
    """
    # One entry per visible story chunk, in order: `rendered[0]` is the prompt,
    # `rendered[1:]` are the actions. `interaction_ids` stays 1:1 with `rendered`
    # (= gametext_arr) — never popped — so it is gametext-aligned, not
    # actions-aligned. Optional[int]: a renderable row could lack an int id.
    rendered: List[str] = []
    interaction_ids: List[Optional[int]] = []
    skipped = 0

    for msg in raw_history:
        role = msg.get("author_role")
        content = (msg.get("content") or "").strip()
        reasoning = (msg.get("reasoning_content") or "").strip()

        if not _is_renderable(role, content, reasoning):
            skipped += 1
            continue

        content = _merge_reasoning(role, content, reasoning)

        if role == "user":
            rendered.append(f"{_INPUT_PLACEHOLDER}{content}{_OUTPUT_PLACEHOLDER}")
        else:  # assistant
            rendered.append(content)

        iid = msg.get("interaction_id")
        interaction_ids.append(iid if isinstance(iid, int) else None)

    # The opening chunk is kobold's `prompt` (story opener, separate from
    # `actions`), but its id stays at `interaction_ids[0]` so the id array
    # remains 1:1 with `gametext_arr` ([prompt, *actions]).
    prompt = rendered[0] if rendered else ""
    actions = rendered[1:]

    savefile: Dict[str, Any] = {
        "gamestarted": True,
        "prompt": prompt,
        "memory": "",
        "authorsnote": "",
        "anotetemplate": "",
        "actions": actions,
        "interaction_ids": interaction_ids,
        "actions_metadata": {},
        "worldinfo": [],
        "wifolders_d": {},
        "wifolders_l": [],
    }
    return savefile, skipped


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
            if msg.get("role") == "assistant" and "tool_calls" in msg:
                for call in msg.get("tool_calls", []):
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
        return msgs
    except (TypeError, ValueError):
        return None


def build_transcript(
    raw_history: List[Dict[str, Any]],
    *,
    ids_with_versions: Optional[Set[int]] = None,
    pending: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Project DERPR history rows into the DP-130 transcript contract.

    Returns `{"chunks": [...]}`. Each chunk is one rendered story turn:

        {
          "interaction_id": <int|null>,   # server-authored identity
          "role": "user|assistant",
          "content": "...",               # reasoning folded into <think> block
          "ephemeral": <bool>,            # true => not yet persisted
          "reasoning": "<str|null>",
          "tool_context": [...]|null,
          "has_versions": <bool>,         # regen/edit archives exist
        }

    **Invariant C1:** every chunk has exactly one `interaction_id` OR
    `ephemeral=true` (never both, never neither).

    `ids_with_versions` marks which interaction ids carry edit/regen archives
    (drives the chevron affordance). `pending`, when supplied, is the live
    parked confirmation for this session — appended as a trailing ephemeral
    chunk (`ephemeral=true`, `interaction_id=null`) carrying its
    `ephemeral_chunk_id`, so a fresh load can render the awaiting-approval text
    without a DB row (invariant C3 on the projection side).
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
            "content": _merge_reasoning(role, content, reasoning),
            "ephemeral": False,
            "reasoning": reasoning or None,
            "tool_context": _parse_tool_context(msg.get("tool_context")),
            "has_versions": iid in versions if iid is not None else False,
        })

    if pending is not None:
        chunks.append({
            "interaction_id": None,
            "ephemeral_chunk_id": pending.get("ephemeral_chunk_id"),
            "role": "assistant",
            "content": pending.get("content") or "",
            "ephemeral": True,
            "reasoning": None,
            "tool_context": pending.get("tool_context"),
            "has_versions": False,
        })

    return {"chunks": chunks}
