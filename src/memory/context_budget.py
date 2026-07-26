# src/memory/context_budget.py
"""Token-budget enforcement shared between chat_system and the kobold engine adapter.

`max_context_tokens` is a persona setting that caps the *total* context
(prompt + reserved response), matching kobold-lite's
`localsettings.max_context_length` semantic. Effective prompt-prune budget
is therefore `max_context_tokens - response_token_limit`.

Today this module only does char/4 estimation + drop-oldest pruning;
future work documented in `memory/project/plans/web_ui_roadmap.md`
(dynamic LTM-depth modulation, real tokenizer swap, retrieval-score-weighted
budget allocation) lands here.
"""

from typing import Any, Dict, List, Optional, Tuple


def estimate_tokens(text: str) -> int:
    """Char/4 token estimate. Empty string → 0.

    Cheap and deterministic — no tokenizer roundtrip. Replace with a real
    tokenizer when char/4 drift becomes load-bearing (likely first observed
    on long CJK content).
    """
    if not text:
        return 0
    return (len(text) + 3) // 4


def _message_tokens(msg: Dict[str, Any]) -> int:
    """Token cost of a single OAI-style message dict.

    Sums `content` (string or list-of-parts) token estimates. Tool/function
    payloads and role overhead are not modelled — char/4 already absorbs the
    rounding error.
    """
    content = msg.get("content")
    if isinstance(content, str):
        return estimate_tokens(content)
    if isinstance(content, list):
        total = 0
        for part in content:
            if not isinstance(part, dict):
                continue
            t = part.get("text")
            if isinstance(t, str):
                total += estimate_tokens(t)
        return total
    return 0


def truncate_messages_to_budget(
    messages: List[Dict[str, Any]],
    max_tokens: Optional[int],
) -> Tuple[List[Dict[str, Any]], int]:
    """Drop oldest non-system messages until total ≤ `max_tokens`.

    Preserves: every `role == "system"` entry, and the most recent
    `role == "user"` entry (so the current turn is never evicted).
    LTM authornote sits inside system / latest-user content by the time
    this runs and is therefore preserved implicitly.

    Returns `(pruned_messages, dropped_count)`. No-op (`dropped_count=0`)
    when `max_tokens` is None / non-positive, or when the input already
    fits the budget. If the preserved set alone exceeds the budget, returns
    just the preserved set with the dropped count of everything else.
    """
    if max_tokens is None or max_tokens <= 0:
        return messages, 0

    total = sum(_message_tokens(m) for m in messages)
    if total <= max_tokens:
        return messages, 0

    last_user_idx: Optional[int] = None
    for i in range(len(messages) - 1, -1, -1):
        if messages[i].get("role") == "user":
            last_user_idx = i
            break

    kept: List[Dict[str, Any]] = []
    dropped = 0
    running = total
    for i, msg in enumerate(messages):
        is_system = msg.get("role") == "system"
        is_last_user = (i == last_user_idx)
        if running <= max_tokens:
            kept.append(msg)
            continue
        if is_system or is_last_user:
            kept.append(msg)
            continue
        running -= _message_tokens(msg)
        dropped += 1

    return kept, dropped


def drop_orphaned_tool_head(
    messages: List[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], int]:
    """Drop a tool sequence left dangling at the head of `messages`.

    A tool turn is three messages long — `user`, `assistant` (with
    `tool_calls`), `tool` (the result) — but every truncation this codebase
    performs counts *messages*, not turns: the DB sliding window
    (`fetch_raw_history`'s row `LIMIT`, expanded into the full sequence by
    `format_raw_history_for_llm`) and `truncate_messages_to_budget` above both
    cut oldest-first and can land mid-sequence. That leaves the history
    *starting* on an `assistant`-with-`tool_calls` or a `tool` message, whose
    triggering user turn is gone.

    Providers reject that shape: Google returns 400 "function call turn must
    come immediately after a user turn" (a `model` Content whose first part is a
    `function_call`, with nothing before it), and the OpenAI/Anthropic wire
    formats reject an unpaired tool result the same way. Since the orphan's
    context is already gone, the repair is to drop it — no valid prompt can be
    built from a half turn.

    Returns `(repaired_messages, dropped_count)`.
    """
    start = 0
    for msg in messages:
        role = msg.get("role")
        if role == "tool" or (role == "assistant" and msg.get("tool_calls")):
            start += 1
            continue
        break
    if start == 0:
        return messages, 0
    return messages[start:], start
