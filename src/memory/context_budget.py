# src/memory/context_budget.py
"""Token-budget enforcement shared between chat_system and the kobold engine adapter.

`max_context_tokens` is a persona setting that caps the *total* context
(prompt + reserved response), matching KoboldCPP's `max_context_length`
semantic. Effective prompt-prune budget
is therefore `max_context_tokens - response_token_limit`.

Today this module only does char/4 estimation + drop-oldest pruning;
future work documented in `memory/project-history/plans/web_ui_roadmap.md`
(dynamic LTM-depth modulation, real tokenizer swap, retrieval-score-weighted
budget allocation) lands here.
"""

import json
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


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

    Sums `content` (string or list-of-parts) token estimates, plus serialized
    `tool_calls` when present. Role overhead is not modelled — char/4 already
    absorbs the rounding error.

    `tool_calls` has to be counted: a call-only assistant message carries no
    `content` key at all, so scoring it 0 would let the pruner drop it without
    reducing the running total while its (non-empty) `tool` results survive.
    `content` is counted beside them: since DP-338 a tool-call message may also
    carry the prose the model wrote with its batch, which is replayed on every
    later turn (DP-441).
    """
    total = 0
    if msg.get("tool_calls"):
        total += estimate_tokens(json.dumps(msg["tool_calls"], default=str))
    content = msg.get("content")
    if isinstance(content, str):
        total += estimate_tokens(content)
    elif isinstance(content, list):
        for part in content:
            if not isinstance(part, dict):
                continue
            t = part.get("text")
            if isinstance(t, str):
                total += estimate_tokens(t)
    return total


def message_tokens(msg: Dict[str, Any]) -> int:
    """Public name for one message's estimated token cost (DP-495 cut)."""
    return _message_tokens(msg)


def truncate_messages_to_budget(
    messages: List[Dict[str, Any]],
    max_tokens: Optional[int],
) -> Tuple[List[Dict[str, Any]], int]:
    """Drop oldest non-system messages until total ≤ `max_tokens`.

    Preserves: every `role == "system"` entry, the most recent
    `role == "user"` entry (so the current turn is never evicted), and a
    trailing assistant message without `tool_calls` — a prefill the model
    resumes (DP-447 continue), which is the current turn too: dropping it
    would make the server answer afresh, and that answer would be stored as
    the continuation. The LTM block rides inside the latest user turn
    (DP-431) and is therefore preserved implicitly.

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
    tail = messages[-1]
    prefill_idx = len(messages) - 1 \
        if tail.get("role") == "assistant" and not tail.get("tool_calls") else None

    kept: List[Dict[str, Any]] = []
    dropped = 0
    running = total
    n = len(messages)
    i = 0
    while i < n:
        msg = messages[i]
        # An assistant message carrying `tool_calls` and the `tool` messages
        # answering it are one atomic unit. Cutting between them leaves an
        # unpaired block at the head of the wire array, and both Anthropic and
        # Gemini reject the whole request for it. DP-296 made such blocks
        # routine — every park and every errored turn now persists one — so a
        # pairing-blind pruner would fail most long conversations, not just
        # those that used tools successfully.
        end = i + 1
        if msg.get("role") == "assistant" and msg.get("tool_calls"):
            while end < n and messages[end].get("role") == "tool":
                end += 1
        span = messages[i:end]

        is_system = msg.get("role") == "system"
        is_last_user = (i == last_user_idx)
        if running <= max_tokens or is_system or is_last_user or i == prefill_idx:
            kept.extend(span)
        else:
            running -= sum(_message_tokens(m) for m in span)
            dropped += len(span)
        i = end

    return kept, dropped


def plan_block_cut(
    rows: Sequence[Tuple[bool, int]],
    *,
    budget: Optional[int],
    fixed_tokens: int = 0,
    keep_at_most: Optional[int] = None,
    high: float = 0.9,
    low: float = 0.5,
    trigger: bool = True,
) -> Optional[int]:
    """How many of the oldest window rows to cut as one block (DP-495).

    `rows` are the prompt's history rows that are not yet in a block, oldest
    first, each `(starts_turn, tokens)`: `starts_turn` marks a row the remaining
    window may begin on (a user turn — a row's tool calls and their results are
    one row, so a row boundary never splits them). `fixed_tokens` is what the
    prompt carries besides history (the new user message, recalled memory).

    Token watermark: when `fixed + window` exceeds `high` of `budget` (and
    `trigger` is set — a new user message arrived), cut the fewest rows that
    bring it to `low` of `budget`. `trigger=False` skips the high check (the
    48 h idle trim), cutting to `low` only.

    History already cut into a block that Hindsight has not finished with is
    still shown but not counted: it is on its way out, and counting it would
    cut a fresh small block every turn while Hindsight is slow — the
    one-turn retains this exists to end. If it pushes the prompt past the
    budget, the caller's oldest-first fallback drops it for that request.

    Count cap: `keep_at_most` (the caller passes half a message-count cap
    that is cutting rows off) cuts the window to at most that many rows, so
    a cap that binds trims in blocks instead of sliding.

    The cut always ends where the remaining window starts on a turn (or at
    the end: everything). Returns the row count to cut, or None for no cut.
    """
    n = len(rows)
    if n == 0:
        return None
    boundaries = [k for k in range(1, n) if rows[k][0]] + [n]
    window = sum(t for _, t in rows)
    suffix = [0] * (n + 1)
    for k in range(n - 1, -1, -1):
        suffix[k] = suffix[k + 1] + rows[k][1]

    cut = 0
    if budget is not None and budget > 0:
        over = fixed_tokens + window > high * budget
        target = low * budget
        if (over or not trigger) and fixed_tokens + window > target:
            cut = next((k for k in boundaries if fixed_tokens + suffix[k] <= target), n)
    if keep_at_most is not None and n > keep_at_most:
        cut = max(cut, next(k for k in boundaries if n - k <= keep_at_most))
    return cut or None


def drop_orphaned_tool_head(
    messages: List[Dict[str, Any]],
    injected: Optional[Iterable[Dict[str, Any]]] = None,
) -> Tuple[List[Dict[str, Any]], int]:
    """Drop a tool sequence left dangling at the head of `messages`.

    A tool turn is three messages long — `user`, `assistant` (with
    `tool_calls`), `tool` (the result) — while the windows that trim history
    count *messages*, not turns, so a cut can land mid-sequence and leave the
    history *starting* on an `assistant`-with-`tool_calls` or a `tool` message
    whose triggering user turn is gone.

    Two of the three producers of that shape are now fixed upstream, and this
    is the backstop rather than the primary defense for them:

    * `truncate_messages_to_budget` above treats an `assistant`+`tool_calls`
      message and its `tool` answers as one atomic span (DP-296), so the token
      pruner no longer splits a pair;
    * `build_conversation_history` refuses to replay an assistant row's
      `tool_context` unless a `user` or `tool` turn precedes it (DP-296), which
      closes the DB sliding-window case at the source.

    The path where this function is still load-bearing is `client_messages`
    (a non-streaming OAI client supplies its own array, which `prepare_request` takes verbatim
    apart from stripping one leading system message) — neither guard above runs
    on it. Keeping the repair as the last stop before the wire also means any
    future producer of an unpaired head fails safe.

    Providers reject that shape: Google returns 400 "function call turn must
    come immediately after a user turn" for a leading `function_call`, and
    "function response turn must come immediately after a function call turn"
    for a leading bare result. Since the orphan's context is already gone, the
    repair is to drop it — no valid prompt can be built from a half turn.

    Two kinds of message may sit *in front* of the orphan without making it
    well-formed, and must be scanned through rather than treated as the head:

    * `role == "system"` entries, which `truncate_messages_to_budget`
      deliberately preserves while dropping the conversation around them, and
      of which the `client_messages` path strips only one;
    * anything in `injected` — head content the request builder prepends after
      truncation (the long-term-memory recall block, a `user` message that is
      not a real conversational turn). Matched by **identity**, so a caller can
      pass a block that may or may not have survived pruning and get the right
      answer either way.

    Neither is a turn the orphan can legally attach to, so skipping them and
    continuing the scan is what makes the repair fire in the cases that
    actually reach a provider.

    Returns `(repaired_messages, dropped_count)`. The skipped preamble is
    preserved in the result; only the orphaned tool messages are removed.
    """
    injected_ids = {id(m) for m in (injected or ())}

    # Walk past preamble that cannot serve as the orphan's originating turn.
    preamble = 0
    for msg in messages:
        if msg.get("role") == "system" or id(msg) in injected_ids:
            preamble += 1
            continue
        break

    start = preamble
    for msg in messages[preamble:]:
        role = msg.get("role")
        if role == "tool" or (role == "assistant" and msg.get("tool_calls")):
            start += 1
            continue
        break

    dropped = start - preamble
    if dropped == 0:
        return messages, 0
    return messages[:preamble] + messages[start:], dropped
