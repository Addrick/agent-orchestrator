# src/memory/retain_blocks.py
"""Block retain primitives (DP-495) — pure, no I/O.

History leaves the prompt in **blocks**, and each block is retained into
Hindsight as one document (`update_mode="replace"`). This module holds the
parts of that which are plain functions of their inputs:

- `RetainScope` / `retain_scope`: which rows one memory mode's prompt covers,
  and the stable key the window state and block numbering hang off. It mirrors
  `RequestBuilder.fetch_raw_history` exactly.
- `block_document_id`: `<persona>:<scope>:b<NNNN>:<first-message UTC>`.
- `split_block`: a block's rendered turns → one or more Hindsight items, each
  at most the bank's chunk size, each carrying the turns before it as
  labelled background in the item `context`.

The cut itself is planned by `context_budget.plan_block_cut`; the state lives
in `MemoryManager` (`Retain_Window`, `Retain_Block`, `Pending_Retain`); the
delivery and completion polling in `TurnPersistence`.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence

from src.persona import MemoryMode
from src.utils.timeutil import to_utc

# Labels the item `context` so the extractor reads it for context and takes no
# facts from it. Hindsight passes `context` to every chunk's extraction prompt.
CONTEXT_PREAMBLE = (
    "Earlier in this conversation (context only, do not extract facts from it):\n"
)


@dataclass(frozen=True)
class RetainScope:
    """The rows one persona's prompt covers under one memory mode.

    `key` names the scope in `Retain_Window`, `Retain_Block` and document ids.
    The filter fields mirror `fetch_raw_history`: None means "not filtered on",
    except `server_id`, which is filtered whenever `filter_server` is set (and
    then None means IS NULL, as the history getters match it).
    """
    persona_name: str
    mode: str
    key: str
    channel: Optional[str] = None
    user_identifier: Optional[str] = None
    server_id: Optional[str] = None
    filter_server: bool = False

    @property
    def has_window(self) -> bool:
        """`ticket` mode shows no history at all: every turn is its own block."""
        return self.mode != MemoryMode.TICKET_ISOLATED.name


def retain_scope(mode: MemoryMode, *, persona_name: str, channel: str,
                 user_identifier: str, server_id: Optional[str]) -> RetainScope:
    """The scope of a request in `mode` — the same rows `fetch_raw_history`
    returns, so a cut never strands turns another scope still shows, nor
    retains turns still in someone else's prompt."""
    server = server_id or None
    if mode == MemoryMode.GLOBAL:
        return RetainScope(persona_name, mode.name, "global")
    if mode == MemoryMode.PERSONAL:
        return RetainScope(persona_name, mode.name, f"user-{user_identifier}",
                           user_identifier=user_identifier)
    if mode == MemoryMode.SERVER_WIDE:
        return RetainScope(persona_name, mode.name, f"server-{server or 'none'}",
                           server_id=server, filter_server=True)
    if mode == MemoryMode.TICKET_ISOLATED:
        return RetainScope(persona_name, mode.name, channel, channel=channel)
    # CHANNEL_ISOLATED: get_channel_history matches a falsy server_id as NULL.
    key = channel if server is None else f"{channel}@{server}"
    return RetainScope(persona_name, mode.name, key, channel=channel,
                       server_id=server, filter_server=True)


def scope_from_row(row: Dict[str, Any]) -> RetainScope:
    """Rebuild a scope stored in `Retain_Window` (column names as there)."""
    return RetainScope(
        persona_name=row["persona_name"], mode=row["memory_mode"], key=row["scope_key"],
        channel=row["channel"], user_identifier=row["user_identifier"],
        server_id=row["server_id"], filter_server=bool(row["filter_server"]),
    )


def block_document_id(persona_name: str, scope_key: str, seq: int,
                      first_ts: datetime, part: Optional[int] = None) -> str:
    """`stagr:web_ui_pandora:b0007:2026-11-02T00:34Z` — persona, scope, block
    number in that scope, UTC time of the block's first message. Parts of an
    oversized block insert `p<N>` after the number (`b0007p2`)."""
    suffix = f"p{part}" if part is not None else ""
    stamp = to_utc(first_ts).strftime("%Y-%m-%dT%H:%MZ")
    return f"{persona_name}:{scope_key}:b{seq:04d}{suffix}:{stamp}"


def context_tail(turns: Sequence[str], max_chars: int) -> str:
    """The last whole turns of `turns` fitting `max_chars` (at least the
    last turn, cut from its front if it alone is longer)."""
    kept: List[str] = []
    total = 0
    for turn in reversed(turns):
        cost = len(turn) + (2 if kept else 0)
        if kept and total + cost > max_chars:
            break
        kept.append(turn)
        total += cost
    text = "\n\n".join(reversed(kept))
    return text[-max_chars:] if len(text) > max_chars else text


def labelled_context(tail: str) -> Optional[str]:
    return f"{CONTEXT_PREAMBLE}{tail}" if tail.strip() else None


def split_block(turns: Sequence[str], chunk_size: int, *, prior: Sequence[str] = (),
                tail_chars: int) -> List[Dict[str, Any]]:
    """Group a block's rendered turns into parts of at most `chunk_size`
    characters (a single larger turn is a part of its own; Hindsight chunks
    it further). Each part is `{"content", "context"}`: the turns joined in
    order, and the labelled tail of whatever precedes the part — `prior`
    (the turns before the block) for the first, the previous part for the rest.
    `start` is the index in `turns` of the part's first turn.
    """
    starts: List[int] = []
    size = 0
    for i, turn in enumerate(turns):
        if not starts or size + 2 + len(turn) > chunk_size:
            starts.append(i)
            size = len(turn)
        else:
            size += 2 + len(turn)

    out: List[Dict[str, Any]] = []
    before: List[str] = list(prior)
    for n, start in enumerate(starts):
        end = starts[n + 1] if n + 1 < len(starts) else len(turns)
        part = list(turns[start:end])
        out.append({
            "content": "\n\n".join(part),
            "context": labelled_context(context_tail(before, tail_chars)),
            "start": start,
        })
        before = part
    return out
