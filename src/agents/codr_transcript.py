# src/agents/codr_transcript.py
"""A codr ticket's opencode session, mirrored into its DERPR channel (DP-484).

Each ticket's session (`derpr:codr:<ticket>`) is copied into the conversation
channel `task:<ticket>` as ordinary interaction rows, so the portal shows it
like any other chat and DERPR keeps it when codr's box is restored:

  - each user message DERPR posted (the ticket prompt, a retry, a nudge)
    becomes a user row;
  - each run of assistant messages between two user messages becomes ONE
    assistant row: every step's reasoning, prose and tool calls (with their
    output) in `tool_context`, in the shape `seal_tool_context` writes, and
    the closing reply as the row's text;
  - a `task` tool call (a subagent) carries the subagent's own session,
    rendered as text, under its result.

Rows are keyed by `platform_message_id` = `opencode:<message id>` (unique in
the table), so mirroring the same messages again updates the rows in place
and never duplicates one. A running turn's row is rewritten as it grows.

Messages the chat path already stores are skipped: a user message carrying
the provider's request key (Adam's message, delivered into the turn or
starting one), and the reply to a turn such a message started.
"""

import asyncio
import hashlib
import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from src.engine.providers.opencode import (
    INJECTED_FIELD, KEY_FIELD, OpencodeClient, is_compaction, message_meta, ticket_channel,
)

logger = logging.getLogger(__name__)

PLATFORM_PREFIX = "opencode:"
AUTHOR_DERPR = "DERPR"
USER_IDENTIFIER = "agent:codr"
SUBAGENT_TOOL = "task"
# A long `cat` or a subagent's whole run would make the row too big for the
# portal to load, so each is cut.
TOOL_OUTPUT_CHARS = 6000
SUBAGENT_CHARS = 20000
SUBAGENT_STEP_CHARS = 600
# A window that holds no user message cannot be split into rows safely (its
# first assistant message may be the middle of one), so look further back.
_WIDE_WINDOW = 20000


@dataclass(frozen=True)
class MirrorRow:
    key: str                    # the opencode message id the row is keyed on
    role: str                   # "user" | "assistant"
    created_ms: int
    content: str
    reasoning: Optional[str] = None
    tool_context: Optional[str] = None


def _role(message: Dict[str, Any]) -> Any:
    return (message.get("info") or {}).get("role")


def _created(message: Dict[str, Any]) -> int:
    return int(((message.get("info") or {}).get("time") or {}).get("created") or 0)


def _text(parts: List[Dict[str, Any]]) -> str:
    return "\n\n".join(
        p.get("text", "").strip() for p in parts
        if p.get("type") == "text" and not p.get("synthetic") and not p.get("ignored") and p.get("text", "").strip()
    )


def _reasoning(parts: List[Dict[str, Any]]) -> str:
    return "\n\n".join(p.get("text", "").strip() for p in parts
                       if p.get("type") == "reasoning" and p.get("text", "").strip())


def _cut(text: str, limit: int) -> str:
    return text if len(text) <= limit else f"{text[:limit]}\n… ({len(text) - limit} more chars)"


def subagent_session_id(part: Dict[str, Any]) -> Optional[str]:
    """The child session a `task` tool call ran (opencode puts it in the
    call's state metadata as soon as the subagent starts)."""
    if part.get("type") != "tool" or part.get("tool") != SUBAGENT_TOOL:
        return None
    sid = ((part.get("state") or {}).get("metadata") or {}).get("sessionId")
    return str(sid) if sid else None


def subagent_ids(msgs: List[Dict[str, Any]]) -> List[str]:
    return [sid for m in msgs for p in m.get("parts", []) if (sid := subagent_session_id(p))]


def render_subagent(msgs: List[Dict[str, Any]]) -> str:
    """A subagent's session as plain text: its reasoning, prose and each tool
    call with the start of its output. The latest work is kept when it is
    too long."""
    lines: List[str] = []
    for m in msgs:
        if _role(m) != "assistant":
            continue
        for p in m.get("parts", []):
            kind = p.get("type")
            if kind == "reasoning" and p.get("text", "").strip():
                lines.append(f"(thinking) {_cut(p['text'].strip(), SUBAGENT_STEP_CHARS)}")
            elif kind == "text" and not p.get("synthetic") and p.get("text", "").strip():
                lines.append(p["text"].strip())
            elif kind == "tool":
                state = p.get("state") or {}
                title = state.get("title") or json.dumps(state.get("input") or {}, ensure_ascii=False)
                out = str(state.get("output") or state.get("error") or f"({state.get('status', 'pending')})")
                lines.append(f"▸ {p.get('tool')}: {_cut(str(title), 200)}\n{_cut(out.strip(), SUBAGENT_STEP_CHARS)}")
    text = "\n\n".join(lines)
    if len(text) > SUBAGENT_CHARS:
        text = f"… (earlier steps cut)\n\n{text[-SUBAGENT_CHARS:]}"
    return text


def _tool_result(part: Dict[str, Any], subagents: Dict[str, str]) -> str:
    state = part.get("state") or {}
    status = state.get("status")
    if status == "error":
        return json.dumps({"error": str(state.get("error") or "error")}, ensure_ascii=False)
    out = _cut(str(state.get("output") or ""), TOOL_OUTPUT_CHARS) if status == "completed" else f"({status or 'pending'}…)"
    sid = subagent_session_id(part)
    if sid and subagents.get(sid):
        out = f"{out}\n\n── subagent session {sid} ──\n\n{subagents[sid]}"
    return out


def _assistant_row(seg: List[Dict[str, Any]], subagents: Dict[str, str]) -> MirrorRow:
    """One row from a run of assistant messages (one per opencode step)."""
    steps: List[Dict[str, Any]] = []
    step_reasoning: List[str] = []
    tail_reasoning: List[str] = []
    replies: List[str] = []
    for m in seg:
        parts = m.get("parts", [])
        reasoning, prose = _reasoning(parts), _text(parts)
        calls = [p for p in parts if p.get("type") == "tool" and p.get("callID")]
        if not calls:
            if prose:
                replies.append(prose)
            if reasoning:
                tail_reasoning.append(reasoning)
            continue
        steps.append({
            "role": "assistant", "content": prose, "reasoning_content": reasoning or None,
            "tool_calls": [{"id": p["callID"], "type": "function", "name": p.get("tool"),
                            "arguments": (p.get("state") or {}).get("input") or {},
                            "function": {"name": p.get("tool"), "arguments": json.dumps(
                                (p.get("state") or {}).get("input") or {}, ensure_ascii=False)}}
                           for p in calls],
        })
        steps += [{"role": "tool", "tool_call_id": p["callID"], "content": _tool_result(p, subagents)}
                  for p in calls]
        if reasoning:
            step_reasoning.append(reasoning)
    # Each step's thinking in order, then what came after the last tool call:
    # the order the transcript peels them back off the cards in.
    reasoning_all = "\n\n".join(step_reasoning + tail_reasoning) or None
    errors = [((m.get("info") or {}).get("error") or {}) for m in seg]
    error = next((e for e in reversed(errors) if e), None)
    content = "\n\n".join(replies)
    if error:
        detail = (error.get("data") or {}).get("message") or ""
        content = f"{content}\n\n[turn error: {error.get('name', 'error')} {detail}]".strip()
    return MirrorRow(key=str((seg[0].get("info") or {}).get("id")), role="assistant",
                     created_ms=_created(seg[0]), content=content, reasoning=reasoning_all,
                     tool_context=json.dumps(steps, ensure_ascii=False) if steps else None)


def fold_session(msgs: List[Dict[str, Any]], subagents: Optional[Dict[str, str]] = None) -> List[MirrorRow]:
    """The rows for `msgs` (oldest first). Starts at the first user message:
    anything before it may be the middle of a row."""
    subagents = subagents or {}
    rows: List[MirrorRow] = []
    seg: List[Dict[str, Any]] = []
    skip = False
    started = False

    def close() -> None:
        if seg and not skip:
            rows.append(_assistant_row(seg, subagents))
        seg.clear()

    for m in msgs:
        role = _role(m)
        if role == "user":
            close()
            started = True
            if message_meta(m, KEY_FIELD) is not None:
                # Adam's message: the chat path stored it. Delivered into a
                # running DERPR turn, that turn carries on; otherwise it
                # started a turn whose reply the chat path stores too.
                if not message_meta(m, INJECTED_FIELD):
                    skip = True
                continue
            text = _text(m.get("parts", []))
            if not text or is_compaction(m):
                continue            # compaction marker or its "continue" prompt: not a new turn (DP-517)
            skip = False
            rows.append(MirrorRow(key=str(m["info"]["id"]), role="user", created_ms=_created(m), content=text))
        elif role == "assistant" and started and not is_compaction(m):
            seg.append(m)
    close()
    return rows


class TranscriptMirror:
    """Copies ticket sessions into their channels (`fold_session`)."""

    def __init__(self, persistence: Any, persona_name: str) -> None:
        self.persistence = persistence          # TurnPersistence: the one history writer
        self.persona_name = persona_name
        self._subagents: Dict[str, Tuple[Any, str]] = {}
        self._stored: Dict[str, str] = {}       # platform id -> digest last written

    async def window(self, client: OpencodeClient, sid: str) -> List[Dict[str, Any]]:
        """The session's recent messages, reaching back to a user message."""
        msgs = await client.turn_messages(sid)
        if not any(_role(m) == "user" for m in msgs):
            msgs = await client.messages(sid, limit=_WIDE_WINDOW)
        return msgs

    async def sync(self, client: OpencodeClient, sid: str, ticket: str,
                   msgs: Optional[List[Dict[str, Any]]] = None) -> int:
        """Mirror `msgs` (default: the session's recent window) into the
        ticket's channel. Returns how many rows were written."""
        if msgs is None:
            msgs = await self.window(client, sid)
        ids = subagent_ids(msgs)
        subagents: Dict[str, str] = {}
        if ids:
            updated = {str(s.get("id")): (s.get("time") or {}).get("updated")
                       for s in await client.request("GET", "/session") or []}
            for child in ids:
                stamp = updated.get(child)
                cached = self._subagents.get(child)
                if cached and stamp is not None and cached[0] == stamp:
                    subagents[child] = cached[1]
                    continue
                text = render_subagent(await client.messages(child, limit=_WIDE_WINDOW))
                self._subagents[child] = (stamp, text)
                subagents[child] = text
        rows = fold_session(msgs, subagents)
        return await asyncio.to_thread(self._store, ticket, rows)

    def _store(self, ticket: str, rows: List[MirrorRow]) -> int:
        channel = ticket_channel(ticket)
        written = 0
        for row in rows:
            pid = f"{PLATFORM_PREFIX}{row.key}"
            digest = hashlib.sha1(json.dumps([row.content, row.reasoning, row.tool_context]).encode()).hexdigest()
            if self._stored.get(pid) == digest:
                continue
            _, changed = self.persistence.upsert_keyed_row(
                platform_message_id=pid, persona_name=self.persona_name, user_identifier=USER_IDENTIFIER,
                channel=channel, role=row.role,
                author_name=AUTHOR_DERPR if row.role == "user" else self.persona_name,
                content=row.content, timestamp=datetime.fromtimestamp(row.created_ms / 1000, tz=timezone.utc),
                reasoning_content=row.reasoning, tool_context_json=row.tool_context)
            written += int(changed)
            self._stored[pid] = digest
        return written
