# src/self_edit/events.py
"""Common, platform-ambivalent event schema for dispatched coding agents (DP-227).

A dispatched agent (Claude Code today, agy/others later) writes a platform-native
log; a per-platform *adapter* converts each native line into the common
``AgentEvent`` below. The bridge tails the native log, runs it through the
adapter, and appends the resulting ``AgentEvent``s to a common JSONL audit log.

Event types
-----------
- ``started``  — the agent process is up (carries session_id, branch, pid).
- ``progress`` — incremental activity (assistant text); logged, never wakes fixr.
- ``question`` — the agent needs a decision; carries session_id for `claude --resume`.
- ``done``     — the agent finished and (usually) opened a PR; carries pr_url/summary.
- ``error``    — the agent failed/blocked.

``WAKE_TYPES`` are the ones that enqueue a fixr turn. ``progress``/``started`` are
recorded only.

Hybrid signalling (the locked design): process *lifecycle* is read from the
platform stream; *question/done/error semantics* come from a sentinel the agent
prints in its final message (``FIXR_QUESTION:`` / ``FIXR_DONE:`` / ``FIXR_ERROR:``).
"""

from __future__ import annotations

import json
import logging
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Type

logger = logging.getLogger(__name__)

# Event type constants (str, so they round-trip through JSONL transparently).
STARTED = "started"
PROGRESS = "progress"
QUESTION = "question"
DONE = "done"
ERROR = "error"

#: Event types that wake fixr (enqueue a supervisor turn). The rest are logged.
WAKE_TYPES = frozenset({QUESTION, DONE, ERROR})

#: Sentinels the dispatched agent prints in its FINAL message so the adapter can
#: classify the terminal result. Kept here so prompt + adapter never drift.
SENTINEL_QUESTION = "FIXR_QUESTION:"
SENTINEL_DONE = "FIXR_DONE:"
SENTINEL_ERROR = "FIXR_ERROR:"


@dataclass
class AgentEvent:
    """One common-schema event. ``payload`` is free-form per type but conventionally
    carries: text (progress/question/error), session_id (started/question/done),
    pr_url + summary (done), branch/pid (started), detail (error)."""

    agent_id: str
    seq: int
    type: str
    payload: Dict[str, Any] = field(default_factory=dict)
    ts: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_jsonl(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)

    @classmethod
    def from_jsonl(cls, line: str) -> "AgentEvent":
        d = json.loads(line)
        return cls(
            agent_id=d["agent_id"],
            seq=int(d.get("seq", 0)),
            type=d["type"],
            payload=d.get("payload", {}) or {},
            ts=d.get("ts") or datetime.now(timezone.utc).isoformat(),
        )

    @property
    def is_wake(self) -> bool:
        return self.type in WAKE_TYPES

    @property
    def is_terminal(self) -> bool:
        """done/error end the agent's life; the bridge stops tailing after one."""
        return self.type in (DONE, ERROR)


def _truncate(text: str, limit: int = 2000) -> str:
    text = text.strip()
    return text if len(text) <= limit else text[:limit] + "…"


class DispatchAdapter(ABC):
    """Converts one platform-native log line into zero or more ``AgentEvent``s.

    Stateful: an instance is created per dispatched agent and may stash context
    (e.g. the session_id seen on init) to enrich later events. ``parse_line``
    returns ``[]`` for lines that carry no fixr-relevant signal."""

    #: short platform id, used to select the adapter for a dispatch.
    name: str = "base"

    def __init__(self, agent_id: str) -> None:
        self.agent_id = agent_id
        self._seq = 0
        self.session_id: Optional[str] = None

    def _next(self, type_: str, payload: Dict[str, Any]) -> AgentEvent:
        ev = AgentEvent(
            agent_id=self.agent_id, seq=self._seq, type=type_, payload=payload
        )
        self._seq += 1
        return ev

    @abstractmethod
    def parse_line(self, raw_line: str) -> List[AgentEvent]:
        ...


class ClaudeStreamAdapter(DispatchAdapter):
    """Adapts Claude Code's ``--output-format stream-json`` lines.

    Lifecycle from the stream (system/init -> started, assistant text ->
    progress, result -> terminal); question/done/error classified from the
    FIXR_* sentinel in the final ``result`` text. Unknown / malformed lines are
    ignored — the adapter never raises on a line it can't read, so one bad line
    can't wedge the bridge."""

    name = "claude"

    def parse_line(self, raw_line: str) -> List[AgentEvent]:
        raw_line = raw_line.strip()
        if not raw_line:
            return []
        try:
            obj = json.loads(raw_line)
        except (json.JSONDecodeError, ValueError):
            logger.debug("[%s] non-JSON stream line ignored: %.120s", self.agent_id, raw_line)
            return []
        if not isinstance(obj, dict):
            return []

        # Capture session_id wherever it appears so resume always has it.
        sid = obj.get("session_id")
        if isinstance(sid, str) and sid:
            self.session_id = sid

        kind = obj.get("type")
        if kind == "system" and obj.get("subtype") == "init":
            return [self._next(STARTED, {"session_id": self.session_id})]
        if kind == "assistant":
            text = self._assistant_text(obj)
            if text:
                return [self._next(PROGRESS, {"text": _truncate(text)})]
            return []
        if kind == "result":
            return [self._classify_result(obj)]
        return []

    @staticmethod
    def _assistant_text(obj: Dict[str, Any]) -> str:
        msg = obj.get("message") or {}
        content = msg.get("content")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts = [
                b.get("text", "")
                for b in content
                if isinstance(b, dict) and b.get("type") == "text"
            ]
            return "\n".join(p for p in parts if p)
        return ""

    def _classify_result(self, obj: Dict[str, Any]) -> AgentEvent:
        result_text = ""
        rt = obj.get("result")
        if isinstance(rt, str):
            result_text = rt.strip()
        is_error = bool(obj.get("is_error")) or str(obj.get("subtype", "")).startswith("error")

        base: Dict[str, Any] = {"session_id": self.session_id}

        if result_text.startswith(SENTINEL_QUESTION):
            base["text"] = result_text[len(SENTINEL_QUESTION):].strip()
            return self._next(QUESTION, base)
        if result_text.startswith(SENTINEL_ERROR) or is_error:
            body = result_text[len(SENTINEL_ERROR):].strip() if result_text.startswith(SENTINEL_ERROR) else result_text
            base["text"] = body or f"agent ended with error ({obj.get('subtype')})"
            base["detail"] = obj.get("subtype")
            return self._next(ERROR, base)
        # DONE (explicit sentinel or a clean finish without one).
        body = result_text[len(SENTINEL_DONE):].strip() if result_text.startswith(SENTINEL_DONE) else result_text
        base["summary"] = body
        base["pr_url"] = _extract_pr_url(body)
        return self._next(DONE, base)


def _extract_pr_url(text: str) -> Optional[str]:
    """Pull the first GitHub PR URL out of the agent's final message, if any."""
    import re
    m = re.search(r"https://github\.com/\S+/pull/\d+", text)
    return m.group(0) if m else None


#: Adapter registry — add agy/other platforms here. Keyed by platform name.
ADAPTERS: Dict[str, Type[DispatchAdapter]] = {
    ClaudeStreamAdapter.name: ClaudeStreamAdapter,
}


def get_adapter(platform: str, agent_id: str) -> DispatchAdapter:
    """Construct the adapter for ``platform`` (defaults to claude)."""
    cls = ADAPTERS.get(platform, ClaudeStreamAdapter)
    return cls(agent_id)
