# src/self_edit/fixr_tools.py
"""Tool handlers for the fixr supervisor persona (DP-227).

Five tools, registered behind the ``fixr`` service binding:

- ``dispatch_fix``   (WRITE → parked): spawn a coding agent for one bug. The
  ConfirmationManager gates this BEFORE the agent starts — the one gate that is
  always on ("gate every dispatch; dial the rest later").
- ``inspect_agents`` (read): the management view over the registry.
- ``answer_agent``   (ungated): resume a waiting agent with a decision. Bounded
  by the already-approved dispatch; gating it would break fixr's woken loop.
- ``kill_agent``     (ungated): stop a stuck/runaway agent — a reduction action.
- ``send_discord``   (ungated): fixr curates + reports on its own judgment (the
  locked autonomy shape). Outbound but low-risk/reversible.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from config import global_config
from src.self_edit.dispatcher import Dispatcher, DispatcherError
from src.self_edit.registry import AgentRecord, AgentRegistry

if TYPE_CHECKING:
    from src.clients.notification import NotificationRouter
    from src.tools.tool_manager import ToolManager

logger = logging.getLogger(__name__)


def _record_view(rec: AgentRecord) -> Dict[str, Any]:
    return {
        "agent_id": rec.agent_id,
        "bug_id": rec.bug_id,
        "status": rec.status,
        "branch": rec.branch,
        "pr_url": rec.pr_url,
        "last_event": rec.last_event,
        "worktree": rec.worktree,
        "has_session": bool(rec.session_id),
    }


class FixrToolHandler:
    def __init__(
        self,
        dispatcher: Dispatcher,
        registry: AgentRegistry,
        notification_router: "NotificationRouter",
    ) -> None:
        self._dispatcher = dispatcher
        self._registry = registry
        self._notifier = notification_router

    def register(self, manager: "ToolManager") -> None:
        manager.register("dispatch_fix", self._dispatch_fix)
        manager.register("inspect_agents", self._inspect_agents)
        manager.register("answer_agent", self._answer_agent)
        manager.register("kill_agent", self._kill_agent)
        manager.register("send_discord", self._send_discord)

    async def _dispatch_fix(self, bug_id: str, description: str) -> Dict[str, Any]:
        logger.info("Tool dispatch_fix: %s", bug_id)
        try:
            rec = await self._dispatcher.dispatch(bug_id, description)
        except DispatcherError as e:
            return {"status": "error", "message": str(e)}
        return {"status": "dispatched", "agent": _record_view(rec)}

    async def _inspect_agents(
        self, agent_id: Optional[str] = None, active_only: bool = False,
    ) -> Dict[str, Any]:
        if agent_id:
            rec = await self._registry.get(agent_id)
            if rec is None:
                return {"found": False, "agent_id": agent_id}
            return {"found": True, "agent": _record_view(rec)}
        recs: List[AgentRecord] = await self._registry.list(active_only=active_only)
        return {"count": len(recs), "agents": [_record_view(r) for r in recs]}

    async def _answer_agent(self, agent_id: str, message: str) -> Dict[str, Any]:
        logger.info("Tool answer_agent: %s", agent_id)
        try:
            rec = await self._dispatcher.answer_agent(agent_id, message)
        except DispatcherError as e:
            return {"status": "error", "message": str(e)}
        return {"status": "resumed", "agent": _record_view(rec)}

    async def _kill_agent(
        self, agent_id: str, remove_worktree: bool = False,
    ) -> Dict[str, Any]:
        logger.info("Tool kill_agent: %s", agent_id)
        ok = await self._dispatcher.kill(agent_id, remove_worktree=remove_worktree)
        return {"status": "killed" if ok else "not_found", "agent_id": agent_id}

    async def _send_discord(
        self, subject: str, body: str, recipient: Optional[str] = None,
    ) -> Dict[str, Any]:
        target = recipient or global_config.CC_FIXR_DISCORD_CHANNEL
        if not target:
            return {
                "status": "error",
                "message": "No Discord recipient: pass `recipient` or set CC_FIXR_DISCORD_CHANNEL.",
            }
        ok = await self._notifier.send("discord", str(target), subject, body)
        return {"status": "sent" if ok else "failed", "recipient": str(target)}
