# src/proposals/service.py

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from src.clients.service_integration import ServiceIntegration
from src.proposals.executor import ProposalExecutor

if TYPE_CHECKING:
    from src.memory.memory_manager import MemoryManager
    from src.tools.tool_manager import ToolManager

logger = logging.getLogger(__name__)

# Single shared operator identity, per the DP-277 decision (one static
# control token / one human). Recorded on reviews and audit rows.
OPERATOR_ID = "operator"


class ProposalIntegration(ServiceIntegration):
    """
    Service integration for the proposal queue (DP-282).

    Personas with service_bindings: ["proposals"] (e.g. joy) gain the human
    review surface: list_proposals / approve_proposal / deny_proposal.
    approve_proposal is a write tool, so it rides the existing universal
    write gate before anything executes.
    """

    def __init__(self, memory_manager: "MemoryManager", executor: ProposalExecutor) -> None:
        self._memory_manager = memory_manager
        self._executor = executor

    @property
    def name(self) -> str:
        return "proposals"

    def register_tools(self, tool_manager: "ToolManager") -> None:
        ProposalToolHandler(self._memory_manager, self._executor).register(tool_manager)


class ProposalToolHandler:
    """Registers the proposal review tools with a ToolManager."""

    def __init__(self, memory_manager: "MemoryManager", executor: ProposalExecutor) -> None:
        self.memory_manager = memory_manager
        self.executor = executor

    def register(self, manager: "ToolManager") -> None:
        manager.register("list_proposals", self._list_proposals)
        manager.register("approve_proposal", self._approve_proposal)
        manager.register("deny_proposal", self._deny_proposal)

    async def _list_proposals(self, status: str = "pending", limit: int = 10) -> Dict[str, Any]:
        logger.info(f"Executing tool: list_proposals status={status} limit={limit}")
        expired = self.memory_manager.expire_stale_proposals()
        list_status: Optional[str] = None if status == "all" else status
        rows = self.memory_manager.list_proposals(status=list_status, limit=limit)
        proposals: List[Dict[str, Any]] = [
            {
                "proposal_id": row["proposal_id"],
                "created_at": str(row.get("created_at", "")),
                "agent": row["agent_name"],
                "status": row["status"],
                "action_type": row["action_type"],
                "args": row["action_args"],
                "rationale": row.get("rationale"),
                "review_note": row.get("review_note"),
                "execution_result": row.get("execution_result"),
            }
            for row in rows
        ]
        return {"expired_now": expired, "count": len(proposals), "proposals": proposals}

    async def _approve_proposal(self, proposal_id: int, note: Optional[str] = None) -> Dict[str, Any]:
        logger.info(f"Executing tool: approve_proposal id={proposal_id}")
        self.memory_manager.expire_stale_proposals()
        proposal = self.memory_manager.get_proposal(proposal_id)
        if not proposal:
            raise ValueError(f"No proposal with id {proposal_id}.")
        if not self.memory_manager.review_proposal(proposal_id, "approved", OPERATOR_ID, note):
            raise ValueError(
                f"Proposal {proposal_id} is not pending (status: {proposal['status']}); nothing executed."
            )
        self.memory_manager.log_audit_event(
            event_type="proposal_approved",
            target_id=proposal_id,
            operator_id=OPERATOR_ID,
            prior_state="pending",
            new_state="approved",
            reason=note,
            metadata={"action_type": proposal["action_type"], "args": proposal["action_args"]},
        )

        try:
            success, result = await self.executor.execute(proposal)
        except Exception as e:
            # An exception here must still land in mark_proposal_executed —
            # otherwise the row is stranded in 'approved' (not pending, never
            # executed) with no retry path short of editing the DB.
            logger.error(f"Proposal {proposal_id} executor raised: {e}", exc_info=True)
            success, result = False, f"executor error: {e}"
        self.memory_manager.mark_proposal_executed(proposal_id, success, result)
        self.memory_manager.log_audit_event(
            event_type="proposal_executed" if success else "proposal_execution_failed",
            target_id=proposal_id,
            operator_id=OPERATOR_ID,
            prior_state="approved",
            new_state="executed" if success else "execution_failed",
            reason=result,
        )
        return {
            "proposal_id": proposal_id,
            "action_type": proposal["action_type"],
            "executed": success,
            "result": result,
        }

    async def _deny_proposal(self, proposal_id: int, reason: str) -> Dict[str, Any]:
        logger.info(f"Executing tool: deny_proposal id={proposal_id}")
        self.memory_manager.expire_stale_proposals()
        proposal = self.memory_manager.get_proposal(proposal_id)
        if not proposal:
            raise ValueError(f"No proposal with id {proposal_id}.")
        if not self.memory_manager.review_proposal(proposal_id, "denied", OPERATOR_ID, reason):
            raise ValueError(
                f"Proposal {proposal_id} is not pending (status: {proposal['status']})."
            )
        self.memory_manager.log_audit_event(
            event_type="proposal_denied",
            target_id=proposal_id,
            operator_id=OPERATOR_ID,
            prior_state="pending",
            new_state="denied",
            reason=reason,
            metadata={"action_type": proposal["action_type"], "args": proposal["action_args"]},
        )
        return {"proposal_id": proposal_id, "status": "denied", "reason": reason}
