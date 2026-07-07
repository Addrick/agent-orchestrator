"""Proposal-queue review tools (service_binding: proposals) — DP-282.

The human review surface over the durable proposal queue that managr (and
future agents) write into. approve_proposal is the only path from a proposal
to an external write, and the executor re-validates args against the fixed
action whitelist at execution time.
"""

from typing import Any, Dict, List


PROPOSAL_TOOLS: List[Dict[str, Any]] = [
    {
        "type": "function",
        "is_write": False,
        "service_binding": "proposals",
        "capabilities": {
            # Rationale/args are derived from ticket content — attacker-influenced text.
            "produces_untrusted": True,
            "irreversible": False,
            "locality": "local",
            "sensitivity": "internal",
        },
        "function": {
            "name": "list_proposals",
            "description": "Lists queued agent proposals awaiting human review (default: pending ones). "
                           "Each entry shows the proposed action, its arguments, and the agent's rationale.",
            "parameters": {
                "type": "object",
                "properties": {
                    "status": {
                        "type": "string",
                        "description": "Filter by status. Defaults to 'pending'.",
                        "enum": ["pending", "approved", "denied", "expired",
                                 "executed", "execution_failed", "all"],
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of proposals to return. Defaults to 10.",
                    },
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "is_write": True,
        "service_binding": "proposals",
        "capabilities": {
            "produces_untrusted": False,
            "irreversible": False,
            "locality": "network",
            "exfil_capable": False,
            "sensitivity": "internal",
        },
        "function": {
            "name": "approve_proposal",
            "description": "Approves a pending proposal and immediately executes its action against Zammad "
                           "(internal note, priority change, or reminder). This is the human approval gate — "
                           "only use it when the operator has explicitly approved the proposal.",
            "parameters": {
                "type": "object",
                "properties": {
                    "proposal_id": {
                        "type": "integer",
                        "description": "The id of the proposal to approve (from list_proposals).",
                    },
                    "note": {
                        "type": "string",
                        "description": "Optional reviewer note recorded with the approval.",
                    },
                },
                "required": ["proposal_id"],
            },
        },
    },
    {
        "type": "function",
        "is_write": True,
        "service_binding": "proposals",
        "capabilities": {
            "produces_untrusted": False,
            "irreversible": False,
            "locality": "local",
            "sensitivity": "internal",
        },
        "function": {
            "name": "deny_proposal",
            "description": "Denies a pending proposal with a reason. Nothing executes; the reason is recorded "
                           "as feedback for the proposing agent.",
            "parameters": {
                "type": "object",
                "properties": {
                    "proposal_id": {
                        "type": "integer",
                        "description": "The id of the proposal to deny (from list_proposals).",
                    },
                    "reason": {
                        "type": "string",
                        "description": "Why the proposal is denied.",
                    },
                },
                "required": ["proposal_id", "reason"],
            },
        },
    },
]
