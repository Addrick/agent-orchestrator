# src/proposals/schemas.py
#
# Fixed action whitelist for the proposal queue (DP-282).
#
# A proposal only ever executes if its action_type is a key here AND its
# args validate against the action's arg spec — checked when the proposal
# is created (emission) and AGAIN at execution time, so a row tampered
# between review and execution still can't smuggle an unknown action or
# argument through. Free text never becomes a proposal.
#
# Phase 1 whitelist is deliberately internal + low-blast: nothing here is
# customer-visible (add_note is hard-forced to an internal article by the
# executor). Customer-facing actions are the last autonomy tier and are
# not represented in this schema at all.

from datetime import datetime
from typing import Any, Dict, List, Optional

ZAMMAD_PRIORITIES = ["1 low", "2 normal", "3 high"]

# DP-290 reflective queue: what the planner may do with one of its own
# still-pending proposals. Deliberately a closed enum — dispositions carry
# ids + a decision + (for revise) whitelist-validated args, never free text
# that could smuggle instructions from tainted ticket content.
DISPOSITION_DECISIONS = ["reaffirm", "revise", "withdraw"]

PROPOSAL_ACTIONS: Dict[str, Dict[str, Any]] = {
    "add_note": {
        "description": "Add an INTERNAL note (never customer-visible) to a ticket.",
        "args": {
            "ticket_number": {"type": int, "required": True,
                              "description": "User-facing ticket number from the board snapshot."},
            "body": {"type": str, "required": True, "max_length": 4000,
                     "description": "Note content."},
        },
    },
    "set_priority": {
        "description": "Change a ticket's priority.",
        "args": {
            "ticket_number": {"type": int, "required": True,
                              "description": "User-facing ticket number from the board snapshot."},
            "priority": {"type": str, "required": True, "enum": ZAMMAD_PRIORITIES,
                         "description": "New priority."},
        },
    },
    "remind": {
        "description": "Park a ticket as 'pending reminder' until a date.",
        "args": {
            "ticket_number": {"type": int, "required": True,
                              "description": "User-facing ticket number from the board snapshot."},
            "pending_until": {"type": str, "required": True, "date": True,
                              "description": "Reminder date, YYYY-MM-DD."},
        },
    },
}

# DP-240: actions a *dispatched subagent* may queue via the MCP bridge, kept in
# a SEPARATE dict from PROPOSAL_ACTIONS on purpose.
#
# PROPOSAL_ACTIONS feeds build_submission_tool_schema, which is managr's
# extraction schema — and managr reads attacker-reachable ticket content. Adding
# a generic tool-call action to that dict would let a poisoned ticket steer
# managr into proposing arbitrary derpr tool calls. The two producers therefore
# get two whitelists, and the scope is an explicit argument at every validation
# site rather than a shared global (so the safe scope is the *default*).
#
# `tool_name` is NOT enum-pinned here: the authoritative check is the executor
# re-resolving the tool against the live MCP ToolPolicy at execution time. A
# name captured in a stored row must never be able to outlive or outrun the
# policy that was in force when it was queued.
AGENT_CALL_ACTIONS: Dict[str, Dict[str, Any]] = {
    "call_derpr_tool": {
        "description": "Execute a derpr tool on behalf of a dispatched subagent.",
        "args": {
            "tool_name": {"type": str, "required": True, "max_length": 128,
                          "description": "Name of the derpr tool to execute."},
            "tool_args": {"type": dict, "required": True,
                          "description": "Arguments passed through to the tool."},
            "agent_id": {"type": str, "required": True, "max_length": 128,
                         "description": "Dispatched agent that requested the call."},
        },
    },
}

# Everything the executor is capable of running. Deliberately not exported as
# the default validation scope — see validate_proposal_args.
EXECUTABLE_ACTIONS: Dict[str, Dict[str, Any]] = {**PROPOSAL_ACTIONS, **AGENT_CALL_ACTIONS}


def validate_proposal_args(
    action_type: str,
    args: Any,
    scope: Optional[Dict[str, Dict[str, Any]]] = None,
) -> List[str]:
    """Validate proposal args against a whitelist. Returns a list of
    human-readable errors; empty list means valid.

    `scope` defaults to PROPOSAL_ACTIONS (board actions only) so every existing
    caller — notably managr's emission path — stays fail-closed against the
    DP-240 agent-call action. The executor passes EXECUTABLE_ACTIONS explicitly,
    because it is the one component allowed to run both families.
    """
    actions = PROPOSAL_ACTIONS if scope is None else scope
    if action_type not in actions:
        return [f"unknown action_type '{action_type}'"]
    if not isinstance(args, dict):
        return ["args must be an object"]

    spec = actions[action_type]["args"]
    errors: List[str] = []

    for key in args:
        if key not in spec:
            errors.append(f"unexpected argument '{key}'")

    for key, rules in spec.items():
        if key not in args:
            if rules.get("required"):
                errors.append(f"missing required argument '{key}'")
            continue
        errors.extend(_validate_value(key, args[key], rules))

    return errors


def _validate_value(key: str, value: Any, rules: Dict[str, Any]) -> List[str]:
    expected = rules["type"]
    # bool is an int subclass; never accept it where int is expected
    if not isinstance(value, expected) or isinstance(value, bool):
        return [f"argument '{key}' must be {expected.__name__}"]
    errors: List[str] = []
    if "enum" in rules and value not in rules["enum"]:
        errors.append(f"argument '{key}' must be one of {rules['enum']}")
    if "max_length" in rules and len(value) > rules["max_length"]:
        errors.append(f"argument '{key}' exceeds {rules['max_length']} chars")
    if rules.get("date"):
        try:
            datetime.strptime(value, "%Y-%m-%d")
        except ValueError:
            errors.append(f"argument '{key}' must be a YYYY-MM-DD date")
    return errors


def build_submission_tool_schema(
    pending_ids: Optional[List[int]] = None,
) -> Dict[str, Any]:
    """Agent-internal tool schema for the proposal-extraction LLM call
    (passed straight to the engine, never routed through ToolManager).
    Derived from PROPOSAL_ACTIONS so the whitelist has one source of truth.

    pending_ids (DP-290): ids of the caller's own still-pending proposals.
    When given, the schema grows a `dispositions` array so the same single
    call can also reaffirm/revise/withdraw them — proposal_id is enum-pinned
    to exactly those ids, so the model cannot address anyone else's rows."""
    action_lines = []
    for name, action in PROPOSAL_ACTIONS.items():
        arg_desc = ", ".join(
            f"{k} ({v['type'].__name__}{', one of ' + str(v['enum']) if 'enum' in v else ''})"
            for k, v in action["args"].items()
        )
        action_lines.append(f"- {name}: {action['description']} Args: {arg_desc}")
    description = ("Queue proposed board actions for human review. "
                   "Allowed action types:\n" + "\n".join(action_lines))

    properties: Dict[str, Any] = {}
    if pending_ids:
        properties["dispositions"] = {
            "type": "array",
            "description": (
                "One entry per proposal listed under YOUR PENDING PROPOSALS: "
                "reaffirm it if it still stands, revise it (full replacement "
                "args, same rules as a new proposal of that action type) if "
                "the situation changed, or withdraw it if no longer needed."
            ),
            "items": {
                "type": "object",
                "properties": {
                    "proposal_id": {"type": "integer", "enum": list(pending_ids)},
                    "decision": {"type": "string", "enum": DISPOSITION_DECISIONS},
                    "args": {
                        "type": "object",
                        "description": "revise only: full replacement arguments.",
                    },
                    "reason": {
                        "type": "string",
                        "description": "revise/withdraw: one short sentence why.",
                    },
                },
                "required": ["proposal_id", "decision"],
            },
        }

    return {
        "type": "function",
        "function": {
            "name": "submit_proposals",
            "description": description,
            "parameters": {
                "type": "object",
                "properties": {
                    **properties,
                    "proposals": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "action_type": {
                                    "type": "string",
                                    "enum": list(PROPOSAL_ACTIONS.keys()),
                                },
                                "args": {
                                    "type": "object",
                                    "description": "Arguments for the action, per the allowed-actions list.",
                                },
                                "rationale": {
                                    "type": "string",
                                    "description": "One sentence: why this action, tied to the report.",
                                },
                            },
                            "required": ["action_type", "args", "rationale"],
                        },
                    },
                },
                "required": ["proposals"],
            },
        },
    }
