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
from typing import Any, Dict, List

ZAMMAD_PRIORITIES = ["1 low", "2 normal", "3 high"]

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


def validate_proposal_args(action_type: str, args: Any) -> List[str]:
    """Validate proposal args against the whitelist. Returns a list of
    human-readable errors; empty list means valid."""
    if action_type not in PROPOSAL_ACTIONS:
        return [f"unknown action_type '{action_type}'"]
    if not isinstance(args, dict):
        return ["args must be an object"]

    spec = PROPOSAL_ACTIONS[action_type]["args"]
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


def build_submission_tool_schema() -> Dict[str, Any]:
    """Agent-internal tool schema for the proposal-extraction LLM call
    (passed straight to the engine, never routed through ToolManager).
    Derived from PROPOSAL_ACTIONS so the whitelist has one source of truth."""
    action_lines = []
    for name, action in PROPOSAL_ACTIONS.items():
        arg_desc = ", ".join(
            f"{k} ({v['type'].__name__}{', one of ' + str(v['enum']) if 'enum' in v else ''})"
            for k, v in action["args"].items()
        )
        action_lines.append(f"- {name}: {action['description']} Args: {arg_desc}")
    description = ("Queue proposed board actions for human review. "
                   "Allowed action types:\n" + "\n".join(action_lines))

    return {
        "type": "function",
        "function": {
            "name": "submit_proposals",
            "description": description,
            "parameters": {
                "type": "object",
                "properties": {
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
