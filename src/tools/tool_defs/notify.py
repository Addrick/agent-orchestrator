"""Operator notification tool (service_binding: notify), DP-512.

`notify_adam` sends text the model wrote to a target the model cannot choose:
the calling persona's entry (or `default`) under `notify.targets` in
agents.json. That is why it is `exfil_capable: False` — the composition rules
guard against data reaching an attacker-reachable sink, and this sink is fixed
in config to the operator (Adam's decision, 2026-11-13; same argument as
Zammad's opt-out). It only holds while a target is somewhere only the operator
reads; the user guide says so.

`is_write: False`: nothing is changed at the destination, so a persona that
enables the tool gets it at `allow`, and its saved policy can set `ask`.
"""

from typing import Any, Dict, List


NOTIFY_TOOLS: List[Dict[str, Any]] = [
    {
        "type": "function",
        "is_write": False,
        "service_binding": "notify",
        "capabilities": {
            "produces_untrusted": False,
            "irreversible": False,
            "locality": "network",
            "sensitivity": "internal",
            "exfil_capable": False,
        },
        "function": {
            "name": "notify_adam",
            "description": (
                "Send Adam a message without ending your turn, e.g. a concrete "
                "question you are blocked on, or something he needs to know now. "
                "It goes to a channel set in config, not to this conversation, and "
                "you do not wait for an answer: keep working after the call. The "
                "result says whether it was sent. If Adam is the one you are "
                "talking to in this conversation, answer in your reply instead."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "message": {
                        "type": "string",
                        "description": "The whole message, in plain language. Must not be empty.",
                    },
                },
                "required": ["message"],
            },
        },
    },
]
