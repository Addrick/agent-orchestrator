"""Voice command timer tools (service_binding: voice)."""

from typing import Any, Dict, List


VOICE_TOOLS: List[Dict[str, Any]] = [
    {
        "type": "function",
        "is_write": False,
        "service_binding": "voice",
        "capabilities": {
            "produces_untrusted": False,
            "irreversible": False,
            "locality": "local",
            "sensitivity": "internal",
        },
        "function": {
            "name": "set_timer",
            "description": (
                "Set a countdown timer that announces in the channel when it "
                "fires. Use for 'remind me in N minutes' / 'set a timer for N'. "
                "Give the duration in natural language ('10 minutes', '30 "
                "seconds', '1 hour')."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "duration": {
                        "type": "string",
                        "description": "Duration in natural language, e.g. '10 minutes', '90 seconds'.",
                    },
                    "label": {
                        "type": "string",
                        "description": "Optional label announced when the timer fires (e.g. 'pasta').",
                    },
                },
                "required": ["duration"],
            },
        },
    },
    {
        "type": "function",
        "is_write": False,
        "service_binding": "voice",
        "capabilities": {
            "produces_untrusted": False,
            "irreversible": False,
            "locality": "local",
            "sensitivity": "internal",
        },
        "function": {
            "name": "list_timers",
            "description": "List the currently pending timers with their remaining time and ids.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "is_write": False,
        "service_binding": "voice",
        "capabilities": {
            "produces_untrusted": False,
            "irreversible": False,
            "locality": "local",
            "sensitivity": "internal",
        },
        "function": {
            "name": "cancel_timer",
            "description": "Cancel a pending timer by its id (from list_timers).",
            "parameters": {
                "type": "object",
                "properties": {
                    "timer_id": {
                        "type": "string",
                        "description": "The id of the timer to cancel.",
                    },
                },
                "required": ["timer_id"],
            },
        },
    },
]
