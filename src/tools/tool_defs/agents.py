"""Agent introspection tools (service_binding: agents)."""

from typing import Any, Dict, List


AGENT_TOOLS: List[Dict[str, Any]] = [
    {
        "type": "function",
        "is_write": False,
        "service_binding": "agents",
        "capabilities": {
            "produces_untrusted": False,
            "irreversible": False,
            "locality": "local",
            "sensitivity": "internal",
        },
        "function": {
            "name": "get_agent_status",
            "description": (
                "Get the current status of registered agents. Returns running state, "
                "last poll time, error counts, and poll statistics. Use without "
                "arguments to see all agents, or specify an agent name for details."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "agent_name": {
                        "type": "string",
                        "description": "Optional: specific agent to query (e.g. 'dispatch'). If omitted, returns all agents.",
                    },
                },
            },
        },
    },
    {
        "type": "function",
        "is_write": False,
        "service_binding": "agents",
        "capabilities": {
            "produces_untrusted": False,
            "irreversible": False,
            "locality": "local",
            "sensitivity": "internal",
        },
        "function": {
            "name": "get_agent_history",
            "description": (
                "Get recent action history for a specific agent. Shows what actions "
                "the agent has taken, their outcomes, timestamps, and any failures. "
                "Optionally filter by ticket or customer."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "agent_name": {
                        "type": "string",
                        "description": "The agent to query history for (e.g. 'dispatch').",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of actions to return. Default: 10.",
                        "default": 10,
                    },
                    "ticket_id": {
                        "type": "string",
                        "description": "Optional: filter to actions related to a specific ticket ID.",
                    },
                    "customer": {
                        "type": "string",
                        "description": "Optional: filter to actions related to a specific customer identifier.",
                    },
                },
                "required": ["agent_name"],
            },
        },
    },
    {
        "type": "function",
        "is_write": True,
        "service_binding": "agents",
        "capabilities": {
            "produces_untrusted": False,
            "irreversible": False,
            "locality": "local",
            "sensitivity": "internal",
        },
        "function": {
            "name": "manage_agent",
            "description": (
                "Start, stop, or restart an autonomous agent. Agents are background "
                "workers that poll for tasks and execute them independently. "
                "This is a write operation and may require confirmation."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "agent_name": {
                        "type": "string",
                        "description": "The agent to manage (e.g. 'dispatch').",
                    },
                    "action": {
                        "type": "string",
                        "enum": ["start", "stop", "restart"],
                        "description": "The lifecycle action to perform.",
                    },
                },
                "required": ["agent_name", "action"],
            },
        },
    },
    {
        "type": "function",
        "is_write": False,
        "service_binding": "agents",
        "capabilities": {
            "produces_untrusted": False,
            "irreversible": False,
            "locality": "local",
            "sensitivity": "internal",
        },
        "function": {
            "name": "lookup_agent_history",
            "description": (
                "Dereference an agent-action series by its action_id. Returns the "
                "parent row, all child step rows, and context tags. Use this after "
                "a recall hit references `action_id:<id>` to fetch the full "
                "trajectory the summary was extracted from."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "action_id": {
                        "type": "integer",
                        "description": "The id of the parent (root) Agent_Actions row.",
                    },
                },
                "required": ["action_id"],
            },
        },
    },
]
