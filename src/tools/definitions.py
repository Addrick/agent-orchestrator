# src/tools/definitions.py

import importlib
import logging
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

"""
This file contains the definitions for all tools available to the LLM.
Each tool is defined as a JSON schema compatible with the function-calling
APIs of major providers like OpenAI, Google, and Anthropic.

These definitions serve as the "contract" that the LLM uses to understand
what a tool does, what parameters it requires, and what it returns.

The actual implementation of these tools is handled by the ToolManager.

Every tool definition carries a `capabilities` block driving the runtime
tool-security framework (see memory/project/plans/tool_security_framework.md):

    "capabilities": {
        "produces_untrusted": bool,    # result may carry attacker-controlled text
        "irreversible": bool,          # effect cannot be trivially undone
        "irreversible_if": str | None, # optional "module:function" classifier
    }

`produces_untrusted` is about *origin* of the data, not network — local
memory tools that surface previously-ingested external content count.
`irreversible_if` is a dotted path to a `(args: dict) -> bool` callable;
when present, runtime ORs its result with `irreversible`.
"""

# A list containing all tool definitions.
# The ToolManager will expose these to the ChatSystem.
#
# Special tool types:
#   "google_grounding" — Not a callable function. Signals the engine to enable
#                        Google's native grounding feature for Gemini models.
#                        Has no effect on other providers or Gemma models.
ALL_TOOL_DEFINITIONS: List[Dict[str, Any]] = [
    {
        "type": "google_grounding",
        "capabilities": {
            "produces_untrusted": True,
            "irreversible": False,
            "locality": "network",
            "sensitivity": "public",
        },
        "function": {
            "name": "google_grounding_search",
            "description": "Enables Google's native Search grounding feature for Gemini models. "
                           "Allows the model to retrieve up-to-date information from the web to "
                           "support its responses. Has no effect on non-Gemini models. "
                           "Subject to Google grounding API costs and rate limits.",
        },
    },
    {
        "type": "function",
        "is_write": False,
        "capabilities": {
            "produces_untrusted": True,
            "irreversible": False,
            "locality": "network",
            "sensitivity": "public",
        },
        "function": {
            "name": "web_search",
            "description": "Searches the web for information using DuckDuckGo. Returns titles, "
                           "URLs, and summaries for the most relevant results. Compatible with "
                           "all model providers.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query.",
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Maximum number of results to return (default: 5).",
                        "default": 5,
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "is_write": False,
        "service_binding": "zammad",
        "capabilities": {
            "produces_untrusted": True,
            "irreversible": False,
            "locality": "network",
            "sensitivity": "internal",
        },
        "function": {
            "name": "get_ticket_details",
            "description": "Retrieves the complete details for a specific Zammad ticket using its user-facing ticket number.",
            "parameters": {
                "type": "object",
                "properties": {
                    "ticket_number": {
                        "type": "integer",
                        "description": "The user-facing number of the ticket (e.g., 53515).",
                    },
                },
                "required": ["ticket_number"],
            },
        },
    },
    {
        "type": "function",
        "is_write": True,
        "service_binding": "zammad",
        "capabilities": {
            "produces_untrusted": False,
            "irreversible": False,
            "locality": "network",
            "sensitivity": "internal",
        },
        "function": {
            "name": "update_ticket",
            "description": "Updates one or more properties of an existing Zammad ticket. Requires the ticket's internal ID. All other fields are optional.",
            "parameters": {
                "type": "object",
                "properties": {
                    "ticket_id": {
                        "type": "integer",
                        "description": "The unique internal numerical ID of the ticket to update.",
                    },
                    "state": {
                        "type": "string",
                        "description": "The new state for the ticket (e.g., 'open', 'closed', 'pending reminder').",
                        "enum": ["new", "open", "pending reminder", "closed"],
                    },
                    "priority": {
                        "type": "string",
                        "description": "The new priority for the ticket.",
                        "enum": ["1 low", "2 normal", "3 high"],
                    },
                    "owner_id": {
                        "type": "integer",
                        "description": "The numerical ID of the agent to assign as the new owner.",
                    },
                    "tags": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "A list of tags to apply to the ticket. This will overwrite existing tags.",
                    },
                },
                "required": ["ticket_id"],
            },
        },
    },
    {
        "type": "function",
        "is_write": True,
        "service_binding": "zammad",
        "capabilities": {
            "produces_untrusted": False,
            "irreversible": False,
            "irreversible_if": "src.tools.classifiers:add_note_irreversible_check",
            "locality": "network",
            "sensitivity": "internal",
        },
        "function": {
            "name": "add_note_to_ticket",
            "description": "Adds a new article (a note or comment) to an existing Zammad ticket. Requires the ticket's internal ID and the note's body.",
            "parameters": {
                "type": "object",
                "properties": {
                    "ticket_id": {
                        "type": "integer",
                        "description": "The unique internal numerical ID of the ticket to add a note to.",
                    },
                    "body": {
                        "type": "string",
                        "description": "The content of the note to be added.",
                    },
                    "internal": {
                        "type": "boolean",
                        "description": "Set to true if the note is for internal agents only, false if it's visible to the customer. Defaults to false.",
                        "default": False,
                    },
                },
                "required": ["ticket_id", "body"],
            },
        },
    },
    {
        "type": "function",
        "is_write": False,
        "service_binding": "zammad",
        "capabilities": {
            "produces_untrusted": True,
            "irreversible": False,
            "locality": "network",
            "sensitivity": "internal",
        },
        "function": {
            "name": "search_tickets",
            "description": "Searches for Zammad tickets using a specific Zammad search query string.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query string. Examples: 'state.name:open AND priority:\"3 high\"', 'customer.email:example@email.com'",
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "is_write": True,
        "service_binding": "zammad",
        "capabilities": {
            "produces_untrusted": False,
            "irreversible": True,
            "locality": "network",
            "sensitivity": "internal",
        },
        "function": {
            "name": "create_ticket",
            "description": "Creates a new Zammad ticket. Requires a title and a body. If 'customer_id' is omitted, the ticket is created for the current user. Use the 'search_user' tool to find the ID for a different user.",
            "parameters": {
                "type": "object",
                "properties": {
                    "title": {
                        "type": "string",
                        "description": "The title of the new ticket."
                    },
                    "body": {
                        "type": "string",
                        "description": "The content of the first message in the ticket."
                    },
                    "customer_id": {
                        "type": "integer",
                        "description": "Optional. The internal ID of the user to create the ticket for. If omitted, the ticket will be created for the user sending the current message."
                    }
                },
                "required": ["title", "body"],
            },
        },
    },
    {
        "type": "function",
        "is_write": False,
        "service_binding": "zammad",
        "capabilities": {
            "produces_untrusted": False,
            "irreversible": False,
            "locality": "network",
            "sensitivity": "pii",
        },
        "function": {
            "name": "search_user",
            "description": "Searches for a Zammad user by a query string (e.g., email address or last name).",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query, e.g., 'john.doe@example.com'."
                    }
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "is_write": True,
        "service_binding": "zammad",
        "capabilities": {
            "produces_untrusted": False,
            "irreversible": False,
            "locality": "network",
            "sensitivity": "pii",
        },
        "function": {
            "name": "create_user",
            "description": "Creates a new customer user in Zammad. The 'firstname', 'lastname', and 'email' parameters are all required. The 'note' is optional.",
            "parameters": {
                "type": "object",
                "properties": {
                    "firstname": {"type": "string", "description": "The user's first name."},
                    "lastname": {"type": "string", "description": "The user's last name."},
                    "email": {"type": "string", "description": "The user's unique email address."},
                    "note": {"type": "string", "description": "An optional note about the user."},
                },
                "required": ["firstname", "lastname", "email"],
            },
        },
    },
    {
        "type": "function",
        "is_write": True,
        "service_binding": "zammad",
        "capabilities": {
            "produces_untrusted": False,
            "irreversible": False,
            "locality": "network",
            "sensitivity": "pii",
        },
        "function": {
            "name": "update_user",
            "description": "Updates an existing user in Zammad. The 'user_id' is required to identify the user. All other parameters are optional. Use the 'search_user' tool first to find the 'user_id' if you don't have it.",
            "parameters": {
                "type": "object",
                "properties": {
                    "user_id": {"type": "integer", "description": "The unique internal ID of the user to update."},
                    "firstname": {"type": "string", "description": "The user's new first name."},
                    "lastname": {"type": "string", "description": "The user's new last name."},
                    "email": {"type": "string", "description": "The user's new unique email address."},
                    "active": {"type": "boolean", "description": "Set to false to deactivate the user, true to reactivate."},
                    "note": {"type": "string", "description": "A new note to add to the user. This will overwrite any existing note."},
                },
                "required": ["user_id"],
            },
        },
    },
    {
        "type": "function",
        "is_write": True,
        "service_binding": "zammad",
        "capabilities": {
            "produces_untrusted": False,
            "irreversible": True,
            "locality": "network",
            "sensitivity": "pii",
        },
        "function": {
            "name": "delete_user",
            "description": "Deletes a user from Zammad. This is a destructive and irreversible action. Requires the unique 'user_id'. Use the 'search_user' tool to find the 'user_id' first to ensure you are deleting the correct user.",
            "parameters": {
                "type": "object",
                "properties": {
                    "user_id": {"type": "integer", "description": "The unique internal ID of the user to delete."},
                },
                "required": ["user_id"],
            },
        },
    },
    {
        "type": "function",
        "is_write": True,
        "service_binding": "zammad",
        "capabilities": {
            "produces_untrusted": False,
            "irreversible": True,
            "locality": "network",
            "sensitivity": "internal",
        },
        "function": {
            "name": "merge_tickets",
            "description": "Merges a source ticket into a target ticket. Moves all conversation history (articles) to the target, links the tickets, and sets the source ticket state to 'merged'. Requires internal numerical IDs for both tickets.",
            "parameters": {
                "type": "object",
                "properties": {
                    "source_ticket_id": {
                        "type": "integer",
                        "description": "The unique internal numerical ID of the ticket to be merged (the duplicate).",
                    },
                    "target_ticket_id": {
                        "type": "integer",
                        "description": "The unique internal numerical ID of the ticket that will receive the content (the original).",
                    },
                },
                "required": ["source_ticket_id", "target_ticket_id"],
            },
        },
    },
    # =========================================================================
    # Agent Management Tools
    # =========================================================================
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
    # =========================================================================
    # Memory Management Tools
    # =========================================================================
    {
        "type": "function",
        "is_write": False,
        "capabilities": {
            "produces_untrusted": True,
            "irreversible": False,
            "locality": "local",
            "sensitivity": "internal",
        },
        "function": {
            "name": "drill_down_memory",
            "description": "Fetch raw episodic memories (Level 2 Archival) under a specific Core Profile. Use this to find missing specific details like dates, links, or verbatim quotes that were consolidated away.",
            "parameters": {
                "type": "object",
                "properties": {
                    "parent_summary_id": {
                        "type": "integer",
                        "description": "The exact ID of the Level 1 Core Profile memory to drill down into.",
                    },
                },
                "required": ["parent_summary_id"],
            },
        },
    },
    {
        "type": "function",
        "is_write": True,
        "capabilities": {
            "produces_untrusted": False,
            "irreversible": False,
            "locality": "local",
            "sensitivity": "internal",
        },
        "function": {
            "name": "update_core_memory",
            "description": "Modify an existing 'Core Fact Profile' (Level 1) when new information contradicts it or adds significant context.",
            "parameters": {
                "type": "object",
                "properties": {
                    "summary_id": {"type": "integer", "description": "The exact ID of the Core Profile to modify."},
                    "new_content": {"type": "string", "description": "The completely revised, comprehensive markdown summary."}
                },
                "required": ["summary_id", "new_content"]
            }
        }
    }
]


def validate_tool_capabilities(tool: Dict[str, Any]) -> None:
    """
    Assert a tool definition carries the required `capabilities` block per
    the tool-security framework. Raises ValueError on any violation.

    - `capabilities` must be a dict with bool `produces_untrusted` and `irreversible`.
    - Optional `irreversible_if` must be a `"module:function"` dotted path that
      resolves to a callable at validation time.
    """
    name = tool.get("function", {}).get("name", "<unknown>")
    caps = tool.get("capabilities")
    if not isinstance(caps, dict):
        raise ValueError(f"Tool '{name}' missing 'capabilities' block")
    for required in ("produces_untrusted", "irreversible", "locality", "sensitivity"):
        if required not in caps:
            raise ValueError(
                f"Tool '{name}' capabilities missing required flag '{required}'"
            )
        if required in ("produces_untrusted", "irreversible"):
            if not isinstance(caps[required], bool):
                raise ValueError(
                    f"Tool '{name}' capability '{required}' must be bool, "
                    f"got {type(caps[required]).__name__}"
                )
        else:
            if not isinstance(caps[required], str):
                raise ValueError(
                    f"Tool '{name}' capability '{required}' must be str, "
                    f"got {type(caps[required]).__name__}"
                )
    classifier = caps.get("irreversible_if")
    if classifier is None:
        return
    if not isinstance(classifier, str) or ":" not in classifier:
        raise ValueError(
            f"Tool '{name}' irreversible_if must be a 'module:function' dotted path"
        )
    module_path, func_name = classifier.split(":", 1)
    try:
        module = importlib.import_module(module_path)
    except ImportError as e:
        raise ValueError(
            f"Tool '{name}' irreversible_if module '{module_path}' could not be imported: {e}"
        ) from e
    func = getattr(module, func_name, None)
    if not callable(func):
        raise ValueError(
            f"Tool '{name}' irreversible_if '{classifier}' did not resolve to a callable"
        )


# Validate at import time so any consumer (engine, tests, tooling) catches
# capability drift immediately.
for _tool in ALL_TOOL_DEFINITIONS:
    validate_tool_capabilities(_tool)


# Model prefixes that do NOT support each tool.
# Uses the same prefix logic as engine.py routing.
# Tools not listed here are compatible with all providers.
GROUNDING_INCOMPATIBLE_PREFIXES = {'gpt', 'claude', 'gemma', 'gemini-3.1', 'local', 'unknown'}

MODEL_INCOMPATIBLE_TOOLS = {
    'google_grounding_search': GROUNDING_INCOMPATIBLE_PREFIXES,
}

# Derived from tool metadata — no manual maintenance needed when adding new tools.
WRITE_TOOLS = {t['function']['name'] for t in ALL_TOOL_DEFINITIONS if t.get('is_write')}

# Tools that ALWAYS require confirmation, even in AUTONOMOUS mode.
# These are typically high-impact or destructive operations.
ALWAYS_CONFIRM_TOOLS = {"merge_tickets", "delete_user"}


def get_tool_capabilities(tool_name: str) -> Dict[str, Any]:
    """
    Retrieves the security capabilities block for a given tool name.
    Returns a default block (all False) if the tool is not found.
    """
    for tool in ALL_TOOL_DEFINITIONS:
        if tool.get("function", {}).get("name") == tool_name:
            from typing import cast
            return cast(Dict[str, Any], tool.get("capabilities", {
                "produces_untrusted": False,
                "irreversible": False,
                "locality": "unknown",
                "sensitivity": "unknown",
            }))
    return {
        "produces_untrusted": False,
        "irreversible": False,
        "locality": "unknown",
        "sensitivity": "unknown",
    }


def is_irreversible(tool_name: str, args: Dict[str, Any]) -> bool:
    """
    Checks if a tool call is irreversible based on its name and arguments.
    """
    caps = get_tool_capabilities(tool_name)
    if caps.get("irreversible"):
        return True

    classifier = caps.get("irreversible_if")
    if not classifier:
        return False

    # Resolve "module:function"
    try:
        module_path, func_name = classifier.split(":", 1)
        module = importlib.import_module(module_path)
        func = getattr(module, func_name)
        return bool(func(args))
    except (ValueError, ImportError, AttributeError, Exception) as e:
        logger.error(f"Error resolving irreversible_if for {tool_name}: {e}")
        return False
