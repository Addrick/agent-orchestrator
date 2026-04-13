# src/tools/definitions.py

from typing import List, Dict, Any

"""
This file contains the definitions for all tools available to the LLM.
Each tool is defined as a JSON schema compatible with the function-calling
APIs of major providers like OpenAI, Google, and Anthropic.

These definitions serve as the "contract" that the LLM uses to understand
what a tool does, what parameters it requires, and what it returns.

The actual implementation of these tools is handled by the ToolManager.
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
        "type": "function",
        "is_write": True,
        "function": {
            "name": "submit_memory_summary",
            "description": "Records observations extracted from a conversation segment for long-term recall. "
                           "Also identifies outlier messages that do not share the primary theme "
                           "of the segment so they can be re-routed to a different cluster.",
            "parameters": {
                "type": "object",
                "properties": {
                    "observations": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Discrete statements capturing what was discussed: facts, preferences, "
                                       "opinions, problems described, solutions provided, advice given, "
                                       "decisions made, and significant emotional context.",
                    },
                    "outlier_ids": {
                        "type": "array",
                        "items": {"type": "integer"},
                        "description": "A list of Interaction IDs for messages that are thematic outliers "
                                       "and should be excluded from this summary.",
                    },
                },
                "required": ["observations", "outlier_ids"],
            },
        },
    },
    {
        "type": "google_grounding",
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
    # =========================================================================
    # Agent Management Tools
    # =========================================================================
    {
        "type": "function",
        "is_write": False,
        "service_binding": "agents",
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

# Model prefixes that do NOT support each tool.
# Uses the same prefix logic as engine.py routing.
# Tools not listed here are compatible with all providers.
GROUNDING_INCOMPATIBLE_PREFIXES = {'gpt', 'claude', 'gemma', 'gemini-3.1', 'local', 'unknown'}

MODEL_INCOMPATIBLE_TOOLS = {
    'google_grounding_search': GROUNDING_INCOMPATIBLE_PREFIXES,
}

# Derived from tool metadata — no manual maintenance needed when adding new tools.
WRITE_TOOLS = {t['function']['name'] for t in ALL_TOOL_DEFINITIONS if t.get('is_write')}
