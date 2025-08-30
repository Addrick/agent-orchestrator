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
ALL_TOOL_DEFINITIONS: List[Dict[str, Any]] = [
    {
        "type": "function",
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
        "function": {
            "name": "update_ticket",
            "description": "Updates one or more properties of an existing Zammad ticket.",
            "parameters": {
                "type": "object",
                "properties": {
                    "ticket_id": {
                        "type": "integer",
                        "description": "The unique numerical ID of the ticket to update.",
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
        "function": {
            "name": "add_note_to_ticket",
            "description": "Adds a new article (a note or comment) to an existing Zammad ticket.",
            "parameters": {
                "type": "object",
                "properties": {
                    "ticket_id": {
                        "type": "integer",
                        "description": "The unique numerical ID of the ticket to add a note to.",
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
        "function": {
            "name": "create_ticket",
            "description": "Creates a new Zammad ticket. Requires a title and a body for the first article. The system will automatically associate it with the current user.",
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
                    }
                },
                "required": ["title", "body"],
            },
        },
    },
    {
        "type": "function",
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
]
