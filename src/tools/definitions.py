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
]
