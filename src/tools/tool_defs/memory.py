"""Core memory tools (no service_binding — core)."""

from typing import Any, Dict, List


MEMORY_TOOLS: List[Dict[str, Any]] = [
    {
        "type": "function",
        "is_write": False,
        "capabilities": {
            # DP-113 / tool_security_framework.md: recall surfaces previously-
            # ingested external content (chat history, tool output) — origin
            # is untrusted even though the read is local.
            "produces_untrusted": True,
            "irreversible": False,
            "locality": "local",
            "sensitivity": "internal",
        },
        "function": {
            "name": "recall_memory",
            "description": (
                "Search the persona's long-term memory bank for facts relevant to a "
                "natural-language query. Returns up to `limit` hits — each is a short "
                "summary of a past conversation or observation. Use when the user "
                "references something you don't see in the recent message window. "
                "Scope (persona, channel, user, server) is inherited from the active "
                "turn — you cannot query another persona's bank."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Natural-language question or topic to recall.",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of memory hits to return (default 10).",
                        "default": 10,
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "is_write": False,
        "capabilities": {
            # DP-424: same trust shape as recall_memory — the unit and its
            # source facts were extracted from ingested (untrusted) content.
            "produces_untrusted": True,
            "irreversible": False,
            "locality": "local",
            "sensitivity": "internal",
        },
        "function": {
            "name": "get_memory",
            "description": (
                "Fetch one long-term memory by the `id` of a recall_memory hit, to "
                "see where it came from. Returns its `document_id` and `chunk_id` "
                "(pass them to get_document for the original text) and, for a "
                "consolidated observation, the `source_memories` it was built from. "
                "Only memories recall_memory could return in this conversation are "
                "visible; anything else reads as not found."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "memory_id": {
                        "type": "string",
                        "description": "The `id` of a recall_memory hit or a source memory.",
                    },
                },
                "required": ["memory_id"],
            },
        },
    },
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
            "name": "get_document",
            "description": (
                "Read the original text a memory was extracted from, usually the "
                "whole past conversation. Use when a recalled fact lacks the detail "
                "you need (exact wording, a link, surrounding discussion). Pass "
                "`chunk_id` to get only the passage that produced the fact; "
                "otherwise the text is returned a page at a time — follow "
                "`next_offset` to read on."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "document_id": {
                        "type": "string",
                        "description": "The `document_id` from a recall_memory hit or get_memory.",
                    },
                    "chunk_id": {
                        "type": "string",
                        "description": "Optional `chunk_id` — return only that passage.",
                    },
                    "offset": {
                        "type": "integer",
                        "description": "Character offset to start reading from (default 0).",
                        "default": 0,
                    },
                    "max_chars": {
                        "type": "integer",
                        "description": "Maximum characters to return (default 6000, max 20000).",
                        "default": 6000,
                    },
                },
                "required": ["document_id"],
            },
        },
    },
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
    },
    {
        "type": "function",
        "is_write": True,
        "capabilities": {
            "produces_untrusted": True,
            "irreversible": False,
            "locality": "local",
            "sensitivity": "user",
        },
        "function": {
            "name": "ingest_path",
            "description": (
                "Ingest a markdown file or directory of notes into the persona's "
                "long-term memory bank. Idempotent: unchanged files are skipped via "
                "a local hash cache. Requires the Hindsight memory backend; on the "
                "SQLite backend the call is a noop with a warning."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "File or directory path to ingest.",
                    },
                    "glob": {
                        "type": "string",
                        "description": "Glob filter applied when path is a directory.",
                        "default": "**/*.md",
                    },
                    "bank": {
                        "type": "string",
                        "description": (
                            "Override target bank id. Defaults to persona.ingest_bank "
                            "if set, otherwise the persona's own name."
                        ),
                    },
                    "force": {
                        "type": "boolean",
                        "description": "Bypass the local hash cache and re-ingest all matches.",
                        "default": False,
                    },
                },
                "required": ["path"],
            },
        },
    },
]
