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
