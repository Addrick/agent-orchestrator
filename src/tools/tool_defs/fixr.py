"""fixr self-improvement supervisor tools (service_binding: fixr)."""

from typing import Any, Dict, List


FIXR_TOOLS: List[Dict[str, Any]] = [
    {
        "type": "function",
        "is_write": True,
        "service_binding": "fixr",
        "capabilities": {
            "produces_untrusted": True,
            "irreversible": False,
            "locality": "local",
            "sensitivity": "internal",
        },
        "function": {
            "name": "dispatch_fix",
            "description": (
                "Spawn a detached Claude Code agent to fix ONE bug in derpr's own "
                "codebase. The agent runs in an isolated git worktree branched off "
                "master, diagnoses, fixes, tests, and opens a PR (it never merges). "
                "This requires human approval before the agent starts. One agent "
                "per bug — do not dispatch duplicates."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "bug_id": {
                        "type": "string",
                        "description": "The DP-XXX task id for this bug (used for the branch + worktree).",
                    },
                    "description": {
                        "type": "string",
                        "description": "Clear statement of the bug: symptom, where it shows, repro if known.",
                    },
                },
                "required": ["bug_id", "description"],
            },
        },
    },
    {
        "type": "function",
        "is_write": False,
        "service_binding": "fixr",
        "capabilities": {
            "produces_untrusted": True,
            "irreversible": False,
            "locality": "local",
            "sensitivity": "internal",
        },
        "function": {
            "name": "inspect_agents",
            "description": (
                "List dispatched bug-fix agents and their status (running, "
                "waiting, done, error, killed), or inspect one by agent_id. The "
                "management view: branch, PR url, last event, whether it can be resumed."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "agent_id": {
                        "type": "string",
                        "description": "Inspect a single agent. Omit to list all.",
                    },
                    "active_only": {
                        "type": "boolean",
                        "description": "When listing, include only running/waiting agents.",
                        "default": False,
                    },
                    "include_archived": {
                        "type": "boolean",
                        "description": "When listing, also show pruned/archived agents (hidden by default).",
                        "default": False,
                    },
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "is_write": False,
        "service_binding": "fixr",
        "capabilities": {
            "produces_untrusted": True,
            "irreversible": False,
            "locality": "local",
            "sensitivity": "internal",
        },
        "function": {
            "name": "answer_agent",
            "description": (
                "Resume a WAITING agent (one that asked a question) with your "
                "decision. The agent continues headlessly from where it paused. "
                "Use this to unblock an agent on a choice you can confidently make; "
                "escalate genuine forks to a human instead."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "agent_id": {
                        "type": "string",
                        "description": "The agent to resume (from inspect_agents).",
                    },
                    "message": {
                        "type": "string",
                        "description": "Your answer/decision/redirection for the agent.",
                    },
                },
                "required": ["agent_id", "message"],
            },
        },
    },
    {
        "type": "function",
        "is_write": False,
        "service_binding": "fixr",
        "capabilities": {
            "produces_untrusted": False,
            "irreversible": False,
            "locality": "local",
            "sensitivity": "internal",
        },
        "function": {
            "name": "kill_agent",
            "description": (
                "Stop a stuck or runaway agent's process and event bridge. "
                "Optionally remove its worktree. Use when an agent is looping, "
                "hung, or you are re-dispatching the bug fresh."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "agent_id": {
                        "type": "string",
                        "description": "The agent to kill.",
                    },
                    "remove_worktree": {
                        "type": "boolean",
                        "description": "Also tear down the agent's git worktree.",
                        "default": False,
                    },
                },
                "required": ["agent_id"],
            },
        },
    },
    {
        "type": "function",
        "is_write": False,
        "service_binding": "fixr",
        "capabilities": {
            "produces_untrusted": False,
            "irreversible": False,
            "locality": "local",
            "sensitivity": "internal",
        },
        "function": {
            "name": "prune_agents",
            "description": (
                "Reap finished agents: delete the on-disk git worktrees of "
                "terminal agents (done/error/killed/orphaned) and archive their "
                "records (kept for audit, hidden from the default list). Use to "
                "free disk after PRs land. Prune one with agent_id, or bound by "
                "max_age_hours. Active agents and bugs with an in-flight agent "
                "are never touched."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "agent_id": {
                        "type": "string",
                        "description": "Prune a single terminal agent. Omit to prune all eligible.",
                    },
                    "max_age_hours": {
                        "type": "number",
                        "description": "Only prune agents untouched for at least this many hours.",
                    },
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "is_write": False,
        "service_binding": "fixr",
        "capabilities": {
            "produces_untrusted": False,
            "irreversible": False,
            "locality": "network",
            "sensitivity": "internal",
        },
        "function": {
            "name": "send_discord",
            "description": (
                "Post a curated report to the team's Discord channel — e.g. an "
                "agent opened a PR (include the link), or a fork needs a human "
                "decision. You decide what is worth reporting; do not forward raw "
                "agent output. Defaults to the configured fixr channel."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "subject": {
                        "type": "string",
                        "description": "Short headline for the report.",
                    },
                    "body": {
                        "type": "string",
                        "description": "The report body (markdown ok).",
                    },
                    "recipient": {
                        "type": "string",
                        "description": "Override Discord channel/recipient id. Defaults to CC_FIXR_DISCORD_CHANNEL.",
                    },
                },
                "required": ["subject", "body"],
            },
        },
    },
]
