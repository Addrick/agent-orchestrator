#!/usr/bin/env python3
"""Claude Code PreToolUse hook: prompt for memory updates before `git commit`.

Wired in `.claude/settings.local.json` against the Bash + PowerShell tools.
Reads tool-input JSON from stdin, inspects the command for a git-commit
invocation, and if found, runs `git diff --cached` to see whether the staged
changes touch only code (src/ or tests/) without any companion memory/
edits.

Decision matrix (exit code → effect):
  0 → allow (no commit, only memory edits, or [skip-memory] in command)
  2 → block; stderr message is injected into Claude's context

If blocked, Claude has the chance to either (a) update memory and re-stage,
or (b) re-run with `[skip-memory]` in the commit command to bypass after
explicit consideration.
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path


# Match `git commit` with optional flags, but not `git commit-tree` or
# `git commit-graph`. Also catches `git -C path commit` and amends.
_GIT_COMMIT_RE = re.compile(r"\bgit(?:\s+-[A-Z]\s+\S+)*\s+commit\b(?!-)")

_SKIP_TOKEN = "[skip-memory]"

_CODE_PREFIXES = ("src/", "tests/", "scripts/", "config/")
_MEMORY_PREFIXES = ("memory/",)


def _is_git_commit(command: str) -> bool:
    return bool(_GIT_COMMIT_RE.search(command))


def _staged_files() -> list[str]:
    try:
        out = subprocess.check_output(
            ["git", "diff", "--cached", "--name-only"],
            text=True,
            stderr=subprocess.DEVNULL,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return []
    return [line.strip() for line in out.splitlines() if line.strip()]


def _classify(files: list[str]) -> tuple[list[str], list[str]]:
    code = [f for f in files if f.startswith(_CODE_PREFIXES)]
    mem = [f for f in files if f.startswith(_MEMORY_PREFIXES)]
    return code, mem


def main() -> int:
    try:
        payload = json.load(sys.stdin)
    except json.JSONDecodeError:
        return 0  # Don't block on malformed input

    tool_name = payload.get("tool_name") or payload.get("toolName") or ""
    if tool_name not in ("Bash", "PowerShell"):
        return 0

    tool_input = payload.get("tool_input") or payload.get("toolInput") or {}
    command = tool_input.get("command", "") or ""

    if not _is_git_commit(command):
        return 0

    if _SKIP_TOKEN in command:
        return 0

    code, mem = _classify(_staged_files())

    if not code:
        return 0  # No code changes — nothing to capture rationale for
    if mem:
        return 0  # Already paired with memory edits

    # Code changes without memory companion — prompt.
    sys.stderr.write(
        "Memory update check: staged code changes have no companion "
        "memory/ edits.\n\n"
        f"Staged code files ({len(code)}):\n"
        + "\n".join(f"  - {f}" for f in code[:10])
        + ("\n  ..." if len(code) > 10 else "")
        + "\n\n"
        "Before committing, consider whether this session captured anything "
        "worth saving in memory:\n"
        "  - A non-obvious design decision or rationale → "
        "memory/project/decisions/YYYY-MM-DD-<slug>.md\n"
        "  - Plan revisions or new sprint state → "
        "memory/project/plans/<plan>.md\n"
        "  - User feedback / corrections → "
        "memory/user/feedback.md\n"
        "  - Architectural changes (new components, data flows) → "
        "memory/codebase/architecture.md and _overview.md\n\n"
        "Then re-stage and re-run the commit. If the change has no "
        "memory-worthy context (trivial fix, mechanical refactor), "
        f"include `{_SKIP_TOKEN}` anywhere in the bash command to bypass "
        "this check.\n"
    )
    return 2


if __name__ == "__main__":
    sys.exit(main())
