---
name: Agy state improvement (future)
description: Transitioning the agy provider from throwaway temp dirs to persistent persona-scoped workspaces to enable caching and CLI continuity
type: project
status: completed
---

# Agy State Improvement — Future Work

Proposed 2026-06-10. Not scheduled.

## Goal

Improve the performance, latency, and context utilization of the local Google Antigravity `agy` CLI integration by moving from stateless, throwaway temp directories to persistent, persona-scoped workspaces (`data/workspaces/agy_{persona_name}/`).

---

## Current Stateless Implementation

Currently, every time the DERPR engine calls `_run_agy_cli` in `src/engine.py`, it performs the following:
1. Creates a unique temporary directory via `tempfile.mkdtemp()`.
2. Spawns `agy` in that directory.
3. Completely deletes the directory and all internal folders (such as `temp_dir/.antigravitycli/`) upon completion.

### The Problem
* **No Cache Retention**: `agy` cannot cache codebase structures, git metadata, or authorization states.
* **Initialization Overhead**: Every execution must re-initialize its local workspace environment from scratch, increasing latency (adding several seconds per tool-loop turn).
* **Context Overhead**: Because `agy` has zero persistent workspace state, DERPR is forced to pass full codebase snapshots and context histories in the prompt on every loop turn.

---

## Proposed Persistent Workspace Design

We can apply the same **Persona-Scoped Workspace** concept proposed for Claude Code to improve the `agy` CLI:

### 1. Workspace Structure
Instead of a random temp path, define a deterministic directory:
```python
workspace_dir = os.path.abspath(f"data/workspaces/agy_{persona_name}")
os.makedirs(workspace_dir, exist_ok=True)
```

### 2. Execution Update (`src/engine.py`)
Modify `_run_agy_cli` to:
* Set `cwd=workspace_dir` when starting the subprocess.
* Skip the `shutil.rmtree` teardown block for the main workspace, leaving `workspace_dir` and the `.antigravitycli/` folder intact.
* Only clean up actual symbolic links inside the directory on completion, rather than wiping the entire cache.

### 3. codebase Synchronization (Optional)
If we want `agy` to have a read-only or read-write view of the local project:
* Set up directory symlinks or git worktrees from the main codebase folder to `data/workspaces/agy_{persona_name}/` on initialization.
* If a write operation is confirmed, sync files back to the main directory.

---

## Benefits

1. **Reduced Latency**: Subprocess execution will execute faster since `agy`'s internal caching layers remain hot between turns.
2. **Stateful Continuity**: `agy` CLI options, configuration presets, and authorization tokens will persist automatically between turns within a conversation session.
3. **Common Architecture**: Alignment with the Claude Code integration workspace model (Option C), sharing identical workspace setup utility classes.

---

## Security Invariants

* **Prompt Constraint**: We continue to pass `"use no other tools/files/shell/web"` in `_render_agy_tool_protocol` so that `agy` acts as a text/reasoning engine rather than trying to execute commands autonomously.
* **OS-level Sandbox**: `agy` is invoked with `--sandbox` (nsjail on Linux, sandbox-exec on macOS) by default (`AGY_SANDBOX`). The cwd alone is *not* a security boundary; the sandbox is. This must stay on before the prompt constraint is ever lifted.

---

## Implemented (DP-208, 2026-06-12)

Shipped as designed, with these deltas from the proposal:

* Persona names are sanitized to a filesystem-safe slug (`agy_{slug}`); empty/missing names fall back to `agy_global`.
* Concurrent calls sharing a workspace are serialized via a per-workspace `asyncio.Lock` (the old temp-dir design was race-free by construction; shared dirs are not).
* The `.antigravitycli` symlink-target cleanup runs **only** for throwaway temp dirs — persistent workspaces keep that state on purpose.
* `--sandbox` added (config `AGY_SANDBOX`, default on).
* Note: newer agy versions store workspace→project mappings centrally in `~/.gemini/antigravity-cli/cache/projects.json` keyed by workspace path — a stable cwd keeps that cache identity stable across calls, which is most of the win.
* Codebase synchronization (section 3) not implemented — deferred until agentic use is turned on.
