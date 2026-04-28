# 🛤️ Parallel Agent Engineering Workflow

This document specifies how multiple AI agents collaborate safely on the `derpr-python` repository.

## 1. Concurrency Model: Git Worktrees

To avoid file-system collisions (e.g., Agent A's `git checkout` changing the files Agent B is currently editing), we use **Git Worktrees**.

### How it works
A worktree allows you to have multiple branches checked out at the same time in different directories, all sharing the same `.git` database.

**Commands for Agents:**
```bash
# 1. Create a branch for the task
git branch feature/DP-101-fix-bug

# 2. Add a worktree in the 'worktrees/' folder
git worktree add worktrees/DP-101 feature/DP-101-fix-bug

# 3. Enter the worktree and work there
cd worktrees/DP-101
# ... implement ...
pytest

# 4. Once merged, clean up
git worktree remove worktrees/DP-101
```

## 2. Parallelizability Matrix (Coaching)

Not all tasks can be done in parallel. Use this matrix to determine if you can "double-up" on work.

| Task A | Task B | Parallelizable? | Rationale |
| :--- | :--- | :--- | :--- |
| **Logic (src/engine.py)** | **UI (portal/)** | ✅ YES | Decoupled components. |
| **DB Schema change** | **DB Schema change** | ❌ NO | Migrations must be sequential to avoid version drift. |
| **Refactor Module X** | **Feature in Module X** | ❌ NO | Massive merge conflicts guaranteed. |
| **Tests for Module Y** | **Implementation of Y** | ⚠️ CAUTION | Use TDD; ideally same agent or very tight coordination. |
| **Docs** | **Anything else** | ✅ YES | Zero code impact. |

### Rule of Thumb:
If the **intersection** of the "Files Affected" lists in the DP tickets is **empty**, you are generally safe to proceed in parallel.

## 3. The Implementation Lifecycle

1. **CLAIM**: Mark a DP ticket as `IN_PROGRESS` and set yourself as `Assignee`.
2. **ISOLATE**: Create a `git worktree`.
3. **IMPLEMENT**: Code, test, and commit using the `✨ DP-XXX: description` format.
4. **AUDIT**: Ask a "Senior Agent" persona to review your worktree changes.
5. **SIGNAL**: Update the DP ticket to `REVIEW` and provide a link to the branch.
6. **MERGE**: Wait for human approval. After the merge, the user or a script will run `git worktree remove`.

## 4. Audit & Quality Gate

Every task MUST pass the following before being presented for human review:
- [ ] `pytest` passes in the worktree.
- [ ] `mypy` passes.
- [ ] `flake8` passes.
- [ ] No secrets committed (checked via internal audit).
- [ ] Documentation updated if necessary.
