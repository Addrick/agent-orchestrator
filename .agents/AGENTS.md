# CLAUDE.md

Async, provider-agnostic LLM orchestration engine (Discord / Gmail / Zammad interfaces).

**This file holds only what is specific to *this* repo.** The workflow and behavioural
rules — ticket IDs, fetch-before-read, worktrees, staging, QA gates, PR/merge policy,
docs-only changes, the memory protocol, working with Adam — live once in
`~/.claude/CLAUDE.md` and are not repeated here. Put new detail behind a pointer rather
than in this file.

**Two agents work this repo** — Claude Code and Antigravity (Gemini). CLAUDE.md is the
single source of truth; `.agents/AGENTS.md` is an auto-generated mirror synced by
`.githooks/pre-commit`. Never hand-edit the mirror, and never fork tool-specific
instructions.

## Commands

```bash
pytest -m "not llm_live" -n auto                    # default test run
flake8 src/ services/                               # lint (services/ is deployed code; CI gates it)
mypy src/ services/ --config-file mypy.ini          # type check
lint-imports                                        # layer contracts (setup.cfg)
python -m src.main                                  # run
```

Those four, plus `scripts/check_missing_deps.py`, are the CI gates
(`.github/workflows/deploy.yml`); `python scripts/ci_check.py` runs all of them, and is the
pre-push hook. ⚠️ CI and the hook run pytest with `-m "not integration"` and only flake8's
hard-error subset — the default run above is the stricter one. A push to `master` that
touches only `docs/`, `readme.md` or `.agents/` skips the pipeline entirely (no rebuild, no
prod restart); **a `CLAUDE.md` edit still deploys**, because the image seeds it into `cc-*`
persona workspaces. Test tiers, markers, and
the **mandatory test requirements** for schema/config/contract/startup changes:
**`docs/testing.md`** — read it before changing any of those four things.

## This repo's parameters

| | |
|---|---|
| Ticket prefix | `DP` — ticket file `memory/project/tasks/DP-XXX.md` |
| Default branch | `master`; worktrees cut from `origin/master` into `worktrees/DP-XXX/` |
| Docs-only set | `docs/`, `readme.md`, `CLAUDE.md`, `.agents/AGENTS.md` — anything under `src/`, `services/`, `config/`, `tests/`, `scripts/` or `.github/` makes it a normal ticket |
| Notes repo | ⚠️ `memory/` is a **separate git repo** (`derpr-private-notes`), gitignored here — "commit the plan / memory / task" means `git -C memory ...`, independent history and branches |

Worktree venv provisioning, hook behaviour and teardown:
`memory/infrastructure/worktree_env_and_hooks.md`.

## Docs — and the duplication problem they exist to solve

| File | Answers |
|---|---|
| `docs/user_guide.md` | what users can do. **Also the spec** — describe new behavior here in plain language *before* implementing |
| `docs/architecture/architecture.md` | how it works internally. **Canonical** (`memory/codebase/architecture.md` is a pointer stub) |
| `docs/capability_map.md` | capability → implementations. The inverse index the other two structurally cannot be |
| `docs/mechanism_ledger.md` | the same question keyed on **mechanism** — trigger · durable store · idempotency · resumer · gate. Regenerated from source, not maintained |

Update `user_guide.md`, `architecture.md` and `capability_map.md` **in the same commit as
the code**. A second implementation of an existing capability is recorded as `by design`
with a decision record, or it is not written.

**Search by mechanism, never by name.** Before adding a capability ask "does anything
already do this job?", not "is there a function called `approve_x`?". Code here is
*regenerated*, not copy-pasted, so a re-derived implementation shares neither the name nor
the vocabulary of the original and no clone detector will find it. The capability map
catches this only when the author recognizes the sibling; the mechanism ledger is what
catches it when they do not — **two rows with the same shape and different module names
are a duplicate.** `python scripts/arch_audit.py similar concepts` covers the
function-level case.

> This is not hypothetical: `docs/mechanism_ledger.md` exists because a fourth copy of the
> deferred-result pipeline shipped 26 days after the capability map was created — *while
> its author was filling in a row of that map*.
