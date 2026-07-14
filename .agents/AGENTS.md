# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Multiple agents work this repo

Both **Claude Code** and **Antigravity (Gemini)** work in this repository, sometimes on the same tasks. To keep instructions from drifting between tools:

- **CLAUDE.md is the single source of truth.** `.agents/AGENTS.md` (the file Antigravity reads) is an auto-generated copy, synced by `.githooks/pre-commit` — edit CLAUDE.md only; never hand-edit the mirror or fork tool-specific instructions.
- Whichever agent you are, follow the same DP-ID / worktree / memory-tier discipline below, so notes and code stay consistent no matter who wrote them.

## Commands

```bash
# Default test run (parallel via pytest-xdist, skips LLM live tests)
pytest -m "not llm_live" -n auto

# Run all tests (live tests auto-skip if no credentials)
pytest

# Unit + integration only (no external services needed)
pytest -m "not zammad_live and not llm_live and not discord_live" -n auto

# Unit tests only
pytest -m "not integration and not zammad_live and not llm_live and not discord_live" -n auto

# Zammad live tests only
pytest -m "zammad_live"

# LLM live tests only
pytest -m "llm_live"

# Run a single test file
pytest tests/test_engine.py

# Run with coverage
pytest --cov=src

# Lint
flake8 src/

# Type check
mypy src/ --config-file mypy.ini

# Run the application
python -m src.main
```

## Architecture

Async, provider-agnostic LLM orchestration engine for chatbot automation (IT support, ticketing, conversational AI). Full architectural detail lives in:

- **`memory/codebase/architecture.md`** — component reference, data flows, schemas, startup sequence
- **`docs/user_guide.md`** — user-facing behavior spec (commands, personas, tools, interfaces)

### Testing

4-tier test organization, ordered by execution:

1. **Unit** (no marker) — single component, everything mocked, no network
2. **Integration** (`@pytest.mark.integration`) — multi-component flows with mocked externals, no network
3. **Zammad Live** (`@pytest.mark.zammad_live`) — requires live Zammad instance (`ZAMMAD_URL` + `ZAMMAD_API_KEY`)
4. **LLM Live** (`@pytest.mark.llm_live`) — real LLM API calls, requires provider API keys

Live tests auto-skip when credentials are absent (via `tests/conftest.py`). Test Zammad credentials are stored in `.env.test` (gitignored), loaded with `override=True` so tests never hit production.

- Test fixtures and mock data in `tests/test_data/`

### Mandatory Test Requirements

When changing any of the following, you MUST add corresponding tests before committing:

**Database schema changes** (`memory_manager.py` CREATE TABLE / ALTER TABLE):
- Add migration tests using the `legacy_mem_manager` fixture pattern in `tests/memory/test_memory_manager.py`
- The fixture creates a DB with the OLD schema (before your change), then tests call `create_schema()` and verify the migration works
- Must test: column/table added, existing data preserved, indexes created, new features usable on migrated DB, idempotent on second run
- Unit tests with `:memory:` always start fresh and will NOT catch migration bugs against existing production databases

**Config schema changes** (`agents.json`, `system_personas.json`, `default_personas.json`, `global_config.py`):
- If adding/renaming/removing a config key: test that code handles the key being absent (old config files) and present (new config files)
- If a config value drives runtime behavior (e.g. `notification_defaults.channel`): test the behavior with realistic config, not just mocks
- Agent config: test via `AgentManager` dependency injection in `tests/agents/test_agent_manager.py`
- Persona config: test loading and field access in `tests/test_persona.py`

**Cross-module contracts** (imports, base class APIs, interface signatures):
- If renaming or moving a class/function: grep for all importers and update them in the same commit
- If changing a base class API (e.g. `Agent` or `AgentLoop`): update all subclasses and their tests in the same commit
- Run `mypy src/ --config-file mypy.ini` before committing any structural change

**Startup registration** (new `ServiceIntegration`, tool handler, or notifier):
- If a component must be registered at startup to function, test that the registration actually happens — not just that the component works in isolation
- The startup wiring test in `tests/integration/test_startup_wiring.py` asserts every tool `service_binding` in `ALL_TOOL_DEFINITIONS` has a registered handler; update it when adding new services


## Parallel Agent Standards

To support multiple agents working concurrently, the following rules are mandatory:

### 1. The DP-ID Anchor
- **Every** branch must follow: `feature/DP-XXX-slug` or `bugfix/DP-XXX-slug`.
- **Every** commit message must follow: `DP-XXX: description` (an optional Conventional-Commits type prefix is fine, e.g. `fix: DP-238 …`; no decorative emoji).
- **No `Co-Authored-By: Claude` trailer** and no "Generated with Claude Code" line — keep commit messages clean of tool attribution.
- **Every** task must have a corresponding file in `memory/project/tasks/DP-XXX.md`.

### 2. Workspace Isolation
- **MUST** use Git Worktrees — for *all* DP-XXX work, including solo single-task
  sessions, not just when agents run concurrently. The main repo directory is a
  shared mutable surface and may already hold another task's uncommitted edits.
- **⚠️ Always `git fetch` first, then branch off `origin/master` — never local
  `master`.** Local `master` drifts stale between sessions; cutting a worktree
  from it has produced duplicate DP-IDs and rework against an old base
  (DP-286). Fetching first also means open/unmerged PRs and stray local
  branches can't pollute your starting point:
  `git fetch origin && git worktree add worktrees/DP-XXX -b bugfix/DP-XXX-slug origin/master`.
  Do all editing, testing, and committing inside that worktree.
- **Never stage or commit from the main repo directory.** Before any
  `git add`/`git commit`, confirm you are inside `worktrees/DP-XXX/` and that
  `git status` shows only files *you* changed. If `git status` lists another
  DP's uncommitted changes (the main tree was left dirty), **STOP** — do not
  `git add` (a path-broad add sweeps that unrelated WIP into your commit).
- **Run `pytest` from inside the worktree** (using its own `.venv`), not from
  the main repo against worktree paths. `pytest`'s `pythonpath = src .` resolves
  relative to the run directory, so a top-level `pytest worktrees/DP-XXX/...`
  imports `src` from the *main* tree, not the worktree — silently masking real
  pass/fail.
- Each worktree gets its **own isolated `.venv`**, built automatically by the
  `post-checkout` hook via `uv` (real venv, packages hardlinked from uv's global
  cache — fast and cheap). No shared venv, no junctions.
- Worktree root: `worktrees/DP-XXX/`.
- Never run `git checkout` or `git pull` in the main repository directory while a parallel agent is working, as this can corrupt their local file state.

### 3. Integration & QA
- No task is considered "QA_READY" until `pytest` passes within its dedicated worktree.
- Verify all CI/CD checks (including `pytest`, `flake8` lint, and `mypy` type check as defined in `.github/workflows/deploy.yml`) after finishing any commit-ready change.
- Human approval is required for all merges to `main` or `develop`.
- After merge, tear down the worktree. Each worktree has its **own real venv**
  (no junction, nothing shared), so teardown is straightforward:
  1. Move every shell's cwd out of the worktree (a cwd inside it holds a Windows
     lock → "Device or resource busy").
  2. `git worktree remove worktrees/DP-XXX` (add `--force` only if you have
     uncommitted changes you intend to discard).
  3. `git worktree prune`.
  `git worktree remove` and `rm -rf` are safe here — the worktree's `.venv` is a
  real directory that touches nothing outside the worktree. (This replaced an
  earlier junction-based scheme whose recursive deletes repeatedly destroyed the
  shared venv; the per-worktree `uv` venv eliminated that footgun.)

## Documentation

Two living documents track the system's design:

- **`docs/user_guide.md`** — Describes what users can do. Serves as both end-user reference and **spec for new features**. When planning new behavior, describe it here first in plain language before implementing. This ensures alignment on what we're building.
- **`memory/codebase/architecture.md`** — Describes how the system works internally. Component reference for Claude's use across sessions.

**Update rules:**
- When implementing a new feature, update `user_guide.md` with the user-facing behavior (commands, tools, modes, etc.)
- When changing internal architecture (new components, changed data flows, new config), update `architecture.md`
- When you notice either doc is stale relative to the code, fix it — don't wait to be asked
- Spec-before-implement: if a design conversation produces a concrete behavior description, add it to `user_guide.md` before writing code

## Memory System — Viking L0/L1/L2 Protocol

A tiered memory system (inspired by [OpenViking](https://github.com/volcengine/OpenViking)) layered on the default auto-memory. Core principle — **progressive disclosure**: load the minimum context to decide, drill deeper only when needed. Use the system prompt's type/frontmatter conventions for individual files; organize and navigate them with the Viking tiers below.

### Structure

> **⚠️ Two-repo layout.** The `memory/` directory is gitignored in this repo and is its own separate git repo (`derpr-private-notes`, sibling directory). Commits to memory content go in that repo, not this one. When the user says "commit the plan/memory/task," operate in `memory/` (use `git -C memory ...` or `cd memory && git ...`). The codebase repo and the memory repo have independent histories and branches.

Memory lives in the project-scoped memory directory with three tiers:

- **L0** — `MEMORY.md` (auto-loaded every session). **Strictly an index** — one-line directory summaries only (~20 lines). No memory content belongs here; only pointers to help decide which L1 files are relevant. When the system prompt instructs you to "add a pointer to MEMORY.md," add a single summary line under the appropriate directory heading, not memory content.
- **L1** — `<dir>/_overview.md` files. Component summaries, relationships, current state. ~200-300 tokens each. Purpose: usually sufficient for decision-making.
- **L2** — Individual detail files. Full content. Purpose: schemas, signatures, implementation specifics. Only read when L1 isn't enough.

```
memory/
├── MEMORY.md              # L0 — always loaded, directory summaries
├── codebase/
│   ├── _overview.md       # L1 — component map, pipeline, key patterns
│   └── architecture.md    # L2 — full structural detail
├── user/
│   ├── _overview.md       # L1 — who Adam is, collaboration guide
│   ├── profile.md         # L2 — full background
│   └── feedback.md        # L2 — behavioral rules (universal + project-specific)
├── project/
│   ├── _overview.md       # L1 — active work, decisions, roadmap summary
│   ├── decisions/         # L2 — immutable records with rationale
│   └── plans/             # L2 — appendable roadmaps
└── external/
    ├── _overview.md       # L1 — external system references
    └── *.md               # L2 — specific external details
```

### Navigation Rules

1. L0 (`MEMORY.md`) is always in context — use it to decide which directories are relevant
2. Read L1 (`_overview.md`) before drilling into L2 files
3. Only read L2 when you need implementation-level detail (schemas, function signatures, DB tables)
4. For code-related work, `codebase/_overview.md` is almost always worth reading

### Mutability Rules

| Category | Rule | Notes |
|----------|------|-------|
| `user/` | Appendable | Profile, preferences, feedback evolve over time |
| `project/decisions/` | Immutable | Historical choices with rationale — never modify, only add new |
| `project/plans/` | Appendable | Active roadmaps, update as work progresses |
| `codebase/` | Regenerable | Can be rebuilt from source if stale — trust code over memory |
| `external/` | Appendable | Update when external facts change |

### Memory Update Triggers

**Automated (hook-enforced):** A git pre-commit hook writes a marker file (`.claude/.memory_update_pending`). A `UserPromptSubmit` hook in `settings.local.json` checks for it and injects a reminder on the next user message. When you see this reminder, review what was committed and update affected L2 → L1 → L0 files before proceeding with new work.

**Self-directed:** the hook only catches commits — be vigilant where none fires. Save: user feedback/corrections on how to work; the *rationale* behind an architectural decision (code shows what, memory stores why); research conclusions (tool evals, security findings, API discoveries); new or revised plans; facts about the user's background/goals; root-cause reasoning for a non-obvious bug. Test: "would a future session benefit, and is it not derivable from code/git?" → if yes, save it.

**Self-check:** if a session involved research, decisions, or feedback and you've written no memory, you're probably missing something — discussions and planning have no commit-hook safety net.

### Update Protocol

When updating memory (whether triggered by hook or self-directed):
1. Identify which memory directories are affected
2. Update affected L2 files (or create new ones)
3. Regenerate affected L1 `_overview.md` bottom-up from L2 content
4. Update L0 `MEMORY.md` if directory-level summaries changed

### Staleness Rule

If an L1 or L2 memory conflicts with what you observe in the code, **trust the code**. Update the memory. Do not act on stale information.

### Scope
Project-scoped. `user/` profile + universal feedback are conceptually global (may need manual sync across projects); codebase/project memories must never leak across projects.

### Hindsight Recall (supplementary)
The `mcp__hindsight__*` tools auto-ingest session transcripts for associative `recall` — **supplementary to Viking, which stays authoritative** (Hindsight has stale entries; trust Viking on conflicts). Use recall for episodic gaps Viking missed; flag stale/wrong-project results in conversation. Mental-model candidates: `project/plans/hindsight_mental_models.md`.
