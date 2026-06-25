# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Multiple agents work this repo

Both **Claude Code** and **Antigravity (Gemini)** operate in this repository, sometimes on the same tasks. To stop instructions from drifting between tools:

- **CLAUDE.md is the canonical, single source of truth.** `.agents/AGENTS.md` (the file Antigravity reads) is an **auto-generated copy** of this file, synced by the `.githooks/pre-commit` hook. **Edit CLAUDE.md only** — never edit `.agents/AGENTS.md` by hand (the hook overwrites it) and never fork tool-specific instructions.
- Regardless of which agent you are, follow the **same** DP-ID / worktree / memory-tier discipline below, so notes and code stay consistent no matter who wrote them.
- `ANTIGRAVITY.md` at the repo root is **not** read by anything (Antigravity uses `.agents/AGENTS.md`; `.antigravity/` is its system-state folder). It is vestigial — ignore it.

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
- **Every** commit message must follow: `<gitmoji> DP-XXX: description`.
- **Every** task must have a corresponding file in `memory/project/tasks/DP-XXX.md`.

### 2. Workspace Isolation
- **MUST** use Git Worktrees — for *all* DP-XXX work, including solo single-task
  sessions, not just when agents run concurrently. The main repo directory is a
  shared mutable surface and may already hold another task's uncommitted edits.
- **Start every DP-XXX by creating its worktree off a clean `master`:**
  `git worktree add worktrees/DP-XXX -b bugfix/DP-XXX-slug master`. Do all
  editing, testing, and committing inside that worktree.
- **Never stage or commit from the main repo directory.** Before any
  `git add`/`git commit`, confirm you are inside `worktrees/DP-XXX/` and that
  `git status` shows only files *you* changed. If `git status` lists another
  DP's uncommitted changes (the main tree was left dirty), **STOP** — do not
  `git add` (a path-broad add will sweep up that WIP into your commit, which is
  exactly how DP-142's first commit was contaminated).
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

This project uses a tiered memory system inspired by [OpenViking](https://github.com/volcengine/OpenViking)'s context database, layered on top of the default auto-memory system. The core principle is **progressive disclosure**: load the minimum context needed to make decisions, and only drill deeper when required. This avoids dumping large documents into every conversation and keeps the context window efficient.

**Relationship to auto-memory:** The system prompt describes a flat memory system with types and frontmatter. This protocol extends it with hierarchical organization and navigation rules. Follow both: use the system prompt's type/frontmatter conventions for individual files, but organize and navigate them using the Viking tiers below.

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

**Self-directed:** The hook only catches commits. You must be vigilant about updating memory in contexts where no hook fires. Common situations:

- User gives feedback or corrections about how to work ("don't do X", "yes that approach was right") — these are high-value and easy to miss
- An architectural decision is made during discussion — capture the *rationale*, not just the choice. The code shows what; memory stores why.
- Research or exploration surfaces conclusions worth keeping (tool evaluations, security findings, API discoveries) — session context vanishes, but the conclusions shouldn't
- A plan is created or substantially revised
- You learn something new about the user's background, role, or goals
- A non-obvious bug is resolved — the root cause reasoning is often not in the commit message or code

When in doubt, ask: "would a future session benefit from knowing this?" If yes, save it. If it's derivable from the code or git history, don't.

**Self-check:** If this conversation involved research, decisions, or user feedback and you have NOT written any memory files this session, you are probably missing something. Review the conversation for saveable context before it ends. Conversations about code changes are covered by the commit hook, but discussions, explorations, and planning sessions have no safety net — if you don't save it, it's gone.

### Update Protocol

When updating memory (whether triggered by hook or self-directed):
1. Identify which memory directories are affected
2. Update affected L2 files (or create new ones)
3. Regenerate affected L1 `_overview.md` bottom-up from L2 content
4. Update L0 `MEMORY.md` if directory-level summaries changed

### Staleness Rule

If an L1 or L2 memory conflicts with what you observe in the code, **trust the code**. Update the memory. Do not act on stale information.

### Project Scope

This memory system is scoped to this project. User profile and universal feedback (preferences that apply across all projects) are stored here but are conceptually global. If working across multiple projects, these may need manual synchronization. Codebase and project memories are correctly project-scoped and should never leak across projects.

### Hindsight Recall (supplementary)

A separate memory layer is available via the `mcp__hindsight__*` tools. Hindsight auto-ingests Claude Code session transcripts and exposes them through associative `recall`. It is **supplementary**, not a replacement for Viking.

- **Viking is authoritative.** Hindsight is new, the DB has stale entries from earlier sessions, and mental-model tuning is in progress. When recall and a Viking note disagree, trust Viking.
- **Recall for episodic gaps** — call it when you hit project-specific context Viking didn't give you (referenced prior work, unfamiliar artifact, surprising local state). Overlap with Viking is fine right now; coverage matters more than minimalism.
- **Flag bad recalls in conversation** — stale, wrong-project, or Viking-contradicting results. No file needed; visibility drives DB and mission tuning.

Mental model candidates under evaluation are tracked in `project/plans/hindsight_mental_models.md`.
