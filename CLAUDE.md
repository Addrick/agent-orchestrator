# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Run all tests (live tests auto-skip if no credentials)
pytest

# Unit + integration only (no external services needed)
pytest -m "not zammad_live and not llm_live and not discord_live"

# Unit tests only
pytest -m "not integration and not zammad_live and not llm_live and not discord_live"

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

Pre-commit hooks run unit + integration tests (excluding live tests) — do not bypass them.

## Architecture

This is an async, provider-agnostic LLM orchestration engine for chatbot automation (IT support, ticketing, conversational AI). The main interfaces are a Discord bot, Gmail bot, and Zammad polling bot. All LLM I/O is async.

### Core Data Flow

```
Interface (Discord/Gmail/Zammad bot)
  → ChatSystem.preprocess_message()   # command detection
  → BotLogic (message_handler.py)     # command dispatch or context building
  → TextEngine (engine.py)            # provider-agnostic LLM call
  → ToolManager (tool_manager.py)     # agentic tool execution loop (max 5 calls)
  → MemoryManager (memory_manager.py) # persist to SQLite
  → Response back to interface
```

### Key Components

**`src/engine.py` — TextEngine**
The LLM abstraction layer. Handles OpenAI, Google (Gemini 2.5/3.1, Gemma), Anthropic, and local OpenAI-compatible endpoints. Manages per-model-family rate limiters (`AsyncLimiter`), translates tool definitions to provider-specific formats, handles image inputs, and retries on empty responses. 429 errors fail fast (no backoff) to preserve quota.

**`src/chat_system.py` — ChatSystem**
Dependency injection hub. Holds references to all `Persona` objects, `TextEngine`, `MemoryManager`, and `ZammadClient`. Routes messages and manages the ticket lifecycle (creates Zammad tickets for support-channel messages, lazy-creates Zammad users on demand).

**`src/message_handler.py` — BotLogic**
All bot commands (`set`, `what`, `remember`, `hello`, `goodbye`, etc.) are parsed and dispatched here. Also builds context windows from `MemoryManager` based on the persona's `MemoryMode`.

**`src/persona.py` — Persona**
Stateful LLM configuration object: model, system prompt, token limit, temperature, `ExecutionMode` (SILENT_ANALYSIS vs ASSISTED_DISPATCH), and `MemoryMode` (CHANNEL_ISOLATED, SERVER_WIDE, PERSONAL, GLOBAL, TICKET_ISOLATED). Runtime `set` commands mutate in-memory state.

**`src/tools/` — Tool System**
`definitions.py` holds JSON schemas for all tools (Zammad CRUD, web search, Google Grounding). `tool_manager.py` executes them. The engine loops on tool_calls until none remain or MAX_TOOL_CALLS (5) is hit. User-facing Zammad ticket numbers are translated to internal IDs inside the tool manager.

**`src/database/memory_manager.py`**
SQLite (`user_memory.db`) with a single persistent connection (`check_same_thread=False`). All DB operations run via `asyncio.to_thread()` to avoid blocking the event loop.

### Persona Persistence

- `config/default_personas.json` and `config/system_personas.json` — tracked in git, seed production
- `data/personas.json` — local only (gitignored), holds runtime state; overrides defaults on startup

### Rate Limits

Configured in `config/global_config.py`. Split by model family (not provider): Gemini 2.5 has both RPM and RPD limiters; other families only RPM.

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
- Add migration tests using the `legacy_mem_manager` fixture pattern in `tests/database/test_memory_manager.py`
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

## Memory System — Viking L0/L1/L2 Protocol

This project uses a tiered memory system inspired by [OpenViking](https://github.com/volcengine/OpenViking)'s context database. The core principle is **progressive disclosure**: load the minimum context needed to make decisions, and only drill deeper when required. This avoids dumping large documents into every conversation and keeps the context window efficient.

### Structure

Memory lives in the project-scoped memory directory with three tiers:

- **L0** — `MEMORY.md` (auto-loaded every session). Directory-level summaries, ~20 lines. Purpose: decide what's relevant without reading anything else.
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

**Automated (hook-enforced):** A post-commit hook writes a marker file that injects a reminder on the next user message. When you see this reminder, review what was committed and update affected L2 → L1 → L0 files before proceeding with new work.

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
