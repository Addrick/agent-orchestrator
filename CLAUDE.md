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
