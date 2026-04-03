---
name: Codebase architecture detail
description: Full structural reference — components, data flows, schemas, interfaces, config, and key implementation details
type: reference
---

## Core Data Flow

```
Interface (Discord/Gmail/Zammad)
  -> Discord bot: preprocess_message() for dev commands (sent to threads, not logged)
  -> ChatSystem.generate_response()
    -> BotLogic.preprocess_message()  -- command detection (set, what, hello, etc.)
    -> _prepare_request()             -- resolve services, build history, filter tools, append user msg
    -> _run_tool_loop()               -- LLM call + tool execution loop (max 5 iterations)
      -> TextEngine.generate_response() -- provider-specific API call
      -> ToolManager.execute_tool()     -- tool execution
    -> MemoryManager.log_message()    -- persist user + assistant messages to SQLite
    -> Response back to interface
```

## Key Components

### `src/chat_system.py` -- ChatSystem
- Dependency injection hub: holds personas, TextEngine, MemoryManager, ToolManager, services
- `generate_response()` -> returns `Tuple[str, ResponseType, Optional[int]]` (text, type, ticket_id)
- `_build_conversation_history()` -> fetches from DB via MemoryManager, formats for LLM
- `_run_tool_loop()` -> up to MAX_TOOL_CALLS iterations, appends tool messages to in-memory history
- `_filter_tools_for_persona()` -> filters by enabled tools, service bindings, model compatibility
- `resume_pending_confirmation()` -> handles CONFIRM mode write-tool approval flow
- `PendingConfirmation` dataclass stores paused state for write-tool confirmation
- `RequestContext` dataclass bundles all pipeline state
- `last_api_requests` dict caches payloads for dump_last/dump_context (keyed by user+persona)
- `ResponseType` enum: DEV_COMMAND, LLM_GENERATION, PENDING_CONFIRMATION

### `src/engine.py` -- TextEngine
- Provider-agnostic LLM abstraction: OpenAI, Anthropic, Google (Gemini/Gemma), local OpenAI-compatible
- `generate_response()` -> returns `Tuple[Dict[str, Any], Optional[Dict[str, Any]]]` (response, payload)
- Response dict: `{"type": "text", "content": "..."}` or `{"type": "tool_calls", "calls": [...]}`
- Per-model-family rate limiters (AsyncLimiter) configured in global_config
- Provider dispatch: `_generate_openai_response`, `_generate_anthropic_response`, `_generate_google_response`
- Each provider builds messages differently but all accept the same context_object format
- Tool message format (used across all providers): `{"role": "tool", "tool_call_id": "...", "name": "...", "content": "..."}`
- Google provider translates to Part-based format in `_build_google_history()`
- 429 errors fail fast (no backoff) to preserve quota

### `src/message_handler.py` -- BotLogic
- Command dispatch: set, what, hello, goodbye, remember, add, delete, detail, dump_last, dump_context, help, update_models
- `preprocess_message()` -> returns command result dict or None (passes through to LLM)
- `set` handlers modify persona state in-memory (model, prompt, tokens, context, temp, tools, memory_mode, etc.)
- `dump_context` -> FILE_RESPONSE format, shows last cached API payload
- `dump_last` -> summary of last API request with turn counts

### `src/persona.py` -- Persona
- Stateful LLM config: model, system prompt, token limit, temperature, top_p, top_k
- `ExecutionMode`: AUTONOMOUS (auto-execute tools), CONFIRM (ask before write tools)
- `MemoryMode`: CHANNEL_ISOLATED, SERVER_WIDE, PERSONAL, GLOBAL, TICKET_ISOLATED
- `get_context_length()` -> supports dynamic override via hello/goodbye commands (increments by 2 per turn)
- Service bindings list determines which tools and services are available
- `get_config_for_engine()` -> returns dict consumed by TextEngine

### `src/tools/` -- Tool System
- `definitions.py` -- JSON schemas for all tools, `WRITE_TOOLS` set, `MODEL_INCOMPATIBLE_TOOLS` dict
- `tool_manager.py` -- ToolManager registry, `execute_tool()`, handler registration pattern
- Tool categories: read-only (search, get) vs write (create, update, close) -- write tools go through confirmation in CONFIRM mode
- `WebSearchHandler` registered at ChatSystem init
- Service-specific tools registered via `ServiceIntegration.register_tools()`

### `src/database/memory_manager.py` -- MemoryManager
- SQLite with single persistent connection, `check_same_thread=False`
- All DB ops run via `asyncio.to_thread()` from async callers
- Thread-safe via `threading.RLock`

**Tables:**
- `User_Interactions` -- interaction_id, user_identifier, persona_name, channel, author_role (user|assistant|system), author_name, content, timestamp, zammad_ticket_id, platform_message_id, server_id
- `Suppressed_Interactions` -- flags messages to exclude from history (FK to User_Interactions)
- `Agent_Actions` -- id, parent_id, agent_name, action_type, trigger_context, action_payload, outcome, outcome_payload, timestamp
- `Agent_Action_Contexts` -- action_id + context_type + context_value (multi-dimensional retrieval)

**Key indexes:** idx_channel_timestamp, idx_platform_message_id (unique), idx_zammad_ticket_id, idx_persona_timestamp, idx_user_persona, idx_server_id_timestamp

**History methods** (all apply SQL LIMIT, fetch DESC then reverse to chronological):
- `get_channel_history(channel, persona_name, server_id, limit)` -- default mode
- `get_server_history(server_id, persona_name, limit)`
- `get_personal_history(user_identifier, persona_name, limit)`
- `get_global_history(persona_name, limit)`
- `get_ticket_history(ticket_id, limit)`

All exclude suppressed messages via `_SUPPRESSION_SUBQUERY`.

`log_message()` -- inserts a row, called by ChatSystem.generate_response()
`suppress_message_by_platform_id()` -- marks a message for exclusion (called on Discord message delete)

### `src/interfaces/discord_bot.py`
- `create_discord_bot()` returns CustomDiscordBot (discord.Client subclass)
- `on_message`: filters out bot's own messages (line 150), thread messages (155), debug channel (151)
- Persona detection: message prefix match ("persona_name message") or channel name match
- Dev commands intercepted BEFORE generate_response (line 177) via `preprocess_message`, responses sent to threads via `_send_dev_response`, then returns -- never logged to DB
- LLM responses: logs user msg + assistant msg to DB via `memory_manager.log_message` (lines 257-290)
- Confirmation flow: reaction-based approval (checkmark/X) with timeout
- `on_message_delete`: calls `suppress_message_by_platform_id`
- Ambient logging: messages in AMBIENT_LOGGING_CHANNELS logged under persona="ambient"
- `_safe_typing`: gracefully handles 429 on typing indicator

### `src/interfaces/gmail_bot.py` (Proof of Concept)
- POC, not fully designed -- expect significant changes
- Polls Gmail API for new messages via history ID tracking
- Persona routing based on email recipient address
- Forces AUTONOMOUS mode (overrides CONFIRM since Gmail has no interactive confirmation)
- Does NOT call `log_message` -- no persistence currently

### `src/interfaces/zammad_bot.py` -- Deprecated stub
- Re-exports from `src/agents/zammad_bot.py` for backward compat only. Candidate for removal.

### `src/interfaces/kobold_api.py` -- Unused
- KoboldCpp local model API wrapper. Not integrated into any pipeline. Candidate for removal.

### `src/agents/` -- Agent Framework

**`base.py` -- Agent (ABC)**
Abstract base for autonomous background workers. Not user-interactive — poll external systems on a schedule and act independently.
- Lifecycle: `start()` → `_on_start()` → `[loop: deploy() every interval]` → `stop()`
- `deploy()`: Abstract — subclass implements one work cycle
- `_build_llm_context()`: Minimal context (user prompt + optional action history injection)
- `_log_step()`: Logs actions/child steps to Agent_Actions table
- `_get_action_history_message()`: Injects recent actions into LLM context if `action_history_limit > 0`
- Auto-loads system personas from file on init (agents invoke system personas for read-only analysis)
- Config: `schedule` (dict, e.g. `{"interval": 30}`), `action_history_limit` (int), `agent_name` (str)

**`agent_manager.py` -- AgentManager**
Central registry and lifecycle controller.
- `register(name, AgentClass, default_config)`: Register agent blueprint
- `start_agent() / stop_agent() / restart_agent()`: Lifecycle control (creates asyncio tasks)
- `auto_start()`: Starts agents where config has `auto_start: true`
- Convention-based DI: inspects agent `__init__` signature, injects `chat_system`, optionally `zammad_client`, `notification_router`, `agent_config`
- Config merging: agents.json < registration defaults < runtime overrides
- `notification_router`: Lazy-set after construction

**`dispatch_agent.py` -- DispatchAgent**
Routes triaged tickets to notifications. Pipeline per ticket:
1. Fetch ticket + triage note from Zammad
2. LLM call to `dispatch_analyst` persona → priority, summary, reasoning (JSON)
3. Route notification via `NotificationRouter` (channel/recipient from agents.json config, not LLM-chosen)
4. Tag ticket as dispatched in Zammad
5. Log action outcome

**`zammad_bot.py` -- ZammadBot**
Multi-stage AI triage pipeline for new, untagged tickets. Uses 4 system personas:
1. `triage_scout` → extract search keywords
2. Search Zammad for related closed tickets (global + per-user)
3. `triage_filter` → relevance scoring of historical tickets
4. `triage_summarizer` → compress long ticket bodies (adaptive, only if context exceeds limit)
5. `triage_analyst` → full analysis with context → internal note posted to ticket
Tags ticket as triaged. No tools used — all LLM calls are read-only.

**`agent_service.py` -- AgentServiceIntegration**
Plugs agent tools into ChatSystem's service binding system. Personas with `service_bindings: ["agents"]` gain access to agent management tools.

**Agents vs Personas:**
- Agents are autonomous background workers — no user interaction, run on intervals
- Personas are conversational — respond to user messages, can use tools including agent management tools
- System personas are read-only LLM configs used by agents for analysis (no tools)
- Agents do NOT spawn other agents; no delegation chains currently

### `src/tools/agent_tool_handler.py` -- AgentToolHandler
Three tools gated behind `service_bindings: ["agents"]`:
- `get_agent_status` (read): Running state, deploy counts, errors for one or all agents
- `get_agent_history` (read): Recent action log with optional ticket_id/customer filters
- `manage_agent` (write): Start/stop/restart — goes through confirmation if persona is in CONFIRM mode

### `src/clients/notification.py` -- Notification System
- `Notifier` (ABC): `async send(recipient, subject, body) → bool`
- `DiscordNotifier`: DM via Discord bot
- `ZammadNotifier`: Internal note on ticket
- `LogNotifier`: Fallback, logs to stdout
- `NotificationRouter`: Routes by channel name, falls back to LogNotifier

### `src/app_manager.py` -- AppManager
Top-level lifecycle coordinator. Starts agent_manager.auto_start(), launches interface tasks (Discord, Gmail), blocks until shutdown.

### `src/clients/`
- `service_integration.py` -- ServiceIntegration abstract base: register_tools, resolve_context, get_tracking_id, prepare_tool_args, on_tool_result, on_message, get_system_messages
- `zammad_client.py` -- ZammadClient: wraps Zammad REST API (HTTP layer)
- `zammad_service.py` -- ZammadIntegration(ServiceIntegration): Zammad-specific tool registration, context resolution, user-facing ticket number ↔ internal ID translation
- The Zammad integration demonstrates the target pattern: separate client (HTTP) from service (tool registration + context). Discord and Gmail should eventually be split the same way (see pending interface refactor decision).

### `src/utils/`
- `google_utils.py` -- `process_grounding_metadata()`: extracts citations and sources from Google grounding responses, inserts inline citation markers
- `message_utils.py` -- `cleanse_message_for_history()`: strips citation markup and source blocks from messages before storing in history. `resolve_redirect_url()`: follows redirects with 429 retry for URL resolution
- `model_utils.py` -- `get_model_prefix()`: maps model names to family prefixes for routing and rate limiting. Model list refresh functions for OpenAI/Google/Anthropic APIs
- `save_utils.py` -- Persona JSON file I/O: load/save personas and models to disk. Handles default + system persona merging on startup

### `src/main.py` -- Startup Sequence
1. MemoryManager (SQLite) + schema migration
2. TextEngine (LLM API router)
3. ZammadClient (optional, fails gracefully if no credentials)
4. ChatSystem (DI hub, injected with memory + engine)
5. Register ZammadIntegration service (if Zammad available)
6. AgentManager + register agent classes (ZammadBot, DispatchAgent)
7. Register AgentServiceIntegration service (agent tools)
8. AppManager + NotificationRouter (Discord/Zammad notifiers)
9. Register interface tasks (Discord bot, Gmail bot)
10. Optional model list refresh on startup
11. `app.start()` → auto_start agents + launch interface tasks

## Config

### `config/global_config.py`
- DEFAULT_CONTEXT_LIMIT = 15 (messages fetched from DB)
- GLOBAL_CONTEXT_LIMIT = 30 (hard cap when history_limit passed)
- MAX_TOOL_CALLS = 5 (per request)
- MAX_CACHED_API_REQUESTS = 128
- DEFAULT_TOKEN_LIMIT = 4096 (LLM output)
- PENDING_CONFIRMATION_TIMEOUT = 300 (seconds)
- Rate limits per model family (RPM/RPD)
- DISCORD_CHAR_LIMIT, DISCORD_STATUS_LIMIT, DISCORD_DEBUG_CHANNEL, AMBIENT_LOGGING_CHANNELS

### Persona Config Files
- `config/default_personas.json` -- tracked in git, seeds production
- `config/system_personas.json` -- tracked in git, system personas (used by agents for read-only LLM analysis)
- `data/personas.json` -- gitignored, local runtime state, overrides defaults on startup

### Agent Config
- `config/agents.json` -- agent definitions, schedule, auto_start, notification_defaults, recipient mappings
- Structure: `{agents: {name: {persona, schedule, action_history_limit, auto_start, notification_defaults}}, recipients: {name: {discord_user_id, email}}}`

## Testing Structure

4-tier, ordered by execution:
1. **Unit** (no marker) -- everything mocked, no network
2. **Integration** (`@pytest.mark.integration`) -- multi-component with mocked externals
3. **Zammad Live** (`@pytest.mark.zammad_live`) -- requires live Zammad instance
4. **LLM Live** (`@pytest.mark.llm_live`) -- real LLM API calls

Key test files:
- `tests/test_chat_system.py` -- ChatSystem with mocked deps, `chat_system_with_mocks` fixture
- `tests/test_engine.py` -- TextEngine provider tests
- `tests/database/test_memory_manager.py` -- schema, CRUD, migration tests (`legacy_mem_manager` fixture)
- `tests/integration/test_full_system_flow.py` -- end-to-end with mocked externals
- `tests/integration/test_memory_modes.py` -- memory mode behavior
- `tests/interfaces/test_discord_bot.py` -- Discord bot event handling
- `tests/interfaces/test_gmail_bot.py` -- Gmail bot message handling

Migration test pattern: `legacy_mem_manager` fixture creates DB with OLD schema (no new columns/tables), test calls `create_schema()` and verifies migration.
