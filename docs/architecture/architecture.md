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
    -> RequestBuilder.prepare_request()  -- resolve services, build history, filter tools, append user msg
      -> build_conversation_history()   -- fetch sliding window from DB
      -> retrieve_memory_block()        -- embed (sliding window + current msg), KNN search,
                                           inject <memory> block as user msg at position 0
                                           (skips memories already in sliding window via
                                            exclude_after_interaction_id; respects include_ambient)
    -> ToolLoop.run()                 -- LLM call + tool execution loop (max 5 iterations)
      -> TextEngine.stream_messages()   -- provider-specific streaming API call
      -> ToolManager.execute_tool()     -- tool execution
      -- yields TokenEvent / ToolCallStartEvent / ToolCallResultEvent / ErrorEvent
      -- + internal _ApiPayloadEvent (orchestrator caches) and _LoopFinishedEvent
         (orchestrator persists assistant turn / parks write calls for audit)
    -> MemoryManager.log_message()    -- persist user + assistant messages (reply_to_id links A->Q)
    -> Response back to interface

Background (SqliteConsolidator, agent_name="memory", every 15min — registered ONLY when SEMANTIC_BACKEND=="sqlite"; Hindsight backend obsoletes it):
  Phase 1 — Embed unembedded messages (loop until exhausted per channel):
  -> get_active_channels()            -- find channels with unprocessed msgs
  -> get_unembedded_messages()        -- fetch msgs not yet in Message_Embeddings
  -> EmbeddingService.encode()        -- token-aware batch chunking
  -> INSERT INTO Message_Embeddings + vec_Message_Embeddings

  Phase 2 — Segment and summarize (loop until clear per channel):
  -> get_unsegmented_embedded_messages()  -- msgs with parent_summary_id IS NULL
  -> _segment_by_similarity()         -- centroid-based topic grouping (threshold 0.80)
                                         bridging: explicit reply_to_id link OR
                                         heuristic (lone user msg + next assistant msg)
  -> _summarize_segment()             -- LLM extracts observations via submit_memory_summary tool
  -> Store segment + summary + vec embedding in Memory_Segments / Memory_Summaries / vec_Memory_Summaries
  -> SET parent_summary_id on processed msgs

Background (MemoryConsolidator, hourly daemon via app.register_task):
  -> cluster similar episodic summaries (L1) by pairwise similarity
  -> LLM merges clusters into core profiles (L2) via submit_core_profile tool
```

## Key Components

### `src/chat_system.py` -- ChatSystem
- Orchestration kernel + public API only (post DP-200/DP-201b). Constructed by `src/bootstrap.create_chat_system` with real deps: `__init__(memory_manager, text_engine, embedding_service=None, *, personas, system_persona_names, tool_manager, models_available=None)`. Borrows `memory_manager.backend` as `self.memory_backend` and pushes the embedding service into it via `set_embedding_service` so `SqliteSemanticBackend.recall` can translate query → embedding.
- Collaborators (DP-200 slice B; DP-201b deleted the transitional private delegate seams — internal and external callers address these directly):
  - `self.request_builder` (`src/request_builder.py` — RequestBuilder): `prepare_request()`, `build_conversation_history()`, `retrieve_memory_block()`, `format_raw_history_for_llm()`, `filter_tools_for_persona()`, `format_memory_block()`, sticky taint map (`conversation_taints` / `set_conversation_taint()`), `resolve_generation_params()`, dry-run `assemble_request()`.
  - `self.turn_persistence` (`src/turn_persistence.py` — TurnPersistence): `log_user_turn()`, `commit_or_update_assistant()`, `retain_turn_safe()`, `store_api_request()` + the `last_api_requests`/`last_api_iterations` caches (DP-202 deleted the mirroring ChatSystem properties; BotLogic receives TurnPersistence as an explicit dep for dump_last/dump_history).
  - `self.confirmations` (`src/confirmations.py` — ConfirmationManager): `pending` park map keyed (user, persona), `park()`, `apply_resume_decision()`; `PendingConfirmation` dataclass lives here.
- `generate_response()` -> returns `Tuple[str, ResponseType, Optional[int], Optional[int]]` (collect-stream wrapper over the kernel)
- `_orchestrate()` -> shared streaming kernel — preprocess → set TurnContext (`src/tools/turn_context.py`) → `request_builder.prepare_request` → user-log → user `retain_turn` → ToolLoop → assistant-commit → assistant `retain_turn` (taint-aware) → DoneEvent → reset TurnContext. `stream_response()` (portal) and `generate_response()` (Discord/Gmail/agents) both delegate here. Token + tool events forwarded as-is; internal `_ApiPayloadEvent` siphons into `turn_persistence.store_api_request`, `_LoopFinishedEvent` drives universal write-audit parking + assistant persistence. Both retain calls are fire-and-forget (sqlite_legacy noop; Hindsight enqueues + returns).
- `resume_pending_confirmation()` / `stream_resume_confirmation()` -> write-tool approval flow for any parked turn (all write tools park for audit, not just CONFIRM mode); decision execution lives in `ConfirmationManager.apply_resume_decision`
- Public request-assembly delegates kept on ChatSystem (portal/inspector surface): `assemble_request()`, `get_view_history()`, `get_session_memory_block()`
- `RequestContext` dataclass (in `src/request_builder.py`) bundles all pipeline state

### `src/generation_events.py` -- streaming event dataclasses
- Lives outside chat_system to break a circular import (chat_system ↔ tool_loop). chat_system re-exports the names so `from src.chat_system import TokenEvent, DoneEvent, ResponseType` still works.
- `TokenEvent(delta)` -- incremental text chunk
- `DoneEvent(text, response_type, assistant_id, user_interaction_id)` -- terminal success
- `ErrorEvent(message)` -- terminal failure
- `ToolCallStartEvent(tool_name, arguments, call_id)` -- tool invocation surfaced mid-stream (tool revamp v1)
- `ToolCallResultEvent(call_id, tool_name, result, error)` -- paired result; `error` populated on tool failure
- `GenerationEvent` Union; `ResponseType` enum: DEV_COMMAND, LLM_GENERATION, PENDING_CONFIRMATION
- `last_api_requests` dict caches payloads for dump_last/dump_context (keyed by user+persona)

### `src/tools/tool_loop.py` -- ToolLoop
- Stream-shaped tool loop extracted from `_orchestrate` (tool revamp v1, supersedes the old `plans/toolloop_extraction.md`).
- `ToolLoop(text_engine, tool_manager, max_iterations=MAX_TOOL_CALLS)` — constructed per-call by `_orchestrate` so tests that swap `chat_system.text_engine` post-init still take effect.
- `run(persona, conversation_history, params, tools, ...)` — async generator. Drives `text_engine.stream_messages` per iteration, forwards `TokenEvent`s, surfaces tool calls as `ToolCallStartEvent` / `ToolCallResultEvent`, mutates `conversation_history` in place (orchestrator + resume path read it back).
- Implements **universal write-audit** (`tool_loop.py:213-285`): on any iteration where a batch contains a `WRITE_TOOLS` call, the loop runs the read calls, then yields `_LoopFinishedEvent(response_type=PENDING_CONFIRMATION)` with `pending_writes` + an `audit_info` block (per-action `irreversible` / `always_confirm` / `service_binding` / `sensitivity` / `enrichment`, plus turn `tainted` / `taint_sources` / `model_reasoning`) and exits. This is unconditional — execution mode is recorded in `audit_info` for display only, never gates parking. Taint (`turn_tainted` from `produces_untrusted` read tools) and irreversibility flags are computed here for the audit surface.
- Tool errors (returned as `{"error": ...}` by `tool_manager.execute_tool`) surface via `ToolCallResultEvent.error`; the loop continues so the model can adapt instead of hard-stopping.
- Internal events `_ApiPayloadEvent` / `_LoopFinishedEvent` are loop-private — they let the orchestrator handle api-payload caching, write-audit `pending_writes` parking, and assistant-row persistence without leaking into the public event surface.

### `src/engine/` — TextEngine + Provider family (DP-244)
- Provider-agnostic LLM abstraction: OpenAI, Anthropic, Google (Gemini/Gemma), local kobold-native, Antigravity (`agy`), Claude Code (`cc-*`)
- **Package layout (DP-244, MERGED):** `src/engine.py` (the old 1586-loc god module) is now the `src/engine/` package. Public import surface is unchanged — `from src.engine import TextEngine, LLMCommunicationError` still works.
  - `driver.py` — the slimmed `TextEngine`: routing, rate limiting, the empty-response retry loop, and 429 model-fallback policy. Owns image policy (`model_supports_images`) and `_FALLBACK_MODELS`.
  - `providers/{base,openai,anthropic,google,local,agy,cc,_shared,_subprocess}.py` — the `Provider` ABC (`base.py`) plus 6 fully-extracted providers, each owning its own streaming / message-build / tool-parse. `agy` and `cc` share the subprocess plumbing in `_subprocess.py`; `_shared.py` holds common helpers.
  - `registry.py` — `ProviderRegistry` / `build_registry(engine)`: the ordered provider list.
- **Provider routing (DP-244):** routing is `ProviderRegistry.resolve(model_name)` (registry.py), which walks an ordered list of `Provider` objects and returns the first whose `matches()` is true (after running its `ensure_supported` host guard). Order preserves the old waterfall precedence — notably `cc-*` before `claude`. Each provider is a fully-extracted `Provider` owning its own canonical streaming generator, emitting the unified event shape `api_payload → text_delta* → [tool_calls] → done`. The policy driver `_stream_response` wraps that stream with image policy, rate limiting, the empty-response retry loop (attempts emit nothing until they produce real content, so invalid attempts retry invisibly), and 429 model fallback. `_get_provider_route` and the `_stream_<provider>_response` methods survive only as back-compat shims delegating into `self._registry` / the resolved provider; the transitional `_EngineProvider` wrapper is retired.
- Entries: `stream_messages()` (chat pipeline — true token deltas; `local` goes straight to `StreamEngine.stream_messages` with GenerationParams/provider_extras intact and no retry policy) and `generate_response()` (one-shot = `collect_stream(_stream_response(...))`; used by agents/BotLogic/consolidation)
- `generate_response()` -> returns `Tuple[Dict[str, Any], Optional[Dict[str, Any]]]` (response, payload)
- Response dict: `{"type": "text", "content": "..."}` or `{"type": "tool_calls", "calls": [...]}`
- Per-model-family rate limiters (AsyncLimiter) configured in global_config
- `LLMCommunicationError` lives in the `src/llm_errors.py` leaf (shared with `stream_engine` without a cycle); `src.engine` re-exports it
- `src/stream_engine.py` (StreamEngine) is NOT a peer engine: it is the kobold-native `local` transport component, constructed by TextEngine itself (chat-template rendering, `<tool_call>` text protocol, per-token SSE; `TextEngine.aclose()` releases its HTTP client). Renders via the `CHAT_TEMPLATES` registry built from kobold-sourced instruct presets (`_KOBOLD_INSTRUCT_PRESETS`) plus a verbatim `alpaca` preset; the per-persona `chat_template` selects the preset (default `chatml`) and per-call `instruct_tags` override it. Exposed to the portal via `GET /api/v1/chat_templates` (DP-140). The old local OpenAI-compat one-shot transport is gone — local one-shot and streaming share one transport and one tool protocol.
- Wire-payload goldens: `tests/test_engine_payload_parity.py` pins the exact request kwargs per provider (frozen dicts; only capture points may move)
- **`agy` route (DP-127, model prefix `agy-*`):** drives the local `agy` CLI via subprocess on the OAuth tier (Gemini 3.5 Flash), not an API key. `_render_agy_prompt` flattens the full history into one role-tagged transcript (stateless, re-flattened each tool-loop turn); `_render_agy_tool_protocol` injects tool descriptions + asks the model to emit a `<tool_call>{json}</tool_call>` block; `_parse_agy_tool_call` parses that block back into the standard `{"id","name","arguments"}` shape so DERPR's own tool loop executes it (engine keeps full policy/CONFIRM/taint control — agy never runs tools itself). `_run_agy_cli` spawns with `stdin=DEVNULL` (or it blocks on an interactive permission prompt), **no** `--dangerously-skip-permissions`/`--add-dir` (so agy's own tools stay gated and it can never run them — DERPR drives every tool itself), in a throwaway tempdir, killing the process group on cleanup. **POSIX-only (DP-138):** `_ensure_agy_supported()` refuses the route on native Windows with a clear error — agy is a TUI that only writes its response to a TTY, but the engine captures piped stdout, so on Windows the response is always empty (no flag/env var changes this; works on macOS/Docker). Run the engine on the POSIX host/WSL/Docker to use agy. Text-only (no images). The Antigravity *SDK* route was rejected (API-key-only, Model-A-native) — see `project/decisions/2026-05-29-agy-sdk-oauth-finding.md`.
- **`cc` route (DP-222, model prefix `cc-*`):** drives the local `claude` CLI via subprocess (one-shot; the `CcProvider` in `providers/cc.py` with a dedicated `_cc_limiter`, resolved ahead of the `"claude" in model_name` branch). `_cc_model_arg` maps `cc-sonnet`→`sonnet` (bare `cc-` falls back to `sonnet`). POSIX-only when sandbox is on (`_ensure_cc_supported`, gated by `CC_SANDBOX`). Structural parity with `agy` (subprocess-per-call, persistent per-persona/global workspace, dedicated rate limiter) but a deliberate divergence: cc **ignores** the engine's `tools` argument and lets Claude Code run its OWN sandboxed tools under `--dangerously-skip-permissions` bounded by the OS sandbox (Seatbelt/bubblewrap via `_build_cc_sandbox_settings`), returning final text — DERPR's tool loop does not wrap it. `_resolve_cc_workspace` precedence: per-call `cc_workspace_override` (used by DP-227 fixr to point at an agent's worktree) > `CC_WORKSPACE_DIR` > per-persona dir > global dir; `_cc_workspace_locks` serialize same-workspace runs.
- Each provider builds messages differently but all accept the same context_object format
- Tool message format (used across all providers): `{"role": "tool", "tool_call_id": "...", "name": "...", "content": "..."}`
- Google provider translates to Part-based format in `_build_google_history()`
- 429 errors fail fast (no backoff) to preserve quota

### `src/message_handler.py` -- BotLogic
- Command dispatch: set, what, hello, goodbye, remember, add, delete, detail, dump_last, dump_context, help, update_models
- Explicit dependencies (DP-202): no ChatSystem reference or import. Rebindable collaborators (`personas`, `visible_personas`, `text_engine`, `tool_manager`, model-catalog get/set) are injected as zero-arg providers; `turn_persistence` (dump caches) and `memory_manager` (trust/untrust) are stable instances. Wired by `ChatSystem.__init__` with closures over `self`; tests use `tests.helpers.make_bot_logic(state)`. Runtime-import rule enforced in `tests/test_module_boundaries.py`.
- `preprocess_message()` -> returns command result dict or None (passes through to LLM)
- `set` handlers modify persona state in-memory (model, prompt, tokens, context, temp, tools, memory_mode, etc.)
- `dump_context` -> FILE_RESPONSE format, shows last cached API payload
- `dump_last` -> summary of last API request with turn counts

### `src/persona.py` -- Persona
- Stateful LLM config: model, system prompt, token limit, temperature, top_p, top_k
- `ExecutionMode`: AUTONOMOUS, CONFIRM — recorded in the audit surface for display, but **does not gate write parking** (all write tools park for audit regardless; see ToolLoop universal write-audit)
- `MemoryMode`: CHANNEL_ISOLATED, SERVER_WIDE, PERSONAL, GLOBAL, TICKET_ISOLATED
- `get_history_messages()` -> supports dynamic override via hello/goodbye commands (increments by 2 per turn)
- Service bindings list determines which tools and services are available
- `include_ambient_memory: bool = True` — whether to include ambient-channel memories in retrieval; controlled by `get_include_ambient_memory()`
- `thinking_level: Optional[str]` — passed through to engine config (e.g. `"minimal"` for Gemma extended thinking)
- `max_context_tokens: int` — total context budget (prompt + reserved response). Defaults to `global_config.DEFAULT_MAX_CONTEXT_TOKENS` (131072). Mirrors kobold-lite's `localsettings.max_context_length` slider semantic so the value round-trips on persona load.
- `chat_template: Optional[str]` — name of the instruct/chat preset used by the local (kobold-native) `StreamEngine` to render the prompt (e.g. `chatml`, `alpaca`, and other kobold-sourced presets incl. thinking variants). Exposed via `get_chat_template()`/`set_chat_template()`, flows through `get_config_for_engine()`, settable via the `set chat_template` dev command and persona GET/PATCH. `None` = engine default (`chatml`).
- **Hindsight bank fields (DP-255):**
  - `retain_mission` / `reflect_mission: Optional[str]` — per-persona missions for the persona's Hindsight bank, honoured only at bank creation (`acreate_bank`). Accessed via `get_retain_mission()` / `get_reflect_mission()`.
  - `disposition: Optional[Dict[str, int]]` — Hindsight extraction disposition `{skepticism|literalism|empathy}`, each clamped 1–5 via `_sanitize_disposition` (`_DISPOSITION_KEYS`; unknown keys dropped, out-of-range/invalid values logged and skipped). Accessed via `get_disposition()`.
- `get_config_for_engine()` -> returns dict consumed by TextEngine (includes thinking_level + chat_template if set)

### `src/tools/` -- Tool System
- `definitions.py` -- JSON schemas for all tools aggregated into the static `ALL_TOOL_DEFINITIONS` seed, which seeds a `ToolDefinitionRegistry` — the live catalog (DP-268): static seed plus post-import `register_tool_definition()` defs (MCP-discovered tools), with `unregister_tool_definition()` to remove them (refuses non-`dynamic` static defs). Name index + write-tool set maintained incrementally; read via `get_tool_definition()` / `get_tool_capabilities()` / `is_write_tool()`; consumers needing the live toolset call `get_all_tool_definitions()`. `MODEL_INCOMPATIBLE_TOOLS` dict.
- `tool_manager.py` -- ToolManager registry, `execute_tool()`, `register()` / `unregister()` handler registration pattern
- Tool categories: read-only (search, get) vs write (create, update, close) -- all write tools are parked for human confirmation regardless of execution mode (universal write-audit, per the security framework)
- `WebSearchHandler` registered at ChatSystem init
- Service-specific tools registered via `ServiceIntegration.register_tools()`
- Memory tools: `submit_memory_summary` (observations[] + outlier_ids[], used by SqliteConsolidator), `drill_down_memory` (retrieves source messages for a summary), `update_core_memory` (edits core profile summaries)
- **`exfil_capable` capability (DP-263):** optional tool-capability flag, default `True`. A network tool that sets `exfil_capable=False` opts out of the exfil-composition policy (`tool_policy.py` Rules 2/3) — used when a write tool's egress carries no model-controlled payload (constrained args to trusted infra), so it is not a data-exfil vector and must not arm those rules. Destructive risk on such tools is still covered by `is_write` (parked for confirmation). Example: proxmox `set_active_model` (arg is a name from a fixed config map). Validated as `bool` at import in `definitions.py`; enforced in `tool_policy.py:114` (`locality == "network" and caps.get("exfil_capable", True)`).

### `src/memory/memory_manager.py` -- MemoryManager
- SQLite with single persistent connection, `check_same_thread=False`
- All DB ops run via `asyncio.to_thread()` from async callers
- Thread-safe via `threading.RLock`
- Uses `sqlite-vec` extension for vector similarity search

**Layer split (DP-108 / Sprint 1 of memory_backend_abc):**
- **Transcript layer (stays on `MemoryManager`)**: system of record. `log_message`, history queries (`get_*_history`), version chevrons (`update_interaction_content`, `list_interaction_versions`, `swap_interaction_version`), suppression, audit (`log_audit_event`, `mark_trusted`, `mark_untrusted`), edit/dedupe entrypoints (`invalidate_summary`, `handle_message_edit`, `handle_portal_retry`).
- **Memory layer (delegated to `MemoryBackend`)**: semantic + episodic surface — embeddings, segments, summaries, agent actions, segment-failure tracking. `MemoryManager.__init__` accepts `backend: MemoryBackend | None` (defaults to `SqliteSemanticBackend(self)`). All semantic/episodic public methods on `MemoryManager` are now thin delegations to `self.backend`.
- The backend shares MemoryManager's connection / lock / `_suppression_filter` (single SQLite connection across both layers; transactions span them).
- Sprint 2 lands `HindsightBackend` against the same ABC; the new-shape methods (`retain_turn`, `recall`, `reflect`, `list_mental_models`, `ensure_bank`, `delete_bank`) are placeholders on the SQLite backend (NotImplementedError or noop) until Hindsight wires them.

### `src/memory/backend/` -- MemoryBackend ABC + impls
- `backend/base.py` — `MemoryBackend` ABC. Carries both legacy SQLite-shape methods (current contract) and new Hindsight-shape methods (placeholders Sprint 2 fills in). Dataclasses: `MemoryHit`, `Experience`, `MentalModel`, `ReflectResult`.
- `backend/sqlite.py` — `SqliteSemanticBackend(memory_manager)`. Wraps existing semantic + episodic SQL. Reaches back through `mm._lock`, `mm._get_connection()`, `mm._suppression_filter()` to share the single connection. Pure refactor — no behavior change.
- `backend/hindsight.py` — `HindsightBackend` (native REST via `HindsightRESTClient`). Retain path is fire-and-forget through one `asyncio.Queue` per `bank_id`. Worker uses **drain-on-tick** coalescing: awaits one item, then drains the rest currently queued and POSTs them as a single upstream `RetainRequest` (`{items: [...], async: true}`). Each item carries `content`, `tags`, `document_id`, `update_mode`, `timestamp`, optional `metadata`. `document_id` is derived from a scope key `{bank_id}:{channel_id}` via `_DocScopeStore` (sibling SQLite at `hindsight_doc_scope.db`); a >24h gap between retains in the same scope opens a new document with `update_mode="replace"`, otherwise items append. `_TrustOverrideStore` (sibling SQLite at `hindsight_overrides.db`) holds operator trust flips and an audit trail; recall post-filters hits through it. `ensure_bank` sends `retain_mission` + `reflect_mission` (and optional `enable_observations` / `observations_mission`) — never the deprecated `mission` / `background` aliases. **Read/list surface (DP-292):** `list_banks`, `list_documents`, `get_document`, `delete_document`, `list_operations`, `get_operation` wrap the upstream bank/document/operation endpoints (client methods `alist_banks`/`alist_documents`/`aget_document`/`adelete_document`/`alist_operations`/`aget_operation`, routes verified against live prod OpenAPI). **Operations are bank-scoped upstream** — no global collection — and the ops-list query param is `type` (client kwarg `op_type`). Unlike recall/reflect (fail-soft to empty), these operator-facing reads propagate `HindsightAPIError`. On the ABC they are default-raising stubs; SQLite inherits the raise.

**Tables:**
- `User_Interactions` -- interaction_id, user_identifier, persona_name, channel, author_role (user|assistant|system), author_name, content, timestamp, zammad_ticket_id, platform_message_id, server_id, tool_context, parent_summary_id (FK to Memory_Summaries), reply_to_id (FK to self, links assistant response to user question)
- `Suppressed_Interactions` -- flags messages to exclude from history (FK to User_Interactions)
- `Message_Embeddings` -- interaction_id (PK, FK), embedding (BLOB), model_name, created_at
- `Interaction_Edit_History` -- edit_id (PK), interaction_id (FK to User_Interactions), old_content, edited_at (archive of prior canonical content, ordered by edited_at ASC, edit_id ASC)
- `Edit_History_Embeddings` -- edit_id (PK, FK to Interaction_Edit_History ON DELETE CASCADE), embedding (BLOB), model_name, created_at (L0 embeddings of archived content — travel with content across chevron swaps; no vec shadow since archives don't participate in k-NN)
- `Memory_Segments` -- segment_id, channel, server_id, persona_name, start_interaction_id, end_interaction_id, message_count, created_at, first_message_at, last_message_at
- `Memory_Summaries` -- summary_id, segment_id (FK), content, embedding (BLOB), model_name, created_at, summary_level (0=legacy, 1=episodic, 2=core), parent_summary_id (FK to self, for consolidation hierarchy)
- `vec_Message_Embeddings` -- sqlite-vec virtual table for KNN search on message embeddings (float[3072])
- `vec_Memory_Summaries` -- sqlite-vec virtual table for KNN search on summary embeddings (float[3072])
- `Agent_Actions` -- id, parent_id, agent_name, action_type, trigger_context, action_payload, outcome, outcome_payload, timestamp
- `Agent_Action_Contexts` -- action_id + context_type + context_value (multi-dimensional retrieval)
- `Audit_Log` -- security/write-audit trail (event_type, target_id, timestamp; indexes idx_audit_event, idx_audit_target) backing `log_audit_event` / `mark_trusted` / `mark_untrusted`
- `Proposals` (DP-282, self-managing queue DP-290) -- proposal_id (PK), created_at, expires_at, agent_name, action_type, action_args, rationale, taint, source_action_id, ticket_number, status (CHECK: pending|approved|denied|expired|executed|execution_failed|withdrawn), reviewed_at, reviewer, review_note, executed_at, execution_result. Indexes: idx_proposal_status, idx_proposal_acceptance, idx_proposal_pending_key (partial UNIQUE on agent/action_type/ticket_number WHERE pending — dedup)

Note: On startup `create_schema()` validates both vec table dimensions against `EMBEDDING_DIMENSION` and auto-syncs any rows missing from the virtual tables.

**Key indexes:**
- User_Interactions: idx_channel_timestamp, idx_platform_message_id (unique), idx_zammad_ticket_id, idx_persona_timestamp, idx_user_persona, idx_server_id_timestamp
- Memory_Segments: idx_segment_channel_persona
- Memory_Summaries: idx_summary_segment, idx_summary_parent
- Agent_Actions: idx_agent_name_timestamp, idx_agent_action_type, idx_agent_parent

**History methods** (all apply SQL LIMIT, fetch DESC then reverse to chronological):
- `get_channel_history(channel, persona_name, server_id, limit)` -- default mode
- `get_server_history(server_id, persona_name, limit)`
- `get_personal_history(user_identifier, persona_name, limit)`
- `get_global_history(persona_name, limit)`
- `get_ticket_history(ticket_id, limit)`

All exclude suppressed messages via `_SUPPRESSION_SUBQUERY`.

**Memory retrieval:**
- `retrieve_relevant_summaries()` -- KNN search via `vec_Memory_Summaries`, filters by channel/persona/model/level, returns top-K summaries by distance; accepts `exclude_after_interaction_id` (skip memories already in sliding window) and `include_ambient` flag
- `get_unembedded_messages()` -- msgs not yet in Message_Embeddings (Phase 1 of SqliteConsolidator)
- `get_unsegmented_embedded_messages()` -- finds embedded msgs with `parent_summary_id IS NULL` for segmentation (Phase 2)
- `get_active_channels()` -- UNION of 3 queries: unembedded msgs, msgs above segment high-water mark, embedded-but-unsummarized msgs
- `get_last_segment_tail_embeddings()` -- tail N embeddings from last segment (for centroid seeding)

`log_message()` -- inserts a row, accepts `reply_to_id` to link assistant response to user question
`suppress_message_by_platform_id()` -- marks a message for exclusion (called on Discord message delete)

### `src/memory/memory_consolidation.py` -- MemoryConsolidator
Promotes episodic summaries (level 1) to core profiles (level 2) by clustering similar summaries.
- O(n^2) pairwise similarity comparison (future: sqlite-vec KNN)
- Union-find clustering on summaries above similarity threshold
- LLM merges clustered summaries into consolidated core profiles
- Writes new level-2 summary with `parent_summary_id` linking back to source episodics

### `src/memory/context_budget.py` -- Context budget enforcement (Phase 3)
Pure helpers for enforcing per-persona `max_context_tokens`. Two callsites today: `request_builder.prepare_request` and `kobold_engine_adapter` `/chat/completions`. Future home for dynamic LTM-depth modulation, retrieval-score-weighted budget allocation, and a real tokenizer swap.
- `estimate_tokens(text)` -> `(len(text)+3)//4`. No tokenizer roundtrip.
- `truncate_messages_to_budget(messages, max_tokens)` -> `(pruned, dropped_count)`. Drops oldest non-system messages until total ≤ budget. Preserves all `role=='system'` messages and the most recent `role=='user'` message (so the current turn is never evicted). No-op when `max_tokens` is None / non-positive / already under budget.
- Budget semantic is **kobold-style total ctx** (prompt + reserved response). Effective prompt prune budget = `persona.max_context_tokens - persona.response_token_limit`.

### `src/embedding_service.py` -- EmbeddingService
- `GeminiEmbeddingProvider` -- Google Gemini Embedding API (model: `gemini-embedding-001`)
- Token-aware chunking: splits text batches to stay under TPM limits
- Rate limiting via `GLOBAL_EMBEDDING_LIMITER` (shared AsyncLimiter)
- `encode_batch()` -- embeds multiple texts, returns list of BLOB embeddings (normalized float32)
- `encode_single()` -- single text embedding
- All embeddings L2-normalized at storage time → dot product = cosine similarity

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

### `src/interfaces/kobold_engine_adapter.py` -- KoboldEngineAdapter
- FastAPI app served from `KoboldAdapter.start()`; mounts the customised kobold-lite at `/portal` (`web_assets/portal.html`) and forwards inference traffic to local KoboldCPP.
- Inference path is OAI-jinja only: the portal runs kobold-lite in kcpp-with-jinja mode, which hijacks `opmode==4` to `/v1/chat/completions`. **Phase D (2026-04-28) split the adapter routes:** `/v1/chat/completions` is now a thin SSE transcoder over `chat_system.stream_response` — the engine rebuilds messages from DB, prunes to budget, archives on retry, and commits the assistant turn; the adapter only formats OAI-shape SSE chunks (and the `event: derpr` frame on `DoneEvent`). Client `data["messages"]` is discarded; `derpr_user_text` (or last-user fallback for non-portal clients) drives the user turn. The native `/api/v1/generate` and `/api/extra/generate/stream` routes remain verbatim passthrough to KoboldCPP because the pre-rendered kobold prompt cannot be safely reconstructed from DB (template-tag drift hazard, see `decisions/2026-04-19-portal-phase2-approach.md`); deprecation queued pending an OAI-feature-parity audit. CORS is wide open (kobold-lite reaches localhost from a different origin).
- Persona endpoints (`/api/v1/model`, `/api/v1/persona/...`, `/v1/models`) read/write through `chat_system.personas` and persist via `save_personas_to_file`. `POST /api/v1/personas` (DP-231) creates a new user persona from the portal's "+ New persona" form: name lowercased + validated as a single `[a-z0-9_-]` token (`_VALID_PERSONA_NAME`), 409 on collision, blank prompt falls back to `"you are in character as <name>"`, body applied through the shared `apply_persona_patch_body` chokepoint (same as PATCH), persisted via `save_personas_to_file` with in-memory rollback on save failure. `GET /api/v1/chat_templates` (DP-140) returns `sorted(StreamEngine.CHAT_TEMPLATES.keys())` for the inspector's template dropdown.
- Phase 2.1 adds `GET /api/v1/session/{persona}/kobold_export?max_turns=K` — returns a kobold-lite v1 savefile JSON built by `interfaces/kobold_export.build_kobold_savefile`. Pulls `chat_system.memory_manager.get_global_history(persona, K)` (no channel scoping; the portal has no channel concept). `K` defaults to the persona's `get_base_context_length()`.
- Phase 2.2 adds `GET /api/v1/session/{persona}/ltm_block?query=...` — delegates to the public `chat_system.get_session_memory_block(persona, user_identifier="portal", channel="web_ui", server_id=None, query=query)` seam (RequestBuilder rebuilds the history + recency cutoff internally) and returns `{"block": str | null}`. The portal wraps `prepare_submit_generation` async to fetch this before each submit, writes the block into kobold-lite's `current_anote` JS variable, and kobold-lite embeds it into the prompt at its normal author's-note position. No server-side prompt mutation occurs. Phase 2.2 also adds `memory_mode` to `GET /api/v1/persona/{name}` and accepts it in `PATCH /api/v1/persona/{name}`, wiring through to `persona.set_memory_mode()`.
- Phase 2.3a–b semantics (sidecar `derpr_user_text` for the user turn, `derpr_retry` for regens, `event: derpr` SSE frame carrying `assistant_id` for chevron hydration, partial-flush on abort) all survive Phase D — they are now implemented inside `chat_system._orchestrate` instead of in the adapter. The adapter handler is purely transcoding: maps `TokenEvent` → OAI `chat.completion.chunk`, `DoneEvent` → `event: derpr` frame + `[DONE]`, `ErrorEvent` → OAI error envelope. `_find_last_user_content(messages)` remains as the non-portal fallback when no sidecar is supplied. `_log_interaction` and `_commit_assistant` adapter helpers are still used by the native passthrough routes (`/api/v1/generate`, `/api/extra/generate/stream`); they are not on the OAI path anymore.
- Phase 3 `max_context_tokens` is exposed on `GET/PATCH /api/v1/persona/{name}`; budget enforcement on the OAI path moved to engine-side as of Phase D (`request_builder.prepare_request` calls `truncate_messages_to_budget` against the rebuilt history). The native passthrough routes do not prune.
- Phase 2.4 closes the portal edit/delete round-trip. `PATCH /api/v1/interaction/{id}` (body `{"content": str}`) calls `update_interaction_content`, which now ALSO drops the row's `Message_Embeddings` + `vec_Message_Embeddings` entries so `SqliteConsolidator._embed_unembedded` re-encodes against the new content on the next batch (the existing LEFT-JOIN-on-`Message_Embeddings` query naturally re-picks rows with stale embeddings cleared); `parent_summary_id` is set to NULL so the next summarizer pass re-groups the row. `DELETE /api/v1/interaction/{id}` calls a new thin wrapper `memory_manager.suppress_interaction(interaction_id)` that inserts a single `Suppressed_Interactions` row (idempotent — second call swallows `IntegrityError` and returns False; the endpoint surfaces this via `already_suppressed: bool`). Suppressed rows are filtered everywhere via `_suppression_filter` (history, sliding window, `kobold_export`, `get_unembedded_messages`, `get_unsegmented_embedded_messages`, `get_active_channels`); reply chains are kept intact (no FK cascade, no nulling of `reply_to_id`), and the segmenter handles orphaned `reply_to_id` references cleanly by falling back to the similarity gate. Portal: `edit_chunk_save` is wrapped to PATCH on non-empty edits and DELETE on empty edits; `interaction_ids[]` is spliced in lockstep with `gametext_arr`. **Archive bloat (option B — dedupe-on-archive)**: `swap_interaction_version` now content-hash-checks before inserting the canonical into `Interaction_Edit_History`; if a row with the same `(interaction_id, old_content)` already exists, the insert is skipped (chevron toggled back to a known content). Stale L0 rows are still dropped because the target restore step overwrites canonical content. This bounds chevron-toggle churn to the unique-content count rather than the click count; genuinely unique edits still grow the archive linearly (intended).
- Phase 2.3b adds version navigation backed by `Edit_History_Embeddings`. Schema: `Edit_History_Embeddings(edit_id PK → Interaction_Edit_History ON DELETE CASCADE, embedding BLOB, model_name, created_at)` — L0 embeddings travel with content across swaps; archives do not participate in k-NN (no `vec_` shadow). Endpoints: `GET /api/v1/interaction/{id}/versions` → `{interaction_id, versions: [{edit_id, content, created_at}, ...]}` with archives ordered `(edited_at ASC, edit_id ASC)` and canonical synthesized last (`edit_id=None`); `POST /api/v1/interaction/{id}/select_version/{k}` invokes `memory_manager.swap_interaction_version(interaction_id, k)` which performs archive-current + restore-target atomically in one txn — archives current canonical (new `Interaction_Edit_History` row, `Message_Embeddings` → `Edit_History_Embeddings`, delete `vec_Message_Embeddings`), restores target archive at position `k` (copy `old_content` → `User_Interactions.content`, move `Edit_History_Embeddings(target)` back into `Message_Embeddings` + `vec_Message_Embeddings`, delete target archive row). Returns `{current_content, interaction_id, total_versions}`; raises `IndexError` (400) on out-of-bounds k and `ValueError` (404) on unknown id with no state mutation. The stream relay in `_setup_routes` now detects `[DONE]` in the upstream chunk, commits the assistant buffer first, and emits `event: derpr\ndata: {"assistant_id": N}\n\n` immediately before forwarding `[DONE]` so the portal can hydrate chevron stacks (`retry_prev_text` / `redo_prev_text`) via `/versions`. Error and abort paths skip the frame (no chevron-able assistant turn). Portal-side: a `window.fetch` wrapper tees the `/chat/completions` response body, parses SSE blocks out-of-band to catch `event: derpr`, and on receipt calls `_derprHydrateVersions(assistant_id)` which fetches `/versions`, populates `derpr_version_chain` + `derpr_cursor`, and derives `retry_prev_text` / `redo_prev_text` from the chain. `btn_back` / `btn_redo` are wrapped to POST `/select_version/{k}` (k looked up by content-match against the current server version list) and advance the cursor; original kobold behavior is the fallback when no assistant_id is known. The kobold-side 10-entry cap on `retry_prev_text` has been removed — full regen history lives in the DB. A `PATCH /api/v1/interaction/{id}` route is also exposed (body `{"content": str}`) for manual edits from the portal; it calls `update_interaction_content` directly and does not archive (the prior content is whatever the caller wants replaced).

- **Phase 4.1 — history contract (DP-130, server port-forward).** Replaces the brittle positional `interaction_ids[]` shadowing (Phase 2.4) with a server-authored per-chunk identity contract. Three server-side pieces (portal.html left untouched — that re-keying is DP-131's Lite stopgap; the bespoke UI is DP-132+). Full spec + invariants C1–C5: `project/decisions/2026-06-02-portal-history-contract.md`.
  - **id-frame on EVERY turn.** `DoneEvent` gained `ephemeral_chunk_id: Optional[str]`. The OAI relay in `kobold_engine_adapter` now emits `event: derpr\ndata: {"user_id", "assistant_id", "response_type", "ephemeral_chunk_id"}\n\n` before `[DONE]` on *every* terminal turn — including a parked CONFIRM write (`assistant_id=None`), a tool-only turn, or an empty generation. Previously the frame was emitted only when `assistant_id is not None`, so parked/tool-only/abort turns sent no frame and the portal's positional id array drifted vs the visible story (root cause of "dispatchr is useless in the web UI"; live evidence `ids_len=11 storyLen=13`, `actions=12 ids=13`). The frame shape is **frozen** as of DP-130 — DP-131/DP-132 consume it.
  - **`ephemeral_chunk_id`.** A stable handle for the rendered-but-unpersisted confirmation chunk on a parked turn. It is the parked `PendingConfirmation.token` (uuid4 hex; reconciles with the DP-127-engine confirm-modal `token` — same correlation id the interactive surface sends back on approve/deny). `PendingConfirmation` also gained `confirmation_text` (the parked chunk's rendered text) so a fresh load can render the awaiting-approval text. Non-parked turns carry `ephemeral_chunk_id=None` and are addressed by `assistant_id`.
  - **`GET /api/v1/session/{persona}/transcript`** — the authoritative projection: `{"chunks": [...]}`, each chunk `{interaction_id|null, role, content, ephemeral, reasoning, tool_context, has_versions}` (invariant **C1**: exactly one `interaction_id` OR `ephemeral=true`). Same persona/global-history scoping as `kobold_export`; suppressed rows already filtered upstream (**C5**). `has_versions` comes from a single batch query `memory_manager.get_ids_with_versions(ids)` (no N+1). A live parked confirmation for `("portal", persona)` is appended as a trailing ephemeral chunk carrying its `ephemeral_chunk_id`. This is the re-sync source for the Lite stopgap and the render source for the bespoke UI — consumers address chunks by identity, never by position, so the array can no longer drift.
  - **`build_kobold_savefile` C2 fix** — see kobold_export section below.
  - **What converges now vs DP-131 (DP-130 is server-only):** the gametext-aligned `kobold_export` (C2) makes the **session restore/reload** path converge with the *unchanged* portal today (`ids_len==storyLen`, verified live). What still needs DP-131 is the **in-session per-turn** path: portal.html line ~33349 guards id pushes on `if (payload.assistant_id)`, so a parked turn mid-session won't advance the array until the next reload — DP-131 drops that guard / re-keys off `/transcript`. The contract test `test_parked_write_emits_id_frame_with_ephemeral_chunk_id` proves the on-wire parked frame deterministically through the real kernel.

- **DP-292 — memory import panel routes.** Operator inventory + ingest for Hindsight banks, all under `/api/v1/memory/*`, reached through the enumerated `_memory_backend` seam (`chat_system.memory_backend`; pinned by `test_adapter_engine_surface_is_enumerated`). Reads are GET (open); mutations pass the DP-277 control-plane auth middleware (`DERPR_CONTROL_TOKEN`). Routes: `GET /banks`, `GET /banks/{id}/documents` (query `q`/`tags` CSV/`limit`/`offset`), `GET /banks/{id}/operations` (query `status`/`type`→`op_type`), `DELETE /banks/{id}/documents/{document_id:path}` (path converter — relpath keys hold slashes), `POST /banks/{id}/upload` (multipart, `.md`/`.txt` decoded → `retain_document`, filename-keyed idempotency; PDF/native `files/retain` deferred), `POST /banks/{id}/ingest_url` (adapter `self._http` fetch → `retain_document`), `POST /banks/{id}/ingest_path` (constructs an `IngestPathHandler`, calls the new turn-context-free `ingest_root(bank_id, root, glob, force)`). Backend failures are mapped by the static helper `_run_backend`: `NotImplementedError`→501 (SQLite backend has no import surface), `HindsightAPIError`→upstream status, `httpx.RequestError`→502. Frontend: the `◈ MEMORY` NavRail dock flips a `view: 'chat'|'memory'` state in `App.tsx` to render `components/MemoryPanel.tsx`; live-only client fns in `api/client.ts` (`listMemoryBanks`, `listMemoryDocuments`, `listMemoryOperations`, `deleteMemoryDocument`, `uploadMemoryFiles`, `ingestMemoryUrl`, `ingestMemoryPath`) — no mock fallback, mutations carry `withAuth`.
- **DP-292 phase 2 — content-date anchoring.** Hindsight derives every extracted memory's `mentioned_at`/`event_date` **solely from the retain request's `timestamp`** (the extraction LLM never reads dates from prose — measured, see `2026-05-14-hindsight-backfill-timestamp-finding`). So the three text-ingest paths (upload, `ingest_url`, `IngestPathHandler`) resolve an anchor date from the document body *before* retain and pass it as `timestamp` instead of upload-time/mtime. `src/memory/date_extraction.py` owns the deterministic core: `extract_regex_dates(text)` scans ISO / `YYYY/MM/DD` / named-month forms (bare-numeric `MM/DD/YYYY` intentionally excluded — locale-ambiguous), `pick_anchor(dates, clamp_now)` drops future (`> clamp_now + 1d`) and pre-`1990` values and returns the **latest** survivor, and `async extract_anchor_date(text, *, fallback_ts, clamp_now=None, llm_tagger=None) -> (datetime, source)` runs regex first, then the optional `llm_tagger` when regex is empty, then `fallback_ts`; `source ∈ {"regex","llm","fallback"}`. The optional LLM fallback is `src/memory/date_tagger.py::DateTagger` — a single-shot forced-tool-schema (`submit_date`) call on the stateless, tool-less `date_tagger` system persona, mirroring `ContentClassifier`: body handed in as untrusted DATA, output constrained to one ISO date or `none`, validated + future-clamped on our side (injection can at worst move the anchor to a plausible past date, never emit instructions or reach a tool). Gated by `DATE_TAGGER_ENABLED` (default on) + `DATE_TAGGER_NAME`; when off, ingest is regex-only. Each retained doc gains `date:<YYYY-MM-DD>` + `date_source:<...>` tags and the same metadata keys. Anchoring is per-document (one item → one `timestamp`; Hindsight stamps every unit it extracts with it — per-fact granularity is not available upstream).

### `src/interfaces/kobold_export.py` -- Savefile builder + transcript projection
- Pure function `build_kobold_savefile(raw_history)` → `(savefile_dict, skipped_count)`.
- **Invariant C2 (DP-130) — gametext alignment:** `interaction_ids` carries one entry per *visible story chunk* including the prompt, so `len(interaction_ids) == len(actions) + 1` (= kobold-lite's `gametext_arr` length, `[prompt, *actions]`). This is the alignment the portal actually uses — `derpr_interaction_ids[modified_turn]` is keyed by the gametext index (index 0 == prompt), and the SSE id-frame path pushes one id per visible chunk. The loop appends a slot (id or `None`) for every renderable row and **never pops the id array**; only `rendered[0]` is split off into `prompt`. **Correction (this is the real C2):** the decision doc's literal `len(actions)==len(interaction_ids)` was itself off-by-one — the reported `actions=12 ids=13` was *correct* gametext alignment (13 chunks = prompt + 12 actions). The actual pre-fix defect was a conditional `if isinstance(iid,int)` append that could drop an id without dropping its chunk; the gametext-aligned rewrite fixes that and makes the **unchanged portal restore correct today** (verified live: `ids_len==storyLen==7`, was 5 vs 6 under a transient actions-aligned implementation). Renderability is shared with the transcript via `_is_renderable` / `_merge_reasoning` helpers.
- `build_transcript(raw_history, *, ids_with_versions, pending)` → `{"chunks": [...]}` — the DP-130 projection (see Phase 4.1 above). Shares the row-filter helpers with the savefile builder so both stay consistent. `pending` (the live parked confirmation, when present) is appended as a trailing `ephemeral=true` chunk.
- Wraps `author_role='user'` rows in the literal placeholders `\n{{[INPUT]}}\n…\n{{[OUTPUT]}}\n`; emits `author_role='assistant'` rows as raw text. System rows, empty-content rows, and unknown-role rows are counted once in `skipped` and dropped. Assistant rows with `tool_context` are treated as normal rows — they're skipped only when content is empty (tool-call-only), otherwise their visible text is preserved. Proper tool-turn rendering is backlog.
- The placeholders mirror kobold-lite's `instructstartplaceholder` / `instructendplaceholder` constants; kobold's render layer substitutes them with the active `instruct_starttag` / `endtag` at submit time, so DERPR never picks the actual instruct tags.
- The first emitted entry becomes the savefile's `prompt`, the rest go into `actions[]`. `memory` is left empty — the persona system prompt is routed into kobold-lite's `instruct_sysprompt` (Sys. Prompt) setting by the UI's `applyPersonaToUI`, keeping the Memory block user-owned.

### `src/agents/` -- Agent Framework

**`base.py` -- Agent (ABC)**
Abstract base for autonomous background workers. Not user-interactive — poll external systems on a schedule and act independently.
- Lifecycle: `start()` → `_on_start()` → `[loop: deploy() every interval]` → `stop()`
- `deploy()`: Abstract — subclass implements one work cycle
- `_build_llm_context()`: Minimal context (user prompt + optional action history injection)
- `_get_action_history_message()`: Injects recent actions into LLM context if `action_history_limit > 0`
- Auto-loads system personas from file on init (agents invoke system personas for read-only analysis)
- Config: `schedule` (dict, e.g. `{"interval": 30}`), `action_history_limit` (int), `agent_name` (str)

Trajectory-logging contract (DP-116a) — root + children written to `Agent_Actions`:
- `_log_task_root(action_type, trigger_context, action_payload, contexts, outcome="pending")`: opens a root row, JSON-serialises + ASCII-safe-truncates `action_payload`, attaches `Agent_Action_Contexts` rows; returns the root `action_id`.
- `_log_step(parent_id, action_type, action_payload, outcome, outcome_payload, contexts)`: child row under a root.
- `_add_contexts(action_id, [(type, value), ...])`: idempotent context-tag insert (drops null/empty).
- `_finalize_action(action_id, outcome, outcome_payload)`: terminal update on a root.
- `_serialize_payload(data)` / `_truncate_ascii(text, max_len=None)`: dict→JSON, ASCII-only (`encode("ascii","replace")`), capped at `MAX_PAYLOAD_CHARS` (4000) unless `max_len` overrides. ASCII-only is defensive against Hindsight's utf-8→latin-1 mangling on some endpoints.

Hindsight bridging (DP-116b) — fire-and-forget bridge of a finalized series into Hindsight:
- `_format_action_series_prose(parent, steps, contexts)`: flattens a root + children + context tags into dense k:v prose (no JSON braces, no nulls). Capped at **16000** chars (Hindsight chunks at 10k internally — a belt on pathological loops, never expected to fire when log-time ref discipline holds).
- `_retain_action_series(action_id)`: called by subclasses after `_finalize_action` on a root. Re-reads the series from `Agent_Actions` (idempotent across restarts), formats prose, enqueues a `retain_experience` POST with stable `document_id=f"agent_action:{action_id}"` (`_action_document_id`, prefix `AGENT_HISTORY_DOC_PREFIX`) + `content_override=<prose>`. Fire-and-forget through `HindsightBackend`'s queue; SQLite-shape backends raise `NotImplementedError`, which is swallowed — bridging is Hindsight-only and opportunistic.
- Class attrs `experience_bank` / `experience_persona` select the destination bank (default: `agent_name`). DispatchAgent points both at `DISPATCH_PERSONA_NAME` so dispatch series mingle into the `dispatch_analyst` persona bank; ReminderAgent + SqliteConsolidator default to their `agent_name`.

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

Trajectory logging (DP-116a): each `_dispatch_ticket()` call writes one `dispatch` root row plus child rows for `tool:zammad.get_ticket`, `tool:zammad.get_ticket_articles`, `llm_step` (dispatch_analyst), `tool:notification.send`, `tool:zammad.add_tag`. Root `action_payload` carries `{ticket_id}`, `outcome_payload` the dispatch result; contexts `ticket_id` + `persona`. On finalize the series is bridged to Hindsight via `_retain_action_series` (into the `dispatch_analyst` bank).

**`reminder_agent.py` -- ReminderAgent**
Scheduled open-ticket nudge agent (shipped — registered in `main.py` when Zammad is available). Polls for tickets needing follow-up and routes a reminder notification via `NotificationRouter`. Uses the same DP-116a/b trajectory logging: `_log_task_root` per cycle + `_retain_action_series` on completion (bridges into its own `reminder` bank; `experience_bank` unset → defaults to `agent_name`).

**`zammad_bot.py` -- ZammadBot**
Multi-stage AI triage pipeline for new, untagged tickets. Uses 4 system personas:
1. `triage_scout` → extract search keywords
2. Search Zammad for related closed tickets (global + per-user)
3. `triage_filter` → relevance scoring of historical tickets
4. `triage_summarizer` → compress long ticket bodies (adaptive, only if context exceeds limit)
5. `triage_analyst` → full analysis with context → internal note posted to ticket
Tags ticket as triaged. No tools used — all LLM calls are read-only.

**`managr_agent.py` -- ManagrAgent** (`agent_name="managr"`, DP-280/282/290)
Autonomous Zammad ticket-triage manager. Per cycle: builds a board snapshot, fans it out to read-only analyst personas for briefs, then a planner persona (`MANAGR_PLANNER_NAME`) produces the Manager's Report. When `proposals_enabled`, a second planner call (`tools=[submit_proposals]`) emits proposed writes into the `src/proposals/` queue instead of writing directly (see that section) — human approves via the portal. DP-290 also gives it reflective dispositions: it can reaffirm/revise/withdraw its own still-pending proposals from a prior cycle. Uses `content_classifier.py` (`ContentClassifier`/`Classification`) to pre-signal-classify ticket content before triage.

**`sqlite_consolidator.py` -- SqliteConsolidator** (formerly `MemoryAgent`; `agent_name="memory"`)
Batch agent that segments conversations by topic, extracts observations via LLM, and stores embedded summaries. **Registered only when `SEMANTIC_BACKEND=="sqlite"`** — the Hindsight backend drives consolidation upstream, and registering this agent under it would crash `deploy()` on the first cycle (legacy SQL ops raise `NotImplementedError`). Two-phase pipeline per channel:

**Phase 1 — Embed (loop until no unembedded messages):**
1. `get_unembedded_messages()` fetches msgs not yet in Message_Embeddings
2. `_chunk_messages()` splits into token-aware batches (item cap + token cap)
3. `EmbeddingService.encode()` batch API call
4. INSERT into `Message_Embeddings` + `vec_Message_Embeddings`

**Phase 2 — Segment + Summarize (loop until clear):**
1. `get_unsegmented_embedded_messages()` fetches embedded msgs with `parent_summary_id IS NULL`
2. `_segment_by_similarity()` groups msgs by centroid-based cosine similarity (threshold from config, default 0.80)
   - Seeds centroid from tail of previous segment for continuity
   - Min segment size configurable (default **1**)
   - Bridging (two mechanisms):
     a. Explicit: msg has `reply_to_id` pointing to a message already in the current segment
     b. Heuristic: fallback for pre-`reply_to_id` data — single user msg immediately followed by assistant msg
3. `_summarize_segment()` sends transcript to LLM via `memory_summarizer` system persona
   - Token guardrail: counts tokens via Google SDK (`count_tokens`), skips segment if > `RATE_LIMIT_GEMMA_4_TPR * 0.95`
   - Pre-processes: strips vertexai grounding redirect URLs from content
   - LLM calls `submit_memory_summary` tool with `observations[]` + `keywords[]` + `outlier_ids[]`
   - Outlier msgs stay `parent_summary_id=NULL` for re-queueing in next batch
   - Fallback: parses plain text if model doesn't use tool call
4. Stores segment + summary + vec embedding in DB, sets `parent_summary_id` on processed msgs

Config in `agents.json`: similarity_threshold, min_segment_size, batch_size (default **100**, capped at 100), allowed_channels, embedding_provider
Rate limiting: shared `GLOBAL_EMBEDDING_LIMITER` (AsyncLimiter) with MemoryConsolidator to prevent double-spending embedding quota

**`agent_service.py` -- AgentServiceIntegration**
Plugs agent tools into ChatSystem's service binding system. Personas with `service_bindings: ["agents"]` gain access to agent management tools.

**Agents vs Personas:**
- Agents are autonomous background workers — no user interaction, run on intervals
- Personas are conversational — respond to user messages, can use tools including agent management tools
- System personas are read-only LLM configs used by agents for analysis (no tools)
- Agents do NOT spawn other agents; no delegation chains currently

### `src/tools/agent_tool_handler.py` -- AgentToolHandler
Four tools gated behind `service_bindings: ["agents"]`:
- `get_agent_status` (read): Running state, deploy counts, errors for one or all agents
- `get_agent_history` (read): Recent action log with optional ticket_id/customer filters
- `manage_agent` (write): Start/stop/restart — parked for audit like any write tool (regardless of execution mode)
- `lookup_agent_history` (read, DP-116b): Dereference a single action series by `action_id` — returns the parent row + child steps + context tags. Used to recover the full trajectory after a Hindsight recall hit surfaces `action_id:<n>` from the bridged experience.

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
- `model_utils.py` -- `get_model_prefix()`: maps model names to family prefixes for routing and rate limiting. Model list refresh functions for OpenAI/Google/Anthropic APIs. **Model list — single source of truth is `chat_system.models_available`** (a snapshot taken at startup via `get_model_list()`, refreshed by the `update_models` command). Both consumers read it: the `what models` command, and the web UI model dropdown (`GET /api/v1/models/list` in both kobold adapters → flattens `chat_system.models_available`). The dropdown used to re-read the cache file via `get_model_list()` per request — unified to `models_available` on 2026-06-02 (DP-127) so it can't drift from `what models`. Underlying cache: `data/personas.json` → `"models"`. `get_model_list(update=True)` is the only path that hits provider APIs (slow) and rewrites the cache. **`STATIC_MODELS`** (`agy-flash`, `local`) are code-known, non-API models merged in on *both* the update and cached-read paths so they're always selectable without an API refresh — fixed 2026-06-01 (DP-127); previously they only existed on the update path and a pre-agy cache hid them.

### `src/personas/`
- `store.py` -- Persona JSON file I/O (moved from `utils/save_utils.py`, DP-203): load/save personas and the model catalog to disk. Handles default-persona auto-seeding, system-persona loading, and DP-128 quarantine-on-load validation

### `src/security/` -- Credential vault + egress scrubber (DP-225)
- `vault.py` — `CredentialVault`: the single inventory of machine secrets. Resolves credential refs (`KNOWN_REFS`: OpenAI/Anthropic/Google/Zammad API keys) via a pluggable source (default `os.environ.get`). `TextEngine` resolves all provider API keys through `get_vault()`. Process-global via `get_vault()`/`reset_vault()`.
- `scrubber.py` — `SecretScrubber`: process-global egress redactor. Registered exact secret values redact to `[REDACTED:<ref>]` (longest-first); a regex fallback catches unregistered secret *shapes* (`sk-ant-…`, `sk-…`, `Token token=…`, `Bearer …`) → `[REDACTED:pattern]`, skipped on strings > `MAX_PATTERN_SCAN_LEN` (100k) to avoid corrupting cached image blobs.
- Enforced at egress seams: `turn_persistence.store_api_request` (API-payload caching), `tool_loop` (audit_info + tool-result strings), `engine`, `zammad_client`.
- Startup wiring: `bootstrap.register_credentials()` calls `get_vault().register_into(get_scrubber())` from within `create_chat_system`.

### `src/self_edit/` -- "fixr" self-improvement supervisor (DP-227)
Event-driven dispatcher that sits ABOVE the engine. One dispatch = one bug = one `git worktree` (via `clone_manager.create_worktree`, off a pristine base clone) + one detached `claude --output-format stream-json` coding-agent subprocess. `dispatcher.py` tails the agent's raw stream-json log through a per-agent bridge task (never the live pipe), normalizes lines to common-schema `AgentEvent`s (`events.py`), and on a `{question,done,error}` wake event calls `on_wake`, which `FixrIntegration` (`integration.py`) wires to `chat_system.generate_response(CC_FIXR_PERSONA, …)`. `registry.py` holds `AgentRegistry`/`AgentRecord`; status persists through an optional `AgentStore` SQLite write-through (`store.py`, DP-233; `CC_FIXR_REGISTRY_DB`) so RUNNING/WAITING agents are reloaded as ORPHANED on restart, and `AgentRecord.archived` (DP-237) backs soft-archive + the `prune_agents` reaper. `FixrIntegration` is a `ServiceIntegration`; personas opt in via `service_bindings: ["fixr"]`. Tools (`fixr_tools.py`, **6**): `dispatch_fix` (`is_write:True` — the one always-gated write, parked for confirmation before the agent starts), `inspect_agents`, `answer_agent`, `kill_agent`, `prune_agents`, `send_discord` (the other five are `is_write:False`/ungated so the woken supervisor can coordinate, prune/reap, and report without a confirmation prompt per event). Approval at two boundaries: ConfirmationManager gates `dispatch_fix`, and a human merges the PR the agent opens — this module never merges or pushes. Config knobs in `global_config`: `CC_FIXR_MODEL_ARG`, `CC_FIXR_CLONE_DIR`, `CC_FIXR_PERSONA`, `CC_FIXR_CHANNEL`, `CC_FIXR_DISCORD_CHANNEL`, `CC_FIXR_REGISTRY_DB`.

### `src/voice/` -- Voice command subsystem (DP-238)
Browser/phone push-to-talk voice commands driving the text LLM. `VoiceIntegration` (`integration.py`) is a `ServiceIntegration` (name `"voice"`) that registers `VOICE_TOOLS` (`src/tools/tool_defs/voice.py` — `set_timer` / `list_timers` / `cancel_timer`, `service_binding: "voice"`); the text LLM owns timers via these tools (no separate voice model reasons over them). Registered in `main.py` at step 7.2. Pipeline: a complete utterance → Moonshine STT (CPU, `MoonshineTranscriber` in `transcriber.py`, lazy `moonshine_onnx` import) → intent → timer/alarm. The live capture source is the browser/phone push-to-talk web endpoint (`web.py`, mounted on the engine adapter's FastAPI app via `attach_web`, gated by `VOICE_WEB_ENABLED`). The Discord voice-receive path (`attach_discord`, `capture.py`) is dead (Discord DAVE encryption) and short-circuits when invoked. Personas opt in via `service_bindings: ["voice"]`.

### `src/proxmox/` -- Proxmox management subsystem (DP-262)
SSH-driven node/guest power ops + koboldcpp model swap on the GPU container's :5001 endpoint. Personas opt in via `service_bindings: ["proxmox"]`.
- `integration.py` — `ProxmoxIntegration(ServiceIntegration)` (name `"proxmox"`), registration-only. **Always** registers (even when `PVE_TOOLS_ENABLED` is false) so the startup-wiring contract holds; the handler short-circuits disabled calls with a clear error instead of attempting SSH.
- `handler.py` — `ProxmoxToolHandler` registers 7 tools; owns a config-driven `SSHRunner`.
- `ssh.py` — `SSHRunner`: runs metacharacter-free argv over SSH to the pve node (rejects shell metacharacters).
- Tools in `src/tools/tool_defs/proxmox.py` (`PROXMOX_TOOLS`, 7): read — `pve_status`, `list_models`; write (parked for confirmation) — `reboot_node` (also `irreversible`), `reboot_guest`, `start_guest`, `stop_guest`, `set_active_model` (`exfil_capable=False`, see DP-263). All carry `produces_untrusted: False`, `locality: "network"`, `sensitivity: "internal"`.
- **Model-availability guard (DP-264):** `list_models` omits configured units whose gguf isn't on disk (a missing file can't be loaded, and enabling it would take :5001 down); `set_active_model` refuses the swap — leaving the current model running — unless the target's model file is present, so it never disables the live model to enable one that can't start (`_model_present` / `_model_path` in `handler.py`).

### `src/tools/mcp_client.py` + `mcp_integration.py` -- MCP client (DP-268)
Consume external MCP tool servers (streamable-HTTP transport). Discovered tools become ordinary derpr tools (`mcp__<server>__<tool>`, `service_binding: "mcp:<server>"`) — parking, taint, composition rules, and per-persona policy all apply with no MCP-specific downstream paths. Personas opt in per tool + bind `mcp:<server>` (no wildcard — see `dynamic` below).
- `mcp_client.py` — `MCPClientManager`, **owned by `main.py`** (voice precedent: `ServiceIntegration` is registration-only, so main.py owns the session lifecycle). Each server runs in a dedicated `_ServerConnection` task that owns the anyio transport contexts (must be entered/exited in one task); `_open_session` is the test seam. `start()` connects every enabled server from `MCP_SERVERS_FILE` (dead server logged + skipped, never breaks startup); `add_server` connects + discovers + registers the tools live and persists config **only on success** (rollback on failure; strict config load so a corrupt file is never clobbered); `remove_server` stops the session, unregisters defs + handlers, persists; `list_servers` reports url/enabled/connected/tools; `call_tool` invokes the server session. Every live (de)registration triggers `revalidate_persona_security` across all personas via `personas_provider`.
- **Translation** (`_translate_tool`): each discovered tool defaults to the most-restrictive metadata (`is_write: True`; `produces_untrusted`/`irreversible: True`, `locality: "network"`, `sensitivity: "pii"`), relaxable only by operator per-tool `tool_overrides` in the config file — server-provided annotations (`readOnlyHint`/`destructiveHint`/…) are logged as hints, never policy. Definitions carry **`dynamic: True`**, the marker that excludes them from `['*']` wildcard expansion in `ToolPolicy.filter_tools` (a new server can never silently widen or quarantine-cascade wildcard personas). Server name is validated, namespaced tool name capped at the 64-char provider limit, description capped at 1024 chars (prompt-injection surface).
- **Hot reload (phase 3):** a background maintenance loop reconnects dead servers and re-discovers a server's toolset when it signals `tools/list_changed` (the periodic tick is the fallback). Network work runs unlocked; registration swaps are all-or-nothing and a no-op when the toolset is unchanged (flap-safe). `MCP_RECONNECT_INTERVAL <= 0` disables the loop. While a server is down its tools stay registered and degrade to per-call `{"error": …}`, so persona policies stay stable across an outage.
- `mcp_integration.py` — `MCPIntegration(ServiceIntegration)` (name `"mcp"`), registration-only: attaches the `ToolManager` to the manager and registers the three management tools. **Always** registers (even when `MCP_ENABLED` is false) so the startup-wiring contract holds; disabled calls short-circuit with a clear error.
- Tools in `src/tools/tool_defs/mcp.py` (`MCP_TOOLS`, `service_binding: "mcp"`): `add_mcp_server` (`is_write` → parked; `url` is model-controlled egress, exfil-capable), `remove_mcp_server` (`is_write` → parked; `exfil_capable: False`), `list_mcp_servers` (read; `produces_untrusted: True`, `exfil_capable: False`).
- Live catalog: `ToolDefinitionRegistry` (`definitions.py`) holds discovered defs; `ToolManager.unregister` drops their handlers; discovered-tool handlers are closures over `MCPClientManager.call_tool`.
- Config (`global_config.py`): `MCP_ENABLED` (default off), `MCP_SERVERS_FILE` (default `DATA_DIR/mcp_servers.json`), `MCP_CONNECT_TIMEOUT` (30s), `MCP_CALL_TIMEOUT` (120s), `MCP_RECONNECT_INTERVAL` (60s). Dependency: `mcp` SDK.

### `src/proposals/` -- durable proposal queue (DP-282, managr Phase 1; self-managing queue DP-290)
Human-approval gating primitive for autonomous-agent writes (replaces ConfirmationManager for this shape — that one is in-memory, one-pending-per-(user,persona), chat-turn-bound). `ManagrAgent` proposes; a human approves via the portal; a separate executor writes. Documented in full in `docs/user_guide.md`.
- `schemas.py` — `PROPOSAL_ACTIONS` whitelist (`add_note` internal-only / `set_priority` / `remind`) + `validate_proposal_args` (required/type/enum/max_length/date, rejects unknown actions and unexpected keys). `build_submission_tool_schema()` derives the agent-internal `submit_proposals` tool from the whitelist.
- `executor.py` — `ProposalExecutor(zammad_client).execute(proposal) -> (bool, str)`. Re-validates args at execution time, resolves `ticket_number` → internal id via search, dispatches to the relevant Zammad write (add note / update priority / update state+pending_time). Errors returned, not raised.
- `service.py` — `ProposalIntegration(ServiceIntegration)`, `name="proposals"`; registers `list_proposals` (read), `approve_proposal`/`deny_proposal` (`is_write:True` → write gate). Approve reviews the row, audits, executes immediately, marks executed. Deny requires a reason. Lazy `expire_stale_proposals()` sweep on list/approve/deny. Registered in `main.py` only when Zammad is configured.
- Store: `Proposals` table in `memory_manager.py` — status pending/approved/denied/expired/executed/execution_failed/withdrawn; partial UNIQUE index on (agent, action_type, ticket_number) WHERE pending enforces dedup.
- Self-managing queue (DP-290): same-key pending proposal is UPSERTed in place instead of duplicated; `ManagrAgent` can also `reaffirm`/`revise`/`withdraw` its own pending proposals each cycle (reflective dispositions) — TTL acts as a GC backstop, not the primary lifecycle.
- Emission (in `ManagrAgent`): config-gated `proposals_enabled`; a second planner call with `tools=[submit_proposals]`; every candidate code-validated before insert, capped at `MANAGR_MAX_PROPOSALS_PER_CYCLE`.

### `src/main.py` -- Startup Sequence
1. MemoryManager (SQLite) + schema migration
2. TextEngine (LLM API router)
3. ZammadClient (optional, fails gracefully if no credentials)
4. EmbeddingService (GeminiEmbeddingProvider) — shared by ChatSystem and SqliteConsolidator
5. ChatSystem (DI hub, injected with memory + engine + embedding_service)
6. Register ZammadIntegration service (if Zammad available)
7. AgentManager + register agent classes (`SqliteConsolidator` only when `SEMANTIC_BACKEND=="sqlite"`; `ZammadBot` + `DispatchAgent` + `ReminderAgent` + `ManagrAgent` only if Zammad available)
8. Register AgentServiceIntegration service (agent tools)
9. AppManager + NotificationRouter (Discord/Zammad notifiers)
9.1 Register FixrIntegration service (fixr dispatch tools, DP-227 — needs ChatSystem + NotificationRouter) [main.py step 7.1]
9.2 Register VoiceIntegration service (voice timer tools, DP-238 — needs NotificationRouter; Discord/web capture late-bound after the interfaces exist) [main.py step 7.2]
9.3 Register ProxmoxIntegration service (proxmox management tools, DP-262 — registration-only, always registers) [main.py step 7.3]
9.4 Register MCPIntegration service (MCP management tools, DP-268 — registration-only, always registers; constructs the main.py-owned `MCPClientManager` with `personas_provider`) [main.py step 7.4]; later `await mcp_manager.start()` connects configured servers and registers their discovered tools (no-op when `MCP_ENABLED` is false), and `mcp_manager.aclose()` runs at shutdown [main.py step 8.2]
9.5 Register ProposalIntegration service (proposal-queue review tools, DP-282 — only if Zammad available; wraps `ProposalExecutor(zammad_client)`) [main.py step 7.5]
10. Register interface tasks (Discord bot, Gmail bot)
11. Register MemoryConsolidator as hourly background daemon (`app.register_task("memory_consolidator", consolidator.start_daemon(check_interval_seconds=3600))`)
12. Optional model list refresh on startup
13. `app.start()` → auto_start agents + launch interface tasks

## Config

### `config/global_config.py`
- DEFAULT_HISTORY_MESSAGES = 15 (messages fetched from DB)
- GLOBAL_HISTORY_MESSAGES = 30 (hard cap when history_limit passed)
- MAX_TOOL_CALLS = 5 (per request)
- MAX_CACHED_API_REQUESTS = 128
- DEFAULT_TOKEN_LIMIT = 4096 (LLM output)
- PENDING_CONFIRMATION_TIMEOUT = 300 (seconds)
- EMPTY_RESPONSE_RETRIES = 3, EMPTY_RESPONSE_RETRY_DELAY = 2
- DISCORD_CHAR_LIMIT, DISCORD_STATUS_LIMIT, DISCORD_DEBUG_CHANNEL, AMBIENT_LOGGING_CHANNELS
- Rate limits per model family (RPM/RPD): GEMINI_25, GEMINI_3, GEMMA_3, GEMMA_4 (also RATE_LIMIT_GEMMA_4_TPR = 256000 tokens per request), OPENAI, ANTHROPIC
- Long-term memory: MEMORY_RETRIEVAL_ENABLED = True, MEMORY_MAX_SUMMARIES_IN_CONTEXT = 5
- Embedding: EMBEDDING_MODEL = 'gemini-embedding-001', EMBEDDING_DIMENSION = 3072
- Embedding rate limits: GEMINI_EMBEDDING_001_RPM = 100, GEMINI_EMBEDDING_001_TPM = 30000, GEMINI_EMBEDDING_001_RPD = 1000

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
- `tests/memory/test_memory_manager.py` -- schema, CRUD, migration tests (`legacy_mem_manager` fixture)
- `tests/memory/test_memory_consolidation.py` -- consolidation clustering and merge tests
- `tests/agents/test_memory_agent.py` -- segmentation, summarization, embedding, deploy cycle
- `tests/test_memory_retrieval.py` -- retrieval relevance, model filtering, level filtering
- `tests/test_embedding_service.py` -- embedding provider, chunking, rate limiting
- `tests/unit/test_memory_glue.py` -- semantic glue (Q/A pair preservation in segmenter)
- `tests/integration/test_full_system_flow.py` -- end-to-end with mocked externals
- `tests/integration/test_memory_modes.py` -- memory mode behavior
- `tests/interfaces/test_discord_bot.py` -- Discord bot event handling
- `tests/interfaces/test_gmail_bot.py` -- Gmail bot message handling

Migration test pattern: `legacy_mem_manager` fixture creates DB with OLD schema (no new columns/tables), test calls `create_schema()` and verifies migration.
