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
      -> _build_conversation_history()  -- fetch sliding window from DB
      -> _retrieve_memory_block()       -- embed (sliding window + current msg), KNN search,
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
- Dependency injection hub: holds personas, TextEngine, MemoryManager, ToolManager, EmbeddingService, services
- Constructor: `__init__(memory_manager, text_engine, embedding_service=None)` — registers WebSearchHandler, MemoryToolHandler, and MemoryRecallHandler at init. Borrows `memory_manager.backend` as `self.memory_backend` and pushes the embedding service into it via `set_embedding_service` so `SqliteSemanticBackend.recall` can translate query → embedding.
- `generate_response()` -> returns `Tuple[str, ResponseType, Optional[int]]` (text, type, ticket_id)
- `_build_conversation_history()` -> fetches from DB via MemoryManager, formats for LLM; returns `(history, oldest_interaction_id)`
- `_retrieve_memory_block()` -> resolves a query string from current message / latest user turn, then calls `self.memory_backend.recall(bank_id=persona, query, k=MEMORY_MAX_SUMMARIES_IN_CONTEXT, tag_filter=[channel:_, user:_, server:_, exclude_after:_])`. Formats the returned `MemoryHit` list into a `<memory>` XML block injected at history[0]. The `exclude_after:N` tag carries the sliding-window cutoff through the backend boundary; SqliteSemanticBackend translates it back into `exclude_after_interaction_id`, Hindsight ignores it. Controlled by `MEMORY_RETRIEVAL_ENABLED` and `MEMORY_MAX_SUMMARIES_IN_CONTEXT`.
- `_prepare_request()` -> builds history, injects memory block, appends user message, then calls `memory.context_budget.truncate_messages_to_budget(messages, max_context_tokens - response_token_limit)` to enforce the persona's total ctx cap. System messages and the latest user message are always preserved.
- `_orchestrate()` -> shared streaming kernel — preprocess → set TurnContext (`src/tools/turn_context.py`) → user-log → user `retain_turn` → ToolLoop → assistant-commit → assistant `retain_turn` (taint-aware) → DoneEvent → reset TurnContext. `stream_response()` (portal) and `generate_response()` (Discord/Gmail/agents) both delegate here. Token + tool events forwarded as-is; internal `_ApiPayloadEvent` siphons into `_store_api_request`, `_LoopFinishedEvent` drives universal write-audit parking + assistant persistence. Both retain calls are fire-and-forget (sqlite_legacy noop; Hindsight enqueues + returns).
- `_retain_turn_safe()` -> wraps `memory_backend.retain_turn(...)` with try/except so a backend hiccup never derails the user turn. Builds scope tags + threads `untrusted` (False for user turns; `ctx.turn_tainted` for assistant turns).
- `_filter_tools_for_persona()` -> filters by enabled tools, service bindings, model compatibility
- `resume_pending_confirmation()` -> handles the write-tool approval flow for any parked turn (all write tools park for audit, not just CONFIRM mode; still uses `_execute_write_calls` / `_append_denied_tool_results` retained for this path)
- `PendingConfirmation` dataclass stores paused state for write-tool confirmation
- `RequestContext` dataclass bundles all pipeline state

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

### `src/engine.py` -- TextEngine
- Provider-agnostic LLM abstraction: OpenAI, Anthropic, Google (Gemini/Gemma), local OpenAI-compatible, Antigravity (`agy`)
- `generate_response()` -> returns `Tuple[Dict[str, Any], Optional[Dict[str, Any]]]` (response, payload)
- Response dict: `{"type": "text", "content": "..."}` or `{"type": "tool_calls", "calls": [...]}`
- Per-model-family rate limiters (AsyncLimiter) configured in global_config
- Provider dispatch: `_generate_openai_response`, `_generate_anthropic_response`, `_generate_google_response`, `_generate_agy_response`
- **`agy` route (DP-127, model prefix `agy-*`):** drives the local `agy` CLI via subprocess on the OAuth tier (Gemini 3.5 Flash), not an API key. `_render_agy_prompt` flattens the full history into one role-tagged transcript (stateless, re-flattened each tool-loop turn); `_render_agy_tool_protocol` injects tool descriptions + asks the model to emit a `<tool_call>{json}</tool_call>` block; `_parse_agy_tool_call` parses that block back into the standard `{"id","name","arguments"}` shape so DERPR's own tool loop executes it (engine keeps full policy/CONFIRM/taint control — agy never runs tools itself). `_run_agy_cli` spawns with `stdin=DEVNULL` (or it blocks on an interactive permission prompt), **no** `--dangerously-skip-permissions`/`--add-dir` (so agy's own tools stay gated and it can never run them — DERPR drives every tool itself), in a throwaway tempdir, killing the process group on cleanup. **POSIX-only (DP-138):** `_ensure_agy_supported()` refuses the route on native Windows with a clear error — agy is a TUI that only writes its response to a TTY, but the engine captures piped stdout, so on Windows the response is always empty (no flag/env var changes this; works on macOS/Docker). Run the engine on the POSIX host/WSL/Docker to use agy. Text-only (no images). The Antigravity *SDK* route was rejected (API-key-only, Model-A-native) — see `project/decisions/2026-05-29-agy-sdk-oauth-finding.md`.
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
- `ExecutionMode`: AUTONOMOUS, CONFIRM — recorded in the audit surface for display, but **does not gate write parking** (all write tools park for audit regardless; see ToolLoop universal write-audit)
- `MemoryMode`: CHANNEL_ISOLATED, SERVER_WIDE, PERSONAL, GLOBAL, TICKET_ISOLATED
- `get_history_messages()` -> supports dynamic override via hello/goodbye commands (increments by 2 per turn)
- Service bindings list determines which tools and services are available
- `include_ambient_memory: bool = True` — whether to include ambient-channel memories in retrieval; controlled by `get_include_ambient_memory()`
- `thinking_level: Optional[str]` — passed through to engine config (e.g. `"minimal"` for Gemma extended thinking)
- `max_context_tokens: int` — total context budget (prompt + reserved response). Defaults to `global_config.DEFAULT_MAX_CONTEXT_TOKENS` (131072). Mirrors kobold-lite's `localsettings.max_context_length` slider semantic so the value round-trips on persona load.
- `get_config_for_engine()` -> returns dict consumed by TextEngine (includes thinking_level if set)

### `src/tools/` -- Tool System
- `definitions.py` -- JSON schemas for all tools, `WRITE_TOOLS` set, `MODEL_INCOMPATIBLE_TOOLS` dict
- `tool_manager.py` -- ToolManager registry, `execute_tool()`, handler registration pattern
- Tool categories: read-only (search, get) vs write (create, update, close) -- all write tools are parked for human confirmation regardless of execution mode (universal write-audit, per the security framework)
- `WebSearchHandler` registered at ChatSystem init
- Service-specific tools registered via `ServiceIntegration.register_tools()`
- Memory tools: `submit_memory_summary` (observations[] + outlier_ids[], used by SqliteConsolidator), `drill_down_memory` (retrieves source messages for a summary), `update_core_memory` (edits core profile summaries)

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
- `backend/hindsight.py` — `HindsightBackend` (native REST via `HindsightRESTClient`). Retain path is fire-and-forget through one `asyncio.Queue` per `bank_id`. Worker uses **drain-on-tick** coalescing: awaits one item, then drains the rest currently queued and POSTs them as a single upstream `RetainRequest` (`{items: [...], async: true}`). Each item carries `content`, `tags`, `document_id`, `update_mode`, `timestamp`, optional `metadata`. `document_id` is derived from a scope key `{bank_id}:{channel_id}` via `_DocScopeStore` (sibling SQLite at `hindsight_doc_scope.db`); a >24h gap between retains in the same scope opens a new document with `update_mode="replace"`, otherwise items append. `_TrustOverrideStore` (sibling SQLite at `hindsight_overrides.db`) holds operator trust flips and an audit trail; recall post-filters hits through it. `ensure_bank` sends `retain_mission` + `reflect_mission` (and optional `enable_observations` / `observations_mission`) — never the deprecated `mission` / `background` aliases.

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
Pure helpers for enforcing per-persona `max_context_tokens`. Two callsites today: `chat_system._prepare_request` and `kobold_adapter` `/chat/completions`. Future home for dynamic LTM-depth modulation, retrieval-score-weighted budget allocation, and a real tokenizer swap.
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

### `src/interfaces/kobold_adapter.py` -- KoboldAdapter
- FastAPI app served from `KoboldAdapter.start()`; mounts the customised kobold-lite at `/portal` (`web_assets/portal.html`) and forwards inference traffic to local KoboldCPP.
- Inference path is OAI-jinja only: the portal runs kobold-lite in kcpp-with-jinja mode, which hijacks `opmode==4` to `/v1/chat/completions`. **Phase D (2026-04-28) split the adapter routes:** `/v1/chat/completions` is now a thin SSE transcoder over `chat_system.stream_response` — the engine rebuilds messages from DB, prunes to budget, archives on retry, and commits the assistant turn; the adapter only formats OAI-shape SSE chunks (and the `event: derpr` frame on `DoneEvent`). Client `data["messages"]` is discarded; `derpr_user_text` (or last-user fallback for non-portal clients) drives the user turn. The native `/api/v1/generate` and `/api/extra/generate/stream` routes remain verbatim passthrough to KoboldCPP because the pre-rendered kobold prompt cannot be safely reconstructed from DB (template-tag drift hazard, see `decisions/2026-04-19-portal-phase2-approach.md`); deprecation queued pending an OAI-feature-parity audit. CORS is wide open (kobold-lite reaches localhost from a different origin).
- Persona endpoints (`/api/v1/model`, `/api/v1/persona/...`, `/v1/models`) read/write through `chat_system.personas` and persist via `save_personas_to_file`.
- Phase 2.1 adds `GET /api/v1/session/{persona}/kobold_export?max_turns=K` — returns a kobold-lite v1 savefile JSON built by `interfaces/kobold_export.build_kobold_savefile`. Pulls `chat_system.memory_manager.get_global_history(persona, K)` (no channel scoping; the portal has no channel concept). `K` defaults to the persona's `get_base_context_length()`.
- Phase 2.2 adds `GET /api/v1/session/{persona}/ltm_block?query=...` — first calls `_build_conversation_history` to get real history + `oldest_id` (needed for the recency filter), then calls `chat_system._retrieve_memory_block(persona, user_identifier="portal", channel="web_ui", server_id=None, conversation_history=history, current_message=query, oldest_interaction_id=oldest_id)` and returns `{"block": str | null}`. The portal wraps `prepare_submit_generation` async to fetch this before each submit, writes the block into kobold-lite's `current_anote` JS variable, and kobold-lite embeds it into the prompt at its normal author's-note position. No server-side prompt mutation occurs. Phase 2.2 also adds `memory_mode` to `GET /api/v1/persona/{name}` and accepts it in `PATCH /api/v1/persona/{name}`, wiring through to `persona.set_memory_mode()`.
- Phase 2.3a–b semantics (sidecar `derpr_user_text` for the user turn, `derpr_retry` for regens, `event: derpr` SSE frame carrying `assistant_id` for chevron hydration, partial-flush on abort) all survive Phase D — they are now implemented inside `chat_system._orchestrate` instead of in the adapter. The adapter handler is purely transcoding: maps `TokenEvent` → OAI `chat.completion.chunk`, `DoneEvent` → `event: derpr` frame + `[DONE]`, `ErrorEvent` → OAI error envelope. `_find_last_user_content(messages)` remains as the non-portal fallback when no sidecar is supplied. `_log_interaction` and `_commit_assistant` adapter helpers are still used by the native passthrough routes (`/api/v1/generate`, `/api/extra/generate/stream`); they are not on the OAI path anymore.
- Phase 3 `max_context_tokens` is exposed on `GET/PATCH /api/v1/persona/{name}`; budget enforcement on the OAI path moved to engine-side as of Phase D (`_prepare_request` calls `truncate_messages_to_budget` against the rebuilt history). The native passthrough routes do not prune.
- Phase 2.4 closes the portal edit/delete round-trip. `PATCH /api/v1/interaction/{id}` (body `{"content": str}`) calls `update_interaction_content`, which now ALSO drops the row's `Message_Embeddings` + `vec_Message_Embeddings` entries so `SqliteConsolidator._embed_unembedded` re-encodes against the new content on the next batch (the existing LEFT-JOIN-on-`Message_Embeddings` query naturally re-picks rows with stale embeddings cleared); `parent_summary_id` is set to NULL so the next summarizer pass re-groups the row. `DELETE /api/v1/interaction/{id}` calls a new thin wrapper `memory_manager.suppress_interaction(interaction_id)` that inserts a single `Suppressed_Interactions` row (idempotent — second call swallows `IntegrityError` and returns False; the endpoint surfaces this via `already_suppressed: bool`). Suppressed rows are filtered everywhere via `_suppression_filter` (history, sliding window, `kobold_export`, `get_unembedded_messages`, `get_unsegmented_embedded_messages`, `get_active_channels`); reply chains are kept intact (no FK cascade, no nulling of `reply_to_id`), and the segmenter handles orphaned `reply_to_id` references cleanly by falling back to the similarity gate. Portal: `edit_chunk_save` is wrapped to PATCH on non-empty edits and DELETE on empty edits; `interaction_ids[]` is spliced in lockstep with `gametext_arr`. **Archive bloat (option B — dedupe-on-archive)**: `swap_interaction_version` now content-hash-checks before inserting the canonical into `Interaction_Edit_History`; if a row with the same `(interaction_id, old_content)` already exists, the insert is skipped (chevron toggled back to a known content). Stale L0 rows are still dropped because the target restore step overwrites canonical content. This bounds chevron-toggle churn to the unique-content count rather than the click count; genuinely unique edits still grow the archive linearly (intended).
- Phase 2.3b adds version navigation backed by `Edit_History_Embeddings`. Schema: `Edit_History_Embeddings(edit_id PK → Interaction_Edit_History ON DELETE CASCADE, embedding BLOB, model_name, created_at)` — L0 embeddings travel with content across swaps; archives do not participate in k-NN (no `vec_` shadow). Endpoints: `GET /api/v1/interaction/{id}/versions` → `{interaction_id, versions: [{edit_id, content, created_at}, ...]}` with archives ordered `(edited_at ASC, edit_id ASC)` and canonical synthesized last (`edit_id=None`); `POST /api/v1/interaction/{id}/select_version/{k}` invokes `memory_manager.swap_interaction_version(interaction_id, k)` which performs archive-current + restore-target atomically in one txn — archives current canonical (new `Interaction_Edit_History` row, `Message_Embeddings` → `Edit_History_Embeddings`, delete `vec_Message_Embeddings`), restores target archive at position `k` (copy `old_content` → `User_Interactions.content`, move `Edit_History_Embeddings(target)` back into `Message_Embeddings` + `vec_Message_Embeddings`, delete target archive row). Returns `{current_content, interaction_id, total_versions}`; raises `IndexError` (400) on out-of-bounds k and `ValueError` (404) on unknown id with no state mutation. The stream relay in `_setup_routes` now detects `[DONE]` in the upstream chunk, commits the assistant buffer first, and emits `event: derpr\ndata: {"assistant_id": N}\n\n` immediately before forwarding `[DONE]` so the portal can hydrate chevron stacks (`retry_prev_text` / `redo_prev_text`) via `/versions`. Error and abort paths skip the frame (no chevron-able assistant turn). Portal-side: a `window.fetch` wrapper tees the `/chat/completions` response body, parses SSE blocks out-of-band to catch `event: derpr`, and on receipt calls `_derprHydrateVersions(assistant_id)` which fetches `/versions`, populates `derpr_version_chain` + `derpr_cursor`, and derives `retry_prev_text` / `redo_prev_text` from the chain. `btn_back` / `btn_redo` are wrapped to POST `/select_version/{k}` (k looked up by content-match against the current server version list) and advance the cursor; original kobold behavior is the fallback when no assistant_id is known. The kobold-side 10-entry cap on `retry_prev_text` has been removed — full regen history lives in the DB. A `PATCH /api/v1/interaction/{id}` route is also exposed (body `{"content": str}`) for manual edits from the portal; it calls `update_interaction_content` directly and does not archive (the prior content is whatever the caller wants replaced).

- **Phase 4.1 — history contract (DP-130, server port-forward).** Replaces the brittle positional `interaction_ids[]` shadowing (Phase 2.4) with a server-authored per-chunk identity contract. Three server-side pieces (portal.html left untouched — that re-keying is DP-131's Lite stopgap; the bespoke UI is DP-132+). Full spec + invariants C1–C5: `project/decisions/2026-06-02-portal-history-contract.md`.
  - **id-frame on EVERY turn.** `DoneEvent` gained `ephemeral_chunk_id: Optional[str]`. The OAI relay in `kobold_engine_adapter` now emits `event: derpr\ndata: {"user_id", "assistant_id", "response_type", "ephemeral_chunk_id"}\n\n` before `[DONE]` on *every* terminal turn — including a parked CONFIRM write (`assistant_id=None`), a tool-only turn, or an empty generation. Previously the frame was emitted only when `assistant_id is not None`, so parked/tool-only/abort turns sent no frame and the portal's positional id array drifted vs the visible story (root cause of "dispatchr is useless in the web UI"; live evidence `ids_len=11 storyLen=13`, `actions=12 ids=13`). The frame shape is **frozen** as of DP-130 — DP-131/DP-132 consume it.
  - **`ephemeral_chunk_id`.** A stable handle for the rendered-but-unpersisted confirmation chunk on a parked turn. It is the parked `PendingConfirmation.token` (uuid4 hex; reconciles with the DP-127-engine confirm-modal `token` — same correlation id the interactive surface sends back on approve/deny). `PendingConfirmation` also gained `confirmation_text` (the parked chunk's rendered text) so a fresh load can render the awaiting-approval text. Non-parked turns carry `ephemeral_chunk_id=None` and are addressed by `assistant_id`.
  - **`GET /api/v1/session/{persona}/transcript`** — the authoritative projection: `{"chunks": [...]}`, each chunk `{interaction_id|null, role, content, ephemeral, reasoning, tool_context, has_versions}` (invariant **C1**: exactly one `interaction_id` OR `ephemeral=true`). Same persona/global-history scoping as `kobold_export`; suppressed rows already filtered upstream (**C5**). `has_versions` comes from a single batch query `memory_manager.get_ids_with_versions(ids)` (no N+1). A live parked confirmation for `("portal", persona)` is appended as a trailing ephemeral chunk carrying its `ephemeral_chunk_id`. This is the re-sync source for the Lite stopgap and the render source for the bespoke UI — consumers address chunks by identity, never by position, so the array can no longer drift.
  - **`build_kobold_savefile` C2 fix** — see kobold_export section below.
  - **What converges now vs DP-131 (DP-130 is server-only):** the gametext-aligned `kobold_export` (C2) makes the **session restore/reload** path converge with the *unchanged* portal today (`ids_len==storyLen`, verified live). What still needs DP-131 is the **in-session per-turn** path: portal.html line ~33349 guards id pushes on `if (payload.assistant_id)`, so a parked turn mid-session won't advance the array until the next reload — DP-131 drops that guard / re-keys off `/transcript`. The contract test `test_parked_write_emits_id_frame_with_ephemeral_chunk_id` proves the on-wire parked frame deterministically through the real kernel.

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
- `save_utils.py` -- Persona JSON file I/O: load/save personas and models to disk. Handles default + system persona merging on startup

### `src/main.py` -- Startup Sequence
1. MemoryManager (SQLite) + schema migration
2. TextEngine (LLM API router)
3. ZammadClient (optional, fails gracefully if no credentials)
4. EmbeddingService (GeminiEmbeddingProvider) — shared by ChatSystem and SqliteConsolidator
5. ChatSystem (DI hub, injected with memory + engine + embedding_service)
6. Register ZammadIntegration service (if Zammad available)
7. AgentManager + register agent classes (`SqliteConsolidator` only when `SEMANTIC_BACKEND=="sqlite"`; `ZammadBot` + `DispatchAgent` + `ReminderAgent` only if Zammad available)
8. Register AgentServiceIntegration service (agent tools)
9. AppManager + NotificationRouter (Discord/Zammad notifiers)
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
