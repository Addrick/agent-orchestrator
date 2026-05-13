---
name: Web UI roadmap (kobold-lite portal)
description: Phased plan for building DERPR's memory-centric web UI on top of kobold-lite; append-only phases + backlog
type: project
---

## Vision

Build a comprehensive web UI for DERPR by extending kobold-lite, focused on **local-only** inference (KoboldCPP). Core value: surface DERPR's memory system (sliding window + semantic LTM retrieval) inside a familiar chat UI. Long-term goal: per-response memory provenance ("show thoughts" for memories).

## Guiding principles

- **Kobold owns templating.** All chat-template / instruct-tag rendering happens in kobold-lite. DERPR never rewraps kobold's output. This is the lesson from Phase 1 (see `decisions/2026-04-19-kobold-portal-passthrough.md`).
- **DERPR owns retrieval and persistence.** Message history, LTM segments, persona settings — DERPR is source of truth.
- **Additive, not invasive.** DERPR contributes to kobold's existing fields (`memory`, `authornote`, session savefile) rather than monkey-patching its internals. Kobold updates stay compatible.
- **Cache-aware placement.** KoboldCPP uses llama.cpp prefix cache only — no mid-prompt slot reservation. Any content that changes per-turn belongs near the end of the prompt (author's note region) to minimize KV recompute.
- **Local-first.** Non-local providers are backlog, not near-term.

## Phase 1 — DONE

Verbatim passthrough adapter: kobold-lite → `kobold_adapter.py` → KoboldCPP. Persona sampling pushed to UI sliders on switch; outgoing request forwarded unmodified. SSE relayed unchanged. Abort forwarded to `/api/extra/abort`.

See decision doc: `decisions/2026-04-19-kobold-portal-passthrough.md`.

## Phase 2 — DERPR Database history source + optional LTM injection

**Goal:** make the DERPR DB an optional history source for kobold-lite sessions, and optionally layer LTM retrieval on top.

Split into **two sub-phases, one session each** (easier testing + cleaner review surface):

### Phase 2.1 — DB as history source (session A) — SHIPPED 2026-04-20 (commit 4977d1b)

User toggles "DERPR Database" → kobold-lite session populated from DERPR message_history. From that point, passthrough logic is unchanged.

**Scope:**

- `GET /api/v1/session/{persona}/kobold_export?max_turns=K` in `kobold_adapter.py`
  - Builds a kobold-lite savefile JSON from `message_history` rows for that persona
  - `K` defaults to persona's configured history limit (reuse existing persona field — pick the one that drives sliding-window size; do not introduce a new setting)
  - Map DERPR roles → kobold instruct-mode turns (user/assistant). Skip tool-call / tool-result rows; log count of skipped rows.
- Frontend toggle: replace the current disabled "History Override" checkbox with a two-state toggle:
  - Label left: "Kobold Native" | Label right: "DERPR Database"
  - Default: Kobold Native (current behavior)
  - On switch → DERPR: fetch export, call kobold-lite's `load_file` (or equivalent public JSON ingest path) with the blob
  - On switch → Kobold Native: confirm-then-clear session
- Sub-checkbox "LTM Generation" rendered under the DERPR side of the toggle, but **disabled and unchecked** in 2.1. Wired in 2.2.

**Tests:**
- Unit: exporter produces schema-valid kobold savefile for a seeded DERPR history
- Integration: round-trip — seed DB → export → load kobold session JSON → compare rendered turns
- Regression: Kobold Native path identical to Phase 1 behavior

**Docs:**
- `docs/user_guide.md`: toggle semantics, what each side does, what gets loaded
- `memory/codebase/architecture.md`: note DB-as-source optional ingestion path

**Definition of done:** User can flip toggle to "DERPR Database," see prior persona conversation materialized in kobold-lite, continue chatting via passthrough. LTM checkbox visible but inert.

### Phase 2.2 — Opt-in LTM injection via author's note + memory-mode dropdown (session B) — SHIPPED 2026-04-20/21

User enables "LTM Generation" checkbox (only available when toggle = DERPR Database). Before each submit, the portal fetches the LTM block from DERPR and sets it as kobold-lite's author's note; kobold then embeds it into the prompt at its normal author's-note position. Adjacent UI addition: a per-persona `memory_mode` dropdown in the Inference Matrix popup.

**As-shipped architecture** (differs from pre-session spec — kobold-lite renders `current_anote` into the prompt client-side; no `authornote` field ever reaches the adapter on the wire):

| # | Item | As shipped |
|---|------|------------|
| 1 | LTM block endpoint | `GET /api/v1/session/{persona}/ltm_block?query=...` → `{"block": str\|null}` |
| 2 | Retrieval call | `_build_conversation_history` first (correct `oldest_id` for recency filter), then `_retrieve_memory_block` with real history context, not empty list |
| 3 | Client injection | `prepare_submit_generation` wrapped async; fetches block, sets `current_anote`, kobold places it near end of prompt |
| 4 | Authornote backup | First LTM submit backs up `current_anote` to localStorage; restored on LTM off |
| 5 | UI indicator | `anotetext` greyed/disabled; "Managed by DERPR LTM" banner |
| 6 | Refresh cadence | Per-turn — every submit with LTM=on refetches |
| 7 | Memory Scope dropdown | CHANNEL_ISOLATED / SERVER_WIDE / PERSONAL / GLOBAL; persists via `PATCH /api/v1/persona/{name}` |
| 8 | `memory_mode` in GET persona | Added to `GET /api/v1/persona/{name}` response |

**Known limitation (addressed in 2.3):** Portal-originated turns are not yet logged to `message_history`, so LTM retrieval in 2.2 only sees Discord / email / Zammad content. Portal conversations are ephemeral until 2.3 lands.

### Phase 2.3 — Portal message logging + retry/edit history

Split into three sub-phases after an in-session scope review:

- **2.3a — user/assistant logging + retry archive on OAI path (SHIPPED)**
- **2.3b — version selection endpoint + chevron hook (SHIPPED 2026-04-22)**
- **2.3c — delete `chat_system.stream_response` (SHIPPED 2026-04-22)**

### Phase 2.3a — OAI-only logging + retry archive — SHIPPED

**As-shipped architecture** (differs from the pre-session spec — portal runs in KCPP-with-jinja mode, so only `/chat/completions` is the canonical path; kobold-native routes were removed rather than dual-logged):

| # | Item | As shipped |
|---|------|------------|
| 1 | Dead-route removal | `/api/v1/generate` and `/api/extra/generate/stream` deleted from `kobold_adapter.py`. Jinja mode routes everything through `/chat/completions`. |
| 2 | User-turn detection | **Sidecar `derpr_user_text` is canonical.** Live testing revealed jinja-hijack's `repack_instruct_history` often produces ZERO user-role entries (`role_tail=['assistant','assistant']`), breaking pure array scanning. Portal's `prepare_submit_generation` override captures `input_text.value` into `window.derpr_last_user_input` *before* kobold clears the textbox; `_derprStampAndConsumeRetry` stamps it as `derpr_user_text` on the outgoing body (not on retry). Adapter prefers sidecar, falls back to `_find_last_user_content(messages)` for non-portal clients. `_strip_envelope` drops the sidecar before upstream forward. |
| 3 | Synchronous logging | `_log_interaction` returns the new `interaction_id` (thread wrapper dropped; SQLite insert is fast under the `MemoryManager` lock). |
| 4 | reply_to_id threading | Stream relay captures `user_interaction_id` in closure scope; `_commit_assistant` passes it as `reply_to_id` when inserting the assistant row. |
| 5 | Abort flush | `CancelledError` branch calls `_commit_assistant` with the partial buffer before forwarding `/api/extra/abort`. |
| 6 | Retry contract | Body field `derpr_retry: true` (boolean — portal does not need to know the interaction id). Portal wraps `btn_retry` to flip `window.derpr_pending_retry`, and wraps `oai_api_sync_req` / `oai_api_stream_sse` to stamp + consume the flag into the outgoing payload. |
| 7 | Retry DB path | `memory_manager.handle_portal_retry(persona, user_identifier, channel)` finds the latest assistant row for the portal session, archives its content into `Interaction_Edit_History`, deletes its `Message_Embeddings` + `vec_Message_Embeddings` rows, and returns the id. `memory_manager.update_interaction_content(id, new)` overwrites the canonical row in place. User row is not re-logged on retry. |
| 8 | Envelope stripping | `_strip_envelope` drops `derpr_retry` before forwarding. |

**Decision notes captured during session:**
- Simplified plan's `derpr_retry_of: <id>` to a boolean `derpr_retry: true`. Portal doesn't know canonical interaction ids without a handshake; server-side "latest assistant row for this portal session" is correct for single-user portal and avoids the round-trip.
- First-turn retry with no prior assistant row is a no-op (logger warning); adapter falls back to normal new-turn logging.
- **Sidecar over messages-array scanning** (post-implementation pivot): First implementation scanned `messages[]` for the last user role. Live test with a fresh "test" submit logged only the assistant reply — DB showed `role_tail=['assistant','assistant']`, user rows never persisted. `repack_instruct_history` in jinja-hijack mode does not reliably preserve user roles in the rebuilt array. Portal has raw input available at submit time, so stamping `derpr_user_text` at the JS boundary is both simpler and decoupled from kobold's repack shape.

### Phase 2.3b — Version selection + chevron hook (pending)

**Goal:** Let the user navigate between regen attempts in the DERPR DB via kobold-lite's client-side version chevrons (`retry_prev_text` / `redo_prev_text` stacks at portal.html:4249). DB is canonical; kobold in-memory stacks become a DB-backed view.

**Locked decisions (2026-04-22):**

| # | Decision | Notes |
|---|----------|-------|
| 1 | Assistant id delivery | SSE final frame `event: derpr\ndata: {"assistant_id": N}\n\n` emitted immediately before `data: [DONE]\n\n`. Emit from `_commit_assistant` return path in `kobold_adapter.py`. |
| 2 | Version indexing | `k=0` = oldest archive, increments by archival time. Canonical = newest, always lives in `User_Interactions.content`. After a swap, indices renumber (portal refetches). |
| 3 | kcpp stacks → DB | Replace `retry_prev_text` + `redo_prev_text` localStorage-backed stacks with server-fetched lists. Hydrate on assistant_id event. Drop kobold's 10-cap (portal.html:27346) — DB has no client-side cap. |
| 4 | Embedding on swap | **L0 embeddings travel with content.** New table `Edit_History_Embeddings(edit_id PK, embedding BLOB, model_name, created_at)` FK → `Interaction_Edit_History(edit_id) ON DELETE CASCADE`. No vec shadow (archives don't participate in retrieval). Swap = atomic move between `Message_Embeddings`+`vec_Message_Embeddings` and `Edit_History_Embeddings`. **L1+ summaries**: match 2.3a (no invalidation in swap path). Future work to refine. |
| 5 | First-turn chevrons | Inherit kobold default: chevrons only render when a regen has occurred. No extra hide logic needed. |

**Derived scope:**

- Schema migration — add `Edit_History_Embeddings` table in `memory_manager.py` `create_schema()`. Idempotent (CREATE IF NOT EXISTS). Migration test required per CLAUDE.md.
- `memory_manager.handle_portal_retry` — update to **move** `Message_Embeddings` row into `Edit_History_Embeddings` (keyed by the new `edit_id`) instead of deleting it. `vec_Message_Embeddings` still deleted (no vec shadow for archives).
- `memory_manager.list_interaction_versions(interaction_id)` — returns `[{edit_id, created_at, content}, ...]` ordered by `edited_at ASC, edit_id ASC` (plus current canonical synthesized as `{edit_id: None, created_at: <canonical>, content: <canonical>}` appended at end).
- `memory_manager.swap_interaction_version(interaction_id, k)` — one transaction:
  1. Archive current canonical: insert new `Interaction_Edit_History` row, move embedding (if any) `Message_Embeddings` → `Edit_History_Embeddings(new_edit_id)`, delete from `vec_Message_Embeddings`.
  2. Locate target archive at position k (1-indexed from oldest after archival step, or 0-indexed over pre-swap archives — spec: index refers to the version the user clicked on, computed pre-swap).
  3. Restore target: copy `old_content` → `User_Interactions.content`, move `Edit_History_Embeddings(target_edit_id)` → `Message_Embeddings` + `vec_Message_Embeddings` if present, delete target `Interaction_Edit_History` row.
  4. Return `{"current_content": str, "interaction_id": int, "total_versions": int}`.
- Endpoints in `kobold_adapter.py`:
  - `GET /api/v1/interaction/{id}/versions` — returns version list (canonical last).
  - `POST /api/v1/interaction/{id}/select_version/{k}` — performs swap, returns new canonical + refreshed version list.
  - SSE `event: derpr` frame with `{"assistant_id": N}` before `[DONE]` on `/chat/completions` stream path.
- Portal hooks in `portal.html`:
  - SSE reader: intercept `event: derpr` frames, capture `assistant_id`, attach to last `gametext_arr` entry.
  - On assistant_id received or on page restore: `GET /versions` to hydrate `retry_prev_text` + `redo_prev_text` from DB.
  - Wrap chevron nav handlers (search for sites that mutate `retry_prev_text`/`redo_prev_text`): call `POST /select_version/{k}`, sync stacks from response.
  - Remove localStorage cap-10 behavior (27346) — not needed once DB-backed.

**Tests:**
- Unit (`tests/memory/test_memory_manager.py`): migration adds `Edit_History_Embeddings`; idempotent on second run.
- Unit: `swap_interaction_version` round-trip — retry → retry → select k=0 restores original text and original embedding blob; `Message_Embeddings` row replaced atomically; `vec_Message_Embeddings` re-populated.
- Unit: out-of-bounds `k` raises `IndexError` / endpoint returns 400; no state mutation.
- Unit: `handle_portal_retry` now preserves embedding in `Edit_History_Embeddings` rather than deleting.
- Unit: `list_interaction_versions` ordering + canonical-at-end invariant.
- Integration (`tests/interfaces/test_kobold_adapter.py`): SSE stream emits `event: derpr` with correct id before `[DONE]`; endpoint round-trip retry → retry → select_version(0) leaves canonical = original.

**Definition of done:** User can click the `<` / `>` chevrons on a portal message with multiple regens; each prior version restores into the DB canonical row (with its original L0 embedding) without losing any attempt. Total version count unchanged across swaps.

**Progress (2026-04-22, SHIPPED):**

Backend + portal + tests + docs complete. 570/570 unit+integration pass.

Done:
- Schema: `Edit_History_Embeddings(edit_id PK, embedding BLOB, model_name, created_at)` added to `create_schema()` in `src/memory/memory_manager.py` (no vec shadow — archives don't participate in k-NN).
- `handle_portal_retry` updated: moves L0 embedding from `Message_Embeddings` into `Edit_History_Embeddings` keyed by the new edit_id. `vec_Message_Embeddings` still dropped for the interaction.
- `list_interaction_versions(interaction_id)` — archives ordered `(edited_at ASC, edit_id ASC)`, canonical synthesized last with `edit_id=None`.
- `swap_interaction_version(interaction_id, k)` — atomic archive-current + restore-target in one txn. Deletes target archive row after restore (cascades any remaining `Edit_History_Embeddings`). Raises `IndexError` out-of-bounds, `ValueError` on unknown id, no state mutation on raise. Returns `{current_content, interaction_id, total_versions}`.
- Migration tests (3) + behavior tests (7) covering: retry embedding preservation, retry with no prior embedding, version list ordering + canonical-last, round-trip embedding preservation on swap, out-of-bounds + unknown-id raises, total-versions stable across multi-swap.
- mypy: no new errors introduced (one pre-existing `no-any-return` in `handle_portal_retry` unchanged).

As-shipped (this session):
- Endpoints added to `kobold_adapter.py`: `GET /api/v1/interaction/{id}/versions` and `POST /api/v1/interaction/{id}/select_version/{k}` (plus `PATCH /api/v1/interaction/{id}` for manual content edits). `_commit_assistant` returns `interaction_id`. Stream relay detects `[DONE]` in the upstream chunk, commits first, and injects `event: derpr\ndata: {"assistant_id": N}\n\n` before forwarding `[DONE]`. Error + abort paths skip the frame.
- Integration tests: SSE frame precedence, empty stream emits no frame, 400 on out-of-bounds `k`, 404 on unknown id, single-version canonical, and retry → retry → select_version(0) round-trip via endpoints.
- portal.html: `window.fetch` wrapped; `/chat/completions` response body is teed so `event: derpr` is parsed out-of-band without disturbing kobold's reader. `_derprHydrateVersions(assistant_id)` fetches `/versions`, populates `derpr_version_chain` + `derpr_cursor`, and derives `retry_prev_text` / `redo_prev_text`. `btn_back` / `btn_redo` wrapped: POST `/select_version/{k}` (k looked up by content match against current server versions), cursor advanced, local stacks refreshed from the chain. Original kobold behavior is the fallback when no assistant_id is known. 10-entry cap at old line 27346 removed.
- Docs: `docs/user_guide.md` Version-chevrons paragraph added; `memory/codebase/architecture.md` lists the new tables + documents the 2.3b flow (SSE derpr event, swap txn, embedding movement).
- 2.3c: `chat_system.stream_response` deleted; `AsyncIterator` import removed. No callers existed. `StreamEngine` wiring left in place (still passed in via `main.py`) — candidate for a future minimal cleanup commit if desired.

### Phase 2.3c — Delete `chat_system.stream_response` (pending)

Standalone cleanup commit — `stream_response` has no external callers (only self-references in `chat_system.py`, confirmed 2026-04-21). It owns prompt construction (incompatible with kobold-owns-templating). Drop the method + the dead-code note in `architecture.md`.

### Phase 2.4 — Edit/delete round-trip — SHIPPED 2026-04-26 (option B)

**Goal:** Round-trip portal message edits and deletions back to the DERPR DB with embedding + summary integrity preserved, and bound archive growth under chevron use.

**As-shipped (undocumented until now):**

| # | Item | As shipped |
|---|------|------------|
| 1 | `interaction_ids[]` parallel array | Tracked alongside `gametext_arr` in `portal.html`, persisted into `storyobj.interaction_ids`, restored on session load (line ~32315) |
| 2 | SSE id delivery | `event: derpr` frame carries `{assistant_id, user_id}` — extends 2.3b's assistant-only payload |
| 3 | Edit endpoint | `PATCH /api/v1/interaction/{id}` → `memory_manager.update_interaction_content` overwrites `User_Interactions.content` and clears `parent_summary_id` (re-queues for next L1 summary batch) |
| 4 | Portal edit hook | `edit_chunk_save` wrapped (`portal.html:32753`) — non-empty edit fires PATCH after stripping `{{[INPUT]}}` / `{{[OUTPUT]}}` placeholders |
| 5 | Empty-edit client-side | Splices `interaction_ids[]` to track `gametext_arr` shrinkage; **no server call** (`// (Optional: call DELETE API here if desired)` placeholder at line ~32764) |

**Locked decisions (2026-04-25):**

| # | Decision | Notes |
|---|----------|-------|
| 1 | L0 re-embed on edit | PATCH path enqueues a fresh embedding via the existing embedding pipeline. Old `Message_Embeddings` + `vec_Message_Embeddings` rows replaced. Async batch acceptable — short staleness window tolerated. |
| 2 | L1+ summary invalidation | Already half-shipped — `update_interaction_content` clears `parent_summary_id`. Edited interactions get re-summarized in the next batch pass, grouped with similar messages by the existing summarizer. No PATCH-path work. |
| 3 | Delete = soft-suppress, reuse `Suppressed_Interactions` | Existing table/infra (`suppress_message_by_platform_id`, `_SUPPRESSION_SUBQUERY`) already filters suppressed rows from history reads. Add `DELETE /api/v1/interaction/{id}` calling a new thin `suppress_interaction(id)` wrapper. Portal empty-edit path replaces the `// (Optional)` placeholder with the DELETE call. |
| 4 | `reply_to_id` integrity on delete | No FK cascade, no nulling. Leave the chain intact. Segmenter updated to allow assistant-only segmentation when its paired user row is suppressed/missing. |

**Open design question — archive bloat under chevron toggling:**

Today `swap_interaction_version` archives the prior canonical on **every** swap, even when the user toggles back-and-forth between the same two contents. Chevron-heavy sessions grow `Interaction_Edit_History` linearly in clicks, not in unique versions.

Options:
- **A. Mutable canonical pointer.** Move all versions into one table, mark canonical via flag (`is_canonical` on `Interaction_Edit_History`) or pointer column on `User_Interactions`. Swap = pointer move, no row creation. Cleanest long-term, largest delta to 2.3b's shipped txn flow.
- **B. Dedupe on archive.** Content-hash check at archive step — if archive row with matching hash for this `interaction_id` exists, swap pointers without inserting. Minimal delta to 2.3b.
- **C. LRU cap per interaction.** Hard cap N archives, evict oldest. Bounds worst case but doesn't solve same-content churn.
- **D. B + C.** Dedupe and cap.

Lean B for minimal disruption. Decision pending.

**Scope (assuming option B):**

- `src/memory/memory_manager.py`:
  - `update_interaction_content` — also enqueue L0 re-embed (delete old `Message_Embeddings` + `vec_Message_Embeddings` rows + queue new embedding via existing pipeline). Idempotent on repeated edits.
  - `suppress_interaction(interaction_id)` — thin wrapper inserting one `Suppressed_Interactions` row. Idempotent (existing `IntegrityError` swallow pattern).
  - `swap_interaction_version` — add content-hash dedup at the archive step. If matching archive row exists, skip insert and re-point against it.
- `src/interfaces/kobold_adapter.py`:
  - `DELETE /api/v1/interaction/{id}` → `suppress_interaction`. Returns `{result: "success", interaction_id: id}`.
- `src/interfaces/web_assets/portal.html`:
  - `edit_chunk_save` empty-edit branch — replace `// (Optional)` comment with `fetch('/api/v1/interaction/${id}', {method:'DELETE'})` after the local splice.
- Segmenter (locate during impl — likely `MemoryAgent` / consolidation module): tolerate `reply_to_id` pointing at a suppressed/absent user row; segment assistant content alone.

**Tests:**

- `tests/memory/test_memory_manager.py`:
  - `update_interaction_content` replaces L0 embedding (old row gone, new row queued/inserted).
  - `suppress_interaction` inserts row; second call idempotent.
  - Suppressed interaction excluded from `get_personal_history`, sliding-window builder, and LTM retrieval.
  - Archive dedupe: swap with content matching existing archive does not grow `Interaction_Edit_History`; total versions stable across N back-and-forth swaps.
- `tests/interfaces/test_kobold_adapter.py`:
  - `PATCH /interaction/{id}` updates content + clears `parent_summary_id`.
  - `DELETE /interaction/{id}` suppresses; second DELETE 200/no-op.
  - Suppressed interaction absent from subsequent `kobold_export` and from outbound `/chat/completions` history.
- Segmenter test: assistant row whose `reply_to_id` is suppressed segments solo without raising.

**Docs:**
- `docs/user_guide.md` — portal section: edit propagates to DB, empty-edit deletes (suppresses); deleted messages stay out of future LTM and exported sessions.
- `memory/codebase/architecture.md` — document `Suppressed_Interactions` reuse for portal-originated deletes, the L0 re-embed hook on PATCH, and the dedup-on-archive rule for chevron swaps.

**Definition of done:** User edits a portal message → DB row updated, L0 embedding refreshed, next summary batch reprocesses. User empty-edits → server marks suppressed; subsequent `kobold_export`, sliding window, and LTM all skip it. Repeated chevron toggles between the same two versions do not grow `Interaction_Edit_History` beyond the unique-content count.

**As-shipped (2026-04-26, option B locked):**

| # | Item | As shipped |
|---|------|------------|
| 1 | Archive-bloat decision | **Option B (dedupe-on-archive).** Smallest delta to 2.3b; chosen over A (pointer rewrite) for maintainability — no schema change, no read-path risk. |
| 2 | `update_interaction_content` | Now also `DELETE FROM Message_Embeddings/vec_*` so the existing LEFT-JOIN query in `MemoryAgent._embed_unembedded` re-picks the row on next batch. `parent_summary_id` clear was already in place from 2.3a. `rowcount` captured before the deletes (was a return-value bug otherwise). |
| 3 | `suppress_interaction(id)` | Thin wrapper. Single INSERT into `Suppressed_Interactions`, swallows `IntegrityError`, returns `True` on insert / `False` on dup. |
| 4 | `swap_interaction_version` dedupe | Pre-INSERT `SELECT 1 FROM Interaction_Edit_History WHERE interaction_id=? AND old_content=? LIMIT 1`. Hit → skip insert (no archival), still drop stale L0 rows. Miss → existing flow. Chevron back/forth bounded to unique-content count. |
| 5 | `DELETE /api/v1/interaction/{id}` | Calls `suppress_interaction`. Returns `{result, interaction_id, already_suppressed: bool}`. Idempotent. |
| 6 | Portal empty-edit path | `edit_chunk_save` empty branch fires `fetch DELETE` after the local `interaction_ids[]` splice. The `// (Optional)` placeholder removed. |
| 7 | Segmenter | **No code change.** `_SUPPRESSION_SUBQUERY` was already on `get_unsegmented_embedded_messages` + `get_unembedded_messages`. Orphan `reply_to_id` (target filtered out) just makes `reply_link` evaluate False; falls back to similarity gate. Regression test added in `tests/agents/test_memory_agent.py::TestSegmentation::test_orphan_reply_to_id_does_not_crash`. |
| 8 | Tests | 6 new in `tests/memory/test_memory_manager.py` (suppress idempotent, history exclusion, PATCH clears L0, dedupe stable archive count under 5× toggle, `total_versions == 2` across N swaps). 4 new in `tests/interfaces/test_kobold_adapter.py` (DELETE success+suppression, DELETE idempotent, PATCH clears L0, suppressed row absent from kobold_export). 1 new segmenter regression. **605/605 unit+integration pass; mypy delta 0.** |
| 9 | Docs | `docs/user_guide.md` Phase 2.4 paragraph on edit/delete behavior. `memory/codebase/architecture.md` Phase 2.4 entry covering PATCH re-embed, DELETE → `Suppressed_Interactions`, segmenter tolerance, dedupe-on-archive. |

## Phase 3 — Persona-driven context budget enforcement — SHIPPED 2026-04-25

**Goal:** add a per-persona token cap (`max_context_tokens`) enforced inside DERPR. Truncation is a global feature applied across both pipelines (chat_system for non-portal, kobold_adapter for portal). LTM authornote takes priority over sliding-window history; on overflow, drop oldest individual messages until under budget.

**As-shipped:**

- `src/memory/context_budget.py` (new): `estimate_tokens` (`(len+3)//4`) + `truncate_messages_to_budget` (drop-oldest non-system, preserve all system + last user, returns `(pruned, dropped)`).
- `src/persona.py`: `max_context_tokens` kwarg, `get_max_context_tokens()` / `set_max_context_tokens()`. Default = `global_config.DEFAULT_MAX_CONTEXT_TOKENS = 131072`. Setter clamps to ≥100, falls back to default on invalid input.
- `src/utils/save_utils.py`: round-trip via `to_dict` and both `load_personas_from_file` / `load_system_personas_from_file`. Missing field → default.
- `src/chat_system.py` `_prepare_request`: prune called after history+memory+user-append. Budget = `max_context_tokens - response_token_limit`. Logs dropped count.
- `src/interfaces/kobold_adapter.py`: GET/PATCH persona include the field. `/chat/completions` prunes outbound `body["messages"]` to budget after `_strip_envelope`. Skip when persona resolution fails.
- `src/message_handler.py`: `what max_context_tokens` + `set max_context_tokens <int>` commands wired through `_what_max_context_tokens` / `_set_max_context_tokens`.
- `src/interfaces/web_assets/portal.html`: new `maxContextTokensInput` field in the Inference Matrix popup. `onPersonaChange` reads the value, `applyPersonaToUI` pushes into `localsettings.max_context_length` + `max_context_length` / `_slide` DOM (replacing the Phase 2.1 deliberate-block comment), `savePersonaFromUI` includes it in PATCH.
- Tests: `tests/memory/test_context_budget.py` (12), `tests/test_persona.py` (5 new), `tests/utils/test_save_utils.py` (2 new), `tests/test_chat_system.py` (1 new prune test), `tests/interfaces/test_kobold_adapter.py` (3 new — GET/PATCH/prune behavior). 594/594 unit+integration pass; mypy delta 0.
- Docs: `docs/user_guide.md` lists `max_context_tokens` in the what/set tables. `memory/codebase/architecture.md` documents the new module + both enforcement sites.

**Locked decisions (2026-04-24):**

| # | Decision | Notes |
|---|----------|-------|
| 1 | Field name | `max_context_tokens` on `Persona`. Distinct from existing `history_messages` (turn count) — semantically a token cap, not a row count. |
| 2 | Default | `DEFAULT_MAX_CONTEXT_TOKENS = 131072` in `global_config.py`. Literal const, not a kcpp fetch — keep startup decoupled. Old persona configs without the field default to this on load. |
| 3 | Token estimator | Char/4 — `estimate_tokens(s) = (len(s) + 3) // 4`. No tokenizer roundtrip. Future work to swap for real tokenizer if precision matters. |
| 4 | Budget semantic | **Total context (kobold-style).** `max_context_tokens` = full ctx including response. Effective prompt prune budget = `max_context_tokens - response_token_limit`. Required so the value stays consistent when synced to kobold-lite's `localsettings.max_context_length` slider. |
| 5 | Truncation policy | Drop oldest non-system message individually (no pair-dropping). Preserve all system messages and the latest user message. LTM authornote is part of the rendered prompt by the time prune runs — kept implicitly because it sits in the latest user content / system role, never evicted. |
| 6 | Shared module | New `src/memory/context_budget.py`. Future home for dynamic LTM-depth modulation, retrieval-score-weighted budget allocation, real tokenizer swap. Two callsites today; logic is pure / interface-agnostic. |
| 7 | Persona → kcpp sync | On persona load in portal, push `max_context_tokens` into `localsettings.max_context_length` + slider DOM. Replaces the deliberate-block from Phase 2.1 (portal.html:32368-32370 comment forbidding the push). That comment was correct when the only persona context field was the turn count; Phase 3 introduces a token-valued field that *should* drive the slider. |

**Scope:**

- `src/memory/context_budget.py` (new):
  - `estimate_tokens(s: str) -> int` — `(len(s) + 3) // 4`. Empty → 0.
  - `truncate_messages_to_budget(messages: list[dict], max_tokens: int) -> tuple[list[dict], int]` — drops oldest non-system messages until total ≤ budget; preserves all system messages and the last user message; returns `(pruned_messages, dropped_count)`. No-op when budget is None or sum already under.
- `src/persona.py`:
  - New `__init__` kwarg `max_context_tokens: Optional[int]` defaulting to `DEFAULT_MAX_CONTEXT_TOKENS`.
  - `get_max_context_tokens()` / `set_max_context_tokens(value)`.
- `src/global_config.py`: `DEFAULT_MAX_CONTEXT_TOKENS = 131072`.
- `src/utils/save_utils.py`: include `max_context_tokens` in persona dict round-trip; tolerate missing field on load (default applied).
- `src/chat_system.py` `_build_conversation_history`:
  - After turn-count slice, compute `prompt_budget = persona.get_max_context_tokens() - persona.get_response_token_limit()`.
  - Build the message list as today, then call `truncate_messages_to_budget(messages, prompt_budget)`.
  - Log `dropped_count` if non-zero (token-prune diagnostic).
- `src/interfaces/kobold_adapter.py`:
  - `GET /api/v1/persona/{name}` response: include `max_context_tokens`.
  - `PATCH /api/v1/persona/{name}`: accept `max_context_tokens`.
  - `/chat/completions` outbound: after `_strip_envelope`, before forward, call `truncate_messages_to_budget(body["messages"], prompt_budget)` where `prompt_budget = persona.max_context_tokens - persona.response_token_limit`. Log dropped count. Skip when persona resolution fails (preserve current passthrough behavior).
- `src/interfaces/web_assets/portal.html`:
  - `onPersonaChange`: read `p.max_context_tokens` into a new popup input (alongside the existing temp/top_p/etc inputs in the Inference Matrix).
  - `applyPersonaToUI`: push the value into `localsettings.max_context_length` + sync `max_context_length` / `max_context_length_slide` DOM via `setBoth`. Remove the 32368-32370 "do not push" comment block — replace with a one-line comment noting Phase 3 ownership.
  - `savePersonaFromUI`: include `max_context_tokens: parseInt(...)` in PATCH payload.
  - New input field in the persona-config popup HTML.

**Tests:**

- `tests/memory/test_context_budget.py` (new):
  - `estimate_tokens` boundaries: empty string → 0; len-4 → 1; len-5 → 2; len-7 → 2; len-8 → 2.
  - `truncate_messages_to_budget`: under-budget no-op (returns input + 0); over-budget drops oldest non-system; multiple system messages all preserved; last user message preserved even when it alone exceeds budget (no-op + logged); returns correct dropped count.
- `tests/test_persona.py`: round-trip `max_context_tokens` through ser/de; missing field on load uses default; PATCH-via-AgentManager analogue if relevant.
- `tests/test_chat_system.py` (or analogue): seed history that exceeds token budget after turn-count slice; verify `_build_conversation_history` returns a message list under budget; log captures dropped count.
- `tests/interfaces/test_kobold_adapter.py`: synthetic `/chat/completions` body with messages totaling above budget → adapter forwards a pruned body; system + last user preserved; pruned count appears in logs.
- Integration: persona load in portal pushes value into `localsettings.max_context_length` (tested via DOM observation or by asserting the GET response shape consumed by the JS path).

**Docs:**
- `docs/user_guide.md`: describe `max_context_tokens` as a per-persona setting, what it caps (full ctx including response), how it interacts with `history_messages` (which still controls turn count fetched from DB).
- `memory/codebase/architecture.md`: document the shared `truncate_messages_to_budget` callpath and both enforcement sites.

**Definition of done:** Setting a low `max_context_tokens` on a persona causes (a) the kobold-lite slider to reflect it on persona load, (b) outbound `/chat/completions` payloads from the portal to be pruned to fit, (c) non-portal interfaces (Discord/Gmail/Zammad) prune the message list before submitting to the model, (d) LTM authornote is never evicted by the prune.

**Future work (noted, not built):**
- Dynamic LTM-depth modulation: allocate the full budget across LTM block + history + system prompt based on retrieval scores rather than a fixed LTM-priority + drop-oldest policy. Lives naturally in `src/memory/context_budget.py`.
- Real tokenizer: swap char/4 for `/api/extra/tokencount` calls or a local tokenizer when char/4 drift becomes a problem (likely first observable on long CJK content).
- Reserve-margin tuning: today the response reservation is exactly `response_token_limit`; could become configurable margin if useful.

## Scope carve-outs for Phase 2

- **Tool-call / tool-result rendering.** The portal has no representation for tool turns. 2.1's exporter skips system rows, empty rows, and tool-call-only assistant rows (tool_context present + content empty). Assistant rows that carry both `tool_context` and visible content are emitted as normal assistant text. Proper tool-turn surfacing (approval gates, tool-result display) is deferred — see Tier 2 "Tool execution UI" in the backlog.
- **Non-instruct opmodes.** Exporter emits instruct-mode placeholders (`{{[INPUT]}}`/`{{[OUTPUT]}}`). Chat / adventure / story modes are not supported; the user guide flags this. Mode-aware wrapping lands with whatever future phase needs it.

## Backlog (unscheduled, unordered within tiers)

Tier 1 — direct follow-ons to Phase 2:
- **Re-port panel feed from `portal_test.html`.** The deleted POC contained a side-panel feed UI (agent activity / dispatch surface). The HTML file was removed in the 2026-04-26 polish pass; the *concept* still belongs in the main portal. Pull the panel feed design back in when wiring agent visibility (see Tier 2 "Agent pipeline visibility").
- **Memory provenance UI.** Per-response collapsible panel showing which LTM segments were injected, styled like "show thoughts". Requires `response_id` threading through SSE and a `GET /api/v1/response/{id}/memories` endpoint.
- **max_context_tokens cap aligned with kobold.** Ensure DB-exported buffer + LTM authornote don't exceed kobold's context cap; mirror kobold's truncation logic on the DERPR side so behavior matches.
- **LTM block placement experiment.** Compare quality of author's note (end) vs prompt-start prepend (full KV recompute). Inform future placement decision.

Tier 2 — adjacent capability:
- **Persona CRUD from UI.** Create/delete personas, not just edit existing. Currently only edit+save exists.
- **Tool execution UI.** Surface tool calls made during generation, approval gates for tool writes, tool result display.
- **Agent pipeline visibility.** Dispatch / autotriage state, queue depth, retry surface.

Tier 3 — broader scope (no near-term commitment):
- **Non-local providers exposed via portal.** OpenAI / Anthropic / Gemini routed through portal. Requires revisiting the "kobold owns templating" principle for providers that use message-array APIs rather than rendered-prompt APIs.
- **Tag-schema adapter.** Was prerequisite for the discarded server-side rebuild approach; no longer on critical path. Keep noted in case future work reopens the question.
- **KV cache reservation / semantic cache.** Blocked on upstream llama.cpp work (chunked prefill, semantic cache). Would unlock prompt-start memory placement without per-turn full recompute. Monitor upstream.
- **Multi-session management.** Named conversation threads per persona, switchable from UI.

## Notes

- Decision doc for Phase 2 architectural choice: `decisions/2026-04-19-portal-phase2-approach.md` (author's note over prompt-start; DB-as-source over server-side rebuild).
- Phase 2 splits across three sessions: 2.1 (shipped), 2.2 (LTM + memory-mode dropdown), 2.3 (portal logging + retry/edit). Each fits one context window with clean test scope.
- 2.2's design questions were locked in a 2026-04-20 prep session with Adam — see the decision table in 2.2's scope block above.
