---
name: Hindsight memory migration
description: Replace current SQLite-based long-term memory (embeddings/segments/summaries) with vectorize-io/hindsight as the retrieval backend; SQLite User_Interactions kept as system-of-record + cold archive
type: project
---

# Hindsight Memory Migration

## Context

Current LTM (`plans/long_term_memory.md`) is OpenViking-inspired: per-message embeddings → centroid sliding-window segmentation → fact extraction summaries → max-similarity retrieval. Functional alpha, never broke production, occasionally useful. ~0.5s perceptible latency on retrieval. Not load-bearing.

Evaluating [vectorize-io/hindsight](https://github.com/vectorize-io/hindsight) (MIT, Python, embedded-or-server, Postgres + pgvector) as a drop-in replacement for the retrieval/consolidation layer. Rationale:

- Biomimetic memory model (world / experiences / mental models) aligns with derpr's persona-as-agent framing.
- LongMemEval SOTA self-reported; one independent reproduction (VT Sanghani). Provides a known-good baseline ceiling to measure custom work against.
- Recall path is pure DB (vector + BM25 + graph + cross-encoder rerank) — sub-second, no LLM in the hot path.
- Retain + reflect are LLM-bound but can run async/scheduled off the user turn.
- Raw content IS persisted (verified: `chunks.chunk_text`, `memory_units.text`, optional `documents.original_text`). Hindsight is a derived index over raw, not a replacement.

**Risk accepted:** new project (created 2025-10-30), high commit cadence, vendor-controlled with SaaS upsell — exactly the supply-chain shape Adam normally avoids. Mitigations: pin image SHA, embedded mode (no `--no-deps` python install), wrap behind `MemoryBackend` ABC for clean rollback, keep SQLite User_Interactions as cold archive.

### Key Design Decisions

- **Cutover, not dual-read.** Memory system is alpha. Flip the flag, observe, evaluate via separate harness later. Skip the dual-read soak from the original sketch.
- **Async-only adapter.** `record_turn` is fire-and-forget into a per-bank queue; `retrieve_context` awaits hindsight recall. No sync paths planned.
- **Recall in tools, not reflect.** "Try to remember about X" tool wraps recall (fast, structured units the engine LLM integrates). Reflect reserved for explicit "deep think" tool + scheduled background consolidation. Reflect on every turn would double-LLM and inflate the store with redundant mental models.
- **SQLite User_Interactions kept.** System-of-record + audit + agent_actions + suppression. Hindsight is the retrieval index. Old `Message_Embeddings` / `Segments` / `Summaries` tables stop being written but remain on disk (cheap; rollback path).
- **Bank ID design — per-persona only.** Existing MemoryModes (CHANNEL_ISOLATED, SERVER_WIDE, PERSONAL, GLOBAL, TICKET_ISOLATED) were a quick patch and are obsoleted by this. New scheme: `bank_id = persona_name`. Single bank per persona. Sub-scoping (channel, user, server, interface) goes in retain `tags` for intra-bank filtering on recall. Cross-persona access (meta-agent) uses fan-out across persona banks (see "Meta-agent access" below).
- **Big model on the backend.** All LLM-bound ops (retain/reflect) run server-side and async. No reason to undershoot model size. Start with 30B+ via `HINDSIGHT_API_LLM_MODEL`, tune later.

## Architecture

```
┌─────────────┐     ┌──────────────────┐     ┌─────────────────┐
│ ChatSystem  │────▶│ MemoryBackend    │────▶│ HindsightBackend│
│  (engine)   │     │ ABC + selector   │     │  (async client) │
└─────────────┘     └──────────────────┘     └────────┬────────┘
                            │                         │
                            ├─▶ NullBackend           ▼
                            │                  ┌─────────────────┐
                            └─▶ SqliteLegacy   │ hindsight-api   │
                                (rollback)     │ + pg0 (Docker)  │
                                               └─────────────────┘

SQLite User_Interactions: kept, system-of-record, untouched cutover
```

### Memory products (server-side tiers)

Hindsight produces three distinct memory classes per bank, each driven by its own mission:

1. **Raw extraction** — `retain` LLM-extracts facts from incoming content per the bank's `retain_mission`. Stored as `memory_units` with `fact_type ∈ {world, experience, observation}`. Steered by `retain_mission` and `retain_extraction_mode` (`concise` | `verbose` | `custom`); chunked server-side per `retain_chunk_size`.
2. **Observations** — when `enable_observations=true`, Hindsight runs a consolidation pass after retain that synthesises stable facts about people/projects per `observations_mission`. Distinct from raw extraction; complements but does not replace mental models.
3. **Mental models** — `reflect` produces these on demand, shaped by `reflect_mission`. Phase 3.3's ReflectionAgent drives this tier.

ReflectionAgent only manages tier 3. Tiers 1–2 run automatically on retain. All three missions are per-bank fields set via `ensure_bank` / `update_bank_config`.

## Phase 1 Start Here (immediate)

Concrete order of operations to begin coding:

1. Create branch `feature/DP-107-hindsight-memory-backend` (or next free DP-ID).
2. New file `src/memory/backend.py` — `MemoryBackend` ABC + `MemoryHit` dataclass.
3. New file `src/memory/null_backend.py` — `NullBackend` no-op impl (smallest first; sanity-check the wiring).
4. New file `src/memory/sqlite_legacy_backend.py` — wrap existing `MemoryManager.retrieve_relevant_summaries` + segment/summary write paths behind the ABC. Pure delegation, no logic changes.
5. Add `memory_backend` field to `global_config.py` (default `sqlite_legacy` for safety).
6. ChatSystem DI: inject the selected backend; replace the direct retrieval-pipeline calls with `backend.retrieve_context(...)` and consolidation calls with `backend.record_turn(...)`. User_Interactions writes stay where they are.
7. Tests: `tests/memory/test_backend_selector.py` confirms each backend slot wires through.
8. New file `src/memory/hindsight_backend.py` — `HindsightBackend` impl. Use `hindsight-client` Python package.
9. Per-bank `asyncio.Queue` + worker task on `HindsightBackend.__aenter__`. Worker handles `httpx.ConnectError` (kobold offline) by logging + dropping.
10. Bank derivation: `bank_id = persona_name`. Tag derivation per §1.4.
11. Phase 2 (compose) can land in parallel — no code dependency on §1's adapter beyond the URL/env.

Stop after step 11. Phase 3 (tools), Phase 4 (backfill), Phase 5 (cleanup) are separate sessions.

## Sprint Status (2026-05-06)

- **Sprint 1 (DP-108)** — MemoryBackend ABC carved out + SqliteSemanticBackend wrapper. Merged into `feature/hindsight-migration`. Pure refactor.
- **Sprint 2 (DP-109)** — Native httpx-based HindsightBackend + paranoid Docker compose (socat egress proxy, embedded pg0, SHA pin, cap_drop, named volume). Untrusted bit threaded through retain → tag → recall round-trip with fail-closed default. Commit `768fdec`. Live test gated on `HINDSIGHT_LIVE_URL`.
- **Sprint 3 (DP-110)** — Landed 2026-05-06. Per-bank `asyncio.Queue` + worker (`__aenter__`/`aclose` lifecycle, ConnectError-tolerant). `mark_trusted`/`mark_untrusted` via parallel SQLite override table (option c — preserves chunk identity, no upstream PR dependency). `user_guide.md` Hindsight section added. `alpine/socat` pinned by index digest. 781 non-live tests green; mypy clean.
- **Sprint 4 (DP-111)** — MemoryRouter fan-out + meta_visible plumbing.
- **Sprint 5 (DP-112)** — retain bundling + retain_mission rename.
- **Sprint 6 (DP-113)** — Engine-side rewire complete. `SqliteSemanticBackend.recall` translates new-shape API to existing `retrieve_relevant_summaries` (real `untrusted` bit surfaced, sliding-window cutoff carried as an `exclude_after:N` scope tag); `retain_turn` is a deliberate noop under sqlite_legacy (the legacy MemoryAgent batch loop continues to drive consolidation). ChatSystem now talks to `self.memory_backend.recall` / `retain_turn` instead of `MemoryManager.retrieve_relevant_summaries` for retrieval / consolidation; transcript-layer calls (`log_message`, suppression, audit) untouched. New `recall_memory` tool (capabilities: `produces_untrusted=True`, `irreversible=False`) exposes recall to the LLM with persona/channel/user/server scope inherited from a per-turn `ContextVar` — the model can't redirect recall across personas. Selector default still `sqlite_legacy`; DP-114 flips it to `hindsight` and wires persona-config `ensure_bank`. 809 non-live tests green; mypy clean.
- **Phase 4 (backfill)** — not yet ticketed; runs post-cutover.
- **Phase 5 (cleanup)** — not yet ticketed; runs after a few weeks of stable Hindsight.

## Phase 1: Adapter + Selector

### 1.1 `MemoryBackend` ABC

`src/memory/backend.py`:

```python
class MemoryBackend(ABC):
    async def record_turn(
        self, persona: str, scope_id: str, role: str, content: str,
        untrusted: bool,                   # required — security framework contract
        metadata: dict, timestamp: datetime,
    ) -> None: ...

    async def retrieve_context(
        self, persona: str, scope_id: str, query: str, limit: int,
    ) -> list[MemoryHit]: ...               # MemoryHit.untrusted: bool

    async def deep_recall(  # optional, reflect-backed
        self, persona: str, scope_id: str, query: str,
    ) -> str: ...

    async def mark_trusted(
        self, persona: str, scope_id: str, hit_id: str,
        operator_id: str, reason: str,
    ) -> None: ...

    async def mark_untrusted(
        self, persona: str, scope_id: str, hit_id: str,
        operator_id: str, reason: str,
    ) -> None: ...
```

The `untrusted` bool is part of the ABC contract per `plans/tool_security_framework.md` + `decisions/2026-05-03-minimum-viable-tool-security.md`. Every backend impl must persist the bit and return it on `MemoryHit`. Reflect-derived rows inherit OR of source bits. `mark_trusted` / `mark_untrusted` flip the bit in place; no bank migration. See security plan for runtime semantics.

Cross-backend conformance test required: record-with-bit → retrieve → assert bit preserved. Catches a backend that silently drops origin metadata.

Three implementations:
- `NullBackend` — no-op, for testing or "memory off" personas
- `SqliteLegacyBackend` — wraps existing MemoryManager retrieval pipeline, no new dev, kept for rollback
- `HindsightBackend` — wraps hindsight Python client

### 1.2 Selector

Global config flag `memory_backend: hindsight|sqlite_legacy|null` in `global_config.py`. Per-persona override optional (rare). Wire through `ChatSystem` DI hub — replaces direct `MemoryManager` access for retrieval/consolidation paths.

User_Interactions writes stay where they are (still SQLite, still via MemoryManager). The backend is *only* the retrieval/consolidation layer.

### 1.3 Per-bank async queue with bundle drain

`HindsightBackend` owns one `asyncio.Queue` per `bank_id` (lazy-init, dict cache). One worker task per queue drains it. **Drain-on-tick coalescing (DP-112):** the worker awaits a single item to wake up, then `get_nowait()`s everything else currently queued and POSTs the lot in one `RetainRequest` (`{items: [...], async: true}`). Bursts batch; sparse traffic posts the lone item immediately. Preserves intra-bank ordering. Failures log + drop the whole batch — alpha, no retry logic.

```python
async def record_turn(...):
    item = self._build_item(...)  # MemoryItem dict (see §1.4)
    queue = self._queues.setdefault(bank_id, asyncio.Queue())
    await queue.put(item)
    # worker task consumes + bundles; user turn doesn't block
```

### 1.4 Bank ID + tags + metadata + document_id

```python
bank_id = persona_name  # single bank per persona

tags = [
    f"channel:{channel_id}",
    f"user:{user_id}",
    f"server:{server_id}",
    f"interface:{interface}",  # discord | zammad | kobold | gmail
]

metadata = {
    "interaction_id": <FK back to User_Interactions row>,
    # plus arbitrary other context — comes back in recall results,
    # NOT filterable at recall time (verified: tags-only for predicates)
}
```

**document_id derivation (DP-112).** Upstream Hindsight grows conversation memory by retaining items that share a stable `document_id` with `update_mode="append"` — the server reprocesses the whole document on each call so cross-turn extraction context is preserved.

- Scope key = `{bank_id}:{channel_id}` (channel sourced from the `channel:<id>` scope tag; falls back to `bank_id` for retains without channel scope, e.g., experiences).
- Session-cut heuristic: a gap of more than `SESSION_GAP_SECONDS` (24h) between retains in the same scope opens a new document. Within-window retains append to the current one.
- Document ID format: `{scope_key}:{session_start_iso}` — human-readable, unique without a counter.
- `update_mode`: `"replace"` on the first retain in a new session, `"append"` thereafter.
- State persisted in a small SQLite table (`Doc_Scope`) sibling to the trust-override store. One indexed upsert per retain, called from the worker that owns the queue.

Recall filters by `tags` + `tags_match` (`any` / `all` / `any_strict` / `all_strict`) and `tag_groups` (compound AND/OR/NOT). Sub-scope queries pick the right tag combination. MemoryMode rework collapses into "what tag predicate does this persona use for recall."

### 1.5 Meta-agent access (cross-persona recall)

Recall is single-bank by URL (`POST /banks/{bank_id}/recall`) — no multi-bank parameter exists. Three patterns considered:

1. **Fan-out and merge (chosen).** Meta-agent recalls from each persona bank in parallel (asyncio.gather), merges by score, takes top-K. N parallel sub-second DB hits → total ≈ slowest single call. Honest mapping of "memories from N personas." Add `MemoryBackend.retrieve_context_multi(banks, query, limit)` helper.
2. Shadow `__shared__` bank with dual retain — rejected: doubles retain LLM cost, dilutes per-persona embedding distribution, mental models bleed.
3. Single shared bank tagged by persona — rejected: collapses entity disambiguation across personas (same name in different persona contexts merges).

## Phase 2: Deployment

### 2.1 Docker compose sidecar

Add to existing compose:

```yaml
hindsight:
  image: ghcr.io/vectorize-io/hindsight@sha256:<pinned>
  ports: ["8888:8888"]
  environment:
    # Local kobold (on-demand) at host port 5001, OAI-compatible at /v1
    HINDSIGHT_API_LLM_PROVIDER: openai
    HINDSIGHT_API_LLM_BASE_URL: http://host.docker.internal:5001/v1
    HINDSIGHT_API_LLM_MODEL: <30B+ local model name>
    HINDSIGHT_API_LLM_API_KEY: kobold  # any string; kobold doesn't auth
  volumes:
    - hindsight-data:/home/hindsight/.pg0
  healthcheck:
    test: ["CMD", "curl", "-f", "http://localhost:8888/health"]

volumes:
  hindsight-data:  # named volume, lives in WSL2 fs (perf)
```

Embedded pg0 (their default). Single container. Postgres lifecycle = hindsight container lifecycle. Trade for simpler ops; revisit if data becomes load-bearing.

**Windows specific:** named volume only (NOT `C:\` bind mount — Postgres on `/mnt/c` is broken under load). Docker Desktop with WSL2 backend.

### 2.2 Backups

Weekly scheduled task on host:
```
docker exec hindsight pg_dump -U <user> hindsight > backups/hindsight-YYYY-MM-DD.sql
```

Restore-test ONCE before trusting. Untested = no backup.

### 2.3 Upgrades

- Pin image SHA, never `latest`.
- Subscribe to releases for changelog awareness before pulling.
- pg_dump before every upgrade (alembic migrations are forward-only).
- Bump deliberately, not on a schedule.

### 2.4 Eventual migration to external Postgres

If/when data becomes load-bearing, switch to compose-managed Postgres service:
- Decoupled hindsight upgrades from data
- Standard pg_dump from host without `docker exec`
- Independent monitoring

Not blocking. Defer until needed.

## Phase 3: Tool Surface

### 3.1 `recall_memory(query, limit=10)` tool

Default for "try to remember about X". Wraps `MemoryBackend.retrieve_context`. Returns structured units the engine LLM integrates into its own response. Sub-second.

### 3.2 `deep_recall(query)` tool (deferred / power-user)

Wraps reflect. Slow (5–15s), premium answer. Gated to explicit invocation, not default behavior. Side effect: writes new mental models to the bank. Skip until there's a clear reason.

### 3.3 ReflectionAgent — DEFERRED (2026-05-07)

**Status: deferred until baseline measurement.** Decision: ship cutover with retain (auto-extraction) + auto-observations + `recall_memory` tool only. Add ReflectionAgent later if measurement shows mental_models improve recall quality.

**Rationale:**
- Hindsight's automatic tiers already cover the critical path: retain triggers fact extraction, `enable_observations=true` runs server-side consolidation producing stable facts. Both happen without an agent.
- `reflect` is the only thing that produces `mental_models` (the third tier — causal chains, contradictions, longer-term patterns). It's on-demand RPC, not auto-scheduled. ReflectionAgent's sole job would be calling reflect periodically without a user query to pre-populate mental_models.
- Mental_models that nobody reads are wasted LLM spend. The hot recall path returns `memory_units` directly; mental_models only matter if (a) the reranker surfaces them in normal recall, or (b) `deep_recall` is invoked frequently. Both are unmeasured.
- Cheaper to measure baseline recall quality without mental_models, then decide if pre-populating them via background reflect is worth the token cost.

**Re-evaluate when:** baseline recall + observations are running stably, and we have data on (a) whether `recall_memory` results feel sparse / shallow, or (b) `deep_recall` usage patterns suggest cold-cache pain.

**Spec parked below for the future ticket. Do not implement until re-evaluated.**

---

<details>
<summary>Parked spec — ReflectionAgent (open if revisiting)</summary>

New file `src/agents/reflection_agent.py`. Inherits `Agent` base.
- `agent_name = "reflection"`
- Drives periodic `MemoryBackend.reflect()` per persona-bank to build mental models offline. Zero user latency.

**Cycle (`deploy()`):**
1. Preflight kobold `GET /v1/models`. Skip cycle if down (info log, no error).
2. Enumerate personas where `memory_backend != null` and reflection enabled.
3. For each persona-bank, bounded fan-out (`max_parallel_banks`, default 4):
   a. Per-persona cadence gate: skip if `now - last_reflect < persona_interval`.
   b. Reflect gate via `GET /banks/{id}/stats` (`BankStatsResponse`):
      - skip if `pending_consolidation > 0` (server still ingesting; reflecting now would miss in-flight content)
      - skip if `last_consolidated_at <= last_reflect_at` (nothing new since last cycle)
      - This replaces the earlier `min_retains_per_cycle` heuristic — the server already tracks consolidation state, no need for a parallel local counter.
   c. `backend.ensure_bank(persona, retain_mission, reflect_mission)` if first sight (see §3.3 Field Naming below).
   d. `backend.reflect(bank_id=persona, query=<seed>)` with per-bank `reflect_timeout_s` (default 300).
   e. Log mental-model count + duration to `agent_actions`.
4. Cycle summary written to action history.

**Reflect-mission seeding:**
- New persona fields `memory_retain_mission` and `memory_reflect_mission`. Default to templated strings derived from persona name + system prompt summary. Pushed to Hindsight via `ensure_bank` on first cycle for that persona. Persona config is the source of truth — re-run `ensure_bank` with current values if the persona file changes (idempotent on Hindsight side).

**Field naming — upstream rename (verified against `hindsight-api-slim/api/http.py` `CreateBankRequest`):**
- Active fields are `retain_mission` (extraction steering) and `reflect_mission` (reflect persona).
- `mission` and `background` are **deprecated aliases** for `reflect_mission`.
- ~~Our current `HindsightRESTClient.acreate_bank` passes `{mission, reflect_mission}`~~ — fixed in DP-112. Client now sends `retain_mission` + `reflect_mission` (and optional `enable_observations` / `observations_mission`); deprecated `mission` / `background` aliases removed.

**Observations tier:**
- ReflectionAgent does **not** manage observations — Hindsight runs them automatically post-retain when `enable_observations=true`. Persona config should expose `enable_observations` + `memory_observations_mission` and pass them through `ensure_bank`. Out of scope for Phase 3.3 implementation but listed for completeness.

**Background-mode query:**
- Hindsight `reflect` requires a `query` per call and ships no default for it (only `reflect_mission` is bank-persistent). Background cycles use a generic static seed (`"summarize new patterns and entities since last reflect"`). Per-persona override via `reflect_seed_query` on the persona.
- **Future experiment — topic-extraction hop:** before calling reflect, run a small LLM pass over recent retains to derive a focused query string ("what was actually discussed this cycle"). Hypothesis: a sharper query yields denser mental models. Cost: +1 LLM call per persona per cycle. Park until we have reflect output to evaluate against — measure mental-model quality with static seed first, then A/B against extracted query. Logged as future work, not Phase 3.3.

**Cadence config — DECIDED: `agents.json`** (closes plan open question on cadence location):

```json
"reflection": {
  "enabled": true,
  "schedule": {"interval": 3600},
  "per_persona_cadence": {
    "high_traffic": 1800,
    "slow_persona": 86400
  },
  "reflect_timeout_s": 300,
  "max_parallel_banks": 4
}
```

Global `schedule.interval` drives the agent loop; per-persona cadence gates inside the cycle. Single agent task, variable per-bank rate. No per-persona task explosion.

**Failure modes:**
- kobold down → skip cycle, info log
- hindsight unreachable → log, `consecutive_errors++`, base-class backoff applies
- per-bank reflect 5xx / timeout → log per-bank, continue other banks (no cycle abort)

**Trust:** reflect-derived mental models inherit OR of source untrusted bits per §1.1. Backend-side; agent passes through.

**Registration:** `main.py`: `agent_manager.register("reflection", ReflectionAgent)` — gated on `memory_backend != "null"` and `reflection.enabled`.

**Tests** (`tests/agents/test_reflection_agent.py`):
- preflight failure skips cycle without erroring
- per-persona cadence skips bank when interval not elapsed
- min_retains gate skips bank below threshold
- reflect failure on bank A does not block bank B (parallel isolation)
- `ensure_bank` called once per persona on first sight; subsequent cycles skip unless persona mission/reflect_mission changed
- per-bank timeout does not hang cycle
- untrusted bit propagates through reflect (cross-backend conformance)

**Out of scope:**
- On-demand reflect via tool (Phase 3.2 `deep_recall`)
- Cross-bank reflect (Hindsight API is single-bank)
- Mental-model pruning/decay (Hindsight server concern)
- Replacing existing OpenViking `MemoryAgent` — Phase 5 cleanup

</details>

## Phase 4: Backfill (post-cutover)

One-shot script `scripts/backfill_hindsight.py`:
- Iterate `User_Interactions` ordered by timestamp, grouped by (persona, scope)
- Call `HindsightBackend.record_turn` with original timestamp + metadata
- Idempotent via Hindsight's content_hash on chunks
- Throughput-bound on extraction LLM; expect hours for full history
- Run when convenient, not blocking

## Phase 5: Cleanup (later)

After Hindsight has run for a few weeks without issue:
- Strip MemoryAgent / segmentation / summarization code from the live path
- Drop `Message_Embeddings`, `Embedding_Vec`, `Segments`, `Segment_Summaries`, `Edit_History_Embeddings` tables (or archive)
- Remove `SqliteLegacyBackend` if no longer needed for rollback
- Update `architecture.md` + `user_guide.md` to drop OpenViking-era memory description

Not in scope for initial cutover.

### Backend LLM is on-demand

Local kobold at `localhost:5001` is started when needed, not always running. Implications:

- Retain queue worker must handle connection refused gracefully — log + drop the payload, don't crash, don't retry-storm. Alpha; no DLQ.
- Backfill script must check kobold is up before starting (one preflight `GET /v1/models`, abort with clear error if down).
- ReflectionAgent should preflight before its scheduled run; skip cycle if kobold is offline (log at info level, not error).
- Recall is unaffected — no LLM in the recall path.

## Prerequisites Status (2026-05-02)

Resolved during planning:

- **Eval harness direction** — repurpose `eval_harnesses/PLAN.md` as a generic multi-paradigm memory evaluator (supports old SQLite pipeline, Hindsight, future systems). Done in a separate plan revision; not blocking Phase 1.
- **Local LLM** — on-demand kobold at `localhost:5001`. Backend wiring assumes this; queue worker tolerates downtime.
- **Tool security framework** — landed 2026-05-03 as minimum-viable model (one bit, two flags, one rule). `MemoryBackend` ABC carries `untrusted: bool` on `record_turn` + `MemoryHit` + `mark_trusted` / `mark_untrusted` ops. Phase 1 ABC + each backend impl owns this contract. New `recall_memory` / `deep_recall` tools (Phase 3) tag `produces_untrusted=True`, `irreversible=False`. See `plans/tool_security_framework.md`.
- **Portal message history bugs** — being addressed elsewhere by Adam.
- **`embedding_pacing_question`** — obsoleted by this migration (Hindsight uses local `bge-small-en-v1.5`, no Gemini API). Close that question on cutover.
- **`internal_tool_schema_cleanup`** — rolled into Phase 5 (the OLD memory pipeline tools get torn out together).
- **MemoryMode rework** — collapses into `bank_id = persona_name` + tag predicates. Sweep MemoryMode references during Phase 1.

**No remaining hard prereqs.** Phase 1 is unblocked.

## Effort

- Phase 1 (adapter + selector + queue): ~2 days
- Phase 2 (compose + backup setup): ~half day
- Phase 3.1 (`recall_memory` tool): ~half day
- Phase 3.3 (ReflectionAgent): **deferred until measured — see §3.3**
- Phase 4 (backfill script): ~half day
- Phase 5 (cleanup): later, separate task

**Total to working cutover: ~3 days focused.**

## Sharp Edges

- **Schema drift.** Hindsight ships alembic migrations weekly (>20 versions in `alembic/versions/`). Pin SHA, plan periodic catch-up windows.
- **Sub-8B local models likely insufficient for retain/reflect.** Their `.env.example` reco is qwen2.5-32b. Adam running 30B+ aligns. Re-evaluate if changing to smaller model.
- **Reflect side effects.** Every reflect call writes new mental models. Background-only by default to avoid bloat from chatty tool use.
- **Hindsight crash = retrieval down.** With embedded pg0, container failure = no recall. NullBackend fallback in selector? TBD if it's worth the complexity for alpha.
- **Tag/metadata filtering for cross-cut queries.** Need to verify hindsight's recall API accepts metadata filters cleanly before locking the bank_id design — if it doesn't, we may need to fall back to multi-bank retains for some scopes.

## Open Questions

- ~~Does Hindsight's recall support cross-bank or metadata-filtered recall?~~ **Resolved 2026-05-02**: single-bank by URL path; rich tag filtering (`tags`, `tags_match`, `tag_groups` with AND/OR/NOT) but metadata values are not filterable at recall time (returned in results only). Decision: per-persona bank + tags for sub-scoping + fan-out for meta-agent.
- Does the LM Studio provider in hindsight handle abrupt disconnects gracefully? Local LLM swap/restart shouldn't crash the queue worker.
- ~~Where does `ReflectionAgent` cadence config live — `agents.json` or per-persona?~~ **Resolved 2026-05-07**: `agents.json` global `schedule.interval` drives the loop, with per-persona overrides in `per_persona_cadence` map. See §3.3.
- ~~Does Hindsight expose a per-bank stats endpoint usable for the reflect gate?~~ **Resolved 2026-05-07**: yes, `GET /banks/{id}/stats` → `BankStatsResponse` with `pending_consolidation` + `last_consolidated_at`. ReflectionAgent uses these directly; no local counter needed.

## Related

- `plans/long_term_memory.md` — current OpenViking-inspired LTM (being replaced)
- `plans/memory_future_work.md` — enhancements to the OLD system; review whether items still apply post-Hindsight
- `eval_harnesses/PLAN.md` — retrieval quality harness; will be the eventual evaluation tool for Hindsight vs old vs hybrid
- External: https://github.com/vectorize-io/hindsight, paper https://arxiv.org/abs/2512.12818
