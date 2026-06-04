---
name: Memory backend ABC — Hindsight-as-drop-in semantic backend
description: Carve a swappable backend layer out of MemoryManager so the semantic+experience+consolidation tier can be served by SQLite (current) or Hindsight (Vectorize, learning-graph memory). Transcript layer stays SQLite-only as system of record.
type: project
---

## Status

**SHIPPED (reconciled 2026-06-04).** Implemented and merged to master via DP-108/109: `src/memory/backend/base.py` (ABC), `sqlite.py` (`SqliteSemanticBackend`), `hindsight.py` (`HindsightBackend`). Sibling to tool-security work — independent.

## Goal

Make the semantic memory layer pluggable so we can run Hindsight (vectorize-io/hindsight) as a full replacement for the current segment/summary/agent-action stack while keeping the option to swap back. Hindsight is self-hosted, local LLM, no external API cost.

## Why Hindsight

- TEMPR retrieval (semantic + keyword + graph + temporal, RRF fusion + cross-encoder rerank). Closer to "agent that learns" than the current LLM-summarized RAG.
- 4 memory types: world / experience / opinion / observation. Maps onto our turns + agent actions cleanly.
- First-class `mental_models` and `directives` APIs — consolidation output is queryable structured data, not opaque blobs.
- Recall is cheap (100–600ms, no LLM). Retain is LLM-heavy (handled via `retain_async=True`).
- Per-bank `mission` + `reflect_mission` — natural home for persona system prompt.
- MIT licensed, Docker + embedded Python options.

## Non-goals

- Replacing the transcript layer. `User_Interactions`, suppression, version chevrons, audit, chronological history queries stay SQLite — different responsibility (system of record, identity, ordering, transactional edits).
- Live dual-backend mirroring. Switching backends requires running the backfill script; no automatic sync.
- Meta-agent implementation. This plan lays groundwork (`meta_visible` flag, `source_persona` tagging, optional metabank) but defers the agent itself.

## Layer split

Two distinct concerns currently smushed in `MemoryManager`:

**Transcript layer (SQLite-only, unchanged):**
- `log_message`, `update_platform_message_id`, `handle_message_edit`, `handle_portal_retry`
- Version chevrons: `update_interaction_content`, `list_interaction_versions`, `swap_interaction_version`
- Suppression: `suppress_interaction`, `suppress_message_by_platform_id`, `_suppression_filter`
- Chronological history: `get_personal_history`, `get_channel_history`, `get_server_history`, `get_global_history`, `get_ticket_history`
- Audit: `log_audit_event`, `mark_trusted/untrusted`

Why these can't move: identity (platform_message_id), strict chronological ordering, transactional edits, Phase 7 audit requirements, Discord version-chevron UI. Hindsight bank recall is semantic — wrong shape, no stable IDs.

**Memory layer (ABC — SQLite or Hindsight):**
- Episodic: `log_agent_action`, `update_agent_action_outcome`, `add_action_contexts`, `get_relevant_agent_actions`, `get_action_steps` → Hindsight Experiences
- Semantic: `store_message_embedding`, `store_segment`, `store_summary`, `retrieve_relevant_summaries`, segment failure tracking → Hindsight Recall
- Consolidation: `memory_consolidation.py` → Hindsight Reflect (noop on Hindsight backend; cron-driven on SQLite)

## ABC

```python
class MemoryBackend(ABC):
    async def retain_turn(bank_id, role, content, *, timestamp, scope_tags, source_persona, metadata) -> str
    async def retain_experience(bank_id, action_type, context, outcome, *, scope_tags, source_persona, metadata) -> str
    async def recall(bank_id, query, *, k, types, tag_filter, max_tokens, budget) -> list[Memory]
    async def recall_experiences(bank_id, query, *, match_contexts, k) -> list[Experience]
    async def reflect(bank_id, query, *, tag_filter) -> ReflectResult  # may noop on sqlite
    async def list_mental_models(bank_id, *, tags) -> list[MentalModel]  # noop on sqlite
    async def ensure_bank(bank_id, *, mission, reflect_mission) -> None
    async def delete_bank(bank_id) -> None
```

Single `bank_id` per call (Hindsight constraint). All retains use `retain_async=True` by default.

## Implementations

1. **`SqliteSemanticBackend`** — wraps existing segment/summary/agent-action code. `mental_models` / `reflect` noop or minimal. Default backend.
2. **`HindsightBackend`** — `hindsight-client` SDK against local server (Docker or embedded). `gpt-oss-20b` via local provider for retain extraction.

## Bank scheme

`bank_id = persona_name`. Scope encoded as **tags**, not separate banks — Hindsight performs better with larger banks (richer graph, better consolidation).

Mandatory tags on every retain:
- `scope:<channel_id|server_id|user_id|global>`
- `mode:<MemoryMode>` (CHANNEL_ISOLATED / SERVER_WIDE / PERSONAL / GLOBAL)
- `source:<discord|gmail|zammad|portal>`
- `source_persona:<persona_name>` (redundant inside persona bank, but consistent w/ metabank schema)

Recall enforces isolation via `tag_filter` matching the persona's MemoryMode. Per-persona `mission` set from system prompt at bank creation.

## Backfill

One-shot script, not automatic mirror. On backend switch:

1. Read `User_Interactions` ordered by timestamp, group per (persona, scope).
2. For each group, call `retain_turn` (or `retain_batch` w/ `document_id = persona:scope:date`).
3. Read `Agent_Actions` + `Agent_Action_Contexts`, call `retain_experience`.
4. Idempotent via stable `document_id`. Re-runnable.

Document explicitly: switching backends loses any new memories written while the other backend was active until backfill rerun.

## Cross-persona / metabank (experimental, opt-in)

Two viable approaches; build groundwork for both, default to fan-out, gate metabank behind flag.

**Fan-out recall (default):**
- Meta queries call `recall` on N visible persona banks in parallel, merge results.
- Latency = max(N) ≈ 600ms.
- Cheap, no extra storage, no extra retain cost.
- Loses cross-persona graph fusion (separate Adam-nodes per bank).

**Metabank dual-write (opt-in, experimental):**
- Personas with `meta_visible: True` dual-write retains to a `_meta` bank.
- `retain_async=True` keeps hot path fast.
- 2× retain LLM compute — acceptable given local LLM + parallel provider capacity.
- True graph fusion across personas. Risk: specialized-persona facts taken out of context during reflection.
- Mitigations:
  - Mandatory `source_persona` tag on every metabank memory (filter, trace, diff vs fan-out).
  - Distinct `set_reflect_mission` on metabank instructing the model to preserve persona-of-origin context when synthesizing.
  - `METABANK_ENABLED` global kill-switch + `delete_bank('_meta')` for clean teardown.

If metabank produces noisy or off-rails insights in testing, drop it. Fan-out remains the fallback.

PERSONAL and TICKET_ISOLATED memories never flow to metabank regardless of `meta_visible` (PII / deprecated mode).

## Config

- `global_config.SEMANTIC_BACKEND: Literal["sqlite", "hindsight"]` (default `sqlite`)
- `global_config.HINDSIGHT_URL: str` (default `http://10.0.0.70:8888` — aux-desktop, since 2026-05-19)
- `global_config.METABANK_ENABLED: bool` (default `False`)
- Per-persona override on backend choice (deferred — single global until proven needed)
- Per-persona `meta_visible: bool` on Persona schema (default `False`, ignored unless `METABANK_ENABLED`)

## Decisions baked in

- **Docker only for Hindsight runtime** (dev + prod). No embedded `HindsightServer`. Single deployment shape. Test tier `@pytest.mark.hindsight_live` auto-skips when `HINDSIGHT_URL` unreachable, matching `zammad_live` / `llm_live` pattern.
- **Consolidation cron: every 2h default**, configurable via `HINDSIGHT_REFLECT_INTERVAL_HOURS`. Per-bank `reflect()` call. SQLite backend keeps existing `memory_consolidation.py` schedule.

## Sprints

Each sprint is sized for a single 200k context window. Each ends green (`pytest` + `mypy` clean) and is independently mergeable. Branch naming: `feature/DP-XXX-memory-backend-abc-sN`.

### Sprint 1 — ABC carve-out + SQLite wrapper

Largest sprint, pure internal refactor. No new dependencies, no Hindsight code yet. Goal: prove the layer split holds without changing behavior.

- Define `MemoryBackend` ABC in `src/memory/backend/base.py` with the full method set from this plan.
- Implement `SqliteSemanticBackend` in `src/memory/backend/sqlite.py` wrapping existing semantic + episodic code paths.
- Move semantic/episodic methods out of `MemoryManager` into the backend; `MemoryManager` constructor takes `backend: MemoryBackend`, defaults to SQLite.
- Transcript layer methods stay on `MemoryManager` unchanged (system of record).
- All callers continue to use `MemoryManager` — backend is internal detail this sprint.
- Tests: existing semantic + agent-action suites run unchanged. Add `tests/memory/test_backend_contract.py` exercising the ABC against `SqliteSemanticBackend`.
- `mypy` clean. No dependency additions.

**Done when:** full pytest passes, `MemoryManager` no longer contains semantic/episodic implementation, only delegates.

### Sprint 2 — Hindsight backend + dev infra

Standalone service work + new backend impl. Self-contained; no caller changes.

- Add `hindsight-client` to `requirements.txt`.
- `docker-compose.hindsight.yml` for local dev (Postgres + Hindsight, port 8888 API / 9999 UI).
- `src/memory/backend/hindsight.py` implementing `MemoryBackend` via `hindsight-client`.
  - `retain_async=True` default on retain calls.
  - Tag enforcement helper: every retain stamps `scope:*`, `mode:*`, `source:*`, `source_persona:*`.
  - `ensure_bank` sets `mission` from persona system prompt + a default `reflect_mission`.
  - Recall translates MemoryMode → tag filter.
- `global_config`: `SEMANTIC_BACKEND` (default `sqlite`), `HINDSIGHT_URL`, `HINDSIGHT_LLM_PROVIDER`.
- Backend selection wired in `MemoryManager.__init__` — config picks SQLite or Hindsight.
- Tests: `@pytest.mark.hindsight_live` tier — full ABC contract against live Docker server. Auto-skip when unreachable via `tests/conftest.py` pattern.
- `docs/user_guide.md` + `memory/codebase/architecture.md` updated for backend selection.

**Done when:** Hindsight backend passes the full ABC contract suite against a live container, SQLite remains default, no caller code changed.

### Sprint 3 — Backfill script + consolidation cron + audit wrap

Operational tooling + Phase 7 integration. Smaller sprint.

- `scripts/backfill_hindsight.py`: reads `User_Interactions` + `Agent_Actions` from SQLite, calls `retain_turn` / `retain_experience` against Hindsight. Idempotent via stable `document_id = persona:scope:date`. Dry-run mode. Per-persona filter flag.
- Tests on a fixture DB → mocked backend.
- Consolidation cron: `HINDSIGHT_REFLECT_INTERVAL_HOURS` (default 2). Background task in `AppManager` calls `backend.reflect()` per active bank. Skip when SQLite backend (existing `memory_consolidation.py` keeps its schedule).
- Tool-security audit (Phase 7): wrap backend recall/retain calls with audit hooks instead of the now-moved SQLite call sites. Update `tests/integration/test_startup_wiring.py` if needed.
- `docs/user_guide.md` operational section: how to run the backfill, what changes when you switch backends.

**Done when:** backfill round-trips a fixture DB into Hindsight with byte-equivalent recall (manual eyeball + scripted check), cron task ticks reflect on schedule, audit logs cover backend-layer reads.

### Sprint 4 — Fan-out recall helper + meta_visible plumbing

Groundwork for cross-persona work. Small, focused.

- `MemoryRouter.recall_many(personas, query, **kwargs)` in `src/memory/router.py` — `asyncio.gather` over backend recalls, merge + dedupe results by entity/timestamp.
- `Persona` schema gains `meta_visible: bool` (default `False`). `default_personas.json` + `system_personas.json` annotated; loader handles absence (test it — see CLAUDE.md mandatory test rule).
- `MemoryRouter.list_visible_personas()` returns the personas eligible for cross-recall.
- No metabank yet. No meta-agent yet. This is groundwork only.
- Tests: unit tests for the merge logic, integration test for fan-out against the SQLite backend (seeded test data across two banks).

**Done when:** `recall_many` works against SQLite backend, `meta_visible` flag round-trips persona load/save, no production caller uses it yet.

### Sprint 5 — Metabank experimental + comparison harness

Opt-in feature behind kill-switch. Validate or kill it within this sprint.

- `METABANK_ENABLED` global config (default `False`).
- `HindsightBackend.retain_*`: when `METABANK_ENABLED` and `persona.meta_visible`, dual-write to `_meta` bank with `retain_async=True`. Mandatory `source_persona` tag. PERSONAL + TICKET_ISOLATED scopes never dual-write regardless.
- Custom `reflect_mission` set on `_meta` bank: explicit instruction to preserve persona-of-origin when synthesizing.
- Backfill script extended: `--include-metabank` flag.
- Comparison harness `scripts/compare_metabank.py`: same query against `fan_out_recall(personas)` vs `metabank.recall(_meta)`, side-by-side output for human eyeball.
- Run harness on a representative query set. Decide: keep, tune, or `delete_bank('_meta')` + flip flag default permanently.
- Document outcome in `decisions/YYYY-MM-DD-metabank-experiment.md`.

**Done when:** decision recorded. Either metabank stays behind opt-in flag with documented use cases, or it's removed with a decision noting why.

## Open questions resolved

- Docker vs embedded: **Docker only.**
- Consolidation cron: **2h default, configurable.**
- Audit wrap: **Sprint 3 — wraps backend layer, not SQLite call sites.**
