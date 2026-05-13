---
name: Hindsight upstream API quirks
description: Verified facts about vectorize-io/hindsight bank/retain/reflect API shape that diverge from what our local HindsightRESTClient assumes. Source-of-truth file paths included.
type: reference
---

# Hindsight upstream API — verified 2026-05-07

Source: `vectorize-io/hindsight` repo, file `hindsight-api-slim/hindsight_api/api/http.py`. Verified against current main.

## Bank create / config — field names

`CreateBankRequest` (http.py L1037):

- **Active**: `retain_mission`, `reflect_mission`, `retain_extraction_mode` (`concise|verbose|custom`), `retain_custom_instructions`, `retain_chunk_size`, `enable_observations`, `observations_mission`.
- **Deprecated aliases for `reflect_mission`**: `mission`, `background`.
- **Deprecated**: top-level `disposition`, `disposition_skepticism|literalism|empathy` (use `update_bank_config`).

**Resolved 2026-05-07 (DP-112):** `HindsightRESTClient.acreate_bank` now sends `retain_mission` + `reflect_mission` (and optional `enable_observations` / `observations_mission`). Deprecated `mission` / `background` aliases are intentionally not accepted by the client.

## Retain shape — bundle, not per-message

`RetainRequest` (http.py L531) is `{items: list[MemoryItem], async: bool}`. Single call accepts multiple items.

`MemoryItem` fields (L437): `content`, `timestamp` (ISO-8601 | `"unset"` | null=now), `context`, `metadata`, `document_id`, `entities`, `tags`, `observation_scopes`, `strategy`, `update_mode` (`"replace"` default | `"append"`).

Growing-conversation idiom:
- Stable `document_id` per conversation
- `update_mode: "append"` adds new content to existing document and reprocesses

**Resolved 2026-05-07 (DP-112):** `HindsightBackend` now bundles items per drain tick into one `RetainRequest`, derives a stable `document_id` per `{bank_id}:{channel_id}` scope (with a >24h gap heuristic that opens a new session), and uses `update_mode="append"` after the first retain in a session. Server-side cross-turn extraction now works.

## Server-side chunking

`retain_chunk_size` is a bank-level config field. Hindsight chunks content during retain per this size. Callers send whole content; server chunks. Don't pre-chunk client-side.

## Three memory-product tiers

1. **Raw extraction** (`memory_units`, `fact_type ∈ {world, experience, observation}`) — driven by `retain_mission` + `retain_extraction_mode`.
2. **Observations** — automatic post-retain consolidation when `enable_observations=true`, driven by `observations_mission`. Stable facts about people/projects.
3. **Mental models** — produced by `reflect`, driven by `reflect_mission`.

ReflectionAgent (Phase 3.3) only drives tier 3. Tiers 1–2 are server-automatic.

## Bank stats endpoint

`GET /banks/{id}/stats` → `BankStatsResponse` (http.py L1463):
- `total_nodes`, `total_links`, `total_documents`, `total_observations`
- `nodes_by_fact_type`, `links_by_link_type`, `links_by_fact_type`, `links_breakdown`
- `pending_operations`, `failed_operations`, `operations_by_status`
- `last_consolidated_at`, `pending_consolidation`, `failed_consolidation`

Reflect-gate signal (used in plan §3.3): skip cycle when `pending_consolidation > 0` (still ingesting) or `last_consolidated_at <= last_reflect_at` (nothing new). No raw "retains since X" counter needed.

## Verbatim mission examples (from CreateBankRequest example schemas)

- `retain_mission`: `"Always include technical decisions and architectural trade-offs. Ignore meeting logistics."`
- `reflect_mission` (from public docs): `"You are a senior engineering assistant. Always ground answers in documented decisions and rationale. Ignore speculation. Be direct and precise."`
- `observations_mission` (from CreateBankRequest example): `"Observations are stable facts about people and projects. Always include preferences and skills."`
