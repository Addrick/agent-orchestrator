---
name: Hindsight upstream API quirks
description: Verified vectorize-io/hindsight bank/retain/reflect API shape. v0.6.1 path map (every bank route is prefixed /v1/default) plus v0.5-era field detail that still holds. The live contract is src/memory/backend/hindsight.py; this page is the reference, not the authority.
type: reference
---

# Hindsight upstream API — v0.6.1 path shape, verified 2026-09-17

Source: `vectorize-io/hindsight`, file `hindsight-api-slim/hindsight_api/api/http.py`.

> ⚠️ **This page was frozen at v0.5.0 until 2026-09-17 and described routes the client had
> already stopped calling.** The **live contract is `src/memory/backend/hindsight.py`**, not this
> file — when they disagree, the code wins. Everything below the path map is v0.5-era field
> detail that is still accurate; the *routes* were the stale part.

## Path map — every bank route moved in v0.6.1

`HindsightRESTClient` was patched in `7d1f7a4` and now prefixes every bank route with
`HINDSIGHT_API_PREFIX = "/v1/default"` (`src/memory/backend/hindsight.py:27`).

| Op | OLD (v0.5) | CURRENT (v0.6.1+) |
|---|---|---|
| Create bank | `POST /banks` (bank_id in body) | `PUT /v1/default/banks/{bank_id}` (body has `name`) |
| Delete bank | `DELETE /banks/{id}` | `DELETE /v1/default/banks/{id}` |
| Retain | `POST /banks/{id}/retain` | `POST /v1/default/banks/{id}/memories` |
| Recall | `POST /banks/{id}/recall` | `POST /v1/default/banks/{id}/memories/recall` |
| Reflect | `POST /banks/{id}/reflect` | `POST /v1/default/banks/{id}/reflect` |
| Bank config | (n/a) | `PATCH /v1/default/banks/{id}/config` |
| Bank stats | (n/a) | `GET /v1/default/banks/{id}/stats` |

**`RecallRequest`** takes `query`, `tags`, `tags_match` (`any|all|any_strict|all_strict`),
`tag_groups`, `types`, `max_tokens`, `budget` (enum `"low"|"mid"|"high"`, **not** a float),
`query_timestamp`, `include`, `trace`. **There is no `k`** — callers slice client-side.

**`RecallResult`** items carry `text` (**not** `content`) and **no `score`** — the server returns
pre-ranked results; synthesize position-based ranking if you need one.

**Retain is still `{items: [...], async: bool}`** — a bare single item without the `items`
wrapper returns 422.

🔴 **A batch may not repeat a `document_id`.** A single `POST .../memories` fails with
HTTP 500 — *"Batch contains duplicate document_ids ... to avoid race conditions"* — which
conflicts with the v0.5 idiom of bundling turns that share a doc_id for `update_mode="append"`.
`HindsightBackend._split_by_document_id` splits each drained bundle into per-doc_id sub-batches;
coalescing still happens at the drain tick, only the on-wire POSTs split.

**Reflect response** is `{text, based_on, structured_output, usage, trace}`; `trace.tool_calls`
is a list of `{tool, input, output, duration_ms, iteration}`. `based_on` is
`{memories, mental_models, directives}`, not per-fact-type buckets.

## ⚠️ An empty bank passes every readiness check

`GET /banks/{id}/stats` returns `200` with `pending_consolidation: 0` for **any** bank name,
including one that does not exist. Read `total_documents` / `total_nodes` as well, or an
un-ingested bank scores as a model failure rather than an empty one.

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
