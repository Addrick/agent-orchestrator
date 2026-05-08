---
name: Hindsight per-hit trust overrides via parallel SQLite (option c)
description: Why mark_trusted/mark_untrusted on HindsightBackend uses a parallel SQLite override table instead of upstream tag-patch or delete+re-retain.
type: project
---

# Hindsight per-hit trust overrides via parallel SQLite

**Date:** 2026-05-06
**Ticket:** DP-110
**Status:** Implemented

## Context

`MemoryBackend` ABC carries `mark_trusted(bank_id, hit_id, ...)` / `mark_untrusted(...)` as part of the tool security framework (decisions/2026-05-03-minimum-viable-tool-security.md, decisions/2026-05-04-operator-trust-overrides.md). DP-109 landed the storage-side bit on retain; DP-110 had to land the operator override path.

Upstream Hindsight v0.5.0 has no documented unit-tag PATCH endpoint. Three options were evaluated in `tasks/DP-110.md`:

- **(a)** Extend upstream with `PATCH /banks/{bank}/units/{id}/tags`, PR upstream.
- **(b)** Supersede in place: delete the unit, re-retain with the flipped tag.
- **(c)** Maintain a parallel SQLite override table; recall post-filters and rewrites the bit.

## Decision

Option **(c)** — parallel SQLite at `src/memory/hindsight_overrides.db`.

Two tables:
- `Unit_Trust_State(bank_id, hit_id, untrusted, updated_at)` — current effective override.
- `Unit_Trust_Audit(audit_id, bank_id, hit_id, prior, new, operator_id, reason, ts)` — append-only flip log.

`HindsightBackend.recall` post-filters: bulk-reads overrides for the result hit IDs, rewrites `MemoryHit.untrusted` where present.

## Why not (a)

Blocks DP-110 on external review timeline. Hindsight is alpha-tier in our stack; not worth the upstream coupling for an operator-facing feature we need now.

## Why not (b)

Delete + re-retain loses chunk identity. The downstream cross-encoder reranker (and any future caching keyed on unit IDs) treats unit IDs as stable; flipping a bit shouldn't invalidate them. Also doubles the LLM round-trips on every flip (re-consolidation).

## Why (c)

- **Decoupled** — works against any Hindsight version, no upstream PR.
- **Cheap** — flips are rare operator actions; one indexed `IN (...)` query per recall.
- **Audit-native** — append-only audit table is the natural shape for the security contract's `(operator_id, reason, prior, new)` requirement.
- **Reversible** — if upstream lands a PATCH endpoint later, we can drop the override table and migrate state with a one-shot script.

## Tradeoffs

- **Two sources of truth.** The Hindsight tag and the override table can disagree. Resolved by always preferring the override on recall (operator wins).
- **Per-deployment file.** Override DB is local — won't survive a `docker compose down -v` unless `src/memory/` is on a persistent volume. Acceptable for alpha; flag for production.
- **`prior` field is "prior override," not "prior effective bit."** First flip records `prior=NULL` even though the unit's storage tag had a value. Audit log captures operator intent (the override history); the storage tag is recoverable independently from Hindsight if needed.

## Files

- `src/memory/backend/hindsight.py` — `_TrustOverrideStore` class + `mark_trusted`/`mark_untrusted` impl + recall post-filter.
- `tests/memory/test_hindsight_backend.py::test_mark_trusted_audit_and_recall_override` — flip → recall → flip back round-trip + audit row assertions.
