---
name: MemoryBackend ABC carries both legacy + new-shape methods
description: Resolution of a tension in the DP-108 / memory_backend_abc plan — the ABC defines the new Hindsight-shape surface (retain_turn / recall / reflect), but MemoryManager's existing semantic+episodic methods (store_summary / retrieve_relevant_summaries / log_agent_action) don't translate cleanly onto that shape. Decision: ABC carries BOTH method sets in Sprint 1.
type: project
---

## Context

`memory/project/plans/memory_backend_abc.md` and `memory/project/tasks/DP-108.md` together define Sprint 1 of the Hindsight migration: carve a `MemoryBackend` ABC out of `MemoryManager`, ship `SqliteSemanticBackend` as a wrapper, no behavior change.

The plan describes the ABC with **Hindsight-shape** methods:

```python
retain_turn(bank_id, role, content, *, timestamp, scope_tags, source_persona, metadata) -> str
retain_experience(bank_id, action_type, context, outcome, *, scope_tags, source_persona, metadata) -> str
recall(bank_id, query, *, k, types, tag_filter, max_tokens, budget) -> list[Memory]
recall_experiences(bank_id, query, *, match_contexts, k) -> list[Experience]
reflect(bank_id, query, *, tag_filter) -> ReflectResult
list_mental_models(bank_id, *, tags) -> list[MentalModel]
ensure_bank(bank_id, *, mission, reflect_mission) -> None
delete_bank(bank_id) -> None
```

But the task spec also says:

> Semantic + episodic methods removed from `MemoryManager` body. `MemoryManager` constructor accepts `backend: MemoryBackend | None`; existing semantic/episodic calls on `MemoryManager` become **thin delegations to `self.backend`**.

The existing methods are SQLite-shaped: `store_segment(channel, server_id, persona, start_id, end_id, ...)`, `store_summary(segment_id, content, embedding, model_name, ...)`, `retrieve_relevant_summaries(persona_name, channel, server_id, user_identifier, memory_mode, ...)`, `log_agent_action(agent_name, action_type, ...)`. They don't translate cleanly onto `retain_turn` / `recall`:

- `store_summary` takes an `embedding: bytes` + `segment_id` — Hindsight has no segment concept and computes embeddings server-side
- `retrieve_relevant_summaries` takes `query_embeddings` (caller-computed) + `memory_mode` (channel/server/personal/global) + an "ambient persona" union — Hindsight takes a string query and tag filters
- `log_agent_action` takes free-form `trigger_context` / `action_payload` strings — Hindsight `retain_experience` takes structured `context: dict` + `outcome`

If the ABC has only new-shape methods, MM's legacy delegations would have to translate on the fly — but the parameter shapes don't admit a clean lossless mapping, and translation belongs in the Hindsight backend (Sprint 2), not MM.

## Decision

The ABC defines **both** method sets in Sprint 1:

1. **Legacy SQLite-shape methods** as `@abstractmethod` — these are the contract every caller uses today. `SqliteSemanticBackend` implements them with the lifted SQL bodies. MM delegates to them via `self.backend`.
2. **New Hindsight-shape methods** with default impls in the ABC — `NotImplementedError` for `retain_turn` / `retain_experience` / `recall` / `recall_experiences` / `delete_bank`; noop returns for `reflect` / `list_mental_models` / `ensure_bank` (matching the plan's "noop / minimal on SQLite" note). HindsightBackend in Sprint 2 will override them with real impls.

Both sets coexist. Legacy is the active contract; new-shape is the forward-looking surface.

## Why

- **Honors the task contract** (MM delegates legacy methods through the ABC) without forcing translation logic into MM where it doesn't belong.
- **Honors the plan** (ABC defines the Hindsight shape) without making Sprint 1 do work that's actually Sprint 2's job.
- **Keeps Sprint 1 pure refactor.** No new behavior, no new translation layer, no new bugs. SQL bodies move; signatures don't.
- **Defers the migration choice.** Eventually callers will move from legacy methods to new-shape. That happens after HindsightBackend exists and we can verify the new shape covers the use cases. Sprint 1 doesn't need to commit to a migration path.

## How to apply

When picking up Sprint 2 (HindsightBackend), the legacy methods on the ABC are an **anti-target** for Hindsight — don't try to implement them on `HindsightBackend`. Instead:

- HindsightBackend implements **only** the new-shape methods (real impls).
- Legacy methods on HindsightBackend either raise `NotImplementedError` OR are routed through a translation shim — to be decided in Sprint 2 based on which callers still need them.
- Eventually MM stops calling legacy methods entirely and uses only new-shape; legacy methods are then dropped from the ABC. That's a Sprint 4+ cleanup, not part of Sprint 2.

## Sharp edge

The ABC currently has a wide surface (15+ legacy methods + 8 new-shape methods). This is **deliberately temporary**. Don't grow it further — additions should go on the new-shape side and migrations should shrink the legacy side.

## Related

- `memory/project/plans/memory_backend_abc.md` — full plan
- `memory/project/plans/hindsight_memory_migration.md` — overarching migration
- `memory/project/tasks/DP-108.md` — Sprint 1 task
- Commit `317a3c9` (feature/DP-108-memory-backend-abc-s1) — Sprint 1 implementation
