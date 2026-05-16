# Bridge Agent_Actions → Hindsight via series-batched retain (Option B)

**Date:** 2026-05-16
**Status:** Accepted (DP-116b)
**Related:** DP-116a (enriched action logging), memory_backend_abc plan, [[hindsight_metadata_strings_only]], [[hindsight_utf8_encoding_bug]]

## Context

After DP-116a, `Agent_Actions` carries real trajectory data: structured `action_payload` / `outcome_payload` on the root, per-tool/LLM child steps under `parent_id`, context tags for entity matching. The data is recallable in SQLite via `get_relevant_agent_actions`, but Hindsight — the system we want for semantic recall ("last time dispatch hit billing, what happened?") — never sees it.

Three options were on the table:

- **A** — leave the gap; rely on SQLite-side action-history tools only.
- **B** — series-batched retain: serialize the full parent+children+contexts series into dense prose at completion, call `retain_experience` once per series.
- **C** — pre-summarization layer: run our own LLM pass on the series, then retain the summary.

## Decision

Adopt **Option B**.

`HindsightBackend.retain_experience` already exists at the right granularity (one item per episodic record, ASCII-safe content, async-queued via `_ensure_worker`). The only missing pieces are:

1. **A call site at series completion** — `Agent._retain_action_series(action_id)` invoked after `_finalize_action` on each root in `dispatch_agent.py` and `reminder_agent.py`.
2. **A dense-prose formatter** — `Agent._format_action_series_prose` flattens the rows to k:v lines, drops nulls/empties, numbers steps, caps at 8000 chars. This avoids feeding raw JSON braces to Hindsight's extractor, which performs best on chat-like prose.
3. **A stable `document_id`** — `f"agent_action:{action_id}"`. Passed explicitly through `retain_experience(..., document_id=...)`, which causes `HindsightBackend._build_item` to bypass `_doc_scope.resolve` and use `update_mode="replace"`. Re-retain on the same id is idempotent.
4. **A `lookup_agent_history(action_id)` tool** under the `agents` service binding, so a persona that hits a bridged summary in recall can dereference back to the full raw rows.

## Why not C

Hindsight's retain pipeline already runs an LLM-driven fact extractor on every item. Adding a derpr-side pre-summarization pass = two LLM calls per series for a marginal density gain over (B + dense prose). Hindsight `reflect` already covers cross-series consolidation into mental models. C duplicates reflect's job. Deferred unless extractor output on serialized series proves unusable in practice.

## Bank target

Series retain into the **dispatch_analyst persona bank** (option 1 from the task plan, not the separate `<persona>__experiences` bank). Rationale: Hindsight is designed for this granularity; mingling agent series alongside chat-prose extractions lets the same `reflect` call surface both. If chunking/density regressions on chat-prose extraction emerge, the fallback (option 2 — separate `<persona>__experiences` bank with raised `retain_chunk_size`, recall as 2-bank fan-out) stays available without code rewrites — only bank wiring changes.

**Open follow-up:** evaluate persona-bank-mingled vs. fan-out approach against a representative LME2 trajectory fixture once that exists.

## Density tweak

The plan flagged stringified action series as syntactically denser than chat prose. Mitigations available:

1. ✅ **Pre-format payload before retain** — implemented (`_format_action_series_prose`).
2. ⏸ Separate experience bank with raised `retain_chunk_size` — deferred; only escalate if (1) is insufficient.
3. ⏸ Raise `retain_chunk_size` on the main persona bank — needs the empirical `{6000, 10000, 16000, 24000}` sweep gated on accumulated real data (or a 50+ synthetic-series fixture). Per-call chunk override is an upstream Hindsight 0.6.1 gap — worth filing.

## What was NOT done

- **`retained_at` column on `Agent_Actions`** — declined. Idempotency via stable `document_id` + `update_mode="replace"` covers the double-retain-after-restart case without a schema migration. Re-flag if the worker-queue replay assumption changes.
- **`tags_match` on mental models including `type:experience`** — config-only change via `apatch_bank_config`, not in this code change. Apply post-deploy once a bank has accumulated bridged series. ([[hindsight_mental_model_tags_match]])
- **Chunk-size sweep** — gated on real or representative-fixture data; can't be run yet.
- **Pre-summarization layer (C)** — only if (B + dense prose) proves insufficient.

## Files touched

- `src/memory/backend/base.py` — `retain_experience` ABC: `document_id` + `content_override` kwargs.
- `src/memory/backend/hindsight.py` — `_build_item` honors explicit `document_id` (replace mode); `retain_experience` threads new kwargs.
- `src/memory/backend/sqlite.py` — `get_agent_action`, `get_action_contexts` helpers.
- `src/memory/memory_manager.py` — delegation + new helpers.
- `src/agents/base.py` — `_format_action_series_prose`, `_retain_action_series`, `experience_bank` / `experience_persona` class attrs.
- `src/agents/dispatch_agent.py`, `src/agents/reminder_agent.py` — wire retain after `_finalize_action`.
- `src/tools/definitions.py`, `src/tools/agent_tool_handler.py` — `lookup_agent_history` tool + handler.
- Tests: `tests/agents/test_base.py`, `tests/memory/test_hindsight_backend.py`, `tests/tools/test_agent_tool_handler.py`, `tests/agents/test_agent_service.py`, `tests/integration/test_dispatch_action_trajectory.py`.
