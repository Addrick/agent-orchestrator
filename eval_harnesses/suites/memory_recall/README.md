# memory_recall suite

Measures whether the right summaries surface for known queries against a
seeded memory state. Bypasses the LLM entirely — recall quality only.

## Status

Stub. Wiring incomplete:

- `driver.recall_driver` calls `MemoryManager.retrieve_relevant_summaries`
  but passes `query_embeddings=None`. Needs an embedding step (use
  `src.embedding_service`) so the call returns hits.
- Hindsight branch is a placeholder.
- `scenarios.json` references segment IDs from the 2026-04-13 audit DB
  (`data/user_memory.db`). Not seeded — run against a copy of that DB.
  TODO: switch to seeded ground truth so the suite is self-contained.

## Run

```powershell
$env:PYTHONPATH="."
python -m eval_harnesses.framework.cli list --suite memory_recall
python -m eval_harnesses.framework.cli run  --suite memory_recall
```
