# memory_recall suite

Measures whether the right summaries surface for known queries against a
seeded memory state. Bypasses the LLM entirely — recall quality only.

Two retrieval shapes share the same scenarios.json:

- **sqlite** — `retrieve_relevant_summaries()` against `Memory_Summaries`.
  Graded by `RetrievalHitsGrader` over `expected_segments` / `anti_segments`.
- **hindsight** — `arecall()` against a Hindsight bank. Graded by
  `SemanticRecallGrader` over fact locators (`expected_facts` /
  `noise_facts`).

## Ground truth — two-tool authoring loop

The "known good" DB is built by two tools that mirror each other:

| Tool | Side | Purpose |
|------|------|---------|
| `draft_scenarios.py` | sqlite upstream | Browse `data/user_memory.db` (`list`, `search`, `show`), then emit a scenario JSON stub for chosen segments (`draft`). |
| `curate.py` | hindsight downstream | Probe a live bank with candidate locators to confirm each resolves to exactly the expected memory(s). |

Author flow:

1. `draft_scenarios search "<topic>"` → find segment_ids whose summaries
   contain the target fact.
2. `draft_scenarios show <seg_id> --with-turns` → confirm the segment
   actually grounds the fact.
3. Pick noise segs in the same channel with adjacent vocab.
4. `draft_scenarios draft --shape both --query "..." --segments A,B
   --noise-segments C --bank test_persona` → JSON stub to stdout.
5. For each hindsight `expected_facts[*].locators[0]`, run
   `curate --bank test_persona --locator "<excerpt>"` until it resolves to
   the intended memory(s) only.
6. Paste into `scenarios.json`. Strip `_seg_id_hint`.

## Frozen fixtures

The suite ships frozen fixtures so it runs without depending on the live
user DB or a re-seeded Hindsight bank on every invocation.

### Sqlite slice — `fixtures/slice.sql`

A `.sql` dump of selected `Memory_Segments` + `Memory_Summaries` (with
embedding BLOBs) + `User_Interactions` + `Message_Embeddings`. Schema is
**not** included — the loader calls `MemoryManager.create_schema()` to get
the current live schema, then exec's the slice INSERTs. Survives schema
migrations.

Rebuild:
```powershell
python -m eval_harnesses.suites.memory_recall.freeze_sqlite `
    --segments 1121,1122,1124,... `
    --out eval_harnesses/suites/memory_recall/fixtures/slice.sql
```

A scenario opts in via the `slice_sql` field (handled by
`framework/fixtures.py:build_fixture`). The materialized DB is cached by
slice-checksum under `.eval_cache/slices/` and reused across runs until
the slice changes.

### Hindsight test_persona bank — `fixtures/test_persona_seed.json`

Persona-fact seed JSON. The `seed_hindsight.py` script reseeds the
`test_persona` Hindsight bank from it. Reseeds are checksum-gated against
`fixtures/.test_persona_seeded.json` so the (slow) LLM-extraction pass
only runs when seed text, bank name, or seeder version changes.

```powershell
python -m eval_harnesses.suites.memory_recall.seed_hindsight
python -m eval_harnesses.suites.memory_recall.seed_hindsight --check   # state only
python -m eval_harnesses.suites.memory_recall.seed_hindsight --force   # reseed anyway
```

## Run

```powershell
$env:PYTHONPATH="."
python -m eval_harnesses.framework.cli list --suite memory_recall
python -m eval_harnesses.framework.cli run  --suite memory_recall
```

For ad-hoc one-off runs against the slice without the full runner:

```powershell
python -m eval_harnesses.suites.memory_recall.sqlite_sanity
```
