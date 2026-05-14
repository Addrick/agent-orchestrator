# Memory recall — grading schema

Fact-anchored grading for semantic banks (Hindsight). Pairs with
`SemanticRecallGrader` in `eval_harnesses/framework/grading.py`.

## Scenario shape

```json
{
  "id": "persona_birthday_multi",
  "description": "birthday query should surface date + year facts",
  "query": "when is Adam's birthday",
  "bank": "test_persona",
  "persona_name": "default",
  "channel": "arbitr",
  "k_sweep": [1, 3, 5, 10],
  "expectations": {
    "expected_facts": [
      {"key": "birthday_date",  "source": "history", "locator": "March 14"},
      {"key": "birthday_year",  "source": "seed",    "seed_key": "adam_dob_year"}
    ],
    "noise_facts": [
      {"key": "work_addr", "source": "history", "locator": "1 Infinite Loop"}
    ]
  },
  "thresholds": {
    "k_pass": 5,
    "recall": 0.5,
    "noise_rate": 0.25
  },
  "graders": ["semantic_recall"]
}
```

### Fact entry

| field      | required | meaning |
|------------|----------|---------|
| `key`      | yes      | stable identifier used in results/diffs |
| `source`   | yes      | `"history"` (already in bank) or `"seed"` (injected from seed file) |
| `locator`  | history  | substring/regex unique enough to identify one memory in the bank |
| `seed_key` | seed     | key into `fixtures/test_persona_seed.json` |

`expected_facts` = must appear in retrieval. `noise_facts` = adjacent /
plausibly-related items that should NOT appear (used to compute `noise_rate`).

## Fixture files

```
eval_harnesses/suites/memory_recall/
  fixtures/
    test_persona_pairs.json   # scenarios (curated query → fact pairs)
    test_persona_seed.json    # intentional facts injected into bank
```

`test_persona_seed.json` schema:

```json
{
  "adam_dob_year": {
    "text": "Adam was born in 1990.",
    "tags": ["persona", "biography"]
  }
}
```

## Resolution

Locators → memory_ids happens once per suite load. Output cached into
`run_output.raw["resolved_ids"]: {fact_key: memory_id}`. Unresolved facts
(zero or multiple matches) → scenario fails loudly with `UNRESOLVED:[keys]`
in grader notes. This is intentional: silent ID drift would mask real bugs.

Resolution targets the **live bank**, not the seed file. Reasons:
- catches bank/seed drift
- handles `source: "history"` which has no seed entry
- failures are debuggable because the resolved map is written into results JSON

## Seed modes (CLI flag `--seed-mode`)

| mode               | behavior                                              | default |
|--------------------|-------------------------------------------------------|---------|
| `assume`           | bank present; resolve only. Fail if locator misses.    | local   |
| `ingest_if_missing`| if bank empty or missing seed facts, run ingestion.    | —       |
| `reseed`           | wipe bank, full reingest from seed + arbitr history.   | CI      |

## Metrics (per K in `k_sweep`)

Given retrieved hits `H = [h_1, ..., h_N]` and expected IDs `E`:

- `precision@k = |{h_i ∈ H[:k] : h_i ∈ E}| / k`
- `recall@k    = |distinct(H[:k]) ∩ E| / |E|`
- `mrr         = 1 / (first_rank_of_expected) or 0`
- `noise_rate@k = |{h_i ∈ H[:k] : h_i ∈ noise_ids}| / k`

Multi-fact queries: denominator of recall is `len(expected_facts)`, so a
query expecting 3 facts that retrieves 1 gets recall = 1/3.

## Pass criterion

```
passed = (no unresolved facts)
       AND recall@k_pass >= thresholds.recall
       AND noise_rate@k_pass <= thresholds.noise_rate
```

Defaults: `k_pass=5, recall=0.5, noise_rate=0.25`. Override per scenario
via `thresholds`. Noise is a soft signal — it lowers score and can fail
the gate, but a single adjacent hit at high K won't nuke the run.

## Why not just segment_ids?

`RetrievalHitsGrader` (segment-id-based) still works for SQLite summary
retrieval. `SemanticRecallGrader` exists because:
1. Hindsight memory_ids are UUIDs that change on reingest — fixtures rot.
2. test_persona is duped arbitr history (large), curated pairs need
   separate storage from the corpus.
3. Multi-fact queries are common; one query → N expected facts is the
   normal case, not the exception.
4. Noise is categorical (any "work address" hit is wrong), not item-specific.

Resolver bridges the gap: humans write stable locators, harness resolves
to current IDs at load time, grader uses IDs.
