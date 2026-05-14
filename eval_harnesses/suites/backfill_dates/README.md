# backfill_dates suite

Measures whether bulk-imported documents end up with the **correct
`mentioned_at` dates** on extracted memories, or whether everything
collapses to the import day.

## Bug under test

`scratch/hindsight_migration/hindsight_import.py` posts items with
`timestamp: "unset"`, which causes the server to stamp every extracted
memory with the import time. `recover_claudecode_hindsight.py` does it
right (passes a per-block anchor timestamp).

## What the suite does (per cell)

1. Provision a fresh scratch bank: `eval_backfill_<scenario>_<variant>_<rand>`
2. POST scenario's documents to `/memories` under one **timestamp strategy**:
   - `unset` — reproduces the bug
   - `explicit` — sets timestamp per item
   - `header_inline` — no timestamp, but doc body starts with `Date: <iso>`
   - `block_anchor` — extracts last ISO date from content as anchor
3. Optional `retain_mission` override (variant) instructs the LLM to read
   dates from prose
4. Recall against the bank, harvest each item's `mentioned_at`
5. `date_attribution` grader compares recalled dates to expected
6. Delete the scratch bank

## Variant matrix

| memory_variant | strategy | mission |
|----------------|----------|---------|
| unset_default_mission | unset | default | (repro)
| explicit_default_mission | explicit | default |
| header_inline_default_mission | header_inline | default |
| block_anchor_default_mission | block_anchor | default |
| unset_backfill_mission | unset | backfill-aware |
| header_inline_backfill_mission | header_inline | backfill-aware |

The interesting comparison is the bottom two vs. their default-mission
twins — does prompt engineering alone fix the bug, or does the timestamp
field have to be set?

## Run

```powershell
$env:PYTHONPATH="."
python -m eval_harnesses.framework.cli list --suite backfill_dates
python -m eval_harnesses.framework.cli run  --suite backfill_dates
```

Requires a reachable Hindsight server (`HINDSIGHT_URL` from
`config/global_config.py`). Each cell creates and deletes its own
scratch bank — no production data touched.

## Caveats

- Uses real Hindsight — no `--live` flag needed; this suite is always live.
- `wait_seconds` after sync POST is a paranoia knob; if recall returns no
  items, bump it.
- Grader treats date-only granularity (YYYY-MM-DD). Time-of-day drift OK.
