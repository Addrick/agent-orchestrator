# ambient_attribution suite

Measures whether multi-speaker transcripts ingested into ambient-style
banks correctly attribute statements to named participants — or fall back
to "ambient said X" / "narrator believes Y".

## Bug under test

`scripts/backfill_hindsight.py` already ships an anti-narrator
retain_mission for the `ambient` bank. This suite measures how well it
works, and lets us iterate on prompt + content format together.

## What the suite does (per cell)

1. Provision a fresh scratch bank with one of the variant retain_missions
2. POST a single-document transcript with multi-speaker turns, formatted
   per the variant's `content_format`
3. Recall against the bank with a "summarize each participant" query
4. `attribution_correctness` grader:
   - Per-fact: did some recalled item contain (speaker_name + key terms)?
   - Anti: count items that read "ambient said X" / "narrator noted Y"
5. Delete the scratch bank

## Variant matrix

| variant | format | mission |
|---------|--------|---------|
| prod_mission_date_speaker | `Date:/Speaker:` headers | shipped prod mission |
| prod_mission_inline | `Name: text` lines | shipped prod mission |
| emphatic_mission_date_speaker | headers | rewritten anti-narrator |
| emphatic_mission_bracketed | `[Name] text` | rewritten anti-narrator |
| emphatic_mission_json | JSON array | rewritten anti-narrator |

The interesting comparisons:
- prod vs emphatic with same format → does prompt rewrite alone help?
- date_speaker vs inline vs bracketed vs json with same mission → does
  format matter when the speaker is structurally explicit?

## Run

```powershell
$env:PYTHONPATH="."
python -m eval_harnesses.framework.cli list --suite ambient_attribution
python -m eval_harnesses.framework.cli run  --suite ambient_attribution
```

3 scenarios × 5 variants = 15 cells. Each cell creates and deletes its
own bank.

## Caveats

- Grader is heuristic: it checks for speaker-name + keyword co-occurrence
  in recalled item text. Doesn't parse a structured "subject" field
  (Hindsight doesn't expose one).
- Bad-attribution detection looks for `<bad_subject>` near speech verbs
  (`said`, `claims`, `noted`, etc.) or as the leading subject of an item.
  False positives possible if a memory legitimately mentions "the
  ambient channel" as topic.
- Recall query phrasing biases what comes back — using "Summarize what
  each participant said" pushes the recall toward attribution-style
  memories. A different probe might surface different items.
