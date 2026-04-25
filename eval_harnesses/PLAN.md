# Eval Harness Plan — Memory Retrieval Quality

## Goal

Build a repeatable harness that measures whether the right memory summaries surface for known queries. Establishes a baseline before any retrieval improvements (spreading activation, decay, keyword search).

## Background

### Why this exists

During a 2026-04-11/12 design review, Adam expressed skepticism about retrieval quality — "laptop questions not tagged as laptop memories." Before investing in retrieval improvements (spreading activation, KNN consolidation, etc.), we need objective measurement. The harness should answer: given a query, does the system return the right memories?

### Prior investigation (2026-04-12 session, chat `e60b73d2`)

1. **JSON pollution bug found and fixed.** 90% of summaries (312/347) stored raw JSON (`{"facts": [...]}`) instead of clean text. Embeddings computed on JSON boilerplate → garbage similarity. Fixed by correcting tool-call response parsing in `_summarize_segment`.

2. **Reset script created** at `scripts/reset_memory_summaries.py`. Backs up DB, wipes `Memory_Summaries`, `Memory_Segments`, `vec_Memory_Summaries`, NULLs `parent_summary_id`. Run with: `python -m scripts.reset_memory_summaries --db data/user_memory.db`

3. **`eval_harnesses/` directory created** with README listing 3 planned harnesses. No code yet.

4. **Agreed sequence:** fix pipeline → reset → re-process → build harness against clean data → establish baseline.

### Data quality audit (2026-04-12/13 session, chat `80b555f0`)

After reset + re-processing with fixed pipeline, audited 228 segments (all `arbitr` channel, 632 embedded msgs, 100% embed rate for this channel).

**Findings:**

#### Good segments (majority)
Most segments correctly group topically related Q/A pairs. Summaries are accurate. Example: SEG 617 groups koboldcpp optimization questions, summary captures hardware specs + BLAS settings.

#### Duplicate questions (medium issue, ~30 segments)
User asks same question 2-4x (retries or rephrases). Segmenter correctly groups them (high similarity). Not a bug — just redundant data. Summaries handle it fine by extracting one answer.

Affected: SEGs 570, 598, 625, 626, 627, 641, 666, 690, 695, 721, and many others.

#### Orphan garbage segments (critical, SEGs 788-795)

Root cause chain:
1. Batch 1 processes most msgs, marks some as outliers (`parent_summary_id` left NULL via `outlier_ids`)
2. Batch 2 picks up orphans — random mix of stray msgs from different topics
3. Orphans get segmented together (only 2 embedded msgs each, spanning huge ID ranges)
4. Summary captures only 1 random topic from the orphan mix

Specific examples:

| SEG | Actual linked msgs | Summary topic | Real content in ID range |
|-----|-------------------|---------------|------------------------|
| 788 | 1 (casino odds assistant answer) | Casino house edge | Surface laptop repair questions |
| 790 | 1 (browser cache assistant answer) | Browser cache viewer | koboldcpp, salmon, garbage disposal, lens wipes |
| 791 | 1 (Word borders assistant answer) | Word paragraph borders | Google workspace, Docker, network switches, vacuum filters |
| 794 | 2 ("use google", "check google") | Openclaw env vars | DDR4 slots, Ubuntu VM, OpenSSH, env vars |
| 795 | 1 (API error message) | Gemini model availability | Copilot hotkey, Uber sushi, steaks, hash browns, 33 topics |

**Key insight:** The ID range stored in segments (`start_interaction_id` to `end_interaction_id`) spans non-arbitr channel msgs too. `general/ambient` has 2196 msgs at only 5% embed rate, and their interaction_ids interleave with arbitr msgs. Diagnostic queries using `BETWEEN start_id AND end_id` sweep unrelated channels — but the segmenter itself is correctly scoped to arbitr channel only. The "mega-segment" appearance is a **display bug in diagnostics**, not a segmentation bug.

**Threshold 0.80 is fine.** Not a threshold problem. Segmenter works correctly on embedded msgs it sees.

#### Pipeline changes made (committed `e4fe74e`)

- `facts` → `observations` in tool schema + prompt (broader extraction: opinions, preferences, problems, not just facts)
- Vertexai grounding redirect URLs stripped from transcript before LLM call (noise reduction)
- `reply_to_id` column added to link assistant msgs to user msgs at generation time
- Orphan behavior kept (not removed) — Adam wants to refine, not eliminate

### Existing diagnostic tooling

**`scripts/memory_diagnostics.py`** — CLI tool with multiple modes:
- `--overview`: high-level stats (msg counts, embed counts, segments, per-channel breakdown)
- `--channel <id>`: per-segment detail with summaries, inter-segment similarity
- `--verbose`: individual messages with similarity scores
- `--threshold <float>`: dry-run re-segmentation with alternate threshold
- `--reset-segments --channel <id>`: delete segments+summaries for a channel (keeps embeddings)
- `--judge <model>`: LLM judge scoring (segmentation coherence, completeness, accuracy, conciseness) using Gemini

The LLM judge already covers summarization fidelity and segmentation coherence. **The missing piece is retrieval quality measurement.**

## Harness Design: Memory Retrieval Quality

### What to measure

Given a query string, does `retrieve_relevant_summaries()` return the expected summaries? Measure:
- **Precision@K**: of top-K results, how many are relevant?
- **Recall@K**: of all relevant summaries, how many appear in top-K?
- **MRR (Mean Reciprocal Rank)**: where does the first relevant result appear?

### Ground truth format

```python
# Each test case: query + set of expected segment IDs + set of anti-expected segment IDs
TEST_CASES = [
    {
        "query": "bass guitar blister",
        "expected_segments": [743],  # bass guitar + blister care
        "anti_segments": [],  # should NOT return Ravens draft talk
        "notes": "SEG 743 contains bass guitar blister advice"
    },
    {
        "query": "koboldcpp context truncation long context",
        "expected_segments": [663, 620],  # koboldcpp context handling
        "anti_segments": [],
        "notes": "SEG 663 = koboldcpp truncation, SEG 620 = KV cache/VRAM"
    },
    {
        "query": "Surface laptop SMC reset power",
        "expected_segments": [573, 574],  # Surface reset + boot failure
        "anti_segments": [],
        "notes": "SEG 573 = Surface hardware reset, SEG 574 = Surface boot loop"
    },
    # More cases needed — see "Building ground truth" below
]
```

### Building ground truth

**Approach:** Use existing DB data. The diagnostics script can dump all segments with summaries. Manually (or semi-automatically) identify queries that should match specific segments.

Good candidate categories from the data:
- **Hardware topics:** Surface laptop (573, 574), EK AIO cooler (721), Dell monitor (695), GPU crashes (626), vapor chamber orientation (720)
- **Software/tools:** koboldcpp (617, 620, 663), llamacpp (778), SillyTavern (781), Thunderbird (712)
- **Gaming:** Noita (697, 698, 699, 702, 708), Rocksmith (found in 795 range)
- **Networking:** WiFi latency (647), hostname resolution (728), network switches (in 791 range)
- **Personal/misc:** bass guitar (743), tax deadlines (641), jury duty (761), sushi safety (763)

**Semi-automated approach:** For each segment, generate 1-2 natural queries from its summary text. Then verify retrieval returns that segment. Can use the LLM judge model for query generation.

### Architecture

```
eval_harnesses/
    retrieval_quality.py    # Main harness script
    ground_truth.json       # Test cases (query → expected segments)
    results/                # Timestamped result files for comparison
```

**`retrieval_quality.py`** should:
1. Load ground truth from JSON
2. For each test case, call `MemoryManager.retrieve_relevant_summaries()`
3. Compare returned segment IDs against expected/anti-expected
4. Compute precision, recall, MRR per case and aggregate
5. Output structured results (JSON + human-readable summary)
6. Optionally generate ground truth semi-automatically from existing segments

### CLI interface

```bash
# Run evaluation against current DB
python -m eval_harnesses.retrieval_quality --db data/user_memory.db

# Generate ground truth candidates from existing segments
python -m eval_harnesses.retrieval_quality --generate-ground-truth --db data/user_memory.db

# Run with specific model name filter
python -m eval_harnesses.retrieval_quality --model gemini-embedding-001

# Compare two result files
python -m eval_harnesses.retrieval_quality --compare results/baseline.json results/after_change.json
```

### Key implementation details

- `retrieve_relevant_summaries` is in `src/memory/memory_manager.py` starting around line 740. It uses `vec_Memory_Summaries` for KNN search.
- Results include `summary_id`, `segment_id`, `content`, `distance`. Distance = L2 on normalized vectors (monotonic with cosine).
- Current retrieval params: `channel`, `persona_name`, `server_id`, `model_name`, `summary_level`, `before_interaction_id`, `window_size`, `limit`.
- For harness: use same params as production retrieval path in `chat_system.py` around line 306.
- The harness DB should be a **copy** of production, not the live DB. Use the backup files in `data/`.

### Data state notes

- `arbitr` channel: 632 msgs, 228 segments, 228 summaries. 100% embed rate. This is the only channel with segments.
- `general/ambient`: 2196 msgs, 5% embed rate, no segments. Not relevant for harness yet.
- DB path: `data/user_memory.db`. Backups exist with timestamps.
- Summarizer is being re-run with new `observations` prompt as of 2026-04-13. Ground truth should be built **after** re-processing completes.

### Open questions

1. Should ground truth be hand-curated only, or is semi-automated generation (LLM generates queries from summaries) acceptable for a first pass?
2. How many test cases needed for meaningful baseline? 20-30 seems reasonable.
3. Should harness test cross-segment retrieval (query matches multiple segments on same topic)?
4. After re-processing, orphan segments (788-795) may look different. Re-audit before finalizing ground truth.
