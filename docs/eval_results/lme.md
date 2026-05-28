# LongMemEval results — Hindsight backend

Per-question evaluation of Hindsight memory recall on the LongMemEval V1 cleaned dataset (`xiaowu0162/longmemeval-cleaned`). 14 banks total: 5 S-tier baseline, 7 M-tier baseline, 2 variant ("v2a") banks ingested under a modified retain mission.

All judging via local `gemini` CLI subprocess (paid OAuth tier) over the ACP transport. A judge meta-eval (below) showed the judge verdict is invariant to judge-model on fixed predictions, so the judge is **locked to `gemini-2.5-flash`** (which benchmarks above the paper's GPT-4o); the answer-model is the experimental variable.

## Run conditions (constant across rows unless noted)

- **Recall**: `arecall(bank, question, tags=[qid])`, `max_tokens=512`, `top_k=10`
- **Answer model**: `gemini-2.5-flash`
- **Judge model**: `gemini-2.5-flash` (substituted for paper's GPT-4o)
- **Scoring**: strict per-paper — every gold fact must appear; abstention required when gold says info is missing.
- **Variant `v2a` retain mission**: "Extract facts from the conversation. For each fact, include both the specific subject (named entities, titles, numbers, dates) and one or more general descriptors (category, type, domain, intent) in the fact text itself. Each fact should remain retrievable whether the reader searches by the specific or by the general." (Hindsight default mission empty otherwise.)

## Results table

| # | qid | qtype | tier | variant | bank | fact_count | ingest_wall | n_retrieved | session_hit | judge | notes |
|---|-----|-------|------|---------|------|-----------:|------------:|------------:|------------:|:-----:|-------|
| 1 | 1c549ce4 | multi-session | S | baseline | `lme_s_1c549ce4` | 1,319 | 36 min | 20 | 1.00 | ✅ | clean — "$140" |
| 2 | 1c0ddc50 | single-session-preference | S | baseline | `lme_s_1c0ddc50` | 1,293 | 39 min | 19 | 1.00 | ❌ | sole S-tier failure; retrieval-ranking layer (vocab mismatch) — gold history-podcast facts present but never enter top-K |
| 3 | a3045048 | temporal-reasoning | S | baseline | `lme_s_a3045048` | 1,042 | 49 min | 18 | 1.00 | ✅ | clean — "7 days" |
| 4 | cc539528 | single-session-assistant | S | baseline | `lme_s_cc539528` | 1,404 | 60 min | 20 | 1.00 | ✅ | "Python, SQL, Ruby, PHP" (gold: Ruby/Python/PHP) — extra item not penalized |
| 5 | 50635ada | knowledge-update | S | baseline | `lme_s_50635ada` | 1,367 | 81 min | 19 | 1.00 | ✅ | "Premier Gold… previous status was Premier…" |
| 6 | 1c0ddc50 | single-session-preference | S | v2a | `lme_s_1c0ddc50_v2a` | 1,613 | 1.7 h | 15 | 1.00 | ❌ | A/B target — variant retain mission **did not fix** the failure; subject+descriptor formatting still leaves history facts off-topic vs. "activities during commute" |
| 7 | 1c549ce4 | multi-session | M | baseline | `lme_m_1c549ce4` | 15,309 | ~6 h | 19 | 1.00 | ✅ | holds at 10× haystack |
| 8 | 91b15a6e | multi-session | M | baseline | `lme_m_91b15a6e` | 13,110 | — | 19 | 1.00 | ✅ | "$5,150" |
| 9 | 8fb83627 | knowledge-update | M | baseline | `lme_m_8fb83627` | 11,866 | — | 13 | 1.00 | ✅ | National Geographic issue count |
| 10 | c9f37c46 | temporal-reasoning | M | baseline | `lme_m_c9f37c46` | 11,525 | — | 15 | 1.00 | ✅ | stand-up comedy duration |
| 11 | gpt4_61e13b3c | temporal-reasoning | M | gpt4-backbone | `lme_m_gpt4_61e13b3c` | 12,467 | — | 12 | 1.00 | ✅ | "Approximately three weeks." (GPT-4 generated haystack) |
| 12 | gpt4_68e94287 | temporal-reasoning | M | gpt4-backbone | `lme_m_gpt4_68e94287` | 11,362 | — | 19 | 1.00 | ✅ | vegan-chili-post ordering |
| 13 | 6aeb4375_abs | knowledge-update | M | abstention | `lme_m_6aeb4375_abs` | 11,856 | — | 15 | 1.00 | ✅ | correctly abstained — "no mention of how many Italian restaurants" |
| 14 | 91b15a6e | multi-session | M | v2a | `lme_m_91b15a6e_v2a` | 15,792 | 11.1 h | 16 | 1.00 | ✅ | no regression from baseline; re-judged 2026-05-24 post-consolidation (was 11,150 facts pre-drain) |
| 15 | 1c549ce4 | multi-session | S | v3a verbose | `lme_s_1c549ce4_v3a` | 1,280 | — | 7 | 1.00 | ✅ | positive control held under verbose; fewer facts in top-K context (20 → 7) because verbose facts are longer |
| 16 | 1c0ddc50 | single-session-preference | S | v3a verbose | `lme_s_1c0ddc50_v3a` | 1,322 | — | 6 | 0.00 | ❌ | **regression** — verbose dropped session_hit from 100 % (baseline & v2a) to 0 %. Longer facts evict gold-session facts from the top-K window; failure is now retrieval-miss, not just ranking |
| 17 | 91b15a6e | multi-session | M | v3a verbose | `lme_m_91b15a6e_v3a` | 11,598 | — | 4 | 1.00 | ✅ | held at M-tier; n_retrieved dropped 19 → 4 (same verbose-fact-length effect) |

## Aggregate scoring

| Slice | n | session_hit% | judge_yes% |
|-------|--:|-------------:|-----------:|
| S baseline | 5 | 100.0% | 80.0% |
| S v2a | 1 | 100.0% | 0.0% |
| S v3a verbose | 2 | 50.0% | 50.0% |
| M baseline | 7 | 100.0% | 100.0% |
| M v2a | 1 | 100.0% | 100.0% |
| M v3a verbose | 1 | 100.0% | 100.0% |
| M granite-extract | 3 | 100.0% | 33.3% |
| **All** | **17** | **94.1%** | **82.4%** |

(The granite-extract row is a separate extraction-model A/B, not folded into
the 17-bank "All" total — see the Granite extraction A/B section below.)

Per-qtype across all rows:

| qtype | n | judge_yes% |
|-------|--:|-----------:|
| multi-session | 6 | 100.0% |
| temporal-reasoning | 4 | 100.0% |
| knowledge-update | 3 | 100.0% |
| single-session-assistant | 1 | 100.0% |
| single-session-preference | 3 | 0.0% |

## Granite extraction A/B (2026-05-27)

Re-ingested 3 m-tier qids with the **granite-4.1-8b-Q5_K_S** local LLM as the
consolidation/extraction model, on the testing hindsight server
`http://10.0.0.70:8890`. The comparison banks (`lme_m_{qid}` on production
`8888`) were extracted with **qwen3-30b-a3b**. Note: the extraction model is
the LLM Hindsight uses during consolidation to turn raw sessions into fact
units — distinct from the gemini answer/judge model in `lme_judge`, which is
held constant across both arms. Granite banks: ~470 docs / ~16.2–16.8k facts
each, fully drained (`pending_consolidation=0`) before scoring. Scored with the
full per-k sweep at temp=0 (`--model-answer lme-t0 --model-judge lme-t0
--per-k-sweep`, `max_tokens=512`). Raw: `.eval_cache/lme_results/granite_3bank.json`
(also archived in the notes repo at `project/eval_results/granite_3bank.json`).

| qid | qtype | gold | n_facts | sess_hit (any k≥3) | judge (k=10) | qwen baseline |
|-----|-------|------|--------:|:------------------:|:------------:|:-------------:|
| 1c549ce4 | multi-session | `$140` | 12 | HIT | **yes** (k≥10) | yes |
| 8fb83627 | knowledge-update | `Five` | 9 | HIT | **no** | yes |
| 7161e7e2 | single-session-assistant | `Admon … 8am-4pm Sundays` | 6 | HIT | **no** | n/a — qwen bank empty (0 nodes), no baseline |

**Aggregate: session_hit 100% (3/3), judge_yes 33% (1/3 at k=10).** Of the two
qids with a qwen baseline (both `yes`), granite **regresses 8fb83627** and
**matches 1c549ce4**. The third (`7161e7e2`) is a granite failure with no qwen
counterpart to compare against — the production qwen bank `lme_m_7161e7e2` is
empty, so it is *not* scored as a regression. Retrieval stays healthy on every
granite bank.

Full per-k matrix (sess_hit / judge per cell):

| qid | k=1 | k=3 | k=5 | k=10 | k=20 |
|-----|-----|-----|-----|------|------|
| 1c549ce4 | miss/no | HIT/no | HIT/no | HIT/**yes** | HIT/**yes** |
| 8fb83627 | HIT/no | HIT/no | HIT/no | HIT/no | HIT/no |
| 7161e7e2 | HIT/no | HIT/no | HIT/no | HIT/no | HIT/no |

Predicted answers at k=10:
- `1c549ce4` → "The total cost of the car cover and detailing spray you purchased is $140 ($120 + $20)." — **correct**, but only once k≥10 holds both price facts simultaneously (budget-gated).
- `8fb83627` → "You have finished reading 3 issues of National Geographic." Gold is "Five". **Wrong, stable across all k** — and (see below) granite never extracted the fact that the count reached five.
- `7161e7e2` → "The provided information states that a rotation table assigns specific agents to each shift…" — describes the table generically, never names Admon's specific Sunday day-shift. **Stable across all k.**

### Fact quality: granite vs qwen (verbatim recall hits)

The judge gap is explained by the *content of the extracted facts*, not by
retrieval. Below are the actual top recall hits (`arecall(question, tags=[qid],
max_tokens=512)`) from each bank, copied verbatim.

**8fb83627 — knowledge-update, gold "Five" (the decisive case).** A
knowledge-update question depends on capturing the *latest* state. The user's
reading count moves from "finished 1–3, on issue 4" to "finished five issues"
across sessions. **qwen extracted both states; granite captured only the stale
one.**

qwen (`lme_m_8fb83627`) top hits — the updated count survived:
```
[0] User is reading National Geographic about the Amazon rainforest and indigenous communities, having finished their fifth issue. | Involving: user
[1] User has been reading about the Amazon rainforest, finishing five issues of National Geographic with great articles on the region. | Involving: user
[2] User is filling in the National Geographic section of a spreadsheet, having finished three issues and currently on the fourth. | When: 2023-04-20 | Involving: user
[3] User enjoys The New Yorker for fiction and National Geographic for science and nature articles, having finished the third issue and currently on the fourth | When: 2023-04-20 | Involving: user
```

granite (`lme_m_8fb83627_granite`) top hits — only the stale "issue 4" state, no "five":
```
[0] User reads news, science, and entertainment magazines; newspapers include The New York Times and The Daily News; enjoys The New Yorker for fiction and National Geographic for science and nature, currently on issue 4 after finishing issues 1-3. | When: 2023-04-20 | Involving: user
[1] Assistant filled in the National Geographic section of the spreadsheet, marking Issues 1‑3 as 'Read' and Issue 4 as 'In Progress' (currently being read). | When: 2023-04-20 | Involving: Assistant ... and User | To help the user organize their National Geographic magazine reading progress in a spreadsheet.
[2] National Geographic Kids and science kits encourage curiosity and exploration of scientific concepts.
[3] National Geographic's Yanomami People provides a comprehensive visual and textual overview of Yanomami culture, history, and the ongoing struggles they face.
```
Granite's recall is dominated by topical/world facts about National Geographic
content (Yanomami, Amazon reading lists) and tops out at the obsolete
"issue 4 / 1–3 read" snapshot. The later "finished five issues" update is
absent from the bank, so no value of k can recover it → the answer is locked at
"3 issues". This is an **extraction-recency failure**: granite dropped the
state-update fact that defines the question.

**1c549ce4 — multi-session, gold "$140" ($120 cover + $20 spray) — granite
matches qwen but less efficiently.** Both banks hold both prices, but granite's
facts are **verbose and duplicated** — nearly every fact appears twice, once
bare and once with a `| Involving: … | To …` annotation tail — which inflates
token cost and separates the two price facts so they only co-occur at k≥10.

granite (`lme_m_1c549ce4_granite`) — the $20 and $120 facts, each emitted twice:
```
[0] User previously used a $20 detailing spray from Amazon that effectively removed tar and bug stains from the car's paint.
[1] Past positive experience: The user previously used a $20 detailing spray from Amazon that effectively removed tar and bug stains from the car's paint. | Involving: user | To illustrate the effectiveness of detailing sprays.
...
[6] Insurance may reimburse a $120 waterproof car cover if it is deemed a valid accessory protecting the vehicle's paint and is covered under the policy.
[8] Insurance may reimburse the $120 waterproof car cover if it's deemed a valid accessory to protect the vehicle's paint and is covered by the policy. | To determine eligibility ...
```

qwen (`lme_m_1c549ce4`) — terser, both prices inside the top 3, no near-duplicate pairs:
```
[0] User previously purchased a detailing spray from Amazon for $20 that successfully removed tar and bug stains from their car's paint | Involving: user
[1] User bought a $20 detailing spray from Amazon that worked well, removing tar and bug stains | Involving: user
[2] User owns a waterproof car cover that cost $120 and protects car paint from elements | When: 2023-05-26 | Involving: user
```
Both ultimately answer "$140", but granite needs a larger k to assemble both
operands because its duplication wastes budget — a precision/efficiency hit, not
a correctness one.

**7161e7e2 — single-session-assistant, gold "Admon … 8am-4pm Sundays" (granite
only; no qwen bank).** Granite captured the rotation **schema** (the shifts
exist, the 7 agents are listed, the table assigns agents Sun–Sat) but not the
**specific cell value** — no fact states Admon→Sunday→8am-4pm:
```
[0] The shift rotation for GM social media agents is set for a 1-week period from Sunday to Saturday, with each of the 7 agents (Admon, Magdy, Ehab, Sara, Mostafa, Nemr, Adam) working one shift per day and having two days off each week. ...
[1] Assistant created a shift‑rotation sheet for 7 GM social‑media agents with four shifts (8 am‑4 pm, 12 pm‑8 pm, 4 pm‑12 am, 12 am‑8 am) ...
[5] The rotation table assigns specific agents to each shift on each day from Sunday to Saturday. ... | To clearly outline who works which shift on which day.
```
The per-agent-per-day assignments — the table's actual contents — were not
emitted as facts; only its structure was. This is a **granularity loss** on a
tabular source. There is no qwen comparison here (the production qwen bank for
this qid is empty), so it stands as a granite observation rather than an A/B.

**Read.** Retrieval (pure-similarity top-k) is decoupled from end-to-end
correctness: the gold session's facts are in context on nearly every cell, yet
2/3 answers are wrong because of *what granite extracted*. Two distinct
fact-quality deficits show up vs qwen: (1) **recency/state-update loss** — the
defining update fact (`finished five issues`) was never extracted (8fb83627);
(2) **verbosity + near-duplicate facts** that waste the context budget
(1c549ce4). A third, qwen-uncompared, is **granularity loss** on tabular data
(7161e7e2). qwen's facts are consistently terser, deduplicated, and retain the
latest state. *Caveat:* single-run verdicts; per the temp=0 + k-repeat protocol
these are unverified across repeats, but the fact-content differences above are
structural (present/absent in the bank), not judge variance.

## Judge meta-eval (2026-05-27) — judge model is not the lever

To test whether the judge model biases verdicts (and whether we need a stronger
judge to match the paper's GPT-4o), we isolated the judge from answer-generation
variance: generate the predicted answer **once** per qid (fixed answer-model),
then grade that *same* frozen prediction with each candidate judge. Harness:
`eval_harnesses/suites/memory_recall/lme_judge_meta.py` (freezes the prediction,
emits a per-qid verdict-per-judge table + pairwise agreement). Raw:
`.eval_cache/lme_results/judge_meta.{json,md}`.

7 qids weighted to the failure-prone qtypes (preference, knowledge-update, a
nuance temporal). Answer-model `lme-t0` (gemini-2.5-flash @ t0). Judges:
`gemini-2.5-flash`, `gemini-2.5-pro`, `gemini-3-flash-preview` (all @ t0).

**Result: all three judges agreed 7/7 (100% pairwise), all verdicts correct on
inspection.** The earlier "flash is unreliable on `*-preference`" instability
was **answer-generation** variance (same question → different predictions across
runs), not the judge disagreeing on a *fixed* prediction. Once the prediction is
frozen, flash judges as well as pro / 3-flash. So the judge is locked to
`gemini-2.5-flash` (cheapest, ≥ GPT-4o on benchmarks) and the `lme-25pro-t0`
alias is kept available for spot-checks. *Limitation:* judges only diverge on
**borderline** predictions (partial credit, heavy paraphrase, extra-info); this
set produced mostly unambiguous predictions, so 100% agreement means "no
difference on clear cases," not "no difference ever." A borderline-engineered
set + hand-labeled `human` column would be needed for a paper-style agreement
number.

## Answer-model A/B (2026-05-27) — gemini-3-flash answerer

Full sweep over all 12 baseline banks (5 S + 7 M) with the answerer swapped to
**`gemini-3-flash-preview` @ t0**, judge held at `gemini-2.5-flash` @ t0,
standard config (`top_k=10`, `max_tokens=512`). Clean A/B vs the
`gemini-2.5-flash`-answerer baseline — only the reader changed. Raw:
`.eval_cache/lme_results/sweep_g3answer_{s,m}.json`.

| tier | n | session_hit% | judge_yes% |
|------|--:|-------------:|-----------:|
| S (g3 answerer) | 5 | 100.0% | 80.0% |
| M (g3 answerer) | 7 | 100.0% | 100.0% |
| **combined** | **12** | **100.0%** | **91.7% (11/12)** |

**Identical to the 2.5-flash-answerer baseline: 11/12, with `1c0ddc50` the sole
failure.** Two confirmations:
- **g3 does not rescue `1c0ddc50`** — still emits the generic "listen to
  podcasts" answer. Consistent with the retrieval-structural diagnosis: the
  history-preference fact never reaches context, so no reader can fix it (unlike
  HyDE-g3, which changed the recall *vector*). A fifth failed intervention by
  proxy — upgrading the reader doesn't touch a retrieval miss.
- **g3 corroborates the granite finding from the reader side**: on the
  qwen-extracted `8fb83627` bank (gold "Five"), g3 answers "finished five
  issues" → yes. The facts are answerable; granite's failure on its own
  `8fb83627` bank was extraction, not the reader.

g3 answers are qualitatively richer (operands, dates, formatting) but
correctness is unchanged. Net: swapping the answerer to gemini-3-flash neither
gains nor regresses on the baseline set.

## Findings

- **Session retrieval is solved at this scale.** Every bank — including M-tier at ~470 sessions / 10× haystack — surfaced at least one gold-aligned session in the top-10. The bottleneck for accuracy is what comes *after* session-level recall.
- **One reproducible failure mode (1c0ddc50, single-session-preference).** Diagnosed in `memory/project/decisions/2026-05-15-lme-1c0ddc50-retrieval-ranking-failure.md` as retrieval-ranking, not extraction or synthesis. Specific gold facts (history podcasts: Hardcore History, Lore, The Dollop, Guns Germs and Steel) are in the bank and surface on lexically-aligned queries, but the natural question vector ("activities during commute") has near-zero embedding proximity to those fact vectors. K-sweep showed no recovery at `max_tokens=2048, top_k=20`.
- **The v2a retain-mission A/B did not move the dial on 1c0ddc50.** Re-ingesting under a mission that explicitly asks for paired specific+general descriptors produced 1,613 facts (vs. 1,293 baseline — 25% more) and pulled in different surface text, but the predicted answer reverted to the same generic "true crime / self-improvement" cluster. Failure is structural to the bi-encoder retrieval layer; mission-level prompting is the wrong lever.
- **v2a did not regress the multi-session M-tier case** (`91b15a6e_v2a`, judge=yes). So the variant mission is safe but inert for this failure mode — a clean negative result.
- **M-tier scaled without degradation.** Same precision/recall envelope at 10× haystack on the cases that pass at S-tier. Per-question ingest cost rises ~10× (S ≈ 30–80 min, M ≈ 6+ h on Gemini embeddings), but recall quality is preserved.
- **HyDE query expansion does not fix `1c0ddc50` either (0/5 at temp=0).** Generating a hypothetical answer and recalling on *that* vector instead of the bare question was the highest-ROI candidate intervention (targets the vocab mismatch directly). Probed via `lme_hyde.py` (baseline-vs-HyDE in one pass). At `gemini-2.5-flash` defaults the result was noisy — one early run surfaced the specific gold titles (Hardcore History, Lore, The Dollop) and looked like a fix — but pinned to temp=0 with k=5 repeats, **both baseline and HyDE score 0/5**. The lucky run was sampling variance, not signal. So four interventions now leave `1c0ddc50` intact: v2a retain-mission, v3a verbose, k-sweep, and HyDE. The failure is structural to bi-encoder ranking plus a strict multi-part gold (history **and** avoid-visual **and** not-generic-genres), not a tuning knob. **Generator-dependence caveat:** re-running HyDE with `gemini-3-flash-preview` (vs 2.5-flash) as the answer/judge model scored **1/3** at temp=0 — the one pass came from a hypothetical doc that invented a *specific* named history podcast ("Revolutions / Mike Duncan"), giving the recall vector enough lexical overlap to surface the gold facts; the two generic hypotheticals missed. So HyDE isn't flatly dead — its efficacy hinges on the generator hallucinating the *right specific entity*, which a stronger model does more often. Still too marginal (and 2× recall cost) to adopt. *Future work: k=10 gemini-3 batch for a real rate.*
- **Single-run judge verdicts on this bucket are unstable; scoring needs temperature pinning + k-repeats.** At default sampling, `1c0ddc50`'s baseline verdict flipped across repeats (1 yes / 3 no over four runs) purely from answer-generation variance — same retrieved facts, different answer completeness. Fix: a global gemini `customAlias` (`lme-t0`) pinning `temperature: 0`, selected via `-m lme-t0`. **Caveat:** temp=0 stabilized the *verdict* but not the *answers* — predicted answers still varied run-to-run at temp=0 (serving-layer nondeterminism: batching / MoE routing), so k-repeats remain necessary even with temperature pinned. The single-number-per-question scoring elsewhere in this doc should be read with that variance in mind for the borderline cases.
- **v3a verbose extraction makes `1c0ddc50` strictly worse.** Verbose mode emits longer facts, so fewer fit in the `max_tokens=512` top-K context window (S-tier `n_retrieved` 20 → 7, M-tier 19 → 4). On the passing cases (`1c549ce4` S+M, `91b15a6e` M) the gold sessions still surface and the answer is unchanged. On `1c0ddc50`, the truncation pushes gold-session facts out of top-K entirely — `session_hit` collapses from 100 % to 0 %. So verbose: (a) confirms density isn't the lever for the structural failure (consistent with v2a result), and (b) reveals a new retrieval-side regression mode driven by fact length × context budget. Future ingest-side experiments should re-tune `top_k` or `max_tokens` rather than assume the baseline context budget is comparable across extraction modes.

## Next actions targeted at 1c0ddc50

The diagnosis points at the retrieval-ranking layer specifically. Promising interventions, in order of expected ROI:
1. ~~**HyDE / query expansion**~~ — **tried, 0/5 at temp=0** (see Findings). Hypothetical-answer recall did not move the verdict. Eliminated.
2. **Hybrid retrieval** — add a sparse (BM25/keyword) channel alongside the dense bi-encoder; lexical fallback when embeddings are off-topic. Now the top remaining candidate.
3. **Entity-conditioned recall** — Hindsight already extracts entities per fact (visible in hit metadata) but exposes no `entities=` query filter; an upstream API change would enable post-hoc entity-anchored expansion.
4. Larger embedding model is *probably not* the fix — the gold and question share no topical surface area, and a stronger encoder still encodes the same semantic distance.

## Reproduction

All result JSONs live in `.eval_cache/lme_results/` (gitignored). Re-run any single judge:

```
python -m eval_harnesses.suites.memory_recall.lme_judge \
  --tier {s|m} --qids <qid> --bank-prefix lme_{s|m} [--bank-suffix _v2a] \
  --out .eval_cache/lme_results/<name>.json
```

Source files:
- `eval_harnesses/suites/memory_recall/lme_judge.py` — recall → answer → judge pipeline (gemini ACP subprocess)
- `eval_harnesses/suites/memory_recall/lme_smoke.py` — Hindsight ingest of a tier into one bank per qid with tag isolation
- `eval_harnesses/suites/memory_recall/lme_ingest_queue.py` — multi-bank queue runner used for the M-tier and v2a ingests
- `eval_harnesses/suites/memory_recall/lme_hyde.py` — HyDE probe: runs baseline (recall on question) and HyDE (recall on an LLM-generated hypothetical answer) side-by-side per qid. Pin `--model-answer lme-t0 --model-judge lme-t0` for temp=0 scoring
- `memory/project/plans/lme_application_sprint.md` — sprint context + state
