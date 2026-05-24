# LongMemEval results — Hindsight backend

Per-question evaluation of Hindsight memory recall on the LongMemEval V1 cleaned dataset (`xiaowu0162/longmemeval-cleaned`). 14 banks total: 5 S-tier baseline, 7 M-tier baseline, 2 variant ("v2a") banks ingested under a modified retain mission.

All judging via local `gemini` CLI subprocess (paid OAuth tier) over the ACP transport — same answer-model and judge-model per run for self-consistency.

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
| **All** | **17** | **94.1%** | **82.4%** |

Per-qtype across all rows:

| qtype | n | judge_yes% |
|-------|--:|-----------:|
| multi-session | 6 | 100.0% |
| temporal-reasoning | 4 | 100.0% |
| knowledge-update | 3 | 100.0% |
| single-session-assistant | 1 | 100.0% |
| single-session-preference | 3 | 0.0% |

## Findings

- **Session retrieval is solved at this scale.** Every bank — including M-tier at ~470 sessions / 10× haystack — surfaced at least one gold-aligned session in the top-10. The bottleneck for accuracy is what comes *after* session-level recall.
- **One reproducible failure mode (1c0ddc50, single-session-preference).** Diagnosed in `memory/project/decisions/2026-05-15-lme-1c0ddc50-retrieval-ranking-failure.md` as retrieval-ranking, not extraction or synthesis. Specific gold facts (history podcasts: Hardcore History, Lore, The Dollop, Guns Germs and Steel) are in the bank and surface on lexically-aligned queries, but the natural question vector ("activities during commute") has near-zero embedding proximity to those fact vectors. K-sweep showed no recovery at `max_tokens=2048, top_k=20`.
- **The v2a retain-mission A/B did not move the dial on 1c0ddc50.** Re-ingesting under a mission that explicitly asks for paired specific+general descriptors produced 1,613 facts (vs. 1,293 baseline — 25% more) and pulled in different surface text, but the predicted answer reverted to the same generic "true crime / self-improvement" cluster. Failure is structural to the bi-encoder retrieval layer; mission-level prompting is the wrong lever.
- **v2a did not regress the multi-session M-tier case** (`91b15a6e_v2a`, judge=yes). So the variant mission is safe but inert for this failure mode — a clean negative result.
- **M-tier scaled without degradation.** Same precision/recall envelope at 10× haystack on the cases that pass at S-tier. Per-question ingest cost rises ~10× (S ≈ 30–80 min, M ≈ 6+ h on Gemini embeddings), but recall quality is preserved.
- **v3a verbose extraction makes `1c0ddc50` strictly worse.** Verbose mode emits longer facts, so fewer fit in the `max_tokens=512` top-K context window (S-tier `n_retrieved` 20 → 7, M-tier 19 → 4). On the passing cases (`1c549ce4` S+M, `91b15a6e` M) the gold sessions still surface and the answer is unchanged. On `1c0ddc50`, the truncation pushes gold-session facts out of top-K entirely — `session_hit` collapses from 100 % to 0 %. So verbose: (a) confirms density isn't the lever for the structural failure (consistent with v2a result), and (b) reveals a new retrieval-side regression mode driven by fact length × context budget. Future ingest-side experiments should re-tune `top_k` or `max_tokens` rather than assume the baseline context budget is comparable across extraction modes.

## Next actions targeted at 1c0ddc50

The diagnosis points at the retrieval-ranking layer specifically. Promising interventions, in order of expected ROI:
1. **HyDE / query expansion** — one extra LLM call to generate a hypothetical answer, then embed that for recall. Targets vocabulary mismatch directly; doubles per-recall cost.
2. **Hybrid retrieval** — add a sparse (BM25/keyword) channel alongside the dense bi-encoder; lexical fallback when embeddings are off-topic.
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
- `memory/project/plans/lme_application_sprint.md` — sprint context + state
