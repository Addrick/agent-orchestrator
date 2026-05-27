"""A-3: hybrid retrieval (sparse BM25 + dense) over the LME banks.

The production retrieval path is dense-only: `arecall` embeds the query and
does ANN over fact vectors. 1c0ddc50 fails on *vocabulary mismatch* — the
question and the gold session use different words, so their embeddings don't
land near each other even though the gold content is in the bank. This harness
adds a sparse keyword channel and measures whether it (or a hybrid merge)
surfaces the gold session that dense retrieval misses.

Corpus choice: BM25 runs over the bank's **raw session documents**
(`GET /banks/{id}/documents` → `original_text`, where the document `id` *is*
the session_id). That keeps the unit of retrieval identical to the eval's
scoring unit (`session_hit@K`) and avoids needing the extracted fact units,
which aren't listable via the REST surface (only reachable through dense
recall). Dense session rankings are derived by mapping `arecall` hits back to
their source session and de-duplicating in rank order.

Three rankings per qid, all scored by `session_hit@K`:
  - sparse  — BM25 over session text          (no LLM, no embeddings)
  - dense   — arecall → session order          (needs the embedding endpoint)
  - hybrid  — reciprocal-rank fusion of both

The default run is **pure retrieval, zero LLM** (only session_hit metrics).
Pass --judge to additionally answer+judge via the gemini CLI (external), using
each method's top-K session text as context — emits lme_compare-diffable rows.

Standalone except for the Hindsight client + (optional) the gemini judge.

Usage:
    # pure retrieval, all three methods, K sweep
    python -m eval_harnesses.suites.memory_recall.lme_bm25 \\
        --tier s --qids 1c0ddc50,1c549ce4 --bank-prefix lme_s \\
        --out .eval_cache/lme_results/bm25_s.json

    # also answer+judge (needs gemini CLI; primary method = hybrid)
    python -m ...lme_bm25 --tier s --qids 1c0ddc50 --bank-prefix lme_s \\
        --judge --primary hybrid --top-k 5 --out .../bm25_s_judged.json
"""
from __future__ import annotations

import argparse
import asyncio
import math
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from config.global_config import HINDSIGHT_URL
from src.memory.backend.hindsight import HINDSIGHT_API_PREFIX, HindsightRESTClient

from .lme_judge import _hit_session_id, _stream_load_qids
from .lme_smoke import TIER_FILES

K_SWEEP = (1, 3, 5, 10, 20)
RRF_K = 60  # standard reciprocal-rank-fusion constant

# Minimal English stopword set — enough to stop BM25 scoring on filler. Kept
# inline so the harness stays dependency-free.
_STOP = frozenset("""
a an and are as at be been being but by can did do does for from had has have
he her him his how i if in into is it its me my no not of on or our so that the
their them then there these they this to was we were what when where which who
will with would you your
""".split())

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokens(text: str) -> List[str]:
    return [t for t in _TOKEN_RE.findall(text.lower())
            if t not in _STOP and len(t) > 1]


class BM25:
    """Okapi BM25 over a fixed document corpus (k1=1.5, b=0.75)."""

    def __init__(self, docs: List[List[str]], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.docs = docs
        self.N = len(docs)
        self.doc_len = [len(d) for d in docs]
        self.avgdl = (sum(self.doc_len) / self.N) if self.N else 0.0
        self.freqs = [Counter(d) for d in docs]
        df: Counter = Counter()
        for d in docs:
            for term in set(d):
                df[term] += 1
        # BM25 idf with +1 to keep it non-negative for common terms
        self.idf = {t: math.log(1 + (self.N - n + 0.5) / (n + 0.5))
                    for t, n in df.items()}

    def scores(self, query: List[str]) -> List[float]:
        out = [0.0] * self.N
        for term in query:
            idf = self.idf.get(term)
            if idf is None:
                continue
            for i in range(self.N):
                f = self.freqs[i].get(term, 0)
                if not f:
                    continue
                denom = f + self.k1 * (
                    1 - self.b + self.b * self.doc_len[i] / (self.avgdl or 1)
                )
                out[i] += idf * (f * (self.k1 + 1)) / denom
        return out


async def _list_documents(client: HindsightRESTClient, bank: str) -> List[Dict[str, Any]]:
    """Page through GET /documents (list shape: no original_text)."""
    items: List[Dict[str, Any]] = []
    offset, limit = 0, 200
    while True:
        r = await client._request(
            "GET",
            f"{HINDSIGHT_API_PREFIX}/banks/{bank}/documents"
            f"?limit={limit}&offset={offset}",
        )
        batch = r.get("items", [])
        items.extend(batch)
        total = r.get("total", len(items))
        offset += len(batch)
        if not batch or offset >= total:
            break
    return items


async def _session_texts(client: HindsightRESTClient, bank: str) -> Dict[str, str]:
    """Map session_id -> original_text. The list endpoint omits original_text,
    so each document is fetched individually via GET /documents/{id}."""
    listed = await _list_documents(client, bank)
    texts: Dict[str, str] = {}
    for d in listed:
        did = d["id"]
        full = await client._request(
            "GET", f"{HINDSIGHT_API_PREFIX}/banks/{bank}/documents/{did}"
        )
        texts[did] = full.get("original_text", "") or ""
    return texts


def _rank(session_scores: List[Tuple[str, float]]) -> List[str]:
    """Session ids ordered by descending score, ties broken by id."""
    return [s for s, sc in sorted(session_scores, key=lambda x: (-x[1], x[0]))
            if sc > 0]


def _rrf(rankings: List[List[str]]) -> List[str]:
    """Reciprocal-rank fusion of several session-id rankings."""
    fused: Counter = Counter()
    for ranking in rankings:
        for rank, sid in enumerate(ranking):
            fused[sid] += 1.0 / (RRF_K + rank)
    return [sid for sid, _ in fused.most_common()]


def _hit_at_k(ranking: List[str], gold: set, k: int) -> bool:
    return bool(set(ranking[:k]) & gold)


def _sweep(ranking: List[str], gold: set) -> Dict[str, bool]:
    return {f"hit@{k}": _hit_at_k(ranking, gold, k) for k in K_SWEEP}


async def _run_one(
    client: HindsightRESTClient, q: Dict[str, Any], bank: str,
    primary: str, top_k: int,
) -> Dict[str, Any]:
    qid = q["question_id"]
    gold = set(q["answer_session_ids"])
    question = q["question"]

    # --- sparse: BM25 over raw session documents ---
    texts = await _session_texts(client, bank)
    sids = list(texts)
    tokenized = [_tokens(texts[s]) for s in sids]
    bm25 = BM25(tokenized)
    sparse_scores = list(zip(sids, bm25.scores(_tokens(question))))
    sparse_rank = _rank(sparse_scores)

    # --- dense: arecall hits → session order ---
    dense_rank: List[str] = []
    dense_error: Optional[str] = None
    try:
        hits = await client.arecall(bank, question, tags=[qid], max_tokens=4000)
        seen: Dict[str, None] = {}
        for h in hits or []:
            sid = _hit_session_id(h)
            if sid and sid not in seen:
                seen[sid] = None
        dense_rank = list(seen)
    except Exception as e:  # noqa: BLE001 — record, keep sparse result
        dense_error = repr(e)[:200]

    # --- hybrid: RRF of whatever rankings we have ---
    inputs = [r for r in (sparse_rank, dense_rank) if r]
    hybrid_rank = _rrf(inputs) if inputs else []

    methods = {
        "sparse": {"ranking": sparse_rank[:20], "sweep": _sweep(sparse_rank, gold)},
        "dense": {"ranking": dense_rank[:20], "sweep": _sweep(dense_rank, gold),
                  "error": dense_error},
        "hybrid": {"ranking": hybrid_rank[:20], "sweep": _sweep(hybrid_rank, gold)},
    }
    prim_rank = {"sparse": sparse_rank, "dense": dense_rank,
                 "hybrid": hybrid_rank}[primary]

    return {
        "qid": qid,
        "question_type": q["question_type"],
        "bank": bank,
        "question": question,
        "gold_session_ids": sorted(gold),
        "n_docs": len(sids),
        "primary_method": primary,
        # flat fields shaped for lme_compare (primary method, at top_k)
        "session_hit_any": _hit_at_k(prim_rank, gold, top_k),
        "session_hit_rate": (
            len(set(prim_rank[:top_k]) & gold) / max(len(gold), 1)
        ),
        "retrieved_session_ids": prim_rank[:top_k],
        "n_facts_retrieved": len(prim_rank),
        "top_k": top_k,
        "methods": methods,
    }


def _print_summary(results: List[Dict[str, Any]]) -> None:
    print("\n=== session_hit@K by method ===", file=sys.stderr)
    hdr = "qid".ljust(12) + "method".ljust(9) + "".join(
        f"@{k}".rjust(6) for k in K_SWEEP)
    print(hdr, file=sys.stderr)
    print("-" * len(hdr), file=sys.stderr)
    for r in results:
        for m in ("sparse", "dense", "hybrid"):
            sw = r["methods"][m]["sweep"]
            cells = "".join(("Y" if sw[f"hit@{k}"] else "·").rjust(6)
                            for k in K_SWEEP)
            print(r["qid"][:11].ljust(12) + m.ljust(9) + cells, file=sys.stderr)
        print(file=sys.stderr)


async def main(
    tier: str, qids: List[str], bank_prefix: str, out: Path,
    bank_suffix: str = "", primary: str = "hybrid", top_k: int = 5,
    judge: bool = False, model_answer: str = "gemini-2.5-flash",
    model_judge: str = "gemini-2.5-flash",
) -> int:
    src = TIER_FILES[tier]
    if not src.exists():
        raise SystemExit(f"missing {tier} JSON at {src}")
    print(f"Streaming {tier} for {len(qids)} qids...", file=sys.stderr)
    by_id = _stream_load_qids(src, set(qids))
    missing = [q for q in qids if q not in by_id]
    if missing:
        raise SystemExit(f"qids not in {tier}: {missing}")

    client = HindsightRESTClient(HINDSIGHT_URL, timeout=120.0)
    results: List[Dict[str, Any]] = []
    for qid in qids:
        q = by_id[qid]
        bank = f"{bank_prefix}_{qid}{bank_suffix}"
        print(f"\n--- {qid} [{q['question_type']}] bank={bank} ---", file=sys.stderr)
        r = await _run_one(client, q, bank, primary, top_k)
        if judge:
            await _judge_methods(client, q, r, top_k, model_answer, model_judge)
        results.append(r)

    out.parent.mkdir(parents=True, exist_ok=True)
    import json
    out.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nWrote {out}", file=sys.stderr)
    _print_summary(results)
    return 0


async def _judge_methods(
    client: HindsightRESTClient, q: Dict[str, Any], row: Dict[str, Any],
    top_k: int, model_answer: str, model_judge: str,
) -> None:
    """Optional: answer+judge each method's top-K session text via gemini.

    Imported lazily so the pure-retrieval path needs no gemini CLI on PATH.
    """
    from .lme_judge import (
        ANSWER_PROMPT, JUDGE_PROMPT, _gemini_call, _parse_verdict,
    )
    bank = row["bank"]
    docs = await _session_texts(client, bank)
    for m, mdata in row["methods"].items():
        top_sids = mdata["ranking"][:top_k]
        context = "\n\n".join(docs.get(s, "") for s in top_sids)
        try:
            predicted = _gemini_call(
                ANSWER_PROMPT.format(context=context, question=q["question"]),
                model=model_answer,
            )
            judge_raw = _gemini_call(
                JUDGE_PROMPT.format(question=q["question"], gold=q["answer"],
                                    predicted=predicted or "(empty)"),
                model=model_judge,
            )
            label = _parse_verdict(judge_raw)
        except Exception as e:  # noqa: BLE001
            predicted, judge_raw, label = "", repr(e)[:200], None
        mdata["predicted_answer"] = predicted
        mdata["judge_label"] = label
        mdata["judge_verdict"] = ("yes" if label is True else
                                  "no" if label is False else "unparseable")
    # promote primary method's verdict to the flat row for lme_compare
    prim = row["methods"][row["primary_method"]]
    row["judge_label"] = prim.get("judge_label")
    row["judge_verdict"] = prim.get("judge_verdict", "unparseable")
    row["predicted_answer"] = prim.get("predicted_answer", "")
    row["answer_model"] = model_answer
    row["judge_model"] = model_judge


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tier", choices=list(TIER_FILES.keys()), default="s")
    ap.add_argument("--qids", required=True, help="comma-separated question_ids")
    ap.add_argument("--bank-prefix", required=True,
                    help="bank name is f'{prefix}_{qid}{suffix}'")
    ap.add_argument("--bank-suffix", default="")
    ap.add_argument("--primary", choices=("sparse", "dense", "hybrid"),
                    default="hybrid", help="method promoted to flat row fields")
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--judge", action="store_true",
                    help="also answer+judge each method via gemini CLI")
    ap.add_argument("--model-answer", default="gemini-2.5-flash")
    ap.add_argument("--model-judge", default="gemini-2.5-flash")
    ap.add_argument("--out", type=Path,
                    default=Path(".eval_cache/lme_results/bm25.json"))
    args = ap.parse_args()
    qids_list = [q.strip() for q in args.qids.split(",") if q.strip()]
    raise SystemExit(asyncio.run(main(
        args.tier, qids_list, args.bank_prefix, args.out, args.bank_suffix,
        args.primary, args.top_k, args.judge, args.model_answer, args.model_judge,
    )))
