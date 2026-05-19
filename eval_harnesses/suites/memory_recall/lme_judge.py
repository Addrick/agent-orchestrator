"""LongMemEval end-to-end scoring via gemini-cli.

Per question, runs:
    1. arecall(bank, question, tags=[qid]) — top-K facts from Hindsight
    2. Answer-generation LLM (gemini CLI, headless `-p`) given question + context
    3. Judge LLM (gemini CLI) grades predicted vs gold answer → boolean

Both LLM calls go through the local `gemini` CLI as a subprocess so billing
hits the user's paid OAuth tier rather than the API-key free quota.
Substituted for the paper's GPT-4o judge — model name is recorded in the
result file for reproducibility.

Inputs:
    --tier {oracle,s,m}          dataset tier
    --qids q1,q2,...             question_ids to score
    --bank-prefix PREFIX         bank name is f"{PREFIX}_{qid}"
    --model-answer NAME          gemini model for answer generation
    --model-judge NAME           gemini model for judging
    --top-k INT                  number of facts to include as context
    --out PATH                   results JSON (overwrites)

Output JSON per question:
    {
      "qid": ..., "question_type": ..., "bank": ...,
      "retrieved_session_ids": [...], "session_hit_rate": float,
      "predicted_answer": str,
      "judge_verdict": "yes"|"no"|"unparseable",
      "judge_raw": str,
      "judge_label": bool,
      "judge_model": str, "answer_model": str,
      "n_facts_retrieved": int,
    }

Aggregated summary printed to stdout at end.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import queue
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

from config.global_config import HINDSIGHT_URL
from src.memory.backend.hindsight import HindsightRESTClient

from .lme_smoke import TIER_FILES


def _stream_load_qids(path: Path, qids: set[str]) -> Dict[str, Dict[str, Any]]:
    """Stream a LongMemEval JSON array and return only items whose
    question_id is in `qids`. Avoids the 2.6GB-into-RAM spike that
    `json.loads(read_text())` causes on the m-tier file.
    """
    decoder = json.JSONDecoder()
    out: Dict[str, Dict[str, Any]] = {}
    buf = ""
    pos = 0  # index into buf
    with open(path, "r", encoding="utf-8") as f:
        # Skip leading whitespace + opening '['
        while True:
            ch = f.read(1)
            if not ch:
                raise ValueError(f"empty file: {path}")
            if ch == "[":
                break
            if not ch.isspace():
                raise ValueError(f"expected '[', got {ch!r}")
        while len(out) < len(qids):
            # Skip whitespace and commas in buf at pos
            while pos < len(buf) and buf[pos] in " \t\n\r,":
                pos += 1
            # Trim consumed prefix to bound memory
            if pos > 65536:
                buf = buf[pos:]
                pos = 0
            # Need more data?
            if pos >= len(buf) or buf[pos] != "{":
                chunk = f.read(1 << 20)  # 1MB
                if not chunk:
                    break  # EOF
                buf += chunk
                continue
            try:
                obj, end = decoder.raw_decode(buf, pos)
            except json.JSONDecodeError:
                # incomplete object — pull more
                chunk = f.read(1 << 20)
                if not chunk:
                    raise
                buf += chunk
                continue
            pos = end
            qid = obj.get("question_id")
            if qid in qids:
                out[qid] = obj
    return out


# Modeled on LongMemEval autoeval. Plain accuracy judge — does the predicted
# answer convey the gold answer's content? Numeric / list answers count as
# correct only if every gold element is present (per paper's strict scoring).
JUDGE_PROMPT = """You are grading a facts-retrieval assistant's answer.

QUESTION:
{question}

GOLD ANSWER:
{gold}

PREDICTED ANSWER:
{predicted}

Does the predicted answer correctly convey the gold answer?
- For factual/numeric/list answers: the prediction must contain every fact
  in the gold answer (no missing items, no contradictions). Paraphrasing is
  fine. Extra info beyond the gold is fine if it doesn't contradict.
- For abstention questions where gold says the answer isn't in the history:
  the prediction is correct only if it abstains or says it doesn't know.

Reply with exactly one token: "yes" if correct, "no" if not. Do not add
explanation, punctuation, or any other text."""


ANSWER_PROMPT = """{context}

QUESTION: {question}
Answer the question concisely using only the data provided above."""


_GEMINI_BIN: Optional[str] = None
_GEMINI_CWD: Optional[str] = None


def _gemini_bin() -> str:
    global _GEMINI_BIN
    if _GEMINI_BIN is None:
        path = shutil.which("gemini")
        if not path:
            raise RuntimeError("`gemini` CLI not on PATH")
        _GEMINI_BIN = path
    return _GEMINI_BIN


def _gemini_cwd() -> str:
    """Empty tmp dir as cwd so the CLI doesn't auto-load workspace context.

    Running from the repo root makes gemini ingest GEMINI.md / source files
    and treat the embedded prompt as session metadata. Running from an empty
    dir + --skip-trust keeps it as a chat endpoint. We also write a
    .geminiignore and a minimal system.md to ensure isolation.
    """
    global _GEMINI_CWD
    if _GEMINI_CWD is None:
        _GEMINI_CWD = tempfile.mkdtemp(prefix="lme_gemini_")
        # Write .geminiignore to the temp dir to prevent context injection
        # from parent directories or unintended auto-loading.
        (Path(_GEMINI_CWD) / ".geminiignore").write_text("*", encoding="utf-8")
        # Write a neutral system.md to override the default CLI persona.
        (Path(_GEMINI_CWD) / "system.md").write_text(
            "You are a facts-retrieval assistant. Use only provided context.",
            encoding="utf-8"
        )
    return _GEMINI_CWD


# Markers that indicate the CLI's agent persona leaked instead of an
# actual answer being produced (workspace introspection, capability
# disclaimers, request-for-input refusals). Retry on detection.
_AGENT_LEAK_MARKERS = (
    "gemini cli",
    "session_context",
    "gemini.md",
    "memory.md",
    "i am ready",
    "i'm ready",
    "i do not have",
    "please provide",
    "workspace is empty",
    "plan mode",
    "i am currently",
)


def _looks_like_agent_leak(out: str) -> bool:
    low = out.lower()
    return any(m in low for m in _AGENT_LEAK_MARKERS)


class GeminiACPClient:
    """Persistent connection to gemini --acp for low-latency multi-prompting."""

    def __init__(self, model: str, cwd: str):
        self.model = model
        self.cwd = cwd
        self.responses: queue.Queue[dict] = queue.Queue()
        self.id_counter = 0
        self.current_response_text: list[str] = []
        self.session_id: Optional[str] = None

        env = os.environ.copy()
        env["GEMINI_SYSTEM_MD"] = str(Path(cwd) / "system.md")

        self.proc = subprocess.Popen(
            [_gemini_bin(), "--acp", "-m", model, "--skip-trust"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            cwd=cwd,
            env=env,
        )

        # Start background reader
        threading.Thread(target=self._reader, daemon=True).start()

        # Protocol Handshake
        self._call("initialize", {"protocolVersion": 1, "capabilities": {}})
        # Start session
        resp = self._call("session/new", {"cwd": cwd, "mcpServers": []})
        self.session_id = resp["result"]["sessionId"]

    def _reader(self):
        for line in self.proc.stdout:
            if not line.strip():
                continue
            try:
                data = json.loads(line)
                if data.get("method") == "session/update":
                    upd = data["params"].get("update", {})
                    if upd.get("sessionUpdate") == "agent_message_chunk":
                        text = upd.get("content", {}).get("text", "")
                        self.current_response_text.append(text)
                self.responses.put(data)
            except json.JSONDecodeError:
                pass

    def _call(self, method: str, params: dict, timeout: float = 300.0) -> dict:
        self.id_counter += 1
        msg = {"jsonrpc": "2.0", "id": self.id_counter, "method": method, "params": params}
        self.proc.stdin.write(json.dumps(msg) + "\n")
        self.proc.stdin.flush()

        start = time.monotonic()
        while time.monotonic() - start < timeout:
            try:
                resp = self.responses.get(timeout=1.0)
                if resp.get("id") == self.id_counter:
                    if "error" in resp:
                        raise RuntimeError(f"ACP Error: {resp['error']}")
                    return resp
            except queue.Empty:
                continue
        raise TimeoutError(f"ACP call '{method}' timed out after {timeout}s")

    def ask(self, prompt: str) -> str:
        self.current_response_text = []
        self._call("session/prompt", {
            "sessionId": self.session_id,
            "prompt": [{"type": "text", "text": prompt}]
        })
        return "".join(self.current_response_text).strip()

    def close(self):
        try:
            self.proc.terminate()
            self.proc.wait(timeout=5)
        except:
            self.proc.kill()


_CLIENT_CACHE: dict[str, GeminiACPClient] = {}


def _gemini_call(
    prompt: str, model: str, timeout: float = 300.0, max_retries: int = 3,
) -> str:
    """One-shot text generation via persistent ACP client.

    Maintains a pool of clients (one per model) to avoid process startup
    overhead. Retries on 'agent leak' markers as a fallback.
    """
    if model not in _CLIENT_CACHE:
        _CLIENT_CACHE[model] = GeminiACPClient(model, _gemini_cwd())
    
    client = _CLIENT_CACHE[model]
    last = ""
    for attempt in range(max_retries):
        out = client.ask(prompt)
        if not _looks_like_agent_leak(out):
            return out
        last = out
        print(f"  [gemini retry {attempt+1}/{max_retries}] agent-leak in ACP: {out[:80]!r}", file=sys.stderr)
        # On leak, we might want to reset the session or restart the client,
        # but for now we just try again (ACP often self-corrects on next turn).
    return last


def _parse_verdict(raw: str) -> Optional[bool]:
    head = (raw or "").strip().lower()
    # Trim trailing punctuation / explanation, take first token.
    head = head.split()[0] if head.split() else ""
    head = head.rstrip(".,!?:;")
    if head in ("yes", "y", "true", "correct"):
        return True
    if head in ("no", "n", "false", "incorrect"):
        return False
    return None


def _fact_text(hit: Dict[str, Any]) -> str:
    # Hindsight recall hits vary in shape; try common keys.
    for k in ("text", "content", "fact_text", "snippet"):
        v = hit.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()
    # Some shapes nest under "fact" / "memory"
    for outer in ("fact", "memory"):
        inner = hit.get(outer) or {}
        if isinstance(inner, dict):
            for k in ("text", "content"):
                v = inner.get(k)
                if isinstance(v, str) and v.strip():
                    return v.strip()
    return ""


def _hit_session_id(hit: Dict[str, Any]) -> Optional[str]:
    sid = (hit.get("source") or {}).get("document_id")
    if sid:
        return sid
    md = hit.get("metadata") or {}
    return md.get("session_id")


async def _score_one(
    client: HindsightRESTClient, q: Dict[str, Any], bank: str,
    top_k: int, max_tokens: int, model_answer: str, model_judge: str,
) -> Dict[str, Any]:
    qid = q["question_id"]
    gold_sessions = set(q["answer_session_ids"])

    t0 = time.monotonic()
    hits = await client.arecall(
        bank, q["question"], tags=[qid], max_tokens=max_tokens,
    )
    t_recall = time.monotonic() - t0
    hits = hits or []
    top = hits[:top_k]

    retrieved_sids = [s for s in (_hit_session_id(h) for h in top) if s]
    unique_sids = list(dict.fromkeys(retrieved_sids))
    sess_hit_rate = (
        len(set(unique_sids) & gold_sessions) / max(len(gold_sessions), 1)
    )

    context = "\n".join(f"- {_fact_text(h)}" for h in top if _fact_text(h))
    ans_prompt = ANSWER_PROMPT.format(context=context, question=q["question"])

    t0 = time.monotonic()
    try:
        predicted = _gemini_call(ans_prompt, model=model_answer)
    except Exception as e:
        predicted = ""
        ans_err = str(e)[:200]
    else:
        ans_err = None
    t_answer = time.monotonic() - t0

    judge_prompt = JUDGE_PROMPT.format(
        question=q["question"], gold=q["answer"], predicted=predicted or "(empty)",
    )
    t0 = time.monotonic()
    try:
        judge_raw = _gemini_call(judge_prompt, model=model_judge)
    except Exception as e:
        judge_raw = ""
        judge_err = str(e)[:200]
    else:
        judge_err = None
    t_judge = time.monotonic() - t0

    verdict = _parse_verdict(judge_raw)
    return {
        "qid": qid,
        "question_type": q["question_type"],
        "bank": bank,
        "question": q["question"],
        "gold_answer": q["answer"],
        "n_facts_retrieved": len(hits),
        "top_k": top_k,
        "retrieved_session_ids": unique_sids,
        "gold_session_ids": sorted(gold_sessions),
        "session_hit_rate": sess_hit_rate,
        "session_hit_any": bool(set(unique_sids) & gold_sessions),
        "predicted_answer": predicted,
        "answer_error": ans_err,
        "judge_raw": judge_raw,
        "judge_error": judge_err,
        "judge_verdict": (
            "yes" if verdict is True else
            "no" if verdict is False else
            "unparseable"
        ),
        "judge_label": verdict,  # True/False/None
        "answer_model": model_answer,
        "judge_model": model_judge,
        "timings_s": {
            "recall": round(t_recall, 2),
            "answer": round(t_answer, 2),
            "judge": round(t_judge, 2),
        },
    }


def _print_summary(results: List[Dict[str, Any]]) -> None:
    by_type: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in results:
        by_type[r["question_type"]].append(r)

    print("\n=== Per-type scoring ===")
    print(f"{'qtype':<28} {'n':>3} {'sess_hit%':>10} {'judge_yes%':>11} "
          f"{'unparse':>8}")
    for qt, rs in sorted(by_type.items()):
        n = len(rs)
        sess_hits = sum(1 for r in rs if r["session_hit_any"]) / n
        ylabels = [r["judge_label"] for r in rs]
        n_yes = sum(1 for v in ylabels if v is True)
        n_unp = sum(1 for v in ylabels if v is None)
        print(f"{qt:<28} {n:>3} {sess_hits*100:>9.1f}% "
              f"{n_yes/n*100:>10.1f}% {n_unp:>8}")

    n = len(results)
    overall_sess = sum(1 for r in results if r["session_hit_any"]) / n
    overall_yes = sum(1 for r in results if r["judge_label"] is True) / n
    overall_unp = sum(1 for r in results if r["judge_label"] is None)
    print(f"\nOVERALL  n={n}  sess_hit={overall_sess*100:.1f}%  "
          f"judge_yes={overall_yes*100:.1f}%  unparseable={overall_unp}")


async def main(
    tier: str, qids: List[str], bank_prefix: str,
    model_answer: str, model_judge: str, top_k: int, max_tokens: int,
    out: Path, bank_suffix: str = "",
) -> int:
    src = TIER_FILES[tier]
    if not src.exists():
        raise SystemExit(f"missing {tier} JSON at {src}")
    print(f"Streaming {tier} for {len(qids)} qids...", file=sys.stderr)
    by_id = _stream_load_qids(src, set(qids))
    missing = [q for q in qids if q not in by_id]
    if missing:
        raise SystemExit(f"qids not in {tier}: {missing}")

    client = HindsightRESTClient(HINDSIGHT_URL, timeout=300.0)
    results: List[Dict[str, Any]] = []
    for qid in qids:
        q = by_id[qid]
        bank = f"{bank_prefix}_{qid}{bank_suffix}"
        print(f"\n--- {qid} [{q['question_type']}] bank={bank} ---", file=sys.stderr)
        r = await _score_one(
            client, q, bank, top_k, max_tokens, model_answer, model_judge,
        )
        results.append(r)
        flag_sess = "OK" if r["session_hit_any"] else "MISS"
        verdict = r["judge_verdict"]
        print(
            f"  sess={flag_sess} judge={verdict}  "
            f"pred={r['predicted_answer'][:100].replace(chr(10),' ')!r}",
            file=sys.stderr,
        )

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nWrote {out}", file=sys.stderr)
    _print_summary(results)
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tier", choices=list(TIER_FILES.keys()), default="s")
    ap.add_argument("--qids", required=True,
                    help="comma-separated question_ids")
    ap.add_argument("--bank-prefix", required=True,
                    help="bank name is f'{prefix}_{qid}'")
    ap.add_argument("--model-answer", default="gemini-2.5-flash")
    ap.add_argument("--model-judge", default="gemini-2.5-flash")
    # Tightened defaults: max_tokens=512 caps the recall budget so the bank
    # can't dump its entire fact-set into context. top_k=10 caps the post-
    # recall slice for the answerer/judge. Either alone is a real constraint;
    # together they force the system to actually rank.
    ap.add_argument("--max-tokens", type=int, default=512,
                    help="Hindsight recall budget (tokens). Lower = harsher precision.")
    ap.add_argument("--top-k", type=int, default=10,
                    help="cap facts shown to answerer + counted in retrieval slot metrics")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--bank-suffix", default="",
                    help="appended to bank name (e.g. '_v2a' for variant)")
    args = ap.parse_args()
    qids_list = [q.strip() for q in args.qids.split(",") if q.strip()]
    raise SystemExit(asyncio.run(main(
        args.tier, qids_list, args.bank_prefix,
        args.model_answer, args.model_judge, args.top_k, args.max_tokens,
        args.out, bank_suffix=args.bank_suffix,
    )))
