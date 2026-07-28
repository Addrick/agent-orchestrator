#!/usr/bin/env python3
"""kcpp-progress — expose KoboldCPP's prompt-ingestion progress over HTTP (DP-311).

KoboldCPP reports per-batch prefill progress ONLY on stdout:

    Processing Prompt [BATCH] (2048 / 24310 tokens)\r
    Generating (17 / 400 tokens)\r

`/api/extra/perf` cannot give you this: every `last_*` field there is frozen for
the duration of a run and only updates on completion. So a live ingestion bar has
exactly one source — the log — and the engine lives on a different host than the
model. This sidecar runs next to KoboldCPP, tails its log, and serves the parsed
counters so the engine can proxy them.

SECURITY: this service is deliberately incapable of leaking prompt content. It
never stores or returns raw log lines — only integers, a phase enum, and
timestamps, all extracted through strict anchored regexes. Anything unmatched is
discarded immediately. Do not add a "last lines" debug endpoint; the log carries
generated text.

Run:
    kcpp_progress.py --log /var/log/kobold/current.log --port 5011
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import re
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict

logger = logging.getLogger("kcpp_progress")

# KCPP writes progress with a CR and no LF, so a reader must split on both. The
# bracketed tag varies by backend path (BLAS / BATCH / no tag at all).
RE_PROMPT = re.compile(r"Processing Prompt\s*(?:\[[^\]]*\]\s*)?\((\d+)\s*/\s*(\d+)\s+tokens\)")
RE_GENERATE = re.compile(r"Generating\s*\((\d+)\s*/\s*(\d+)\s+tokens\)")
# End-of-run markers. KCPP prints a CtxLimit summary after every completed
# generation; the EOS notice appears only when the model emitted a stop token.
RE_DONE = re.compile(r"CtxLimit:|EOS token triggered|Generation Aborted")

PHASE_IDLE = "idle"
PHASE_PREFILL = "prefill"
PHASE_GENERATE = "generate"

# A run that stops emitting lines for this long is treated as over. The sidecar
# cannot see KCPP's internal state, only its output, so a crashed or aborted run
# would otherwise leave the UI showing "prefill 8192/24310" forever.
STALE_AFTER_S = 30.0


class ProgressState:
    """Last-known ingestion/generation counters. Guarded by a lock: the tailer
    thread writes, HTTP handler threads read."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._phase = PHASE_IDLE
        self._processed = 0
        self._total = 0
        self._gen = 0
        self._gen_total = 0
        self._updated_at = 0.0
        self._runs = 0

    def note_prompt(self, processed: int, total: int) -> None:
        with self._lock:
            # A new run's first batch is (0 / N); count runs on that edge so a
            # consumer can tell "still the same prefill" from "a new one".
            if self._phase != PHASE_PREFILL or processed < self._processed:
                self._runs += 1
            self._phase = PHASE_PREFILL
            self._processed, self._total = processed, total
            self._gen = self._gen_total = 0
            self._updated_at = time.time()

    def note_generate(self, done: int, total: int) -> None:
        with self._lock:
            self._phase = PHASE_GENERATE
            # Prefill finished implicitly — KCPP does not print a final
            # (N / N) batch when the last chunk is short.
            if self._total:
                self._processed = self._total
            self._gen, self._gen_total = done, total
            self._updated_at = time.time()

    def note_done(self) -> None:
        with self._lock:
            self._phase = PHASE_IDLE
            self._updated_at = time.time()

    def snapshot(self) -> Dict[str, Any]:
        with self._lock:
            phase, age = self._phase, time.time() - self._updated_at
            if phase != PHASE_IDLE and age > STALE_AFTER_S:
                # Do not report a run as live when its output went silent.
                phase = PHASE_IDLE
            return {
                "phase": phase,
                "processed": self._processed,
                "total": self._total,
                "generated": self._gen,
                "generate_total": self._gen_total,
                "age_s": round(age, 2) if self._updated_at else None,
                "run": self._runs,
                "source": "log",
            }


def tail_log(path: str, state: ProgressState, poll_s: float = 0.25) -> None:
    """Follow `path` forever, feeding matched counters into `state`.

    Handles the file being absent, rotated, or truncated in place (KCPP's log is
    capped and something truncates it — see the infra notes). Reads bytes rather
    than lines because progress records end in CR, so `readline()` would block
    until an unrelated writer emitted an LF — which is the exact 20-30s clumping
    this sidecar exists to avoid.
    """
    fh = None
    inode = None
    buf = ""
    while True:
        try:
            if fh is None:
                if not os.path.exists(path):
                    time.sleep(1.0)
                    continue
                fh = open(path, "r", encoding="utf-8", errors="replace")
                fh.seek(0, os.SEEK_END)  # only care about what happens next
                inode = os.fstat(fh.fileno()).st_ino
                buf = ""

            chunk = fh.read()
            if chunk:
                buf += chunk
                # Split on CR and LF alike; keep the trailing partial record.
                parts = re.split(r"[\r\n]", buf)
                buf = parts.pop()
                if len(buf) > 8192:  # a line this long is not one of ours
                    buf = ""
                for line in parts:
                    _consume(line, state)
                # Also evaluate the still-incomplete tail. KCPP's end-of-run
                # summary ("[hh:mm:ss] CtxLimit:… Generated:8/8 …") is the LAST
                # thing it writes and its newline does not arrive until the next
                # run starts — so waiting for a terminator leaves the phase stuck
                # at `generate` until the staleness timeout, which is exactly the
                # bug this line fixes. Re-consuming the same record when its
                # terminator finally lands is harmless: every _consume path is
                # idempotent for identical input.
                if buf:
                    _consume(buf, state)
                continue

            # No new bytes: check for rotation/truncation before sleeping.
            try:
                st = os.stat(path)
                if st.st_ino != inode or st.st_size < fh.tell():
                    fh.close()
                    fh = None
                    continue
            except FileNotFoundError:
                fh.close()
                fh = None
                continue
            time.sleep(poll_s)
        except Exception:
            logger.exception("tail loop error; reopening in 2s")
            try:
                if fh:
                    fh.close()
            except Exception:
                pass
            fh = None
            time.sleep(2.0)


def _consume(line: str, state: ProgressState) -> None:
    """Extract counters from one log record. The record itself is never kept."""
    m = RE_PROMPT.search(line)
    if m:
        state.note_prompt(int(m.group(1)), int(m.group(2)))
        return
    m = RE_GENERATE.search(line)
    if m:
        state.note_generate(int(m.group(1)), int(m.group(2)))
        return
    if RE_DONE.search(line):
        state.note_done()


class Handler(BaseHTTPRequestHandler):
    state: ProgressState

    def do_GET(self) -> None:  # noqa: N802 (stdlib API)
        path = self.path.split("?", 1)[0].rstrip("/") or "/"
        if path in ("/progress", "/"):
            self._json(200, self.state.snapshot())
        elif path == "/healthz":
            self._json(200, {"ok": True})
        else:
            self._json(404, {"error": "not found"})

    def _json(self, code: int, body: Dict[str, Any]) -> None:
        raw = json.dumps(body).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        # The engine may poll this from a browser-adjacent context.
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(raw)

    def log_message(self, fmt: str, *args: Any) -> None:
        """Silence per-request logging — this endpoint is polled ~1/s."""


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--log", default="/var/log/kobold/current.log")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=5011)
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    state = ProgressState()
    threading.Thread(target=tail_log, args=(args.log, state), daemon=True).start()

    Handler.state = state
    srv = ThreadingHTTPServer((args.host, args.port), Handler)
    logger.info("kcpp-progress serving on %s:%d, tailing %s", args.host, args.port, args.log)
    srv.serve_forever()


if __name__ == "__main__":
    main()
