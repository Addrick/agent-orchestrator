"""kcpp-progress log parser (DP-311).

The sidecar's whole job is turning KoboldCPP stdout into integers, so these
tests feed it the exact records a live KCPP 1.115 emits — captured from
CT101, not invented — and assert both what it extracts and what it refuses
to retain.
"""
import importlib.util
from pathlib import Path

import pytest

_MOD_PATH = Path(__file__).resolve().parents[2] / "services" / "kcpp-progress" / "kcpp_progress.py"
_spec = importlib.util.spec_from_file_location("kcpp_progress", _MOD_PATH)
assert _spec and _spec.loader
kcpp_progress = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(kcpp_progress)

ProgressState = kcpp_progress.ProgressState
_consume = kcpp_progress._consume


@pytest.fixture
def state():
    return ProgressState()


def test_starts_idle_with_no_counters(state):
    snap = state.snapshot()
    assert snap["phase"] == "idle"
    assert snap["processed"] == 0 and snap["total"] == 0


@pytest.mark.parametrize("line,expected", [
    # The BATCH tag is what CT101 emits; older/other paths use BLAS or no tag.
    ("Processing Prompt [BATCH] (2048 / 24310 tokens)", (2048, 24310)),
    ("Processing Prompt [BLAS] (0 / 5000 tokens)", (0, 5000)),
    ("Processing Prompt (512 / 512 tokens)", (512, 512)),
])
def test_parses_prefill_progress_variants(state, line, expected):
    _consume(line, state)
    snap = state.snapshot()
    assert snap["phase"] == "prefill"
    assert (snap["processed"], snap["total"]) == expected


def test_parses_generation_progress(state):
    _consume("Generating (17 / 400 tokens)", state)
    snap = state.snapshot()
    assert snap["phase"] == "generate"
    assert (snap["generated"], snap["generate_total"]) == (17, 400)


def test_generation_implies_prefill_complete(state):
    """KCPP does not print a final (N / N) batch when the last chunk is short,
    so the bar would stick at 22528/24310 forever without this."""
    _consume("Processing Prompt [BATCH] (22528 / 24310 tokens)", state)
    _consume("Generating (1 / 8 tokens)", state)
    snap = state.snapshot()
    assert snap["processed"] == snap["total"] == 24310


@pytest.mark.parametrize("marker", [
    "CtxLimit:24310/163840, Amt:1/8, Process:35.02s",
    "(EOS token triggered! ID:248046)",
    "Generation Aborted",
])
def test_end_markers_return_to_idle(state, marker):
    _consume("Generating (17 / 400 tokens)", state)
    _consume(marker, state)
    assert state.snapshot()["phase"] == "idle"


def test_run_counter_distinguishes_a_new_prefill_from_progress(state):
    """A consumer must be able to tell "same run, further along" from "new run"
    — otherwise a fresh (0 / N) reads as a bar jumping backwards."""
    _consume("Processing Prompt [BATCH] (0 / 100 tokens)", state)
    first = state.snapshot()["run"]
    _consume("Processing Prompt [BATCH] (50 / 100 tokens)", state)
    assert state.snapshot()["run"] == first, "progress within a run must not bump it"
    _consume("Processing Prompt [BATCH] (0 / 900 tokens)", state)
    assert state.snapshot()["run"] == first + 1


def test_stale_run_reverts_to_idle(state, monkeypatch):
    """The sidecar sees output, not state: a crashed or aborted run that stops
    printing must not pin the bar at 8192/24310 forever."""
    _consume("Processing Prompt [BATCH] (8192 / 24310 tokens)", state)
    assert state.snapshot()["phase"] == "prefill"
    real_time = kcpp_progress.time.time
    monkeypatch.setattr(kcpp_progress.time, "time",
                        lambda: real_time() + kcpp_progress.STALE_AFTER_S + 1)
    assert state.snapshot()["phase"] == "idle"


def test_snapshot_never_carries_log_text(state):
    """The log contains generated text and, on error paths, prompts. The
    snapshot must be integers/enums only — no field may echo the record."""
    secret = "the user's private prompt about their medical results"
    _consume(f"Processing Prompt [BATCH] (1 / 2 tokens) {secret}", state)
    _consume(f"some unrelated kobold chatter: {secret}", state)
    blob = repr(state.snapshot())
    assert "medical" not in blob and "private" not in blob
    assert set(state.snapshot()) == {
        "phase", "processed", "total", "generated",
        "generate_total", "age_s", "run", "source",
    }


def test_end_marker_is_the_real_timestamped_ctxlimit_line(state):
    """Verbatim from CT101. KCPP's end-of-run summary is the last thing it
    writes and its newline does not land until the NEXT run starts, so the
    tailer consumes it as an unterminated tail — meaning this exact string must
    match, prefix and all."""
    _consume("Generating (8 / 8 tokens)", state)
    _consume(
        "[00:43:16] CtxLimit:11/163840, Init:0.03s, Processed:3 in 0.02s "
        "(150.00T/s), Generated:8/8 in 0.56s (14.26T/s), Total:0.61s",
        state,
    )
    assert state.snapshot()["phase"] == "idle", (
        "end-of-run summary not recognized — phase would hang at `generate` "
        "until the staleness timeout"
    )


def test_consume_is_idempotent_for_repeated_records(state):
    """The tailer re-consumes an unterminated record once its terminator
    arrives. Doing so must not bump the run counter or move any number."""
    for _ in range(4):
        _consume("Processing Prompt [BATCH] (4096 / 24310 tokens)", state)
    snap = state.snapshot()
    assert snap["run"] == 1, "a re-read of the same record must not look like a new run"
    assert (snap["processed"], snap["total"]) == (4096, 24310)


def test_unmatched_lines_do_not_change_state(state):
    _consume("Processing Prompt [BATCH] (2048 / 24310 tokens)", state)
    before = state.snapshot()
    for noise in ["load_tensors: offloaded 97/97 layers to GPU",
                  "KV Save State 4: Created SaveState of 24279 tokens",
                  "Processing Prompt but not really (x / y tokens)"]:
        _consume(noise, state)
    after = state.snapshot()
    assert (after["phase"], after["processed"], after["total"]) == \
           (before["phase"], before["processed"], before["total"])
