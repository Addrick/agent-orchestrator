"""Unit tests for the Claude Code -> Hindsight session ingester.

The bug these exist for (DP-336): `--backfill` used to POST transcripts that
were still being written and then record them as ingested forever, so a third
of the corpus was the opening minutes of a session and nothing could ever
repair it.
"""
import json
import os
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List

import pytest

from scripts import ingest_claudecode_session as ing


def write_session(
    project_dir: Path,
    session_id: str,
    turns: int = 6,
    start: datetime = datetime(2026, 8, 1, 12, 0, tzinfo=timezone.utc),
    body: str = "some durable content about the project",
) -> Path:
    """Write a transcript with alternating user/assistant turns.

    `load_session` needs >=1 user turn and >=4 turns total, so the default of 6
    is comfortably above the filter.
    """
    path = project_dir / f"{session_id}.jsonl"
    rows: List[Dict[str, Any]] = []
    for i in range(turns):
        rows.append({
            "type": "user" if i % 2 == 0 else "assistant",
            "timestamp": (start + timedelta(minutes=i)).isoformat().replace("+00:00", "Z"),
            "message": {"content": f"turn {i} {body}"},
        })
    path.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    return path


@pytest.fixture
def project_dir(tmp_path: Path) -> Path:
    d = tmp_path / "C--Users-Adam-Programming-Python-derpr-python"
    d.mkdir()
    return d


@pytest.fixture
def state_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    p = tmp_path / "hindsight_ingested.json"
    monkeypatch.setattr(ing, "STATE_FILE", p)
    return p


def age_file(path: Path, seconds: int) -> None:
    old = time.time() - seconds
    os.utime(path, (old, old))


class TestStateFile:
    def test_v1_list_migrates_to_map_without_lengths(self, state_file: Path) -> None:
        state_file.write_text(json.dumps({"ingested_sessions": ["aaa", "bbb"]}), encoding="utf-8")
        state = ing.load_state()
        assert state == {"aaa": {}, "bbb": {}}
        # No recorded length means the growth check must not fire on them.
        assert state["aaa"].get("chars") is None

    def test_round_trip_keeps_records(self, state_file: Path) -> None:
        ing.save_state({"aaa": {"chars": 10}})
        assert ing.load_state()["aaa"]["chars"] == 10
        assert json.loads(state_file.read_text(encoding="utf-8"))["version"] == 2

    def test_unreadable_state_is_treated_as_empty(self, state_file: Path) -> None:
        state_file.write_text("{ not json", encoding="utf-8")
        assert ing.load_state() == {}

    def test_missing_state_is_empty(self, state_file: Path) -> None:
        assert not state_file.exists()
        assert ing.load_state() == {}


class TestActiveGuard:
    def test_recently_written_transcript_is_active(self, project_dir: Path) -> None:
        path = write_session(project_dir, "live")
        assert ing.is_active(path) is True

    def test_old_transcript_is_not_active(self, project_dir: Path) -> None:
        path = write_session(project_dir, "done")
        age_file(path, ing.ACTIVE_GRACE_SECONDS + 60)
        assert ing.is_active(path) is False

    def test_backfill_defers_a_live_session(self, project_dir: Path) -> None:
        write_session(project_dir, "live")
        assert ing.collect_sessions(project_dir, None, {}) == []

    def test_backfill_takes_a_finished_session(self, project_dir: Path) -> None:
        path = write_session(project_dir, "done")
        age_file(path, ing.ACTIVE_GRACE_SECONDS + 60)
        got = ing.collect_sessions(project_dir, None, {})
        assert [s[0] for s in got] == ["done"]

    def test_explicit_session_id_bypasses_the_guard(self, project_dir: Path) -> None:
        # What the SessionEnd hook passes: the session is over, mtime is now.
        write_session(project_dir, "live")
        got = ing.collect_sessions(project_dir, "live", {})
        assert [s[0] for s in got] == ["live"]

    def test_missing_transcript_for_explicit_id_is_not_an_error(self, project_dir: Path) -> None:
        assert ing.collect_sessions(project_dir, "nope", {}) == []


class TestGrowthDetection:
    def test_already_ingested_at_same_length_is_skipped(self, project_dir: Path) -> None:
        path = write_session(project_dir, "done")
        age_file(path, ing.ACTIVE_GRACE_SECONDS + 60)
        chars = len(ing.collect_sessions(project_dir, None, {})[0][3])
        state = {"done": {"chars": chars}}
        assert ing.collect_sessions(project_dir, None, state) == []

    def test_grown_transcript_is_re_ingested(self, project_dir: Path) -> None:
        path = write_session(project_dir, "done", turns=6)
        age_file(path, ing.ACTIVE_GRACE_SECONDS + 60)
        short = len(ing.collect_sessions(project_dir, None, {})[0][3])
        write_session(project_dir, "done", turns=20)
        age_file(path, ing.ACTIVE_GRACE_SECONDS + 60)
        got = ing.collect_sessions(project_dir, None, {"done": {"chars": short}})
        assert [s[0] for s in got] == ["done"]

    def test_v1_entry_is_left_to_recheck(self, project_dir: Path) -> None:
        # A migrated record has no length; guessing would re-POST the whole
        # corpus, so backfill must leave it alone.
        path = write_session(project_dir, "legacy")
        age_file(path, ing.ACTIVE_GRACE_SECONDS + 60)
        assert ing.collect_sessions(project_dir, None, {"legacy": {}}) == []


class TestFindTruncated:
    def _docs(self, sid: str, text_length: int) -> Dict[str, Dict[str, Any]]:
        return {sid[:8]: {"id": f"claudecode:session_{sid[:8]}:ts", "text_length": text_length}}

    def test_shorter_on_server_is_reported(self, project_dir: Path) -> None:
        path = write_session(project_dir, "truncated")
        age_file(path, ing.ACTIVE_GRACE_SECONDS + 60)
        stale = ing.find_truncated(project_dir, self._docs("truncated", 10))
        assert len(stale) == 1
        doc_id, parsed, have, want = stale[0]
        assert (doc_id, parsed[0], have) == ("claudecode:session_truncate:ts", "truncated", 10)
        assert want > have

    def test_full_length_document_is_left_alone(self, project_dir: Path) -> None:
        path = write_session(project_dir, "complete")
        age_file(path, ing.ACTIVE_GRACE_SECONDS + 60)
        full = len(ing.collect_sessions(project_dir, None, {})[0][3])
        assert ing.find_truncated(project_dir, self._docs("complete", full)) == []

    def test_live_session_is_not_repaired_mid_flight(self, project_dir: Path) -> None:
        # Its document is short because the session is still going, not broken.
        write_session(project_dir, "live")
        assert ing.find_truncated(project_dir, self._docs("live", 10)) == []

    def test_session_with_no_document_is_not_a_repair_candidate(self, project_dir: Path) -> None:
        path = write_session(project_dir, "neverposted")
        age_file(path, ing.ACTIVE_GRACE_SECONDS + 60)
        assert ing.find_truncated(project_dir, {}) == []


class TestProjectDirResolution:
    def test_encodes_a_windows_path(self) -> None:
        assert ing.encode_project_dir(r"C:\Users\Adam\Programming\Python\derpr-python") == \
            "C--Users-Adam-Programming-Python-derpr-python"

    def test_bare_encoded_name_is_used_as_is(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(ing, "PROJECTS_ROOT", tmp_path)
        (tmp_path / "C--some-project").mkdir()
        assert ing.resolve_project_dir("C--some-project") == tmp_path / "C--some-project"
