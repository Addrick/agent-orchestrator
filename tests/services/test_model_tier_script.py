"""The eviction rules in `services/pve/derpr-model-tier`, run as real bash.

This is the half of DP-340 that Python never executes and that no handler test
can reach — and it is the half that deletes 24 GB files. Everything here drives
the actual script with `HOT_DIR`/`ARCHIVE_DIR` pointed at temp directories, so
the LRU ordering, the pin rules and the never-delete-an-unarchived-copy
invariant are exercised as deployed rather than described.

`HOT_CAPACITY_BYTES` is what makes it testable: `df` on a temp directory reports
the whole disk, so without a logical cap the eviction loop would never fire.

Skipped where bash is unavailable; CI is ubuntu and dev boxes have git-bash.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

_BASH = shutil.which("bash")
pytestmark = pytest.mark.skipif(_BASH is None, reason="needs bash to run the script")

_SCRIPT = Path(__file__).resolve().parents[2] / "services" / "pve" / "derpr-model-tier"

MB = 1024 * 1024


@pytest.fixture
def tier(tmp_path):
    """A working tier layout plus a runner bound to it."""
    hot = tmp_path / "hot"
    archive = tmp_path / "archive" / "models"
    root = tmp_path / "archive"
    hot.mkdir(parents=True)
    archive.mkdir(parents=True)

    def write(where: Path, name: str, size: int) -> Path:
        p = where / name
        p.write_bytes(b"\0" * size)
        return p

    def run(*args: str, capacity: int | None = None, margin: int = 0):
        env = dict(os.environ)
        env.update({
            "HOT_DIR": str(hot),
            "ARCHIVE_DIR": str(archive),
            "ARCHIVE_ROOT": str(root),
            "MARGIN_BYTES": str(margin),
            # The script needs real coreutils (stat, du, sha256sum, cp). On
            # Windows the inherited PATH is the Windows one and bash finds none
            # of them, so put bash's own bindir first. On Linux this is a no-op
            # (dirname(bash) is already /usr/bin or /bin).
            # No `pct` is on this PATH either, so active_hot_file() finds
            # nothing and no model counts as in-use.
            "PATH": os.path.dirname(_BASH) + os.pathsep
                    + os.environ.get("PATH", "/usr/bin:/bin"),
        })
        if capacity is not None:
            env["HOT_CAPACITY_BYTES"] = str(capacity)
        return subprocess.run(
            [str(_BASH), str(_SCRIPT), *args],
            env=env, capture_output=True, text=True,
        )

    class Tier:
        pass

    t = Tier()
    t.hot, t.archive, t.root, t.run, t.write = hot, archive, root, run, write
    return t


def _hot_names(tier) -> set[str]:
    return {p.name for p in tier.hot.glob("*.gguf")}


def _job(tier, job_id: str) -> dict:
    return json.loads((tier.root / ".jobs" / f"{job_id}.json").read_text())


def _set_served(tier, name: str, when: int) -> None:
    served = tier.root / ".tier" / "served"
    served.mkdir(parents=True, exist_ok=True)
    f = served / name
    f.touch()
    os.utime(f, (when, when))


# -- list --------------------------------------------------------------------

def test_list_reports_both_tiers(tier):
    tier.write(tier.hot, "hot.gguf", 4 * MB)
    tier.write(tier.archive, "hot.gguf", 4 * MB)
    tier.write(tier.archive, "cold.gguf", 4 * MB)
    out = tier.run("list")
    assert out.returncode == 0, out.stderr
    payload = json.loads(out.stdout)
    tiers = {m["file"]: m["tier"] for m in payload["models"]}
    assert tiers == {"hot.gguf": "hot", "cold.gguf": "cold"}


def test_a_file_in_both_tiers_reports_hot(tier):
    """The hot copy is the servable one; the archive copy is its backing."""
    tier.write(tier.hot, "m.gguf", 4 * MB)
    tier.write(tier.archive, "m.gguf", 4 * MB)
    payload = json.loads(tier.run("list").stdout)
    assert [m["tier"] for m in payload["models"]] == ["hot"]


def test_list_is_valid_json_when_empty(tier):
    """Hand-rolled JSON with no jq — the empty case is where that breaks."""
    payload = json.loads(tier.run("list").stdout)
    assert payload["models"] == []
    assert payload["status"] == "ok"


# -- pin / unpin -------------------------------------------------------------

def test_pin_and_unpin_round_trip(tier):
    tier.write(tier.archive, "m.gguf", MB)
    assert tier.run("pin", "m.gguf").returncode == 0
    assert json.loads(tier.run("list").stdout)["models"][0]["pinned"] is True
    assert tier.run("unpin", "m.gguf").returncode == 0
    assert json.loads(tier.run("list").stdout)["models"][0]["pinned"] is False


def test_pinning_an_unknown_model_fails(tier):
    assert tier.run("pin", "nope.gguf").returncode != 0


def test_pinning_is_idempotent(tier):
    """Double-pinning must not put the name in the file twice — unpin removes
    every occurrence, but a stale duplicate would still be misleading."""
    tier.write(tier.archive, "m.gguf", MB)
    tier.run("pin", "m.gguf")
    tier.run("pin", "m.gguf")
    pinned = (tier.root / ".tier" / "pinned").read_text().split()
    assert pinned.count("m.gguf") == 1


# -- promotion + eviction ----------------------------------------------------

def test_promote_copies_and_marks_served(tier):
    tier.write(tier.archive, "m.gguf", 4 * MB)
    out = tier.run("run-promote", "m.gguf", "job1", capacity=100 * MB)
    assert out.returncode == 0, out.stderr
    assert _hot_names(tier) == {"m.gguf"}
    assert _job(tier, "job1")["state"] == "done"


def test_eviction_takes_the_least_recently_served_first(tier):
    for name, when in [("old.gguf", 1000), ("mid.gguf", 2000), ("new.gguf", 3000)]:
        tier.write(tier.hot, name, 4 * MB)
        tier.write(tier.archive, name, 4 * MB)
        _set_served(tier, name, when)
    tier.write(tier.archive, "want.gguf", 4 * MB)
    # Room for four 4 MB files; three are resident, so exactly one must go.
    out = tier.run("run-promote", "want.gguf", "job1", capacity=14 * MB)
    assert out.returncode == 0, out.stderr
    assert "old.gguf" not in _hot_names(tier)
    assert {"mid.gguf", "new.gguf", "want.gguf"} <= _hot_names(tier)


def test_a_never_served_model_is_evicted_before_an_old_one(tier):
    """An install nobody ever ran should go before a model that at least once
    earned its place."""
    tier.write(tier.hot, "served.gguf", 4 * MB)
    tier.write(tier.archive, "served.gguf", 4 * MB)
    _set_served(tier, "served.gguf", 1000)
    tier.write(tier.hot, "never.gguf", 4 * MB)      # no served marker at all
    tier.write(tier.archive, "never.gguf", 4 * MB)
    tier.write(tier.archive, "want.gguf", 4 * MB)
    out = tier.run("run-promote", "want.gguf", "job1", capacity=10 * MB)
    assert out.returncode == 0, out.stderr
    assert "never.gguf" not in _hot_names(tier)
    assert "served.gguf" in _hot_names(tier)


def test_only_enough_is_evicted(tier):
    """Capacity rule, not a fixed count: evicting more than needed would throw
    away minutes of future copy time for nothing."""
    for i, when in enumerate([1000, 2000, 3000]):
        tier.write(tier.hot, f"m{i}.gguf", 4 * MB)
        tier.write(tier.archive, f"m{i}.gguf", 4 * MB)
        _set_served(tier, f"m{i}.gguf", when)
    tier.write(tier.archive, "want.gguf", 4 * MB)
    tier.run("run-promote", "want.gguf", "job1", capacity=14 * MB)
    assert len(_hot_names(tier)) == 3   # one out, one in


def test_a_pinned_model_is_never_evicted(tier):
    tier.write(tier.hot, "pinned.gguf", 4 * MB)
    tier.write(tier.archive, "pinned.gguf", 4 * MB)
    _set_served(tier, "pinned.gguf", 1)          # oldest by far
    tier.run("pin", "pinned.gguf")
    tier.write(tier.hot, "other.gguf", 4 * MB)
    tier.write(tier.archive, "other.gguf", 4 * MB)
    _set_served(tier, "other.gguf", 9999)
    tier.write(tier.archive, "want.gguf", 4 * MB)
    out = tier.run("run-promote", "want.gguf", "job1", capacity=10 * MB)
    assert out.returncode == 0, out.stderr
    assert "pinned.gguf" in _hot_names(tier)
    assert "other.gguf" not in _hot_names(tier)


def test_all_pinned_refuses_instead_of_overruling_a_pin(tier):
    """A pin the system may overrule is not a pin. Refusing is the whole point
    of the rule, and the reason says which pins to release."""
    for name in ("a.gguf", "b.gguf"):
        tier.write(tier.hot, name, 4 * MB)
        tier.write(tier.archive, name, 4 * MB)
        tier.run("pin", name)
    tier.write(tier.archive, "want.gguf", 4 * MB)
    out = tier.run("run-promote", "want.gguf", "job1", capacity=10 * MB)
    assert out.returncode != 0
    assert _job(tier, "job1")["reason"] == "hot_tier_full_all_pinned"
    # Nothing was deleted on the way to refusing.
    assert {"a.gguf", "b.gguf"} <= _hot_names(tier)


def test_an_unarchived_hot_file_is_never_deleted(tier):
    """THE invariant. Eviction is only a cache operation because the archive
    copy is authoritative; without one, deleting the hot copy is data loss."""
    tier.write(tier.hot, "orphan.gguf", 4 * MB)      # no archive copy
    _set_served(tier, "orphan.gguf", 1)
    tier.write(tier.archive, "want.gguf", 4 * MB)
    out = tier.run("run-promote", "want.gguf", "job1", capacity=6 * MB)
    assert out.returncode != 0
    assert _job(tier, "job1")["reason"] == "unarchived_victim"
    assert "orphan.gguf" in _hot_names(tier)


def test_a_mismatched_archive_copy_blocks_eviction(tier):
    """Same invariant, subtler case: an archive copy that exists but differs is
    not a backing copy, and trusting the filename alone would lose the bytes."""
    tier.write(tier.hot, "drift.gguf", 4 * MB)
    tier.archive.joinpath("drift.gguf").write_bytes(b"\1" * (4 * MB))
    _set_served(tier, "drift.gguf", 1)
    tier.write(tier.archive, "want.gguf", 4 * MB)
    out = tier.run("run-promote", "want.gguf", "job1", capacity=6 * MB)
    assert out.returncode != 0
    assert _job(tier, "job1")["reason"] == "unarchived_victim"
    assert "drift.gguf" in _hot_names(tier)


def test_an_already_hot_identical_file_is_a_no_op(tier):
    tier.write(tier.hot, "m.gguf", 4 * MB)
    tier.write(tier.archive, "m.gguf", 4 * MB)
    out = tier.run("run-promote", "m.gguf", "job1", capacity=100 * MB)
    assert out.returncode == 0, out.stderr
    assert _job(tier, "job1")["state"] == "done"


def test_a_differing_hot_file_is_refused_not_overwritten(tier):
    """koboldcpp may have the hot file mapped. Same name, different bytes is a
    refusal — the same rule the installer applies to downloads."""
    tier.hot.joinpath("m.gguf").write_bytes(b"\1" * (4 * MB))
    tier.write(tier.archive, "m.gguf", 4 * MB)
    out = tier.run("run-promote", "m.gguf", "job1", capacity=100 * MB)
    assert out.returncode != 0
    assert _job(tier, "job1")["reason"] == "hot_copy_differs"
    assert tier.hot.joinpath("m.gguf").read_bytes()[:1] == b"\1"


def test_no_partial_file_is_left_behind_on_success(tier):
    tier.write(tier.archive, "m.gguf", 4 * MB)
    tier.run("run-promote", "m.gguf", "job1", capacity=100 * MB)
    assert list(tier.hot.glob("*.part")) == []


def test_promote_refuses_a_duplicate_job_id(tier):
    tier.write(tier.archive, "m.gguf", MB)
    (tier.root / ".jobs").mkdir(parents=True, exist_ok=True)
    (tier.root / ".jobs" / "job1.json").write_text("{}")
    assert tier.run("promote", "m.gguf", "job1").returncode != 0


def test_promote_refuses_a_model_with_no_archive_copy(tier):
    tier.write(tier.hot, "m.gguf", MB)
    assert tier.run("promote", "m.gguf", "job1").returncode != 0


@pytest.mark.parametrize("bad", [
    "../../etc/passwd", "/srv/models/m.gguf", "m.txt", "m.gguf.part", "",
])
def test_promote_rejects_bad_filenames(tier, bad):
    assert tier.run("promote", bad, "job1").returncode != 0


# --- the active-model guard ------------------------------------------------
#
# Every test above runs with no `pct` on PATH, so active_hot_file() finds
# nothing and no model counts as in-use. That is convenient and it is also how
# a real bug shipped: the extraction only understood `--model=<path>`, the
# units on the node spell it `--model <path>`, and so the one guard standing
# between an eviction and deleting the weights out from under a running
# koboldcpp returned empty on every real unit. These tests put a `pct` stub on
# PATH that answers with systemd's actual `ExecStart` rendering.

_PCT_STUB = """#!/bin/sh
# Stands in for `pct exec <ct> -- ...` on a node.
case "$*" in
  *list-unit-files*) echo "koboldcpp-ff711-q6k.service enabled enabled" ;;
  *ExecStart*)
    echo "{ path=/opt/koboldcpp/koboldcpp ; argv[]=/opt/koboldcpp/koboldcpp \
%(flags)s ; ignore_errors=no ; start_time=[n/a] ; pid=91 ; status=0/0 }" ;;
  *) exit 1 ;;
esac
"""


def _with_pct(tier, flags: str):
    """Return a `run` that has a pct stub on PATH reporting `flags`."""
    binpath = tier.root / "stubbin"
    binpath.mkdir(exist_ok=True)
    pct = binpath / "pct"
    pct.write_text(_PCT_STUB % {"flags": flags})
    pct.chmod(0o755)

    def run(*args: str, **kw):
        old = os.environ.get("PATH", "")
        os.environ["PATH"] = str(binpath) + os.pathsep + old
        try:
            return tier.run(*args, **kw)
        finally:
            os.environ["PATH"] = old
    return run


@pytest.mark.parametrize("flags", [
    "--model /opt/koboldcpp/models/live.gguf --port 5001",
    "--model=/opt/koboldcpp/models/live.gguf --port 5001",
])
def test_the_served_model_is_never_evicted(tier, flags):
    """Both `--model <path>` and `--model=<path>` must be understood.

    The space-separated form is what the node actually emits; handling only
    the `=` form is what made this guard a no-op in production.
    """
    for name in ("live.gguf", "other.gguf"):
        tier.write(tier.hot, name, 4 * MB)
        tier.write(tier.archive, name, 4 * MB)
    # `live` is the least-recently-served, so LRU alone would take it first.
    _set_served(tier, "live.gguf", 1000)
    _set_served(tier, "other.gguf", 9000)
    tier.write(tier.archive, "want.gguf", 4 * MB)

    out = _with_pct(tier, flags)("run-promote", "want.gguf", "job1",
                                 capacity=10 * MB)
    assert out.returncode == 0, out.stderr
    assert "live.gguf" in _hot_names(tier), "evicted the model being served"
    assert "other.gguf" not in _hot_names(tier)


def test_a_draft_model_flag_is_not_mistaken_for_the_served_weights(tier):
    """`--draftmodel` also ends in .gguf and must not shadow `--model`."""
    for name in ("live.gguf", "draft.gguf"):
        tier.write(tier.hot, name, 4 * MB)
        tier.write(tier.archive, name, 4 * MB)
    _set_served(tier, "live.gguf", 1000)
    _set_served(tier, "draft.gguf", 9000)
    tier.write(tier.archive, "want.gguf", 4 * MB)

    flags = ("--draftmodel /opt/koboldcpp/models/draft.gguf "
             "--model /opt/koboldcpp/models/live.gguf --port 5001")
    out = _with_pct(tier, flags)("run-promote", "want.gguf", "job1",
                                 capacity=10 * MB)
    assert out.returncode == 0, out.stderr
    assert "live.gguf" in _hot_names(tier)


def test_list_reports_the_active_model(tier):
    tier.write(tier.hot, "live.gguf", MB)
    flags = "--model /opt/koboldcpp/models/live.gguf --port 5001"
    out = _with_pct(tier, flags)("list")
    assert out.returncode == 0, out.stderr
    assert json.loads(out.stdout)["active"] == "live.gguf"
