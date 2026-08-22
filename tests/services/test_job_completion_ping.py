"""DP-343 — the completion ping, as the node actually sends it.

The Python half of this feature (tests/huggingface/test_job_completion.py) starts
at a job id. Getting a job id to derpr at all is bash: `write_status` fires
`ping_derpr` when a job reaches `done` or `failed`, from inside the detached
half of an install or a promotion. Nothing in Python executes that, and it is
the half that decides whether the whole feature ever runs.

Both scripts are driven as real bash with a fake `curl` on PATH that records the
request instead of making it — so what is asserted is the argv the node builds,
including the token header and the body, rather than a description of it.

Skipped where bash is unavailable; CI is ubuntu and dev boxes have git-bash.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

_BASH = shutil.which("bash")
pytestmark = pytest.mark.skipif(_BASH is None, reason="needs bash to run the script")

_SERVICES = Path(__file__).resolve().parents[2] / "services" / "pve"
_INSTALL = _SERVICES / "derpr-model-install"
_TIER = _SERVICES / "derpr-model-tier"

URL = "http://10.0.0.70:5003/api/v1/model_job/complete"
TOKEN = "n0de-callback-token"


def _fake_bin(tmp_path: Path, *, curl_exit: int = 0) -> Path:
    """A PATH directory of shims: curl records, pct/logger/systemd-run succeed.

    `curl` writes one line per invocation into `curl.log` and, when given `-o`,
    creates the requested file so the installer's download path completes.
    """
    bindir = tmp_path / "bin"
    bindir.mkdir()
    log = tmp_path / "curl.log"
    (bindir / "curl").write_text(
        "#!/bin/bash\n"
        f'printf "%s\\n" "$*" >> "{log.as_posix()}"\n'
        "out=''\n"
        "while [ $# -gt 0 ]; do\n"
        "  case \"$1\" in -o) out=\"$2\"; shift 2 ;; *) shift ;; esac\n"
        "done\n"
        f'[ -n "$out" ] && printf "%s" "$(cat "{(tmp_path / "payload").as_posix()}")" > "$out"\n'
        f"exit {curl_exit}\n",
        encoding="utf-8",
    )
    for name in ("pct", "logger", "systemd-run", "systemctl"):
        (bindir / name).write_text("#!/bin/bash\nexit 0\n", encoding="utf-8")
    for f in bindir.iterdir():
        f.chmod(0o755)
    return bindir


def _env(tmp_path: Path, bindir: Path, *, url: str = URL,
         token: str | None = TOKEN) -> dict:
    """Environment for a script run whose cwd is `tmp_path`.

    Every path the scripts are given is RELATIVE, and every run sets
    `cwd=tmp_path`. That is not tidiness: `sha256sum` escapes its output line
    with a leading backslash when the filename contains one, so an absolute
    Windows path makes the installer's own digest check compare an escaped
    digest against a bare one and fail every download. The node has no backslashes in a
    path; a dev box does, and this is what keeps the test testing the script
    rather than the path separator.
    """
    env = dict(os.environ)
    token_file = tmp_path / "callback.token"
    if token is not None:
        token_file.write_text(token, encoding="utf-8")
    env.update({
        "DERPR_CALLBACK_URL": url,
        "DERPR_CALLBACK_TOKEN_FILE": "callback.token" if token is not None
                                     else "no-such.token",
        "DERPR_CALLBACK_TIMEOUT": "1",
        # Shims first, then real coreutils (a Windows PATH has none), then
        # whatever the box already had.
        "PATH": os.pathsep.join(
            [str(bindir), os.path.dirname(_BASH), env.get("PATH", "")]
        ),
    })
    return env


def _requests(tmp_path: Path) -> list[str]:
    log = tmp_path / "curl.log"
    if not log.exists():
        return []
    return [line for line in log.read_text(encoding="utf-8").splitlines() if line]


def _job_file(tmp_path: Path, job: str) -> dict:
    """The job document the script wrote.

    `sha256sum` prefixes its output with a backslash when the path it hashed
    contains one, and on a Windows dev box the temp directory does — so the sha
    field can carry a stray leading escape that is an artifact of the box, never
    of the node. Strip it rather than assert around it; the digest is not what
    these tests are about.
    """
    raw = (tmp_path / "archive" / ".jobs" / f"{job}.json").read_text(encoding="utf-8")
    return json.loads(raw.replace('"sha256":"' + chr(92), '"sha256":"'))


def _pings(tmp_path: Path) -> list[str]:
    """Only the callback requests — the installer's download uses curl too."""
    return [r for r in _requests(tmp_path) if URL in r or "model_job/complete" in r]


# ---------------------------------------------------------------------------
# promotion (derpr-model-tier)
# ---------------------------------------------------------------------------

def _tier_layout(tmp_path: Path, size: int = 4096) -> dict:
    hot = tmp_path / "hot"
    root = tmp_path / "archive"
    archive = root / "models"
    hot.mkdir()
    archive.mkdir(parents=True)
    (archive / "m.gguf").write_bytes(b"\0" * size)
    return {"HOT_DIR": "hot", "ARCHIVE_DIR": "archive/models",
            "ARCHIVE_ROOT": "archive", "JOBS_DIR": "archive/.jobs",
            "STATE_DIR": "archive/.tier", "MARGIN_BYTES": "0"}


def _run_promote(tmp_path: Path, env: dict, job: str = "m-1") -> subprocess.CompletedProcess:
    return subprocess.run(
        [_BASH, str(_TIER), "run-promote", "m.gguf", job],
        env=env, cwd=tmp_path, capture_output=True, text=True,
    )


def test_promotion_pings_derpr_when_it_finishes(tmp_path):
    bindir = _fake_bin(tmp_path)
    env = _env(tmp_path, bindir)
    env.update(_tier_layout(tmp_path))

    res = _run_promote(tmp_path, env)
    assert res.returncode == 0, res.stderr

    pings = _pings(tmp_path)
    assert len(pings) == 1, f"expected exactly one ping, got {pings}"
    ping = pings[0]
    assert f"Authorization: Bearer {TOKEN}" in ping
    # The body is the job id and nothing else — derpr re-reads the job over SSH.
    assert '{"job_id":"m-1"}' in ping
    assert URL in ping


def test_promotion_failure_pings_too(tmp_path):
    """A failed promotion is exactly when a human needs telling.

    Same-name-different-bytes in the hot tier is a refusal, never an overwrite —
    and before DP-343 that refusal sat in a job file nobody read.
    """
    bindir = _fake_bin(tmp_path)
    env = _env(tmp_path, bindir)
    layout = _tier_layout(tmp_path)
    env.update(layout)
    # A hot copy with the same name and different bytes.
    (tmp_path / "hot" / "m.gguf").write_bytes(b"\1" * 4096)

    res = _run_promote(tmp_path, env)
    assert res.returncode == 1

    job = _job_file(tmp_path, "m-1")
    assert job["state"] == "failed" and job["reason"] == "hot_copy_differs"
    assert len(_pings(tmp_path)) == 1


def test_no_ping_configured_is_silent(tmp_path):
    """The pre-DP-343 behaviour, and the right state for a node that cannot
    reach derpr: no URL, no ping, no failure."""
    bindir = _fake_bin(tmp_path)
    env = _env(tmp_path, bindir, url="")
    env.update(_tier_layout(tmp_path))

    res = _run_promote(tmp_path, env)

    assert res.returncode == 0, res.stderr
    assert _pings(tmp_path) == []


def test_missing_token_file_does_not_ping(tmp_path):
    """No credential = no request. Posting unauthenticated would only produce a
    401 the node cannot act on, and would put the job id on the wire anyway."""
    bindir = _fake_bin(tmp_path)
    env = _env(tmp_path, bindir, token=None)
    env.update(_tier_layout(tmp_path))

    res = _run_promote(tmp_path, env)

    assert res.returncode == 0, res.stderr
    assert _pings(tmp_path) == []


def test_unreachable_derpr_does_not_fail_the_promotion(tmp_path):
    """THE property that makes this safe to add to a script that moves 24 GB.

    The copy is finished and verified before the ping is attempted. A derpr that
    is down, restarting, or unreachable must cost the announcement and nothing
    else — the job state on the node stays `done`.
    """
    bindir = _fake_bin(tmp_path, curl_exit=7)
    env = _env(tmp_path, bindir)
    layout = _tier_layout(tmp_path)
    env.update(layout)

    res = _run_promote(tmp_path, env)

    assert res.returncode == 0, res.stderr
    job = _job_file(tmp_path, "m-1")
    assert job["state"] == "done" and job["step"] == "promoted"
    assert (tmp_path / "hot" / "m.gguf").exists()


def test_running_states_do_not_ping(tmp_path):
    """Only terminal states ring the doorbell.

    A promotion writes `running` at every step (evict, copy, verify); an install
    republishes `downloaded_bytes` every few seconds. Pinging on those would
    wake a persona once per progress tick.
    """
    bindir = _fake_bin(tmp_path)
    env = _env(tmp_path, bindir)
    layout = _tier_layout(tmp_path)
    env.update(layout)

    _run_promote(tmp_path, env)

    # The job passed through running/evict, running/copy and running/verify on
    # its way to done, and pinged exactly once.
    assert len(_pings(tmp_path)) == 1


# ---------------------------------------------------------------------------
# install (derpr-model-install)
# ---------------------------------------------------------------------------

def _install_env(tmp_path: Path, bindir: Path, payload: bytes) -> dict:
    archive = tmp_path / "archive" / "models"
    archive.mkdir(parents=True)
    (tmp_path / "payload").write_bytes(payload)
    template = tmp_path / "unit.in"
    template.write_text("[Service]\nExecStart=@@KCPP_DIR@@ @@MODEL_PATH@@\n",
                        encoding="utf-8")
    env = _env(tmp_path, bindir)
    env.update({
        "ARCHIVE_DIR": "archive/models",
        "JOBS_DIR": "archive/.jobs",
        "TEMPLATE": "unit.in",
        "GGUF_HEADER": "no-such-header.py",
        "TOKEN_FILE": "no-such-hf.token",
        "PROGRESS_INTERVAL": "1",
        "HF_BASE": "https://hf.invalid",
    })
    return env


def _run_install(tmp_path: Path, env: dict, sha: str, size: int,
                 job: str = "newmodel-1"):
    return subprocess.run(
        [_BASH, str(_INSTALL), "run", "owner/repo", "model.gguf", "newmodel",
         "8192", str(size), sha, job],
        env=env, cwd=tmp_path, capture_output=True, text=True,
    )


def test_finished_install_pings_derpr(tmp_path):
    payload = b"gguf-bytes"
    bindir = _fake_bin(tmp_path)
    env = _install_env(tmp_path, bindir, payload)

    res = _run_install(tmp_path, env, hashlib.sha256(payload).hexdigest(), len(payload))
    assert res.returncode == 0, res.stderr

    job = _job_file(tmp_path, "newmodel-1")
    assert job["state"] == "done"
    pings = _pings(tmp_path)
    assert len(pings) == 1
    assert '{"job_id":"newmodel-1"}' in pings[0]
    assert f"Authorization: Bearer {TOKEN}" in pings[0]


def test_sha_mismatch_pings_the_failure(tmp_path):
    """The download that must never be reported as an install.

    A digest mismatch deletes the partial file and fails the job; the ping is
    what turns that into something a human hears about instead of a job file on
    the archive disk.
    """
    payload = b"gguf-bytes"
    bindir = _fake_bin(tmp_path)
    env = _install_env(tmp_path, bindir, payload)

    res = _run_install(tmp_path, env, "0" * 64, len(payload))
    assert res.returncode == 1

    job = _job_file(tmp_path, "newmodel-1")
    assert job["state"] == "failed" and job["reason"] == "sha256_mismatch"
    assert len(_pings(tmp_path)) == 1
