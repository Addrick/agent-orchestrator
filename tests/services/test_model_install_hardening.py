"""DP-349 — the node installer's hardening, exercised as a script.

These are `bash` runs of `services/pve/derpr-model-install`, not Python unit
tests, because every defect here lives in shell: a byte cap standing in for an
element cap, a precheck separated from its commit, an unbounded `curl`, a
colliding destination name, and steps reported done off a discarded exit status.
None of it is reachable from derpr's side, and the node is deployed by hand —
so a test that mocks the script proves nothing about the file that ships.

DP-360 added a second reason to keep this file: the gguf-header gate lives in
shell too, and it silently discarded every refusal fragment for months while
four Python tests of the *producer* passed. A boundary with tests on only one
side of it is a boundary nothing tests.

The container is a shim directory on `PATH`. It answers the two probes the
installer now demands positive answers from, which is itself the point of
defect 6: a `pct exec` that fails is evidence about the *container*, never about
the unit, and reading it as "absent" disarms the guard exactly when the box
cannot be consulted.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

_SERVICES = Path(__file__).resolve().parents[2] / "services" / "pve"
_INSTALL = _SERVICES / "derpr-model-install"
_BASH = shutil.which("bash") or "/bin/bash"

pytestmark = pytest.mark.skipif(
    not Path(_BASH).exists() or not _INSTALL.exists(),
    reason="needs bash and the node installer script",
)

PAYLOAD = b"gguf-bytes-for-the-hardening-tests"
SHA = hashlib.sha256(PAYLOAD).hexdigest()
SIZE = len(PAYLOAD)


def _set_job_running(bindir: Path, running: bool) -> None:
    """`systemctl is-active modelinstall-<job>` is the installer's only witness
    that a reservation still belongs to something alive. `running=False` is the
    crash/reboot case: the claim outlived the process that made it."""
    shim = bindir / "systemctl"
    shim.write_text(
        "#!/bin/bash\n"
        'case "$*" in\n'
        f"  *is-active*) exit {0 if running else 3} ;;\n"
        "esac\n"
        "exit 0\n",
        encoding="utf-8",
    )
    shim.chmod(0o755)


def _shims(tmp_path: Path, *, unit_present: bool = False,
           unit_probe_broken: bool = False, reload_rc: int = 0,
           load_state: str = "loaded") -> Path:
    """A PATH directory standing in for the node's tools."""
    bindir = tmp_path / "bin"
    bindir.mkdir()
    (tmp_path / "payload").write_bytes(PAYLOAD)

    # curl honours --max-filesize the way the real one does: refuse to write
    # more than the caller allowed.
    (bindir / "curl").write_text(
        "#!/bin/bash\n"
        f'printf "%s\\n" "$*" >> "{(tmp_path / "curl.log").as_posix()}"\n'
        "out=''; cap=''\n"
        "while [ $# -gt 0 ]; do\n"
        '  case "$1" in\n'
        '    -o) out="$2"; shift 2 ;;\n'
        '    --max-filesize) cap="$2"; shift 2 ;;\n'
        "    *) shift ;;\n"
        "  esac\n"
        "done\n"
        f'body="$(cat "{(tmp_path / "payload").as_posix()}")"\n'
        '[ -n "$cap" ] && [ "${#body}" -gt "$cap" ] && exit 63\n'
        '[ -n "$out" ] && printf "%s" "$body" > "$out"\n'
        "exit 0\n",
        encoding="utf-8",
    )

    present = "present" if unit_present else "absent"
    probe = "" if unit_probe_broken else f"echo {present}"
    # `push` keeps a copy of what was pushed, so a test can read the unit the
    # installer actually rendered (DP-364) rather than trusting the template.
    (bindir / "pct").write_text(
        "#!/bin/bash\n"
        'case "$*" in\n'
        f'  push*) cp "$3" "{tmp_path.as_posix()}/pushed-$(basename "$4")" ;;\n'
        f'  *LoadState*) echo {load_state} ;;\n'
        f'  *"echo present"*) {probe or ":"} ;;\n'
        f'  *daemon-reload*) exit {reload_rc} ;;\n'
        "esac\n"
        "exit 0\n",
        encoding="utf-8",
    )
    _set_job_running(bindir, True)
    for name in ("logger", "systemd-run", "flock"):
        (bindir / name).write_text("#!/bin/bash\nexit 0\n", encoding="utf-8")
    for f in bindir.iterdir():
        f.chmod(0o755)
    return bindir


def _env(tmp_path: Path, bindir: Path) -> dict:
    archive = tmp_path / "archive" / "models"
    archive.mkdir(parents=True)
    (tmp_path / "unit.in").write_text(
        "[Service]\nExecStart=@@KCPP_DIR@@ @@MODEL_PATH@@\n", encoding="utf-8")
    # DP-364: a header reader has to exist -- the installer reads the cache
    # mode off the file and refuses when it cannot. This default reports a
    # dense model and no header shape; tests that need another replace it.
    reader = tmp_path / "default-reader"
    reader.write_text('#!/bin/bash\n[ "$1" = --ssm-layers ] && printf 0\nexit 0\n',
                      encoding="utf-8")
    (bindir / "python3").write_text('#!/bin/bash\nexec bash "$@"\n', encoding="utf-8")
    (bindir / "python3").chmod(0o755)
    env = dict(os.environ)
    env.update({
        "PATH": f"{bindir.as_posix()}{os.pathsep}{env.get('PATH', '')}",
        "MODELS_DIR": "archive/models",
        "JOBS_DIR": "archive/.jobs",
        "TEMPLATE": "unit.in",
        "GGUF_HEADER": reader.as_posix(),
        "PROGRESS_INTERVAL": "1",
        "HF_BASE": "https://hf.invalid",
        "DERPR_CALLBACK_URL": "",
    })
    return env


def _run(tmp_path: Path, env: dict, verb: str, *args: str,
         size: int = SIZE, sha: str = SHA, name: str = "newmodel",
         job: str = "newmodel-1",
         kv: str = "q8") -> subprocess.CompletedProcess:
    argv = [_BASH, str(_INSTALL), verb, "owner/repo", "model-Q4_K_M.gguf",
            name, "8192", str(size), sha, job, kv]
    return subprocess.run(argv + list(args), env=env, cwd=tmp_path,
                          capture_output=True, text=True)


def _job(tmp_path: Path, job: str = "newmodel-1") -> dict:
    return json.loads((tmp_path / "archive" / ".jobs" / f"{job}.json").read_text())


# -- defect 4: the destination basename ---------------------------------------

def test_destination_is_named_for_the_unit_not_the_repo_file(tmp_path):
    """Two repos publishing `model-Q4_K_M.gguf` must not fight over one path.

    The repo file name is not unique across repos; `$NAME` is, because the unit
    stem `koboldcpp-$NAME.service` must not already exist.
    """
    env = _env(tmp_path, _shims(tmp_path))
    res = _run(tmp_path, env, "run")
    assert res.returncode == 0, res.stderr

    archive = tmp_path / "archive" / "models"
    assert (archive / "newmodel.gguf").exists()
    assert not (archive / "model-Q4_K_M.gguf").exists()


def test_the_unit_points_at_the_same_renamed_file(tmp_path):
    """The rename is only safe if the unit template follows it."""
    env = _env(tmp_path, _shims(tmp_path))
    assert _run(tmp_path, env, "run").returncode == 0
    assert _job(tmp_path)["state"] == "done"
    # @@MODEL_PATH@@ is expanded from BASENAME, so a divergence here would be a
    # unit pointing at a file that was never written.
    log = (tmp_path / "curl.log").read_text()
    assert "model-Q4_K_M.gguf" in log, "the URL still uses the repo's file name"


# -- defect 3: bounded download and a byte-count pre-filter -------------------

def test_curl_is_bounded_by_the_approved_size(tmp_path):
    env = _env(tmp_path, _shims(tmp_path))
    assert _run(tmp_path, env, "run").returncode == 0
    assert f"--max-filesize {SIZE}" in (tmp_path / "curl.log").read_text()


def test_a_short_file_fails_on_the_byte_count_before_the_digest(tmp_path):
    """`stat` knows what hashing 30 GB would take minutes to discover."""
    env = _env(tmp_path, _shims(tmp_path))
    res = _run(tmp_path, env, "run", size=SIZE + 999)
    assert res.returncode == 1
    job = _job(tmp_path)
    assert job["state"] == "failed"
    assert job["reason"] == "size_mismatch", "the digest ran anyway"


def test_a_repo_serving_more_than_it_advertised_is_refused(tmp_path):
    """--max-filesize makes the overrun a download failure, not a full disk."""
    env = _env(tmp_path, _shims(tmp_path))
    res = _run(tmp_path, env, "run", size=4, sha=hashlib.sha256(b"abcd").hexdigest())
    assert res.returncode == 1
    assert _job(tmp_path)["state"] == "failed"
    assert not (tmp_path / "archive" / "models" / "newmodel.gguf.part").exists()


# -- defect 6: no step reported done off a discarded exit status --------------

def test_a_failed_daemon_reload_fails_the_job(tmp_path):
    """A unit systemd has not re-read does not exist to `list-unit-files`,
    which is exactly where `list_models` discovers units."""
    env = _env(tmp_path, _shims(tmp_path, reload_rc=1))
    res = _run(tmp_path, env, "run")
    assert res.returncode == 1
    job = _job(tmp_path)
    assert job["state"] == "failed" and job["reason"] == "daemon_reload_failed"


def test_a_unit_that_did_not_load_is_not_reported_installed(tmp_path):
    env = _env(tmp_path, _shims(tmp_path, load_state="not-found"))
    res = _run(tmp_path, env, "run")
    assert res.returncode == 1
    job = _job(tmp_path)
    assert job["state"] == "failed" and job["reason"] == "unit_not_loaded"


def test_an_unreachable_container_is_not_read_as_unit_absent(tmp_path):
    """The guard must refuse, not proceed, when the box cannot be consulted.

    A failing `pct exec test -f` is equally "container stopped, locked, or
    unreachable"; treating it as "no such unit" disarms the check at exactly
    the moment it matters.
    """
    env = _env(tmp_path, _shims(tmp_path, unit_probe_broken=True))
    res = _run(tmp_path, env, "install")
    assert res.returncode == 1
    assert "could not determine whether" in res.stderr


def test_an_existing_unit_is_still_refused(tmp_path):
    env = _env(tmp_path, _shims(tmp_path, unit_present=True))
    res = _run(tmp_path, env, "install")
    assert res.returncode == 1
    assert "already exists" in res.stderr


# -- defect 2: the precheck and its commit are one critical section ----------

def test_install_reserves_the_space_it_promised(tmp_path):
    env = _env(tmp_path, _shims(tmp_path))
    assert _run(tmp_path, env, "install").returncode == 0
    reservation = tmp_path / "archive" / ".jobs" / ".reservations" / "newmodel-1"
    assert reservation.exists(), "nothing claimed the space"
    lines = reservation.read_text().split("\n")
    assert lines[0] == str(SIZE)
    assert lines[1].endswith("newmodel.gguf.part")
    assert lines[2] == "newmodel"


def test_a_second_job_installing_the_same_name_is_refused(tmp_path):
    """The unit check upstream cannot catch this: the second job starts before
    the first has written a unit at all."""
    env = _env(tmp_path, _shims(tmp_path))
    assert _run(tmp_path, env, "install").returncode == 0
    res = _run(tmp_path, env, "install", job="newmodel-2")
    assert res.returncode == 1
    assert "already installing newmodel" in res.stderr


def test_a_concurrent_job_must_fit_beside_what_is_already_promised(tmp_path):
    """N jobs each verifying the same free bytes against themselves is the
    whole defect: they were all admitted into space that fits one."""
    env = _env(tmp_path, _shims(tmp_path))
    assert _run(tmp_path, env, "install").returncode == 0

    # A second, differently-named job whose size exceeds the disk once the
    # first job's claim is subtracted.
    huge = 1 << 62
    res = _run(tmp_path, env, "install", size=huge, name="othermodel",
               job="othermodel-1")
    assert res.returncode == 1
    assert "insufficient space" in res.stderr
    assert "reserved by running jobs" in res.stderr


def test_a_stale_reservation_is_pruned_with_its_orphaned_part(tmp_path):
    """A job killed by a crash or a reboot leaves a claim that is never coming
    good, and a `.part` occupying the space it claimed."""
    bindir = _shims(tmp_path)
    env = _env(tmp_path, bindir)
    assert _run(tmp_path, env, "install").returncode == 0

    reservations = tmp_path / "archive" / ".jobs" / ".reservations"
    part = tmp_path / "archive" / "models" / "newmodel.gguf.part"
    part.write_bytes(b"half a download")
    assert (reservations / "newmodel-1").exists()

    # The first job's unit goes away, which is what a crash or a reboot looks
    # like from here: the reservation is on disk, the process behind it is not.
    _set_job_running(bindir, False)
    res = _run(tmp_path, env, "install", name="othermodel", job="othermodel-1")
    assert res.returncode == 0, res.stderr
    assert not (reservations / "newmodel-1").exists()
    assert not part.exists(), "the orphaned .part kept its space"


def test_a_job_that_never_started_does_not_keep_its_claim(tmp_path):
    """If systemd-run fails, the bytes are not in flight and the reservation
    must not outlive the attempt -- pruning only fires on the NEXT install."""
    bindir = _shims(tmp_path)
    (bindir / "systemd-run").write_text("#!/bin/bash\nexit 1\n", encoding="utf-8")
    (bindir / "systemd-run").chmod(0o755)
    env = _env(tmp_path, bindir)

    res = _run(tmp_path, env, "install")
    assert res.returncode == 1
    reservation = tmp_path / "archive" / ".jobs" / ".reservations" / "newmodel-1"
    assert not reservation.exists()


def test_a_finished_job_releases_its_claim(tmp_path):
    env = _env(tmp_path, _shims(tmp_path))
    assert _run(tmp_path, env, "install").returncode == 0
    assert _run(tmp_path, env, "run").returncode == 0
    assert _job(tmp_path)["state"] == "done"
    reservation = tmp_path / "archive" / ".jobs" / ".reservations" / "newmodel-1"
    assert not reservation.exists(), "a finished job still holds its space"


# -- DP-360: the gguf header gate, exercised in the shell it lives in ---------
#
# This section exists because of how DP-360's bug was found. `gguf_header.py`
# had four Python-side tests covering `kv_shape_note` and all four passed,
# while the shell `case` that decides whether the fragment is kept required a
# leading ,"n_layer" and silently dropped every refusal note ever emitted --
# for months. Nothing tested the boundary, so nothing could catch it.
#
# The producer is shimmed rather than run for real: `python3` may not exist on
# a dev box, and what is under test here is the installer's side of the
# contract, not the reader's. `test_gguf_header.py` owns the reader.


def _header_shim(bindir: Path, tmp_path: Path, script: str) -> str:
    """Stand in for `python3 $GGUF_HEADER <file>` and return the header path.

    The gate reads `if hdr="$(python3 "$GGUF_HEADER" "$DEST" 2>/dev/null)"`, so
    both halves have to be shimmed: `python3` becomes a trampoline that runs
    its first argument under bash, and `$GGUF_HEADER` becomes the bash script
    standing in for the reader. `script` is that reader's body — it writes a
    fragment on stdout and exits with the status being tested.
    """
    producer = tmp_path / "header-producer"
    # DP-364: the installer also asks `--ssm-layers` for the cache mode. A
    # script that does not answer it is a dense model, so these fragment tests
    # keep testing the fragment and nothing else.
    if "--ssm-layers" not in script:
        script = '[ "$1" = --ssm-layers ] && { printf 0; exit 0; }\n' + script
    producer.write_text("#!/bin/bash\n" + script, encoding="utf-8")
    producer.chmod(0o755)
    trampoline = bindir / "python3"
    trampoline.write_text('#!/bin/bash\nexec bash "$@"\n', encoding="utf-8")
    trampoline.chmod(0o755)
    return producer.as_posix()


def _run_with_header(tmp_path, script: str) -> dict:
    bindir = _shims(tmp_path)
    env = _env(tmp_path, bindir)
    env["GGUF_HEADER"] = _header_shim(bindir, tmp_path, script)
    res = _run(tmp_path, env, "run")
    assert res.returncode == 0, res.stderr
    return _job(tmp_path)


def test_a_refusal_note_reaches_the_job_status(tmp_path):
    """The live bug, at the layer it lived in.

    gemma4 emits ,"kv_shape_note":... as its FIRST key. Under the old gate --
    `case "$hdr" in ,\\"n_layer\\"*)` -- that fragment matched nothing and was
    dropped, so the one output that exists specifically to STOP a wrong
    estimate never reached derpr. Four passing Python tests said otherwise.
    """
    job = _run_with_header(
        tmp_path,
        'printf \'%s\' \',"kv_shape_note":"sliding-window","ssm_layers":0\'\n'
        "exit 0\n",
    )
    assert job["kv_shape_note"] == "sliding-window"
    assert job["ssm_layers"] == 0


def test_a_shape_fragment_reaches_the_job_status(tmp_path):
    job = _run_with_header(
        tmp_path,
        'printf \'%s\' \',"n_layer":16,"n_kv_head":4,"head_dim":256'
        ',"ssm_layers":48\'\n'
        "exit 0\n",
    )
    assert job["n_layer"] == 16
    assert job["ssm_layers"] == 48


def test_a_key_this_installer_has_never_seen_is_still_kept(tmp_path):
    """The regression-proofing the widened glob did not actually buy.

    A pattern gate has to be re-widened for every key the reader ever grows,
    and the failure mode when it is not is silence. Gating on the exit status
    means the installer never has to know the key names at all -- which is the
    point, because it is the half of the pair that gets deployed late.
    """
    job = _run_with_header(
        tmp_path,
        'printf \'%s\' \',"some_future_key":7,"n_layer":16\'\n'
        "exit 0\n",
    )
    assert job["n_layer"] == 16


def test_a_reader_that_dies_mid_write_contributes_nothing(tmp_path):
    """A partial fragment spliced into the status makes the WHOLE job record
    unparseable, which derpr reports to the operator as "no readable status" --
    strictly worse than having no header facts.

    The old glob accepted this: `,"n_layer":48,"n_kv` starts with a comma, a
    quote and lowercase letters, and contains a '":'. Nothing about a pattern
    match can tell a complete fragment from a truncated one; the exit status
    can, and it is the producer's to set.
    """
    job = _run_with_header(
        tmp_path,
        'printf \'%s\' \',"n_layer":48,"n_kv\'\n'
        "exit 1\n",
    )
    assert job["state"] == "done"
    assert "n_layer" not in job
    assert "n_kv_head" not in job


def test_a_reader_that_reads_nothing_is_not_a_failed_install(tmp_path):
    """A header quirk must never fail an install whose bytes verified. Empty
    output with exit 0 is the reader saying "I could not read this file", which
    is a normal outcome, not an error."""
    job = _run_with_header(tmp_path, "exit 0\n")
    assert job["state"] == "done"
    assert "n_layer" not in job
    assert "ssm_layers" not in job


def test_a_missing_reader_no_longer_finishes_the_install(tmp_path):
    """DP-364 reversed this. A node without `gguf_header.py` used to finish the
    install with no header facts; now the reader is what decides the unit's
    cache mode, and guessing it writes a unit that is silently wrong for one of
    the two architectures. So the job fails -- with the bytes kept, so the
    retry after deploying the reader does not download again."""
    env = _env(tmp_path, _shims(tmp_path))
    env["GGUF_HEADER"] = "no-such-header.py"
    assert _run(tmp_path, env, "run").returncode == 1
    job = _job(tmp_path)
    assert job["state"] == "failed" and job["reason"] == "cache_mode_unknown"
    assert (tmp_path / "archive" / "models" / "newmodel.gguf").exists()


# -- DP-364: the unit is written complete -------------------------------------
#
# CT101 carried a policy wrapper over the koboldcpp binary because the template
# hardcoded `--quantkv 1 --multiuser 4 --smartcache 4` into every unit, and a
# per-model fix made by hand was reverted by the next reinstall. These render
# the REAL template through the installer, so what is asserted is the unit the
# node would push -- the only artefact the process ever reads.

_TEMPLATE = _SERVICES / "koboldcpp-model.service.in"

_HYBRID = (
    '[ "$1" = --ssm-layers ] && { printf 48; exit 0; }\n'
    "printf '%s' ',\"ssm_layers\":48'\n"
)
#: A reader that can walk no header: exit 0 with nothing to say, which is the
#: reader's own "I could not tell" (see gguf_header.ssm_layers).
_UNREADABLE = '[ "$1" = --ssm-layers ] && exit 0\nexit 0\n'


def _render(tmp_path: Path, kv: str = "q8", header: str | None = None,
            **extra_env: str) -> subprocess.CompletedProcess:
    """`header` None keeps _env's default reader, which reports a dense model."""
    bindir = _shims(tmp_path)
    env = _env(tmp_path, bindir)
    env["TEMPLATE"] = _TEMPLATE.as_posix()
    env.update(extra_env)
    if header is not None:
        env["GGUF_HEADER"] = _header_shim(bindir, tmp_path, header)
    return _run(tmp_path, env, "run", kv=kv)


def _exec_start(tmp_path: Path) -> str:
    unit = (tmp_path / "pushed-koboldcpp-newmodel.service").read_text()
    return next(line for line in unit.splitlines() if line.startswith("ExecStart="))


def test_the_template_hardcodes_no_per_model_flag():
    text = _TEMPLATE.read_text(encoding="utf-8")
    assert "@@QUANTKV@@" in text and "@@CACHE_FLAGS@@" in text
    assert "--quantkv 1" not in text
    assert "--smartcache 4" not in text
    assert "--multiuser 4" not in text


@pytest.mark.parametrize("kv,quantkv", [
    ("q8", "--quantkv 1"), ("f16", "--quantkv 0"), ("q4", "--quantkv 2"),
])
def test_the_unit_carries_the_approved_precision(tmp_path, kv, quantkv):
    res = _render(tmp_path, kv)
    assert res.returncode == 0, res.stderr
    line = _exec_start(tmp_path)
    assert f" {quantkv} " in f"{line} "
    assert "--multiuser 1" in line
    job = _job(tmp_path)
    assert job["kv_precision"] == kv


def test_a_dense_model_gets_swap_slots(tmp_path):
    """A dense KV cache truncates, so fast-forward already survives an edit;
    swap slots are what let it switch conversations without reprocessing."""
    res = _render(tmp_path)
    assert res.returncode == 0, res.stderr
    line = _exec_start(tmp_path)
    assert line.endswith("--smartcache 4")
    assert "--smartcachegrid" not in line
    assert _job(tmp_path)["cache_mode"] == "swap"


def test_a_hybrid_gets_the_grid_and_no_swap_slots(tmp_path):
    """A recurrent state cannot be rewound, so only a checkpoint survives an
    edit. And never both flags: koboldcpp.py drops --smartcache whenever
    --smartcachegrid is set, so writing both would only look like both."""
    res = _render(tmp_path, header=_HYBRID)
    assert res.returncode == 0, res.stderr
    line = _exec_start(tmp_path)
    assert line.endswith("--smartcachegrid 32768")
    assert "--smartcache " not in line
    assert _job(tmp_path)["cache_mode"] == "grid"


def test_the_cache_budgets_are_site_settings(tmp_path):
    res = _render(tmp_path, header=_HYBRID, SMARTCACHEGRID_MB="16384")
    assert res.returncode == 0, res.stderr
    assert _exec_start(tmp_path).endswith("--smartcachegrid 16384")


def test_a_model_whose_architecture_cannot_be_read_gets_no_unit(tmp_path):
    """Either guess writes a unit that is silently wrong for one of the two
    architectures, so the node refuses. The verified bytes stay, so a retry
    is not a second multi-GB download."""
    res = _render(tmp_path, header=_UNREADABLE)
    assert res.returncode == 1
    job = _job(tmp_path)
    assert job["state"] == "failed" and job["reason"] == "cache_mode_unknown"
    assert not (tmp_path / "pushed-koboldcpp-newmodel.service").exists()
    assert (tmp_path / "archive" / "models" / "newmodel.gguf").exists()


@pytest.mark.parametrize("kv", ["q8-grid", "int8", "Q8"])
def test_an_unknown_precision_is_refused_before_anything_moves(tmp_path, kv):
    env = _env(tmp_path, _shims(tmp_path))
    res = _run(tmp_path, env, "install", kv=kv)
    assert res.returncode == 1
    assert "invalid kv precision" in res.stderr
    assert not (tmp_path / "archive" / ".jobs" / "newmodel-1.json").exists()


# -- defect 5: the token plumbing is gone -------------------------------------

def test_no_authorization_header_reaches_curls_argv(tmp_path):
    """The bearer used to sit on curl's argv, world-readable through
    /proc/<pid>/cmdline for the length of a multi-GB download. Deleted, not
    fixed: the token was never a wanted feature (DP-347)."""
    env = _env(tmp_path, _shims(tmp_path))
    assert _run(tmp_path, env, "run").returncode == 0
    log = (tmp_path / "curl.log").read_text()
    assert "Authorization" not in log and "Bearer" not in log
    script = _INSTALL.read_text(encoding="utf-8")
    assert "derpr-model-install.token" not in script
    assert "curl_auth" not in script
