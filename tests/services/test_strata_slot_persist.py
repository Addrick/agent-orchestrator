"""strata-slot-persist (DP-405B): CT102's save-on-stop / restore-on-start, against a fake Strata.

Runs the real shell script. What matters is what it must never do: block a stop, restore a
half-written or another model's session, or restore the same session twice.
"""
import json
import os
import shutil
import subprocess
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "services" / "strata" / "strata-slot-persist"
SH, CURL = shutil.which("sh"), shutil.which("curl")
pytestmark = pytest.mark.skipif(not (SH and CURL), reason="needs sh and curl")
MODEL = "qwen3.8-flash-next-dau-abl-tb-iq3_s"


class FakeStrata:
    """/v1/models plus llama-server's /slots/0 save/restore, writing into a real directory."""

    def __init__(self, slot_dir, statuses=("loaded",), save_status=200):
        self.slot_dir, self.statuses, self.save_status = slot_dir, list(statuses), save_status
        self.calls = []
        fake = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                status = fake.statuses.pop(0) if len(fake.statuses) > 1 else fake.statuses[0]
                self._json(200, {"object": "list", "data": [{"id": MODEL, "status": {"value": status}}]})

            def do_POST(self):
                payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                action = self.path.split("action=")[1]
                fake.calls.append((action, payload["filename"], self.headers["Content-Type"]))
                path = Path(fake.slot_dir) / payload["filename"]
                if action == "save" and fake.save_status == 200:
                    path.write_bytes(b"session")
                code = fake.save_status if action == "save" else (200 if path.exists() else 404)
                self._json(code, {})

            def _json(self, code, payload):
                body = json.dumps(payload).encode()
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *_):
                pass

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}"

    def close(self):
        self.server.shutdown()


def run(action, url, slot_dir, **env):
    full = {**os.environ, "STRATA_URL": url, "STRATA_SLOT_DIR": str(slot_dir), "STRATA_POLL_S": "1", **env}
    return subprocess.run([SH, str(SCRIPT), action], env=full, capture_output=True, text=True, timeout=60)


@pytest.fixture
def strata(tmp_path):
    fakes = []

    def make(**kw):
        fakes.append(FakeStrata(tmp_path, **kw))
        return fakes[-1]
    yield make
    for fake in fakes:
        fake.close()


def test_save_writes_under_a_temporary_name_then_renames(strata, tmp_path):
    fake = strata()
    assert run("save", fake.url, tmp_path).returncode == 0
    assert fake.calls == [("save", f"shutdown-live-{MODEL}.bin.part", "application/json")]
    assert (tmp_path / f"shutdown-live-{MODEL}.bin").read_bytes() == b"session"
    assert not (tmp_path / f"shutdown-live-{MODEL}.bin.part").exists()


def test_nothing_live_drops_an_older_copy(strata, tmp_path):
    """Strata answers 400 "no complete session to save": restoring an older session would be wrong."""
    (tmp_path / f"shutdown-live-{MODEL}.bin").write_bytes(b"old")
    fake = strata(save_status=400)
    assert run("save", fake.url, tmp_path).returncode == 0
    assert not list(tmp_path.iterdir())


def test_a_failed_save_never_leaves_a_file_to_restore(strata, tmp_path):
    fake = strata(save_status=500)
    (tmp_path / f"shutdown-live-{MODEL}.bin.part").write_bytes(b"half")
    assert run("save", fake.url, tmp_path).returncode == 0
    assert not (tmp_path / f"shutdown-live-{MODEL}.bin.part").exists()


def test_a_dead_server_does_not_block_the_stop(tmp_path):
    result = run("save", "http://127.0.0.1:9", tmp_path)
    assert result.returncode == 0
    assert "nothing to save" in result.stdout


def test_restore_waits_for_the_model_then_restores_once(strata, tmp_path):
    (tmp_path / f"shutdown-live-{MODEL}.bin").write_bytes(b"session")
    fake = strata(statuses=["loading", "loading", "loaded"])
    assert run("restore", fake.url, tmp_path).returncode == 0
    assert fake.calls == [("restore", f"shutdown-live-{MODEL}.bin", "application/json")]
    assert (tmp_path / f"shutdown-live-{MODEL}.bin.restored").exists()
    assert not (tmp_path / f"shutdown-live-{MODEL}.bin").exists()


def test_restore_ignores_another_models_session(strata, tmp_path):
    (tmp_path / "shutdown-live-some-other-model.bin").write_bytes(b"session")
    fake = strata()
    assert run("restore", fake.url, tmp_path).returncode == 0
    assert fake.calls == []


def test_restore_gives_up_when_the_model_never_loads(strata, tmp_path):
    (tmp_path / f"shutdown-live-{MODEL}.bin").write_bytes(b"session")
    fake = strata(statuses=["loading"])
    result = run("restore", fake.url, tmp_path, STRATA_READY_TIMEOUT_S="2")
    assert result.returncode == 0
    assert fake.calls == []
