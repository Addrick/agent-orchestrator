# tests/test_opencode_provider.py
#
# DP-448 — the opencode provider, driven against a fake `opencode serve` that
# implements just the endpoints derpr calls, with the shapes opencode 1.19.3
# returns (captured from the live sandbox server).

import asyncio
import base64
import itertools
import os
import re
import shutil
import subprocess

import pytest
from aiohttp import web
from aiohttp.test_utils import TestServer

from config import global_config
from src.engine import LLMCommunicationError, TextEngine
from src.engine.providers import opencode
from src.engine.providers.opencode import OpencodeProvider, new_messages, opencode_model
from src.utils.model_utils import get_model_prefix

ROOT = "/home/agent/work"


class FakeOpencode:
    """In-memory opencode: sessions per directory, a workspace tree, and a
    configurable reply for each posted prompt."""

    def __init__(self) -> None:
        self.dirs = {"hello.py": "file"}
        self.sessions = {}      # id -> session dict
        self.messages = {}      # id -> list of {info, parts}
        self.busy = set()
        self.calls = []         # (method, path, directory, body)
        self.reply_text = "done."
        self.reply_error = None
        self.reply_delay = 0.0
        self.scripts = []       # (directory, script) run through /shell
        self.shell_output = ""  # what a non-mkdir script prints
        self.file_root = None   # local Path that /file/content serves from
        self._ids = itertools.count(1)

    def add(self, sid, role, parts, **info):
        """Store a message the way opencode returns it: ids on every part."""
        mid = f"msg_{next(self._ids)}"
        stored = [{"id": f"prt_{next(self._ids)}", "sessionID": sid, "messageID": mid, **p} for p in parts]
        message = {"info": {"id": mid, "role": role, **info}, "parts": stored}
        self.messages[sid].append(message)
        return message

    def app(self) -> web.Application:
        app = web.Application()
        app.router.add_route("*", "/{tail:.*}", self.handle)
        return app

    async def handle(self, request: web.Request) -> web.Response:
        body = await request.json() if request.can_read_body else None
        directory = request.query.get("directory")
        path = request.path
        self.calls.append((request.method, path, directory, body))
        parts = path.strip("/").split("/")

        if path == "/file":
            assert directory == ROOT
            return web.json_response([
                {"name": n, "path": n, "absolute": f"{ROOT}/{n}", "type": t, "ignored": False}
                for n, t in self.dirs.items()
            ])
        if path == "/file/content":
            data = (self.file_root / request.query["path"]).read_bytes()
            try:
                # Like opencode 1.19: a text file loses its trailing newlines.
                return web.json_response({"type": "text", "content": data.decode().rstrip("\n")})
            except UnicodeDecodeError:
                return web.json_response({"type": "binary", "encoding": "base64",
                                          "content": base64.b64encode(data).decode()})
        if path == "/session" and request.method == "POST":
            sid = f"ses_{next(self._ids)}"
            self.sessions[sid] = {"id": sid, "directory": directory, "title": body.get("title"),
                                  "permission": body.get("permission"),
                                  "time": {"created": 1, "updated": next(self._ids)}}
            self.messages[sid] = []
            return web.json_response(self.sessions[sid])
        if path == "/session" and request.method == "GET":
            return web.json_response([s for s in self.sessions.values() if s["directory"] == directory])
        if path == "/instance/dispose":
            return web.json_response(True)
        if path == "/session/status":
            return web.json_response({sid: {"type": "busy"} for sid in self.busy})
        return await self.session_route(request.method, parts, directory, body)

    async def session_route(self, method, parts, directory, body) -> web.Response:
        """/session/{id}[/action[/…]]."""
        sid = parts[1]
        if len(parts) == 2 and method == "DELETE":
            self.sessions.pop(sid)
            return web.json_response(True)
        action = parts[2]
        if action == "shell":
            return self.shell(directory, body["command"])
        if action == "prompt_async":
            self.add(sid, "user", body["parts"])
            return web.Response(status=204)
        if action == "message" and len(parts) == 6 and method == "PATCH":
            part = next(p for m in self.messages[sid] for p in m["parts"] if p["id"] == parts[5])
            part.update(body)
            return web.json_response(part)
        if action == "abort":
            self.busy.discard(sid)
            return web.json_response(True)
        if action == "message" and method == "GET":
            return web.json_response(self.messages[sid])
        if action == "message" and method == "POST":
            self.add(sid, "user", body["parts"])
            self.busy.add(sid)
            await asyncio.sleep(self.reply_delay)
            self.busy.discard(sid)
            info = {"time": {"created": 1, "completed": 2}, "finish": "stop"}
            if self.reply_error:
                info["error"] = self.reply_error
            reply = self.add(sid, "assistant", [
                {"type": "step-start"},
                {"type": "text", "text": self.reply_text},
                {"type": "text", "text": "(synthetic)", "synthetic": True},
            ], **info)
            return web.json_response(reply)
        raise AssertionError(f"unexpected {method} /{'/'.join(parts)}")

    def shell(self, directory, command) -> web.Response:
        encoded = re.fullmatch(r"echo (\S+) \| base64 -d \| bash", command).group(1)
        script = base64.b64decode(encoded).decode()
        self.scripts.append((directory, script))
        made = re.fullmatch(r"mkdir -p -- '([^']+)'", script)
        if made:
            self.dirs[made.group(1)] = "directory"
        output = "" if made else (self.shell_output(script) if callable(self.shell_output)
                                  else self.shell_output)
        return web.json_response({"info": {"role": "assistant"}, "parts": [
            {"type": "tool", "tool": "bash", "state": {"status": "completed", "output": output}}]})

    def posts(self):
        return [c for c in self.calls if c[0] == "POST" and c[1].endswith("/message")]


def real_bash(cwd):
    """A `shell_output` that runs each script in a real bash in `cwd`, for the
    tests whose point is what the script does."""
    bash = shutil.which("bash")
    if bash is None:
        pytest.skip("needs bash")

    def run(script):
        # On stdin, as `run_shell` sends it (and a Windows command line caps at 32K).
        done = subprocess.run([bash], input=script, cwd=cwd, capture_output=True, text=True)
        return done.stdout + done.stderr
    return run


@pytest.fixture
async def fake(monkeypatch):
    fake = FakeOpencode()
    server = TestServer(fake.app())
    await server.start_server()
    monkeypatch.setattr(global_config, "OPENCODE_SERVER_URL", str(server.make_url("")).rstrip("/"))
    monkeypatch.setattr(global_config, "OPENCODE_SERVER_PASSWORD", "pw")
    monkeypatch.setattr(global_config, "OPENCODE_WORKSPACES_ROOT", ROOT)
    yield fake
    await server.close()


def _history(*turns):
    return {"persona_prompt": "You are codr.", "message_history": [
        {"role": role, "content": text} for role, text in turns
    ], "current_message": {"text": "", "image_url": None}}


async def _generate(provider, history, *, request_id="req-1", model="opencode", persona="codr"):
    config = {"model_name": model, "persona_name": persona, "request_id": request_id}
    return await provider.generate(config, history, None)


# --- routing & parsing ------------------------------------------------------


def test_routes_before_the_claude_substring_match(monkeypatch):
    monkeypatch.setattr(global_config, "OPENCODE_SERVER_URL", "http://sandbox:4096")
    provider = TextEngine()._registry.resolve("opencode-anthropic/claude-sonnet-5")
    assert isinstance(provider, OpencodeProvider)
    assert get_model_prefix("opencode-anthropic/claude-x") == "opencode"


def test_route_refused_without_a_server_url(monkeypatch):
    monkeypatch.setattr(global_config, "OPENCODE_SERVER_URL", "")
    with pytest.raises(LLMCommunicationError, match="OPENCODE_SERVER_URL"):
        TextEngine()._registry.resolve("opencode")


def test_model_names():
    assert opencode_model("opencode") is None
    assert opencode_model("opencode-kcpp/local") == {"providerID": "kcpp", "modelID": "local"}
    for bad in ("opencode-kcpp", "opencodex/y", "opencode-/x"):
        with pytest.raises(LLMCommunicationError):
            opencode_model(bad)


def test_only_messages_after_the_last_reply_are_new():
    h = [{"role": "user", "content": "a"}, {"role": "assistant", "content": "b"},
         {"role": "user", "content": "c"}, {"role": "user", "content": "d"}]
    assert [m["content"] for m in new_messages(h)] == ["c", "d"]
    assert new_messages(h[:1]) == h[:1]          # no reply yet: the whole window


# --- workspace & session ----------------------------------------------------


async def test_first_call_creates_workspace_and_session_then_reuses_both(fake):
    provider = OpencodeProvider(TextEngine())
    result, payload = await _generate(provider, _history(("user", "build it")))

    assert result == {"type": "text", "content": "done."}   # synthetic part dropped
    assert fake.dirs["codr"] == "directory"
    (session,) = fake.sessions.values()                       # bootstrap session deleted
    assert session["directory"] == f"{ROOT}/codr"
    assert session["title"] == "derpr:codr"
    assert session["permission"] == opencode.SESSION_PERMISSIONS
    assert payload["session"] == session["id"] and payload["resumed"] is False

    # A fresh provider (a derpr restart) finds the same session by its title.
    await _generate(OpencodeProvider(TextEngine()),
                    _history(("user", "build it"), ("assistant", "done."), ("user", "next")),
                    request_id="req-2")
    assert len(fake.sessions) == 1
    assert len([c for c in fake.calls if c[1] == "/session/" + session["id"] + "/shell"]) == 0
    assert fake.scripts == [(ROOT, "mkdir -p -- 'codr'")]


async def test_existing_workspace_is_not_recreated(fake):
    fake.dirs["codr"] = "directory"
    await _generate(OpencodeProvider(TextEngine()), _history(("user", "hi")))
    assert not [c for c in fake.calls if c[1].endswith("/shell")]


async def test_forwards_only_new_messages_with_system_model_and_key(fake):
    history = _history(("user", "old"), ("assistant", "old reply"), ("user", "new one"))
    await _generate(OpencodeProvider(TextEngine()), history, model="opencode-kcpp/local")
    (_, _, directory, body), = fake.posts()
    assert directory == f"{ROOT}/codr"
    assert body["parts"] == [{"type": "text", "text": "User: new one",
                              "metadata": {"derpr_key": "req-1"}}]
    assert body["system"] == "You are codr."
    assert body["model"] == {"providerID": "kcpp", "modelID": "local"}


# --- retries, errors, timeouts ---------------------------------------------


async def test_a_driver_retry_resumes_instead_of_posting_again(fake):
    provider = OpencodeProvider(TextEngine())
    history = _history(("user", "long task"))
    await _generate(provider, history)
    _, payload = await _generate(provider, history)          # same request_id
    assert len(fake.posts()) == 1
    assert payload["resumed"] is True


async def test_a_retry_resumes_even_after_another_calls_message(fake):
    """The lock is released between driver attempts, so a second call's turn
    can land after ours; our retry must still find its own turn, not re-post,
    and must read its own reply, not the other call's."""
    provider = OpencodeProvider(TextEngine())
    history = _history(("user", "long task"))
    await _generate(provider, history, request_id="a")
    fake.reply_text = "other reply."
    await _generate(provider, _history(("user", "other")), request_id="b")
    result, payload = await _generate(provider, history, request_id="a")
    assert len(fake.posts()) == 2
    assert payload["resumed"] is True
    assert result["content"] == "done."


async def test_same_text_on_a_new_call_is_a_new_turn(fake):
    provider = OpencodeProvider(TextEngine())
    history = _history(("user", "again"))
    await _generate(provider, history, request_id="a")
    await _generate(provider, history, request_id="b")        # e.g. a user regenerate
    assert len(fake.posts()) == 2


async def test_a_failed_reply_raises_even_with_partial_text(fake):
    fake.reply_error = {"name": "MessageAbortedError", "data": {"message": "aborted"}}
    with pytest.raises(LLMCommunicationError, match="MessageAbortedError"):
        await _generate(OpencodeProvider(TextEngine()), _history(("user", "x")))


async def test_a_timed_out_turn_is_aborted_on_the_server(fake, monkeypatch):
    monkeypatch.setattr(global_config, "OPENCODE_CALL_TIMEOUT_SECONDS", 0.2)
    fake.reply_delay = 1.0
    with pytest.raises(LLMCommunicationError, match="timed out") as exc:
        await _generate(OpencodeProvider(TextEngine()), _history(("user", "x")))
    assert [c for c in fake.calls if c[1].endswith("/abort")]
    assert exc.value.api_payload["workspace"] == f"{ROOT}/codr"


async def test_a_cancelled_call_aborts_its_turn_on_the_server(fake):
    """The portal's stop button cancels the call; the agent must stop too."""
    fake.reply_delay = 5.0
    task = asyncio.create_task(_generate(OpencodeProvider(TextEngine()), _history(("user", "x"))))
    while not fake.posts():
        await asyncio.sleep(0.01)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert [c for c in fake.calls if c[1].endswith("/abort")]


async def test_http_error_carries_the_payload(fake, monkeypatch):
    monkeypatch.setattr(global_config, "OPENCODE_WORKSPACES_ROOT", "/elsewhere")
    with pytest.raises(LLMCommunicationError) as exc:
        await _generate(OpencodeProvider(TextEngine()), _history(("user", "x")))
    assert exc.value.api_payload["model"] == "opencode"


# --- through the driver -----------------------------------------------------


async def test_driver_streams_the_reply_and_keeps_request_id_across_retries(fake, monkeypatch):
    engine = TextEngine()
    seen = []
    real = OpencodeProvider.generate

    async def flaky(self, config, history_object, tools=None):
        seen.append(config["request_id"])
        if len(seen) == 1:
            raise LLMCommunicationError("transient")
        return await real(self, config, history_object, tools)

    monkeypatch.setattr(OpencodeProvider, "generate", flaky)
    monkeypatch.setattr("src.engine.driver.EMPTY_RESPONSE_RETRY_DELAY", 0)
    events = [ev async for ev in engine._stream_response(
        {"model_name": "opencode", "persona_name": "codr"}, _history(("user", "hi")))]
    assert events[-1] == {"type": "done", "full_text": "done."}
    assert len(seen) == 2 and seen[0] == seen[1]


# --- a running codr task (DP-449) ---------------------------------------------


def _auto_turn(fake, sid, *, finished=False):
    fake.add(sid, "user", [{"type": "text", "text": "task", "metadata": {opencode.AUTO_FIELD: {"task": "t1"}}}])
    fake.add(sid, "assistant", [{"type": "tool", "tool": "bash", "state": {"status": "completed"}}],
             time={"created": 1}, finish="tool-calls")
    if finished:
        fake.add(sid, "assistant", [{"type": "text", "text": "all done"}],
                 time={"created": 1, "completed": 2}, finish="stop")


async def test_a_message_during_a_codr_task_is_delivered_into_it(fake):
    provider = OpencodeProvider(TextEngine())
    await _generate(provider, _history(("user", "hi")), request_id="a")
    (sid,) = fake.sessions
    _auto_turn(fake, sid)
    fake.busy.add(sid)

    result, payload = await _generate(provider, _history(("user", "how is it going?")), request_id="b")
    assert result == {"type": "reaction", "content": opencode.INJECTED_ACK}
    assert payload["delivered_into_task"]
    assert len(fake.posts()) == 1                              # no blocking turn posted
    delivered = fake.messages[sid][-1]["parts"][0]
    assert delivered["text"] == "User: how is it going?"
    assert delivered["metadata"] == {"derpr_key": "b", opencode.INJECTED_FIELD: True}

    # A driver retry of the same call finds it and does not deliver it twice.
    await _generate(provider, _history(("user", "how is it going?")), request_id="b")
    assert len(fake.messages[sid]) == 5


async def test_a_message_after_a_codr_task_finished_is_an_ordinary_turn(fake):
    provider = OpencodeProvider(TextEngine())
    await _generate(provider, _history(("user", "hi")), request_id="a")
    (sid,) = fake.sessions
    _auto_turn(fake, sid, finished=True)
    result, _ = await _generate(provider, _history(("user", "and now?")), request_id="b")
    assert result["content"] == "done."
    assert len(fake.posts()) == 2


async def test_a_task_just_started_counts_as_running_before_the_server_says_busy(fake):
    """prompt_async returns before the session shows as busy; a chat message in
    that gap must not post a blocking turn behind the task."""
    provider = OpencodeProvider(TextEngine())
    await _generate(provider, _history(("user", "hi")), request_id="a")
    (sid,) = fake.sessions
    fake.add(sid, "user", [{"type": "text", "text": "task", "metadata": {opencode.AUTO_FIELD: {"task": "t1"}}}])
    result, _ = await _generate(provider, _history(("user", "quick q")), request_id="b")
    assert result["content"] == opencode.INJECTED_ACK


async def test_run_shell_sends_the_script_base64_encoded(fake):
    """opencode's /shell expands `$` before running the command, so the
    script must not reach it as plain text."""
    from aiohttp import ClientSession
    fake.shell_output = "ok\n"
    async with ClientSession(headers=opencode.auth_headers()) as http:
        client = opencode.OpencodeClient(http, f"{ROOT}/codr")
        out = await client.run_shell('false; echo "exit=$?"')
    assert out == "ok\n"
    assert fake.scripts == [(f"{ROOT}/codr", 'false; echo "exit=$?"')]
    shell_call = next(c for c in fake.calls if c[1].endswith("/shell"))
    assert "$" not in shell_call[3]["command"]
    assert not fake.sessions                                   # throwaway session deleted


async def test_write_file_sends_any_size_in_pieces_under_the_argument_cap(fake, tmp_path):
    """One shell command over 128 KiB is an HTTP 500 on the real server, so a
    big file must arrive in several, and land whole (DP-455)."""
    from aiohttp import ClientSession
    fake.shell_output = real_bash(tmp_path)
    data = os.urandom(300_000)
    async with ClientSession(headers=opencode.auth_headers()) as http:
        client = opencode.OpencodeClient(http, f"{ROOT}/codr")
        await client.write_file("sub/dir/big.bin", data)
    assert (tmp_path / "sub/dir/big.bin").read_bytes() == data
    assert not (tmp_path / "sub/dir/big.bin.derpr-part").exists()
    commands = [c[3]["command"] for c in fake.calls if c[1].endswith("/shell")]
    assert len(commands) > 5 and max(len(c) for c in commands) < 128 * 1024


async def test_write_file_raises_when_the_written_file_does_not_match(fake):
    """The shell reports a failed command as output, so a write that did not
    land must be caught by the check, not trusted."""
    from aiohttp import ClientSession
    fake.shell_output = "sha256sum: No space left on device\n"
    async with ClientSession(headers=opencode.auth_headers()) as http:
        client = opencode.OpencodeClient(http, f"{ROOT}/codr")
        with pytest.raises(LLMCommunicationError, match="writing big.bin failed"):
            await client.write_file("big.bin", b"x" * 10)


async def test_read_file_returns_any_file_byte_for_byte(fake, tmp_path):
    """opencode strips a text file's trailing newlines, so a read that went
    through its text path would be short."""
    from aiohttp import ClientSession
    fake.shell_output, fake.file_root = real_bash(tmp_path), tmp_path
    binary, text = bytes([0, 255]) + b"bundle", "café\n\n".encode()
    (tmp_path / "out.bundle").write_bytes(binary)
    (tmp_path / "notes.md").write_bytes(text)
    async with ClientSession(headers=opencode.auth_headers()) as http:
        client = opencode.OpencodeClient(http, f"{ROOT}/codr")
        assert await client.read_file("out.bundle") == binary
        assert await client.read_file("notes.md") == text
    assert sorted(p.name for p in tmp_path.iterdir()) == ["notes.md", "out.bundle"]
