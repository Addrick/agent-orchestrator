# tests/test_opencode_provider.py
#
# DP-448 — the opencode provider, driven against a fake `opencode serve` that
# implements just the endpoints derpr calls, with the shapes opencode 1.19.3
# returns (captured from the live sandbox server).

import asyncio
import itertools

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
        self._ids = itertools.count(1)

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
        if path == "/session" and request.method == "POST":
            sid = f"ses_{next(self._ids)}"
            self.sessions[sid] = {"id": sid, "directory": directory, "title": body.get("title"),
                                  "permission": body.get("permission"),
                                  "time": {"created": 1, "updated": next(self._ids)}}
            self.messages[sid] = []
            return web.json_response(self.sessions[sid])
        if path == "/session" and request.method == "GET":
            return web.json_response([s for s in self.sessions.values() if s["directory"] == directory])
        if path == "/session/status":
            return web.json_response({sid: {"type": "busy"} for sid in self.busy})
        sid = parts[1]
        if len(parts) == 2 and request.method == "DELETE":
            self.sessions.pop(sid)
            return web.json_response(True)
        action = parts[2]
        if action == "shell":
            name = body["command"].split("'")[1]
            self.dirs[name] = "directory"
            return web.json_response({"info": {"role": "assistant"}, "parts": []})
        if action == "abort":
            self.busy.discard(sid)
            return web.json_response(True)
        if action == "message" and request.method == "GET":
            return web.json_response(self.messages[sid])
        if action == "message" and request.method == "POST":
            self.messages[sid].append({"info": {"role": "user"}, "parts": body["parts"]})
            self.busy.add(sid)
            await asyncio.sleep(self.reply_delay)
            self.busy.discard(sid)
            info = {"role": "assistant", "time": {"created": 1, "completed": 2}}
            if self.reply_error:
                info["error"] = self.reply_error
            reply = {"info": info, "parts": [
                {"type": "step-start"},
                {"type": "text", "text": self.reply_text},
                {"type": "text", "text": "(synthetic)", "synthetic": True},
            ]}
            self.messages[sid].append(reply)
            return web.json_response(reply)
        raise AssertionError(f"unexpected {request.method} {path}")

    def posts(self):
        return [c for c in self.calls if c[0] == "POST" and c[1].endswith("/message")]


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
