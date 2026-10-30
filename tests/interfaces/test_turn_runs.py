"""DP-487: a portal turn outlives the connection that started it.

The adapter half runs under real uvicorn on a real port: TestClient does not
deliver a client disconnect the way uvicorn does (it is what DP-239 found), and
"the reply survives a dropped connection" is exactly a claim about that path.
"""

import asyncio
import contextlib
import json
import socket
from typing import Any, AsyncIterator, List, Optional

import httpx
import pytest
import uvicorn

from src.interfaces.turn_runs import DONE_FRAME, ReplayGone, TurnRuns
from tests.interfaces.test_kobold_engine_adapter import (
    KoboldAdapter,
    _fetch_portal_rows,
    _make_real_adapter,
)


@pytest.fixture(autouse=True)
def _bypass_control_plane_auth(monkeypatch):
    monkeypatch.setattr(KoboldAdapter, "_valid_control_token", lambda self, tok: True)


# ---------------------------------------------------------------------------
# TurnRuns — the registry on its own
# ---------------------------------------------------------------------------

KEY = ("p", "portal", "web_ui", None)


def _frames(ev: Any) -> List[bytes]:
    return [f"data: {ev}\n\n".encode()]


def _errors(msg: str) -> List[bytes]:
    return [f"data: ERR {msg}\n\n".encode(), DONE_FRAME]


async def _collect(it: AsyncIterator[bytes]) -> List[bytes]:
    return [f async for f in it]


async def test_follower_leaving_does_not_stop_the_run():
    gate = asyncio.Event()
    finished: List[str] = []

    async def events():
        yield "a"
        await gate.wait()
        yield "b"

    runs = TurnRuns()
    run = runs.start(KEY, "chat", events, _frames, error_frames=_errors,
                     on_finish=finished.append)
    follower = run.follow(0)
    first = await follower.__anext__()
    assert b"derpr-run" in first
    assert await follower.__anext__() == b"data: a\n\n"
    await follower.aclose()  # the client went away

    assert runs.active(KEY) is run
    gate.set()
    assert run.task is not None
    await run.task
    assert finished == ["done"]
    # A late follower gets the whole turn, from the start.
    replay = await _collect(run.follow(0))
    assert replay[1:] == [b"data: a\n\n", b"data: b\n\n"]
    assert runs.active(KEY) is None


async def test_cancel_closes_the_engine_generator_and_ends_followers():
    closed: List[str] = []

    async def events():
        try:
            yield "a"
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            closed.append("cancelled")
            raise

    runs = TurnRuns()
    run = runs.start(KEY, "chat", events, _frames, error_frames=_errors)
    follower = asyncio.create_task(_collect(run.follow(0)))
    await asyncio.sleep(0.05)
    assert await runs.cancel(run.run_id) is True
    assert closed == ["cancelled"]
    got = await asyncio.wait_for(follower, 2)
    assert got[-1] == DONE_FRAME
    assert await runs.cancel(run.run_id) is False  # already finished


async def test_engine_exception_ends_the_turn_with_an_error_frame():
    async def events():
        yield "a"
        raise RuntimeError("boom")

    runs = TurnRuns()
    run = runs.start(KEY, "chat", events, _frames, error_frames=_errors)
    got = await _collect(run.follow(0))
    assert got[-2].startswith(b"data: ERR")
    assert got[-1] == DONE_FRAME


async def test_trimmed_window_refuses_replay_but_keeps_live_tail():
    gate = asyncio.Event()

    async def events():
        for i in range(10):
            yield "x" * 50 + str(i)
        await gate.wait()
        yield "last"

    runs = TurnRuns(max_run_bytes=300)
    run = runs.start(KEY, "chat", events, _frames, error_frames=_errors)
    await asyncio.sleep(0.05)
    assert run.base > 0
    with pytest.raises(ReplayGone):
        await _collect(run.follow(0))
    live = asyncio.create_task(_collect(run.follow(run.end)))
    await asyncio.sleep(0.01)
    gate.set()
    assert await asyncio.wait_for(live, 2) == [b"data: last\n\n"]


async def test_finished_runs_are_purged_after_the_grace_period():
    async def events():
        yield "a"

    runs = TurnRuns(finished_grace_s=0.0)
    run = runs.start(KEY, "chat", events, _frames, error_frames=_errors)
    assert run.task is not None
    await run.task
    await asyncio.sleep(0.01)
    assert runs.get(run.run_id) is None


async def test_shutdown_cancels_every_running_turn():
    cancelled: List[str] = []

    async def events():
        try:
            yield "a"
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            cancelled.append("x")
            raise

    runs = TurnRuns()
    runs.start(KEY, "chat", events, _frames, error_frames=_errors)
    runs.start(("q", "portal", "web_ui", None), "chat", events, _frames, error_frames=_errors)
    await asyncio.sleep(0.05)
    await runs.shutdown()
    assert cancelled == ["x", "x"]


@pytest.mark.parametrize("stop", ["cancel", "shutdown"])
async def test_a_run_stopped_before_its_first_step_still_ends(stop):
    # Cancelling a task before it first runs never enters its coroutine, so
    # _drive's finally (which marks the run finished) never runs. Unsealed,
    # followers hang and active() refuses every later send in the conversation.
    async def events():
        yield "a"

    runs = TurnRuns()
    run = runs.start(KEY, "chat", events, _frames, error_frames=_errors)
    if stop == "cancel":
        assert await runs.cancel(run.run_id) is True
    else:
        await runs.shutdown()
    assert run.done
    assert runs.active(KEY) is None
    frames = await asyncio.wait_for(_collect(run.follow(0)), 1.0)
    assert frames[-1] == DONE_FRAME


# ---------------------------------------------------------------------------
# The adapter, under real uvicorn
# ---------------------------------------------------------------------------

def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


@contextlib.asynccontextmanager
async def _serve(adapter: Any) -> AsyncIterator[str]:
    port = _free_port()
    server = uvicorn.Server(uvicorn.Config(adapter.app, host="127.0.0.1", port=port,
                                           log_level="warning"))
    task = asyncio.create_task(server.serve())
    try:
        for _ in range(100):
            if server.started:
                break
            await asyncio.sleep(0.05)
        assert server.started, "uvicorn did not start"
        yield f"http://127.0.0.1:{port}"
    finally:
        server.should_exit = True
        with contextlib.suppress(asyncio.TimeoutError):
            await asyncio.wait_for(task, timeout=10)


def _gated_engine(gate: asyncio.Event):
    """LLM stub: writes 'first ', waits for `gate`, then 'second'."""
    async def stream_messages(*args, **kwargs):
        yield {"type": "api_payload", "payload": {}}
        yield {"type": "text_delta", "text": "first "}
        await gate.wait()
        yield {"type": "text_delta", "text": "second"}
        yield {"type": "done", "full_text": "first second"}
    return stream_messages


def _body(text: str = "hi") -> dict:
    return {"stream": True, "derpr_user_text": text, "messages": []}


async def _start_and_drop(client: httpx.AsyncClient, base: str) -> str:
    """Open a turn, read until its first token, then drop the connection
    (the phone screen going dark). Returns the run id."""
    run_id: Optional[str] = None
    async with client.stream("POST", f"{base}/v1/chat/completions", json=_body()) as r:
        assert r.status_code == 200
        buf = ""
        async for chunk in r.aiter_text():
            buf += chunk
            if run_id is None and "derpr-run" in buf:
                line = buf.split("event: derpr-run\ndata: ", 1)[1].split("\n", 1)[0]
                run_id = json.loads(line)["run_id"]
            if "first " in buf:
                break
    assert run_id
    return run_id


async def _wait_for_reply(mm: Any, text: str) -> List[Any]:
    for _ in range(100):
        rows = _fetch_portal_rows(mm, "test_persona")
        if len(rows) == 2 and rows[1]["content"] == text:
            return rows
        await asyncio.sleep(0.05)
    raise AssertionError(f"reply never saved; rows={_fetch_portal_rows(mm, 'test_persona')}")


async def test_reply_survives_a_dropped_connection():
    gate = asyncio.Event()
    adapter, mm, _, _ = _make_real_adapter(stream_messages=_gated_engine(gate))
    async with _serve(adapter) as base, httpx.AsyncClient(timeout=10) as client:
        await _start_and_drop(client, base)
        await asyncio.sleep(0.3)
        # Before DP-487 the drop saved "first " as a stopped turn right here.
        rows = _fetch_portal_rows(mm, "test_persona")
        assert [r["author_role"] for r in rows] == ["user"]
        gate.set()
        rows = await _wait_for_reply(mm, "first second")
    assert rows[1]["reply_to_id"] == rows[0]["interaction_id"]
    mm.close()


async def test_reattach_replays_what_was_missed_then_follows_live():
    gate = asyncio.Event()
    adapter, mm, _, _ = _make_real_adapter(stream_messages=_gated_engine(gate))
    async with _serve(adapter) as base, httpx.AsyncClient(timeout=10) as client:
        run_id = await _start_and_drop(client, base)

        active = (await client.get(f"{base}/api/v1/session/test_persona/active_turn",
                                   params={"channel": "web_ui"})).json()
        assert active["run"]["run_id"] == run_id

        async def reattach() -> str:
            out = ""
            async with client.stream("GET", f"{base}/api/v1/turns/{run_id}/events",
                                     params={"from": 0}) as r:
                assert r.status_code == 200
                async for chunk in r.aiter_text():
                    out += chunk
            return out

        follower = asyncio.create_task(reattach())
        await asyncio.sleep(0.2)
        gate.set()
        wire = await asyncio.wait_for(follower, 10)
    assert wire.index("derpr-run") < wire.index("first ") < wire.index("second")
    assert wire.rstrip().endswith("data: [DONE]")
    mm.close()


async def test_a_send_while_a_turn_runs_is_refused_with_the_run():
    gate = asyncio.Event()
    adapter, mm, _, _ = _make_real_adapter(stream_messages=_gated_engine(gate))
    async with _serve(adapter) as base, httpx.AsyncClient(timeout=10) as client:
        run_id = await _start_and_drop(client, base)
        r = await client.post(f"{base}/v1/chat/completions", json=_body("again"))
        assert r.status_code == 409
        assert r.json()["error"]["code"] == "turn_running"
        assert r.json()["derpr_run"]["run_id"] == run_id
        gate.set()
        await _wait_for_reply(mm, "first second")
    # The refused send wrote nothing.
    assert [r["content"] for r in _fetch_portal_rows(mm, "test_persona")] == ["hi", "first second"]
    mm.close()


async def test_stop_saves_what_the_turn_had():
    gate = asyncio.Event()  # never set: only a stop ends this turn
    adapter, mm, _, _ = _make_real_adapter(stream_messages=_gated_engine(gate))
    async with _serve(adapter) as base, httpx.AsyncClient(timeout=10) as client:
        run_id = await _start_and_drop(client, base)
        r = await client.post(f"{base}/v1/chat/cancel", json={"run_id": run_id})
        assert r.json() == {"stopped": True}
        # Saved by the time the stop returns — no re-sync race for the portal.
        rows = _fetch_portal_rows(mm, "test_persona")
        assert rows[-1]["author_role"] == "assistant"
        assert rows[-1]["content"] == "first "
        active = (await client.get(f"{base}/api/v1/session/test_persona/active_turn")).json()
        assert active["run"] is None
    mm.close()


async def test_events_for_an_unknown_run_is_404():
    adapter, mm, _, _ = _make_real_adapter()
    async with _serve(adapter) as base, httpx.AsyncClient(timeout=10) as client:
        r = await client.get(f"{base}/api/v1/turns/nope/events")
    assert r.status_code == 404
    mm.close()


async def test_a_stop_that_beats_its_turn_means_the_turn_never_runs():
    """The portal names the run, so a stop pressed before the first frame
    still finds it: the server remembers the stop and refuses the turn."""
    adapter, mm, _, _ = _make_real_adapter()
    async with _serve(adapter) as base, httpx.AsyncClient(timeout=10) as client:
        rid = "0123456789abcdef0123456789abcdef"
        r = await client.post(f"{base}/v1/chat/cancel", json={"run_id": rid})
        assert r.json() == {"stopped": False}
        r = await client.post(f"{base}/v1/chat/completions",
                              json={**_body(), "derpr_run_id": rid})
        assert r.status_code == 409
        assert r.json()["error"]["code"] == "stopped"
    assert _fetch_portal_rows(mm, "test_persona") == []
    mm.close()


async def test_the_run_takes_the_name_the_client_gave_it():
    gate = asyncio.Event()
    adapter, mm, _, _ = _make_real_adapter(stream_messages=_gated_engine(gate))
    rid = "abcdef0123456789abcdef0123456789"
    async with _serve(adapter) as base, httpx.AsyncClient(timeout=10) as client:
        async with client.stream("POST", f"{base}/v1/chat/completions",
                                 json={**_body(), "derpr_run_id": rid}) as r:
            async for chunk in r.aiter_text():
                assert f'"run_id": "{rid}"' in chunk
                break
        r = await client.post(f"{base}/v1/chat/cancel", json={"run_id": rid})
        assert r.json() == {"stopped": True}
    mm.close()


async def test_an_approval_continuation_survives_a_dropped_connection():
    from src.persona import ExecutionMode

    gate = asyncio.Event()
    calls = {"n": 0}

    async def stream_messages(*args, **kwargs):
        # 1: propose the write (parks) · 2: the parking turn's own reply ·
        # 3: the approval continuation, held until the client is gone.
        calls["n"] += 1
        yield {"type": "api_payload", "payload": {}}
        if calls["n"] == 1:
            yield {"type": "tool_calls", "calls": [
                {"id": "w1", "name": "create_ticket", "arguments": {"title": "t", "body": "b"}}]}
            yield {"type": "done", "full_text": ""}
            return
        text = "Proposed." if calls["n"] == 2 else "Ticket opened."
        if calls["n"] > 2:
            await gate.wait()
        yield {"type": "text_delta", "text": text}
        yield {"type": "done", "full_text": text}

    adapter, mm, persona, chat_system = _make_real_adapter(stream_messages=stream_messages)
    persona.set_execution_mode(ExecutionMode.CONFIRM)
    persona.set_enabled_tools(["*"])
    executed: List[str] = []

    async def fake_execute(name, _offered, **kwargs):
        executed.append(name)
        return {"ok": True}
    chat_system.tool_manager.execute_tool = fake_execute  # type: ignore[method-assign]

    async with _serve(adapter) as base, httpx.AsyncClient(timeout=10) as client:
        async with client.stream("POST", f"{base}/v1/chat/completions",
                                 json=_body("open a ticket")) as r:
            park = "".join([c async for c in r.aiter_text()])
        token = json.loads(park.split("event: derpr-confirm\ndata: ", 1)[1].split("\n", 1)[0])["token"]

        async with client.stream("POST", f"{base}/api/v1/persona/test_persona/confirm",
                                 json={"approved": True, "token": token}) as r2:
            seen = ""
            async for chunk in r2.aiter_text():
                seen += chunk
                if "derpr-run" in chunk:
                    break  # drop the connection before the continuation writes
        active = (await client.get(f"{base}/api/v1/session/test_persona/active_turn")).json()
        assert active["run"] is not None, (r2.status_code, seen, calls["n"])
        assert active["run"]["kind"] == "confirm"
        gate.set()
        await _wait_for_assistant(mm, "Ticket opened.")
    assert "create_ticket" in executed
    mm.close()


async def _wait_for_assistant(mm: Any, text: str) -> None:
    for _ in range(100):
        rows = _fetch_portal_rows(mm, "test_persona")
        if any(r["author_role"] == "assistant" and r["content"] == text for r in rows):
            return
        await asyncio.sleep(0.05)
    raise AssertionError(f"{text!r} never saved; rows={_fetch_portal_rows(mm, 'test_persona')}")
