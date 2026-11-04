"""inference-router (DP-483): routing rules, plus the real proxy over real sockets.

The socket tests exist because the rules are the easy half: what breaks a client is a
streamed reply arriving in one lump, or a closed client leaving a box generating.
"""
import asyncio
import importlib.util
import json
import sys
from pathlib import Path

import pytest

_MOD_PATH = Path(__file__).resolve().parents[2] / "services" / "inference-router" / "inference_router.py"
_spec = importlib.util.spec_from_file_location("inference_router", _MOD_PATH)
assert _spec and _spec.loader
ir = importlib.util.module_from_spec(_spec)
sys.modules["inference_router"] = ir
_spec.loader.exec_module(ir)

MOVE = 1000
QUIET = 120.0
NOW = 10_000.0
MODEL = "flash-next"


def box(name, **kw):
    kw.setdefault("healthy", True)
    kw.setdefault("model", MODEL)
    return ir.Backend(name, "127.0.0.1", 1, **kw)


def body(*texts):
    return json.dumps({"messages": [{"role": "user", "content": t} for t in texts]}).encode()


def chain(*texts):
    return ir.prefix_chain(body(*texts))


BIG = "x" * (MOVE * 2)
OTHER_BIG = "y" * (MOVE * 2)


def choose(boxes, c):
    return ir.choose(boxes, c, NOW, MOVE, QUIET)


def test_prefix_chain_is_shared_only_up_to_the_first_different_message():
    first, second = chain("a", "b", "c"), chain("a", "b", "other")
    assert first[1] == second[1]
    assert first[2] != second[2]


@pytest.mark.parametrize("payload", [
    {"messages": [{"role": "user", "content": "a"}]},
    {"system": "s", "messages": [{"role": "user", "content": "a"}]},
    {"input": [{"role": "user", "content": "a"}]},
    {"prompt": "a"},
])
def test_prefix_chain_reads_every_api_shape(payload):
    assert ir.prefix_chain(json.dumps(payload).encode())


@pytest.mark.parametrize("raw", [b"", b"not json", b"[1]", b"{}"])
def test_an_unreadable_body_is_a_new_conversation_not_an_error(raw):
    assert ir.prefix_chain(raw) == []


def test_new_conversation_prefers_the_primary():
    assert choose([box("r9700"), box("dt21")], chain("hi")).name == "r9700"


def test_conversation_returns_to_the_box_holding_its_cache():
    r9700, dt21 = box("r9700"), box("dt21")
    dt21.remember(chain(BIG))
    assert choose([r9700, dt21], chain(BIG, "next")) is dt21


def test_the_box_serving_the_newer_prefix_wins_after_a_move():
    r9700, dt21 = box("r9700"), box("dt21")
    r9700.remember(chain(BIG))
    dt21.remember(chain(BIG, "moved here"))
    assert choose([r9700, dt21], chain(BIG, "moved here", "next")) is dt21


def test_big_conversation_moves_to_a_quiet_box_rather_than_wait():
    r9700, dt21 = box("r9700", inflight=1), box("dt21", last_active=NOW - QUIET)
    r9700.remember(chain(BIG))
    assert choose([r9700, dt21], chain(BIG, "next")) is dt21


def test_big_conversation_waits_rather_than_evict_a_conversation_between_steps():
    """Three conversations on two boxes must not trade places and re-read forever."""
    r9700, dt21 = box("r9700", inflight=1), box("dt21", last_active=NOW - 1)
    r9700.remember(chain(BIG))
    dt21.remember(chain(OTHER_BIG))
    assert choose([r9700, dt21], chain(BIG, "next")) is r9700


def test_new_conversation_does_not_take_a_box_mid_conversation_when_one_is_quiet():
    """stagr between tool steps on r9700: codr must start on dt21, or stagr has to move."""
    r9700, dt21 = box("r9700", last_active=NOW - 1), box("dt21")
    assert choose([r9700, dt21], chain("new")) is dt21


def test_small_conversation_takes_any_free_box_rather_than_wait():
    r9700, dt21 = box("r9700", inflight=1), box("dt21", last_active=NOW - 1)
    assert choose([r9700, dt21], chain("new")) is dt21


def test_all_busy_a_new_conversation_joins_the_shortest_queue():
    r9700, dt21 = box("r9700", inflight=3), box("dt21", inflight=1)
    assert choose([r9700, dt21], chain("new")) is dt21


def test_work_the_router_did_not_send_counts_as_busy():
    r9700, dt21 = box("r9700", external_busy=True), box("dt21")
    assert choose([r9700, dt21], chain("new")) is dt21


def test_a_box_with_another_model_is_never_chosen():
    """CT101 swapped in on the R9700: dt21's Strata is no longer the same model."""
    r9700, dt21 = box("r9700", model="27b", inflight=5), box("dt21")
    assert ir.eligible([r9700, dt21], "27b", 10) == [r9700]


def test_a_down_primary_fails_over_only_to_its_own_model():
    down, same, other = box("r9700", healthy=False), box("dt21"), box("omen", model="27b")
    assert ir.eligible([down, same, other], MODEL, 10) == [same]


def test_a_prompt_too_big_for_a_box_stays_off_it():
    big, small = box("r9700", n_ctx=262144), box("dt21", n_ctx=65536)
    assert ir.eligible([big, small], MODEL, 100_000) == [big]


def test_a_prompt_too_big_for_every_box_still_gets_a_server_error():
    assert ir.eligible([box("r9700", n_ctx=10)], MODEL, 100)


def test_seen_prefixes_are_bounded(monkeypatch):
    monkeypatch.setattr(ir, "MAX_SEEN_PREFIXES", 2)
    backend = box("r9700")
    backend.remember(chain("a", "b", "c"))
    assert len(backend.seen) == 2


@pytest.mark.parametrize("entry, healthy", [
    ({"id": MODEL, "status": {"value": "loaded"}, "meta": {"n_ctx": 5}}, True),
    ({"id": "kcpp-reports-no-status"}, True),
    ({"id": MODEL, "status": {"value": "unloaded"}}, False),  # refuse rather than load 55 GB on his dev box
    ({"id": MODEL, "status": {"value": "loading"}}, False),
])
def test_only_a_loaded_model_takes_traffic(entry, healthy):
    router = ir.Router([box("r9700", healthy=False)], MOVE, QUIET)
    router.note_poll(router.primary, {"data": [entry]}, None)
    assert router.primary.healthy is healthy


def test_expected_model_survives_the_primary_going_down():
    router = ir.Router([box("r9700"), box("dt21")], MOVE, QUIET)
    router.note_poll(router.primary, {"data": [{"id": MODEL}]}, None)
    for _ in range(ir.POLL_FAILURES_DOWN):
        router.note_poll(router.primary, None, None)
    assert router.primary.healthy is False
    assert router.expected_model == MODEL


def test_one_missed_poll_does_not_move_a_conversation_off_its_cache():
    """A busy R9700 missed a live /metrics poll on 2026-11-02."""
    router = ir.Router([box("r9700")], MOVE, QUIET)
    for _ in range(ir.POLL_FAILURES_DOWN - 1):
        router.note_poll(router.primary, None, None)
    assert router.primary.healthy is True
    router.note_poll(router.primary, {"data": [{"id": MODEL}]}, None)
    assert router.primary.poll_failures == 0


def test_strata_generating_for_someone_else_is_busy():
    router = ir.Router([box("r9700")], MOVE, QUIET)
    router.note_poll(router.primary, {"data": [{"id": MODEL}]}, {"live": {"state": "generating"}})
    assert router.primary.external_busy is True


def test_parse_backends():
    first, second = ir.parse_backends("r9700=127.0.0.1:5006, dt21=10.0.0.69:5001")
    assert (first.name, first.port, second.host) == ("r9700", 5006, "10.0.0.69")


@pytest.mark.parametrize("spec", ["", "r9700", "r9700=host", "=host:1"])
def test_parse_backends_rejects_bad_config(spec):
    with pytest.raises(ValueError):
        ir.parse_backends(spec)


# --- the real proxy ---------------------------------------------------------------


class FakeEngine:
    """A model server that streams, records requests, and notices a closed connection."""

    def __init__(self, status=b"200 OK", headers=b"Content-Type: text/event-stream\r\n"):
        self.status = status
        self.headers = headers
        self.requests = []
        self.release = asyncio.Event()
        self.disconnected = asyncio.Event()
        self.hold = False

    async def handle(self, reader, writer):
        head = await reader.readuntil(b"\r\n\r\n")
        length = next((int(line.split(b":")[1]) for line in head.split(b"\r\n")
                       if line.lower().startswith(b"content-length")), 0)
        request_body = await reader.readexactly(length) if length else b""
        self.requests.append((head, request_body))
        writer.write(b"HTTP/1.1 " + self.status + b"\r\n" + self.headers + b"\r\n")
        writer.write(b"first")
        await writer.drain()
        if self.hold:
            await self.release.wait()
            if not await reader.read(1):
                self.disconnected.set()
        writer.write(b"-last")
        await writer.drain()
        writer.close()

    async def start(self):
        self.server = await asyncio.start_server(self.handle, "127.0.0.1", 0)
        return self.server.sockets[0].getsockname()[1]


class Harness:
    def __init__(self, engines):
        self.engines = engines

    async def __aenter__(self):
        boxes = []
        for name, engine in self.engines.items():
            boxes.append(ir.Backend(name, "127.0.0.1", await engine.start(), healthy=True, model=MODEL))
        self.router = ir.Router(boxes, MOVE, QUIET)
        self.router.expected_model = MODEL
        self.server = await asyncio.start_server(ir.Server(self.router).handle, "127.0.0.1", 0)
        self.port = self.server.sockets[0].getsockname()[1]
        return self

    async def __aexit__(self, *_):
        self.server.close()
        for engine in self.engines.values():
            engine.server.close()

    async def open(self, payload=b"", method="POST", path="/v1/chat/completions"):
        reader, writer = await asyncio.open_connection("127.0.0.1", self.port)
        writer.write(f"{method} {path} HTTP/1.1\r\nHost: 10.0.0.70:5001\r\nConnection: keep-alive\r\n"
                     f"Content-Length: {len(payload)}\r\n\r\n".encode() + payload)
        await writer.drain()
        return reader, writer

    async def fetch(self, payload=b"", **kw):
        reader, writer = await self.open(payload, **kw)
        raw = await asyncio.wait_for(reader.read(), 5)
        writer.close()
        return raw


async def _until(predicate):
    for _ in range(200):
        if predicate():
            return
        await asyncio.sleep(0.01)
    raise AssertionError("condition not reached")


@pytest.mark.asyncio
async def test_proxy_relays_the_response_and_names_the_box():
    async with Harness({"r9700": FakeEngine(), "dt21": FakeEngine()}) as h:
        raw = await h.fetch(body("hi"))
    assert raw.startswith(b"HTTP/1.1 200 OK\r\nX-Inference-Backend: r9700\r\n")
    assert raw.endswith(b"first-last")


@pytest.mark.asyncio
async def test_proxy_preserves_the_host_header_and_closes_upstream_per_request():
    """Strata answers 403 to a Host outside allowed_hosts."""
    engine = FakeEngine()
    async with Harness({"r9700": engine}) as h:
        await h.fetch(body("hi"))
    head = engine.requests[0][0]
    assert b"Host: 10.0.0.70:5001" in head
    assert b"Connection: close" in head
    assert b"keep-alive" not in head


@pytest.mark.asyncio
async def test_concurrent_conversations_run_on_different_boxes():
    """The point of the ticket: nothing waits while a box is free."""
    r9700, dt21 = FakeEngine(), FakeEngine()
    r9700.hold = True
    async with Harness({"r9700": r9700, "dt21": dt21}) as h:
        reader, writer = await h.open(body("stagr"))
        await asyncio.wait_for(reader.readuntil(b"first"), 5)
        second = await h.fetch(body("codr"))
        assert b"X-Inference-Backend: dt21" in second
        assert h.router.primary.inflight == 1
        writer.close()
        r9700.release.set()


@pytest.mark.asyncio
async def test_a_request_waits_for_the_first_free_box_rather_than_join_a_queue():
    """Live, 2026-11-02: a request sat 208 s queued on the R9700 with dt21 idle after 3 s."""
    r9700, dt21 = FakeEngine(), FakeEngine()
    r9700.hold = dt21.hold = True
    async with Harness({"r9700": r9700, "dt21": dt21}) as h:
        first = await h.open(body("one"))
        await asyncio.wait_for(first[0].readuntil(b"first"), 5)
        second = await h.open(body("two"))
        await asyncio.wait_for(second[0].readuntil(b"first"), 5)
        third = await h.open(body("three"))
        await _until(lambda: h.router.waiting == 1)
        assert len(r9700.requests) + len(dt21.requests) == 2

        second[1].close()  # dt21 frees up; r9700 stays busy
        raw = await asyncio.wait_for(third[0].readuntil(b"first"), 5)
        assert b"X-Inference-Backend: dt21" in raw
        assert h.router.waiting == 0
        for reader_writer in (first, third):
            reader_writer[1].close()
        r9700.release.set()
        dt21.release.set()


@pytest.mark.asyncio
async def test_a_client_leaving_while_held_never_reaches_a_box():
    r9700 = FakeEngine()
    r9700.hold = True
    async with Harness({"r9700": r9700}) as h:
        first = await h.open(body("one"))
        await asyncio.wait_for(first[0].readuntil(b"first"), 5)
        second = await h.open(body("two"))
        await _until(lambda: h.router.waiting == 1)
        second[1].close()
        await _until(lambda: h.router.waiting == 0)
        first[1].close()
        r9700.release.set()
        await _until(lambda: h.router.primary.inflight == 0)
    assert len(r9700.requests) == 1


@pytest.mark.asyncio
async def test_a_control_call_does_not_wait_behind_a_generation():
    r9700 = FakeEngine()
    r9700.hold = True
    async with Harness({"r9700": r9700}) as h:
        first = await h.open(body("one"))
        await asyncio.wait_for(first[0].readuntil(b"first"), 5)
        r9700.hold = False
        raw = await h.fetch(b"{}", path="/v1/messages/count_tokens")
        assert raw.endswith(b"first-last")
        assert h.router.primary.inflight == 1
        first[1].close()
        r9700.release.set()


@pytest.mark.asyncio
async def test_a_client_leaving_before_the_first_byte_closes_the_upstream():
    """A non-streaming reply sends nothing until it is done."""
    release = asyncio.Event()
    closed = asyncio.Event()

    class Silent(FakeEngine):
        async def handle(self, reader, writer):
            await reader.readuntil(b"\r\n\r\n")
            release.set()
            await reader.read()  # returns at EOF: the router closed its side
            closed.set()
            writer.close()

    async with Harness({"r9700": Silent()}) as h:
        _, writer = await h.open(body("hi"))
        await asyncio.wait_for(release.wait(), 5)
        writer.close()
        await asyncio.wait_for(closed.wait(), 5)
        await _until(lambda: h.router.primary.inflight == 0)


@pytest.mark.asyncio
async def test_the_log_reports_what_was_cached_before_this_request(caplog):
    async with Harness({"r9700": FakeEngine()}) as h:
        with caplog.at_level("INFO", logger="inference_router"):
            await h.fetch(body("hi"))
    assert "cached 0 of" in caplog.text


@pytest.mark.asyncio
async def test_a_reply_streams_before_it_finishes():
    engine = FakeEngine()
    engine.hold = True
    async with Harness({"r9700": engine}) as h:
        reader, writer = await h.open(body("hi"))
        assert b"first" in await asyncio.wait_for(reader.readuntil(b"first"), 5)
        writer.close()
        engine.release.set()


@pytest.mark.asyncio
async def test_a_client_closing_closes_the_upstream_so_the_generation_stops():
    engine = FakeEngine()
    engine.hold = True
    async with Harness({"r9700": engine}) as h:
        reader, writer = await h.open(body("hi"))
        await asyncio.wait_for(reader.readuntil(b"first"), 5)
        writer.close()
        await _until(lambda: h.router.primary.inflight == 0)
        engine.release.set()
        await asyncio.wait_for(engine.disconnected.wait(), 5)


@pytest.mark.asyncio
async def test_a_box_that_cannot_be_reached_fails_over_to_the_next():
    engine = FakeEngine()
    async with Harness({"r9700": FakeEngine(), "dt21": engine}) as h:
        h.router.primary.port = 1  # refused
        raw = await h.fetch(body("hi"))
        assert b"X-Inference-Backend: dt21" in raw
        assert h.router.primary.healthy is False


NO_ENGINE = b"Content-Type: application/json\r\nX-Proxy-Error: r9700-proxy\r\n"  # r9700-lb.conf's @no_engine


@pytest.mark.asyncio
@pytest.mark.parametrize("headers", [NO_ENGINE, b"Content-Type: text/html\r\n"])
async def test_a_502_from_the_box_proxy_fails_over(headers):
    """r9700-proxy answers 502 when neither CT is running."""
    async with Harness({"r9700": FakeEngine(b"502 Bad Gateway", headers), "dt21": FakeEngine()}) as h:
        raw = await h.fetch(body("hi"))
        assert b"X-Inference-Backend: dt21" in raw
        assert h.router.primary.healthy is False


@pytest.mark.asyncio
@pytest.mark.parametrize("status", [b"502 Bad Gateway", b"503 Service Unavailable", b"500 Internal Server Error"])
async def test_an_engine_error_is_the_requests_failure_not_the_boxs(status):
    """Strata's 502 structured_output_failed once marked the R9700 down and moved the chat (2026-11-04)."""
    r9700, dt21 = FakeEngine(status, b"Content-Type: application/json\r\n"), FakeEngine()
    async with Harness({"r9700": r9700, "dt21": dt21}) as h:
        raw = await h.fetch(body("hi"))
        assert raw.startswith(b"HTTP/1.1 " + status + b"\r\nX-Inference-Backend: r9700\r\n")
        assert b"Content-Type: application/json" in raw and raw.endswith(b"first-last")
        assert h.router.primary.healthy is True
    assert not dt21.requests


@pytest.mark.parametrize("head, failure", [
    (NO_ENGINE + b"\r\n", True),
    (b"Content-Type: text/html\r\nServer: nginx\r\n\r\n", True),
    (b"Content-Type: application/json\r\n\r\n", False),
    (b"\r\n", False),
])
def test_proxy_failure(head, failure):
    assert ir.proxy_failure(head) is failure


@pytest.mark.asyncio
async def test_nothing_up_is_a_503_not_a_hang():
    async with Harness({"r9700": FakeEngine()}) as h:
        h.router.primary.healthy = False
        raw = await h.fetch(body("hi"))
    assert raw.startswith(b"HTTP/1.1 503")


@pytest.mark.asyncio
async def test_a_status_read_goes_to_the_primary_and_counts_for_nothing():
    r9700 = FakeEngine()
    async with Harness({"r9700": r9700, "dt21": FakeEngine()}) as h:
        raw = await h.fetch(method="GET", path="/metrics")
        assert b"X-Inference-Backend: r9700" in raw
        assert h.router.primary.last_active == float("-inf")


@pytest.mark.asyncio
async def test_router_status_reports_every_box():
    async with Harness({"r9700": FakeEngine(), "dt21": FakeEngine()}) as h:
        raw = await h.fetch(method="GET", path="/router/status")
    status = json.loads(raw.split(b"\r\n\r\n", 1)[1])
    assert [b["name"] for b in status["backends"]] == ["r9700", "dt21"]


@pytest.mark.asyncio
async def test_a_chunked_request_body_is_read_and_resent_with_a_length():
    engine = FakeEngine()
    async with Harness({"r9700": engine}) as h:
        reader, writer = await asyncio.open_connection("127.0.0.1", h.port)
        writer.write(b"POST /v1/chat/completions HTTP/1.1\r\nHost: x\r\nTransfer-Encoding: chunked\r\n\r\n"
                     b"5\r\nhello\r\n0\r\n\r\n")
        await writer.drain()
        await asyncio.wait_for(reader.read(), 5)
        writer.close()
    assert engine.requests[0][1] == b"hello"
    assert b"Transfer-Encoding" not in engine.requests[0][0]


# --- background work: Hindsight on the R9700 behind chat (DP-405B) ----------------


def test_a_background_job_waits_out_the_hold_off_after_chat():
    router = ir.Router([box("r9700", strata=True, last_active=NOW - 30)], MOVE, QUIET, holdoff_s=60)
    assert router.admit_background(NOW) is None
    assert router.admit_background(NOW + 30) is router.primary


def test_the_hold_off_defaults_to_the_quiet_window():
    assert ir.Router([box("r9700")], MOVE, QUIET).holdoff_s == QUIET


@pytest.mark.parametrize("kw", [
    {"strata": False},  # kcpp on the R9700: cancel-on-disconnect and strata_checkpoint unverified there
    {"strata": True, "inflight": 1},
    {"strata": True, "external_busy": True},
    {"strata": True, "healthy": False},
])
def test_a_background_job_only_takes_an_idle_strata_box(kw):
    router = ir.Router([box("r9700", **kw)], MOVE, QUIET)
    assert router.admit_background(NOW) is None


def test_one_background_job_per_box():
    router = ir.Router([box("r9700", strata=True)], MOVE, QUIET)
    router.primary.background = asyncio.Event()
    assert router.admit_background(NOW) is None


def test_queued_chat_blocks_a_new_background_job():
    router = ir.Router([box("r9700", strata=True)], MOVE, QUIET)
    router.waiting = 1
    assert router.admit_background(NOW) is None


def test_background_work_stays_off_boxes_not_named_for_it():
    """dt21's :5001 is Adam's own Strata; only the boxes configured for it take Hindsight."""
    router = ir.Router([box("r9700", strata=True, inflight=1), box("dt21", strata=True)], MOVE, QUIET)
    assert router.admit_background(NOW) is None
    router = ir.Router([box("r9700", strata=True), box("dt21", strata=True)], MOVE, QUIET,
                       background_names=["dt21"])
    assert router.admit_background(NOW).name == "dt21"


def test_a_box_running_only_a_background_job_counts_as_idle_for_chat():
    r9700 = box("r9700", strata=True)
    r9700.background = asyncio.Event()
    assert choose([r9700, box("dt21", last_active=NOW - 1)], chain("new")) is r9700


def test_strata_is_recognised_by_its_metrics_engine_block():
    router = ir.Router([box("r9700")], MOVE, QUIET)
    router.note_poll(router.primary, {"data": [{"id": MODEL}]}, {"engine": {"version": "0.1.52"}, "live": {}})
    assert router.primary.strata is True
    router.note_poll(router.primary, {"data": [{"id": MODEL}]}, None)  # kcpp: no /metrics JSON
    assert router.primary.strata is False


def test_our_own_background_job_is_not_someone_elses_work():
    router = ir.Router([box("r9700")], MOVE, QUIET)
    busy = {"engine": {}, "live": {"state": "generating"}}
    router.primary.background = asyncio.Event()
    router.note_poll(router.primary, {"data": [{"id": MODEL}]}, busy, now=NOW)
    assert router.primary.external_busy is False
    router.primary.background, router.primary.background_ended = None, NOW
    router.note_poll(router.primary, {"data": [{"id": MODEL}]}, busy, now=NOW + 1)  # still cancelling
    assert router.primary.external_busy is False
    router.note_poll(router.primary, {"data": [{"id": MODEL}]}, busy, now=NOW + ir.BACKGROUND_SETTLE_S)
    assert router.primary.external_busy is True


def test_background_body_turns_off_strata_checkpoints():
    sent = json.loads(ir.background_body(json.dumps({"messages": [], "strata_checkpoint": True}).encode()))
    assert sent["strata_checkpoint"] is False
    assert ir.background_body(b"not json") == b"not json"


def test_with_host_replaces_the_pools_host():
    lines = ir.with_host([b"POST / HTTP/1.1", b"Host: kobold-proxy", b"X: y"], "10.0.0.70")
    assert lines == [b"POST / HTTP/1.1", b"Host: 10.0.0.70", b"X: y"]


def _reply(payload, status=b"200 OK", headers=b"Content-Type: application/json\r\n"):
    return b"HTTP/1.1 " + status + b"\r\n" + headers + b"\r\n" + payload


@pytest.mark.parametrize("raw, complete", [
    (_reply(b'{"a": 1}', headers=b"Content-Length: 8\r\n"), True),
    (_reply(b'{"a": 1', headers=b"Content-Length: 8\r\n"), False),  # cut off mid-body
    (_reply(b'8\r\n{"a": 1}\r\n0\r\n\r\n', headers=b"Transfer-Encoding: chunked\r\n"), True),
    (_reply(b'8\r\n{"a": 1}\r\n', headers=b"Transfer-Encoding: chunked\r\n"), False),
    (_reply(b'{"a": 1}'), True),  # unframed, ended by EOF: must parse
    (_reply(b'{"choices": [{"message": {"content": "{\\"facts\\": ['), False),
    (_reply(b"data: {}\n\ndata: [DONE]\n\n", headers=b"Content-Type: text/event-stream\r\n"), True),
    (_reply(b"data: {}\n\n", headers=b"Content-Type: text/event-stream\r\n"), False),
    (_reply(b"<html>", status=b"502 Bad Gateway", headers=b"Content-Type: text/html\r\n"), True),
    (b"HTTP/1.1 200 OK\r\nContent-Le", False),
    (b"", False),
])
def test_response_complete(raw, complete):
    assert ir.response_complete(raw) is complete


class JsonEngine:
    """A non-streaming model server: nothing is sent until the reply is done, like Strata's."""

    def __init__(self, reply=b'{"choices": [{"message": {"content": "{\\"facts\\": []}"}}]}', truncate=False,
                 status=b"200 OK"):
        self.reply, self.truncate, self.status = reply, truncate, status
        self.requests = []
        self.started = asyncio.Event()
        self.release = asyncio.Event()
        self.disconnected = asyncio.Event()
        self.hold = False

    async def handle(self, reader, writer):
        head = await reader.readuntil(b"\r\n\r\n")
        length = next((int(line.split(b":")[1]) for line in head.split(b"\r\n")
                       if line.lower().startswith(b"content-length")), 0)
        self.requests.append((head, await reader.readexactly(length) if length else b""))
        self.started.set()
        if self.hold:
            hung_up = asyncio.ensure_future(reader.read(1))
            released = asyncio.ensure_future(self.release.wait())
            await asyncio.wait({hung_up, released}, return_when=asyncio.FIRST_COMPLETED)
            released.cancel()
            if hung_up.done() and not hung_up.result():
                self.disconnected.set()
                writer.close()
                return
            hung_up.cancel()
        declared = len(self.reply) + (100 if self.truncate else 0)
        writer.write(b"HTTP/1.1 " + self.status + b"\r\nContent-Type: application/json\r\n"
                     b"Content-Length: %d\r\n\r\n" % declared + self.reply)
        await writer.drain()
        writer.close()

    async def start(self):
        self.server = await asyncio.start_server(self.handle, "127.0.0.1", 0)
        return self.server.sockets[0].getsockname()[1]


class BackgroundHarness(Harness):
    """The chat listener and the background listener over one router, as main() wires them."""

    async def __aenter__(self):
        await super().__aenter__()
        for backend in self.router.backends:
            backend.strata = True
        self.bg_server = await asyncio.start_server(ir.Server(self.router, background=True).handle, "127.0.0.1", 0)
        self.bg_port = self.bg_server.sockets[0].getsockname()[1]
        return self

    async def __aexit__(self, *exc):
        self.bg_server.close()
        await super().__aexit__(*exc)

    async def open_bg(self, payload):
        reader, writer = await asyncio.open_connection("127.0.0.1", self.bg_port)
        writer.write(b"POST /v1/chat/completions HTTP/1.1\r\nHost: kobold-proxy\r\n"
                     b"Content-Length: %d\r\n\r\n" % len(payload) + payload)
        await writer.drain()
        return reader, writer

    async def fetch_bg(self, payload):
        reader, writer = await self.open_bg(payload)
        raw = await asyncio.wait_for(reader.read(), 5)
        writer.close()
        return raw


HS_JOB = json.dumps({"model": "qwen3.6-35b-a3b", "messages": [{"role": "user", "content": "extract"}]}).encode()
VERIFY = json.dumps({"messages": [{"role": "user", "content": "Say ok"}], "max_completion_tokens": 512}).encode()


@pytest.mark.asyncio
async def test_a_background_job_gets_the_whole_reply_and_leaves_no_trace_on_chat_state():
    engine = JsonEngine()
    async with BackgroundHarness({"r9700": engine}) as h:
        raw = await h.fetch_bg(HS_JOB)
        assert raw.startswith(b"HTTP/1.1 200 OK\r\nX-Inference-Backend: r9700\r\n")
        assert raw.endswith(engine.reply)
        assert h.router.primary.last_active == float("-inf")  # the hold-off and the quiet rule see chat only
        assert h.router.primary.background is None
    head, sent = engine.requests[0]
    assert json.loads(sent)["strata_checkpoint"] is False
    assert b"Host: 10.0.0.70" in head and b"kobold-proxy" not in head


@pytest.mark.asyncio
async def test_chat_preempts_a_background_job_which_gets_a_503_never_a_short_200():
    engine = JsonEngine()
    engine.hold = True
    async with BackgroundHarness({"r9700": engine}) as h:
        bg_reader, bg_writer = await h.open_bg(HS_JOB)
        await asyncio.wait_for(engine.started.wait(), 5)
        engine.hold = False
        chat = asyncio.ensure_future(h.fetch(body("user turn")))
        bg_raw = await asyncio.wait_for(bg_reader.read(), 5)
        assert bg_raw.startswith(b"HTTP/1.1 503")
        await asyncio.wait_for(engine.disconnected.wait(), 5)  # the box saw the hang-up: Strata cancels here
        chat_raw = await asyncio.wait_for(chat, 1)  # the chat starts at once, not after the job
        assert chat_raw.startswith(b"HTTP/1.1 200 OK")
        assert h.router.background_counts["preempted"] == 1
        bg_writer.close()


@pytest.mark.asyncio
async def test_a_background_job_is_refused_at_once_while_chat_runs():
    """Never held: Hindsight's startup probe (10 s cap) and its pool must not wait on chat."""
    engine = JsonEngine()
    engine.hold = True
    async with BackgroundHarness({"r9700": engine}) as h:
        _, chat_writer = await h.open(body("long chat"))
        await asyncio.wait_for(engine.started.wait(), 5)
        raw = await asyncio.wait_for(h.fetch_bg(VERIFY), 1)
        assert raw.startswith(b"HTTP/1.1 503")
        assert len(engine.requests) == 1  # the probe never reached the box
        chat_writer.close()
        engine.release.set()


@pytest.mark.asyncio
async def test_a_background_job_is_refused_during_the_hold_off_after_chat():
    engine = JsonEngine()
    async with BackgroundHarness({"r9700": engine}) as h:
        await h.fetch(body("user turn"))
        raw = await h.fetch_bg(HS_JOB)
        assert raw.startswith(b"HTTP/1.1 503")
        assert h.router.background_counts["refused"] == 1
    assert len(engine.requests) == 1


@pytest.mark.asyncio
async def test_a_cut_off_reply_reaches_hindsight_as_a_502():
    async with BackgroundHarness({"r9700": JsonEngine(truncate=True)}) as h:
        raw = await h.fetch_bg(HS_JOB)
        assert raw.startswith(b"HTTP/1.1 502")
        assert h.router.background_counts["incomplete"] == 1


@pytest.mark.asyncio
async def test_a_background_5xx_is_passed_on_and_does_not_take_the_box_out_of_chat():
    """Strata answers 502 when a reply fails its JSON schema; that is the job's failure, not the box's."""
    failed = b'{"error": {"type": "structured_output_failed"}}'
    async with BackgroundHarness({"r9700": JsonEngine(failed, status=b"502 Bad Gateway")}) as h:
        raw = await h.fetch_bg(HS_JOB)
        assert raw.startswith(b"HTTP/1.1 502 Bad Gateway\r\nX-Inference-Backend: r9700")
        assert h.router.primary.healthy is True


@pytest.mark.asyncio
async def test_hindsight_hanging_up_cancels_its_job():
    engine = JsonEngine()
    engine.hold = True
    async with BackgroundHarness({"r9700": engine}) as h:
        _, writer = await h.open_bg(HS_JOB)
        await asyncio.wait_for(engine.started.wait(), 5)
        writer.close()
        await asyncio.wait_for(engine.disconnected.wait(), 5)
        await _until(lambda: h.router.primary.background is None)


@pytest.mark.asyncio
async def test_the_chat_port_never_runs_background_rules():
    engine = JsonEngine()
    async with BackgroundHarness({"r9700": engine}) as h:
        await h.fetch(HS_JOB)
    assert "strata_checkpoint" not in json.loads(engine.requests[0][1])
