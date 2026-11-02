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
    router.note_poll(router.primary, None, None)
    assert router.primary.healthy is False
    assert router.expected_model == MODEL


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

    def __init__(self, status=b"200 OK"):
        self.status = status
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
        writer.write(b"HTTP/1.1 " + self.status + b"\r\nContent-Type: text/event-stream\r\n\r\n")
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


@pytest.mark.asyncio
async def test_a_502_from_the_box_proxy_fails_over():
    """r9700-proxy answers 502 when neither CT is running."""
    async with Harness({"r9700": FakeEngine(b"502 Bad Gateway"), "dt21": FakeEngine()}) as h:
        raw = await h.fetch(body("hi"))
    assert b"X-Inference-Backend: dt21" in raw


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
