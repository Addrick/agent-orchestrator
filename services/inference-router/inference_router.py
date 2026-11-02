#!/usr/bin/env python3
"""Front door for local inference (:5001): spreads conversations over model servers (DP-483).

Strata runs one request at a time and keeps a conversation's prompt cache on the box
that served it, so a blind balancer gives the worst of both: a 200k-token conversation
re-read for minutes each time it hops, or two conversations queued on one box while the
other sits idle. nginx can hash (ignores busy) or count connections (ignores the cache),
not both. This router does both, using only the request's own messages, so clients need
no session header and no configuration.

Standard library only: it runs from a bind-mounted file in a stock python image, and
must survive derpr deploys. It never logs or stores message content, only hashes.
"""
import asyncio
import hashlib
import json
import logging
import os
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Iterable, Optional, Sequence
from urllib.parse import urlsplit

log = logging.getLogger("inference_router")

CONNECT_TIMEOUT_S = 2.0  # a stopped box may drop packets silently rather than refuse
READ_TIMEOUT_S = 1200.0  # same as the nginx proxies: a big prompt can read for minutes
POLL_INTERVAL_S = 3.0
POLL_TIMEOUT_S = 5.0  # above r9700-proxy's 2 s connect failover to its backup CT
MAX_HEAD_BYTES = 64 * 1024
MAX_BODY_BYTES = 64 * 1024 * 1024
MAX_SEEN_PREFIXES = 20000
CHARS_PER_TOKEN = 4  # estimate, only used to keep a prompt off a smaller-context box
RETRY_STATUSES = {b"502", b"503", b"504"}  # the box's proxy has no live engine behind it
HOP_HEADERS = {b"connection", b"keep-alive", b"proxy-connection", b"transfer-encoding", b"content-length"}


@dataclass
class Backend:
    name: str
    host: str
    port: int
    healthy: bool = False
    model: Optional[str] = None
    n_ctx: Optional[int] = None
    inflight: int = 0
    external_busy: bool = False  # the server reports work this router did not send
    last_active: float = float("-inf")
    seen: "OrderedDict[str, int]" = field(default_factory=OrderedDict)  # prefix hash -> prefix chars

    @property
    def idle(self) -> bool:
        return self.inflight == 0 and not self.external_busy

    def matched(self, chain: Sequence[tuple[str, int]]) -> int:
        """Chars of this conversation's leading messages this box has already served."""
        for digest, chars in reversed(chain):
            if digest in self.seen:
                return chars
        return 0

    def remember(self, chain: Sequence[tuple[str, int]]) -> None:
        for digest, chars in chain:
            self.seen[digest] = chars
            self.seen.move_to_end(digest)
        while len(self.seen) > MAX_SEEN_PREFIXES:
            self.seen.popitem(last=False)


def prefix_chain(body: bytes) -> list[tuple[str, int]]:
    """Cumulative (hash, chars) per leading message: the identity of a prompt prefix.

    Handles chat completions, Anthropic messages, Responses input and raw prompts.
    Anything else is an unknown prompt, routed as a new conversation.
    """
    try:
        payload = json.loads(body)
    except ValueError:
        return []
    if not isinstance(payload, dict):
        return []
    parts: list[Any] = []
    if payload.get("system"):
        parts.append(payload["system"])
    for key in ("messages", "input", "prompt"):
        value = payload.get(key)
        if value:
            parts.extend(value if isinstance(value, list) else [value])
            break
    digest = hashlib.sha256()
    chars = 0
    chain = []
    for part in parts:
        text = json.dumps(part, sort_keys=True, ensure_ascii=False)
        digest.update(text.encode("utf-8", "replace"))
        chars += len(text)
        chain.append((digest.copy().hexdigest(), chars))
    return chain


def eligible(backends: Sequence[Backend], expected_model: Optional[str], prompt_tokens: int) -> list[Backend]:
    """Boxes that can serve this prompt: up, the same model, and a big enough context."""
    same_model = [b for b in backends if b.healthy and (expected_model is None or b.model == expected_model)]
    fits = [b for b in same_model if b.n_ctx is None or prompt_tokens <= b.n_ctx]
    return fits or same_model


def choose(
    candidates: Sequence[Backend],
    chain: Sequence[tuple[str, int]],
    now: float,
    move_chars: int,
    quiet_s: float,
) -> Optional[Backend]:
    """Pick a box. `candidates` is in preference order, primary first.

    Keeps a conversation with its cache unless that would make it wait while another
    box is free. A box "quiet" for quiet_s has no conversation between steps on it, so
    moving there evicts nobody. Requiring that for a big conversation stops three
    conversations on two boxes trading places and re-reading their prompts forever.
    """
    if not candidates:
        return None

    def is_quiet(backend: Backend) -> bool:
        return backend.idle and now - backend.last_active >= quiet_s

    home = max(candidates, key=lambda b: b.matched(chain))  # max keeps the first on a tie
    if home.matched(chain) >= move_chars:  # expensive to move
        if home.idle:
            return home
        return next((b for b in candidates if is_quiet(b)), home)

    # Cheap to place: take the free box that disturbs the least. Quiet beats merely idle.
    def rank(backend: Backend) -> tuple[int, int, int]:
        tier = 0 if is_quiet(backend) else 1 if backend.idle else 2
        return (tier, backend.inflight if tier == 2 else 0, -backend.matched(chain))

    return min(candidates, key=rank)


class Router:
    def __init__(self, backends: Sequence[Backend], move_chars: int, quiet_s: float) -> None:
        self.backends = list(backends)
        self.primary = self.backends[0]
        self.move_chars = move_chars
        self.quiet_s = quiet_s
        self.expected_model: Optional[str] = None  # the primary's, kept while it is down

    def route(self, body: bytes, now: float) -> tuple[list[Backend], list[tuple[str, int]]]:
        """The chosen box first, then the others to fall back to if it cannot be reached."""
        chain = prefix_chain(body)
        tokens = (chain[-1][1] if chain else 0) // CHARS_PER_TOKEN
        candidates = eligible(self.backends, self.expected_model, tokens)
        chosen = choose(candidates, chain, now, self.move_chars, self.quiet_s)
        if chosen is None:
            return [], chain
        return [chosen] + [b for b in candidates if b is not chosen], chain

    def note_poll(self, backend: Backend, models: Optional[dict[str, Any]], metrics: Optional[dict[str, Any]]) -> None:
        entry = _first_model(models)
        status = (entry.get("status") or {}).get("value") if entry else None
        # A lazy or unloaded server must refuse rather than load on a router request.
        backend.healthy = entry is not None and status in (None, "loaded")
        if entry is None:
            backend.external_busy = False
            return
        backend.model = entry.get("id")
        backend.n_ctx = (entry.get("meta") or {}).get("n_ctx")
        if backend is self.primary and backend.healthy:
            self.expected_model = backend.model
        live = (metrics or {}).get("live") or {}
        backend.external_busy = backend.inflight == 0 and live.get("state", "idle") not in ("idle", "unloaded")

    def status(self) -> dict[str, Any]:
        now = time.monotonic()
        return {
            "expected_model": self.expected_model,
            "backends": [{
                "name": b.name, "url": f"http://{b.host}:{b.port}", "healthy": b.healthy, "model": b.model,
                "n_ctx": b.n_ctx, "inflight": b.inflight, "external_busy": b.external_busy,
                "idle_s": None if b.last_active == float("-inf") else round(now - b.last_active, 1),
            } for b in self.backends],
        }


def _first_model(models: Optional[dict[str, Any]]) -> Optional[dict[str, Any]]:
    data = (models or {}).get("data")
    return data[0] if isinstance(data, list) and data and isinstance(data[0], dict) else None


def parse_backends(spec: str) -> list[Backend]:
    """`r9700=127.0.0.1:5006,dt21=10.0.0.69:5001`, primary first."""
    backends = []
    for item in filter(None, (part.strip() for part in spec.split(","))):
        name, _, address = item.partition("=")
        parsed = urlsplit(f"//{address}")
        if not name or not parsed.hostname or not parsed.port:
            raise ValueError(f"bad backend {item!r}: want name=host:port")
        backends.append(Backend(name, parsed.hostname, parsed.port))
    if not backends:
        raise ValueError("no backends configured")
    return backends


def rewrite_head(head_lines: Iterable[bytes], body_length: int) -> bytes:
    """One request per connection: EOF ends the response, and closing it cancels the generation."""
    lines = iter(head_lines)
    out = [next(lines)]
    out += [line for line in lines if line.split(b":", 1)[0].strip().lower() not in HOP_HEADERS]
    out += [b"Connection: close", b"Content-Length: %d" % body_length]
    return b"\r\n".join(out) + b"\r\n\r\n"


async def read_request(reader: asyncio.StreamReader) -> Optional[tuple[list[bytes], bytes]]:
    try:
        head = await asyncio.wait_for(reader.readuntil(b"\r\n\r\n"), READ_TIMEOUT_S)
    except (asyncio.IncompleteReadError, asyncio.LimitOverrunError, asyncio.TimeoutError, ConnectionError):
        return None
    lines = head[:-4].split(b"\r\n")
    headers = {k.strip().lower(): v.strip() for k, _, v in (line.partition(b":") for line in lines[1:])}
    if b"chunked" in headers.get(b"transfer-encoding", b"").lower():
        return lines, await _read_chunked(reader)
    length = int(headers.get(b"content-length", b"0") or 0)
    if length > MAX_BODY_BYTES:
        raise ValueError("request body too large")
    return lines, await reader.readexactly(length) if length else b""


async def _read_chunked(reader: asyncio.StreamReader) -> bytes:
    body = bytearray()
    while True:
        size = int((await reader.readline()).split(b";")[0].strip() or b"0", 16)
        if size == 0:
            await reader.readline()
            return bytes(body)
        if len(body) + size > MAX_BODY_BYTES:
            raise ValueError("request body too large")
        body += await reader.readexactly(size)
        await reader.readline()


async def _send(backend: Backend, request: bytes) -> Optional[tuple[asyncio.StreamReader, asyncio.StreamWriter, bytes]]:
    """Connect and read the status line, the last point a request can still go elsewhere."""
    writer = None
    try:
        reader, writer = await asyncio.wait_for(asyncio.open_connection(backend.host, backend.port), CONNECT_TIMEOUT_S)
        writer.write(request)
        await writer.drain()
        status = await asyncio.wait_for(reader.readline(), READ_TIMEOUT_S)
    except (OSError, asyncio.TimeoutError) as exc:
        log.warning("%s unreachable: %r", backend.name, exc)
        if writer:
            writer.close()
        return None
    parts = status.split()
    if len(parts) < 2 or parts[1] in RETRY_STATUSES:
        log.warning("%s answered %r", backend.name, status[:60])
        writer.close()
        return None
    return reader, writer, status


async def relay(
    upstream: asyncio.StreamReader, client_reader: asyncio.StreamReader, client: asyncio.StreamWriter,
) -> None:
    """Copy the response until either side ends. The caller closing upstream cancels the gen."""
    async def pump() -> None:
        while True:
            chunk = await asyncio.wait_for(upstream.read(65536), READ_TIMEOUT_S)
            if not chunk:
                return
            client.write(chunk)
            await client.drain()

    async def client_gone() -> None:
        while await client_reader.read(4096):
            pass

    tasks = [asyncio.ensure_future(pump()), asyncio.ensure_future(client_gone())]
    done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
    for task in pending:
        task.cancel()
    await asyncio.gather(*pending, return_exceptions=True)
    for task in done:
        task.exception()  # a reset on either side only ends this request


class Server:
    def __init__(self, router: Router) -> None:
        self.router = router

    async def handle(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        try:
            request = await read_request(reader)
            if request:
                await self._serve(request[0], request[1], reader, writer)
        except (ValueError, asyncio.IncompleteReadError):
            _respond(writer, b"400 Bad Request", {"error": "unreadable request"})
        except ConnectionError:
            pass
        finally:
            writer.close()

    async def _serve(
        self, lines: list[bytes], body: bytes, reader: asyncio.StreamReader, writer: asyncio.StreamWriter,
    ) -> None:
        method, target = (lines[0].split() + [b"", b""])[:2]
        if target == b"/router/status":
            _respond(writer, b"200 OK", self.router.status())
            return
        now = time.monotonic()
        if method == b"POST":
            order, chain = self.router.route(body, now)
        else:  # status reads, model lists: the primary answers, as it did before the router
            chain = []
            order = sorted((b for b in self.router.backends if b.healthy), key=self.router.backends.index)
        if not order:
            _respond(writer, b"503 Service Unavailable", {"error": "no inference server is up"})
            return
        await self._forward(order, chain, rewrite_head(lines, len(body)) + body, method, target, reader, writer)

    async def _forward(
        self, order: list[Backend], chain: list[tuple[str, int]], request: bytes, method: bytes, target: bytes,
        reader: asyncio.StreamReader, writer: asyncio.StreamWriter,
    ) -> None:
        for backend in order:
            sent = await self._dispatch(backend, chain, request, method, target, reader, writer)
            if sent:
                return
            if method == b"POST":
                backend.healthy = False  # the next poll restores it
        _respond(writer, b"503 Service Unavailable", {"error": "no inference server answered"})

    async def _dispatch(
        self, backend: Backend, chain: list[tuple[str, int]], request: bytes, method: bytes, target: bytes,
        reader: asyncio.StreamReader, writer: asyncio.StreamWriter,
    ) -> bool:
        counted = method == b"POST"
        if counted:
            backend.inflight += 1
            backend.last_active = time.monotonic()
        upstream_writer = None
        try:
            sent = await _send(backend, request)
            if sent is None:
                return False
            upstream, upstream_writer, status = sent
            if counted:
                backend.remember(chain)
                log.info("%s %s -> %s (matched %d of %d chars, %d in flight)", method.decode(), target.decode(),
                         backend.name, backend.matched(chain), chain[-1][1] if chain else 0, backend.inflight)
            writer.write(status + b"X-Inference-Backend: " + backend.name.encode() + b"\r\n")
            await relay(upstream, reader, writer)
            return True
        finally:
            if upstream_writer:
                upstream_writer.close()
            if counted:
                backend.inflight -= 1
                backend.last_active = time.monotonic()


def _respond(writer: asyncio.StreamWriter, status: bytes, payload: dict[str, Any]) -> None:
    body = json.dumps(payload).encode()
    writer.write(b"HTTP/1.1 " + status + b"\r\nContent-Type: application/json\r\nConnection: close\r\n"
                 b"Content-Length: %d\r\n\r\n" % len(body) + body)


async def _get_json(backend: Backend, path: str) -> Optional[dict[str, Any]]:
    writer = None
    try:
        reader, writer = await asyncio.wait_for(asyncio.open_connection(backend.host, backend.port), POLL_TIMEOUT_S)
        host = f"Host: {backend.host}\r\n" if backend.host != "127.0.0.1" else "Host: 10.0.0.70\r\n"
        writer.write(f"GET {path} HTTP/1.1\r\n{host}Connection: close\r\n\r\n".encode())
        raw = await asyncio.wait_for(reader.read(), POLL_TIMEOUT_S)
        head, _, body = raw.partition(b"\r\n\r\n")
        if b" 200" not in head.split(b"\r\n", 1)[0]:
            return None
        if b"chunked" in head.lower():
            body = _dechunk(body)
        parsed = json.loads(body)
        return parsed if isinstance(parsed, dict) else None
    except (OSError, ValueError, asyncio.TimeoutError):
        return None
    finally:
        if writer:
            writer.close()


def _dechunk(body: bytes) -> bytes:
    out = bytearray()
    while body:
        size_line, _, rest = body.partition(b"\r\n")
        size = int(size_line.split(b";")[0] or b"0", 16)
        if size == 0:
            break
        out += rest[:size]
        body = rest[size + 2:]
    return bytes(out)


async def poll_forever(router: Router) -> None:
    while True:
        for backend in router.backends:
            models = await _get_json(backend, "/v1/models")
            metrics = await _get_json(backend, "/metrics") if models else None
            was = backend.healthy
            router.note_poll(backend, models, metrics)
            if was != backend.healthy:
                log.info("%s is %s (%s)", backend.name, "up" if backend.healthy else "down", backend.model)
        await asyncio.sleep(POLL_INTERVAL_S)


async def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    router = Router(
        parse_backends(os.environ.get("ROUTER_BACKENDS", "r9700=127.0.0.1:5006,dt21=10.0.0.69:5001")),
        move_chars=int(os.environ.get("ROUTER_MOVE_CHARS", "40000")),
        quiet_s=float(os.environ.get("ROUTER_QUIET_S", "120")),
    )
    poller = asyncio.ensure_future(poll_forever(router))
    server = await asyncio.start_server(
        Server(router).handle, os.environ.get("ROUTER_HOST", "0.0.0.0"), int(os.environ.get("ROUTER_PORT", "5001")),
        limit=MAX_HEAD_BYTES,
    )
    log.info("routing to %s", ", ".join(b.name for b in router.backends))
    async with server:
        try:
            await server.serve_forever()
        finally:
            poller.cancel()


if __name__ == "__main__":
    asyncio.run(main())
