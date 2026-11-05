#!/usr/bin/env python3
"""Front door for local inference (:5001): spreads conversations over model servers (DP-483).

Strata runs one request at a time and keeps a conversation's prompt cache on the box
that served it, so a blind balancer gives the worst of both: a 200k-token conversation
re-read for minutes each time it hops, or two conversations queued on one box while the
other sits idle. nginx can hash (ignores busy) or count connections (ignores the cache),
not both. This router does both, using only the request's own messages, so clients need
no session header and no configuration.

Background work (DP-405B): Hindsight reaches the router on a second port. A job there runs
only on an idle Strata box with no chat for a hold-off, is refused with a 503 otherwise (its
nginx pool hands it to another worker), and is preempted -- its upstream closed, which Strata
turns into a cancel -- the moment a chat request wants the box. Its reply is held whole and
sent on only when complete, so a cut-off job is always a 5xx, never a short 200.

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
POLL_FAILURES_DOWN = 3  # a busy server can miss one poll; a false "down" moves a big conversation off its cache
MAX_HEAD_BYTES = 64 * 1024
MAX_BODY_BYTES = 64 * 1024 * 1024
MAX_SEEN_PREFIXES = 20000
CHARS_PER_TOKEN = 4  # estimate, only used to keep a prompt off a smaller-context box
RETRY_STATUSES = {b"502", b"503", b"504"}  # fail over only when the box's proxy says so (proxy_failure)
# Only these occupy a box. Anything else (token counts, load, abort) must not wait behind a generation.
GENERATION_PATHS = {b"/v1/chat/completions", b"/v1/completions", b"/v1/messages", b"/v1/responses",
                    b"/api/v1/generate", b"/api/extra/generate/stream"}
HOP_HEADERS = {b"connection", b"keep-alive", b"proxy-connection", b"transfer-encoding", b"content-length"}
# The Host a loopback backend (r9700-proxy on CT100) is reached as: Strata answers 403 outside allowed_hosts.
LOOPBACK_HOST = os.environ.get("ROUTER_PUBLIC_HOST", "10.0.0.70")
# After a background job ends, Strata may still report the cancelled generation for a moment;
# that is not someone else's work. Strata's own disconnect watch polls every 0.5 s.
BACKGROUND_SETTLE_S = 2.0
# Sent with every background request: Strata keeps no conversation checkpoint for the call, so
# a burst of Hindsight jobs cannot evict a parked chat (measured 2026-11-04, DP-405B).
BACKGROUND_FIELDS = {"strata_checkpoint": False}
# Who a request is for, for the portal statusline (DP-499). A client names itself in this header
# (derpr sends the persona, codr's opencode config sends "codr"); otherwise the background
# listener's work is Hindsight's and anything else is shown as the caller's address.
REQUESTER_HEADER = b"x-inference-requester"
BACKGROUND_REQUESTER = "hindsight"
MAX_REQUESTER_CHARS = 48
# The parts of a box's /metrics the status carries: the live run and the last completed one.
# Everything else (per-request history) stays on the box.
STATUS_METRICS_KEYS = ("time", "live", "totals")


@dataclass
class Job:
    """One request occupying, or waiting for, a box. Never holds message content."""
    requester: str
    kind: str  # "chat" or "background"
    started: float
    prompt_chars: int = 0

    def describe(self, now: float) -> dict[str, Any]:
        return {"requester": self.requester, "kind": self.kind, "elapsed_s": round(now - self.started, 1),
                "prompt_chars": self.prompt_chars}


def requester_of(head_lines: Sequence[bytes], peer: Optional[str], background: bool) -> str:
    """The name the statusline shows for this request: its header, else Hindsight or the caller's IP."""
    raw = _headers(head_lines[1:]).get(REQUESTER_HEADER, b"").decode("ascii", "replace")
    name = "".join(c for c in raw if c.isalnum() or c in "-_.:@ ").strip()[:MAX_REQUESTER_CHARS]
    if name:
        return name
    if background:
        return BACKGROUND_REQUESTER
    return peer or "unknown"


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
    poll_failures: int = 0
    last_active: float = float("-inf")  # chat only: background jobs hold no conversation
    seen: "OrderedDict[str, int]" = field(default_factory=OrderedDict)  # prefix hash -> prefix chars
    strata: bool = False  # cancels on disconnect and honours strata_checkpoint: may take background work
    background: Optional[asyncio.Event] = None  # set while a background job runs here; set() preempts it
    background_ended: float = float("-inf")
    jobs: list[Job] = field(default_factory=list)  # chat and background requests running here now
    metrics: Optional[dict[str, Any]] = None  # last /metrics sample (Strata), trimmed to STATUS_METRICS_KEYS
    perf: Optional[dict[str, Any]] = None  # last /api/extra/perf sample, for a box with no Strata metrics

    @property
    def idle(self) -> bool:
        """No chat in progress. A background job does not count: chat preempts it."""
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
    def __init__(
        self, backends: Sequence[Backend], move_chars: int, quiet_s: float,
        holdoff_s: Optional[float] = None, background_names: Optional[Iterable[str]] = None,
    ) -> None:
        self.backends = list(backends)
        self.primary = self.backends[0]
        self.move_chars = move_chars
        self.quiet_s = quiet_s
        # A chat between steps must not find a Hindsight job on its box: same window as "quiet".
        self.holdoff_s = quiet_s if holdoff_s is None else holdoff_s
        self.background_names = set(background_names) if background_names is not None else {self.primary.name}
        unknown = self.background_names - {b.name for b in self.backends}
        if unknown:  # a typo here would silently send every Hindsight job elsewhere
            raise ValueError(f"background boxes {sorted(unknown)} are not in ROUTER_BACKENDS")
        self.background_counts = {"admitted": 0, "refused": 0, "preempted": 0, "incomplete": 0}
        self.expected_model: Optional[str] = None  # the primary's, kept while it is down
        self.changed = asyncio.Event()  # replaced on each wake; waiters resume in arrival order
        self.waiting = 0
        self.held: list[Job] = []  # chat requests waiting in _hold, oldest first (DP-499 statusline)

    def admit_background(self, now: float) -> Optional[Backend]:
        """A box a background job may start on now, or None: then it is refused, never held.

        Refusing at once is what keeps Hindsight's startup probe and its nginx pool fast: a
        held job would sit on a pool slot that another worker could have served.
        """
        if self.waiting:  # chat is queued: it gets the next free box
            return None
        for backend in self.backends:
            if (backend.name in self.background_names and backend.healthy and backend.strata and backend.idle
                    and backend.background is None and now - backend.last_active >= self.holdoff_s):
                return backend
        return None

    def preempt(self, backend: Backend) -> None:
        """Chat wants this box: stop its background job, if one runs and is not already stopping."""
        job = backend.background
        if job is not None and not job.is_set():
            log.info("chat wants %s: preempting its background job", backend.name)
            self.background_counts["preempted"] += 1
            job.set()

    def wake(self) -> None:
        """A box may have freed up: held requests look again, oldest first."""
        changed, self.changed = self.changed, asyncio.Event()
        changed.set()

    def route(self, body: bytes, now: float) -> tuple[list[Backend], list[tuple[str, int]]]:
        """The chosen box first, then the others to fall back to if it cannot be reached."""
        chain = prefix_chain(body)
        tokens = (chain[-1][1] if chain else 0) // CHARS_PER_TOKEN
        candidates = eligible(self.backends, self.expected_model, tokens)
        chosen = choose(candidates, chain, now, self.move_chars, self.quiet_s)
        if chosen is None:
            return [], chain
        return [chosen] + [b for b in candidates if b is not chosen], chain

    def note_poll(
        self, backend: Backend, models: Optional[dict[str, Any]], metrics: Optional[dict[str, Any]],
        now: Optional[float] = None, perf: Optional[dict[str, Any]] = None,
    ) -> None:
        now = time.monotonic() if now is None else now
        entry = _first_model(models)
        if entry is None:
            backend.poll_failures += 1
            if backend.poll_failures >= POLL_FAILURES_DOWN:
                backend.healthy = False
                backend.external_busy = False
                backend.strata = False
                backend.metrics = backend.perf = None
            return
        backend.poll_failures = 0
        # A lazy or unloaded server must refuse rather than load on a router request.
        backend.healthy = (entry.get("status") or {}).get("value") in (None, "loaded")
        backend.model = entry.get("id")
        backend.n_ctx = (entry.get("meta") or {}).get("n_ctx")
        if backend is self.primary and backend.healthy:
            self.expected_model = backend.model
        # Strata's /metrics is JSON with an "engine" block; kcpp has none. Only Strata is known to
        # cancel on disconnect (verified on CT102 2026-11-04) and to honour strata_checkpoint.
        backend.strata = isinstance((metrics or {}).get("engine"), dict)
        backend.metrics = ({k: metrics[k] for k in STATUS_METRICS_KEYS if k in metrics}
                           if backend.strata and metrics else None)
        if backend.metrics is not None:
            last = (metrics or {}).get("requests")
            if isinstance(last, list) and last:
                backend.metrics["requests"] = last[:1]  # the last completed run, for the stop/gen chips
        backend.perf = perf if isinstance(perf, dict) and "idle" in perf else None
        live = (metrics or {}).get("live") or {}
        ours = (backend.inflight > 0 or backend.background is not None
                or now - backend.background_ended < BACKGROUND_SETTLE_S)
        backend.external_busy = not ours and live.get("state", "idle") not in ("idle", "unloaded")

    def status(self) -> dict[str, Any]:
        now = time.monotonic()
        return {
            "expected_model": self.expected_model,
            "waiting": self.waiting,
            "background": {"boxes": sorted(self.background_names), "holdoff_s": self.holdoff_s,
                           **self.background_counts},
            "held": [job.describe(now) for job in self.held],
            "backends": [{
                "name": b.name, "url": f"http://{b.host}:{b.port}", "healthy": b.healthy, "model": b.model,
                "n_ctx": b.n_ctx, "inflight": b.inflight, "external_busy": b.external_busy,
                "strata": b.strata, "background_running": b.background is not None,
                "idle_s": None if b.last_active == float("-inf") else round(now - b.last_active, 1),
                "primary": b is self.primary,
                "jobs": [job.describe(now) for job in b.jobs],
                "metrics": b.metrics, "perf": b.perf,
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


def upstream_host(backend: Backend) -> str:
    """The Host a backend must be addressed as when the caller's own Host means nothing to it."""
    return LOOPBACK_HOST if backend.host == "127.0.0.1" else backend.host


def with_host(head_lines: Sequence[bytes], host: str) -> list[bytes]:
    """Replace the Host header. Hindsight's nginx pool sends its own container name, which Strata refuses."""
    lines = [head_lines[0]] + [line for line in head_lines[1:] if not line.lower().startswith(b"host:")]
    return lines[:1] + [b"Host: " + host.encode()] + lines[1:]


def background_body(body: bytes) -> bytes:
    """Mark a background request as one Strata must not keep as a conversation."""
    try:
        payload = json.loads(body)
    except ValueError:
        return body  # the server's 400 says more than ours would
    if not isinstance(payload, dict):
        return body
    payload.update(BACKGROUND_FIELDS)
    return json.dumps(payload, ensure_ascii=False).encode("utf-8")


async def _read_all(reader: asyncio.StreamReader) -> bytes:
    """The whole reply: the request said Connection: close, so the server's EOF ends it."""
    raw = bytearray()
    while True:
        chunk = await asyncio.wait_for(reader.read(65536), READ_TIMEOUT_S)
        if not chunk:
            return bytes(raw)
        raw += chunk
        if len(raw) > MAX_BODY_BYTES:
            raise ValueError("reply too large")


def response_complete(raw: bytes) -> bool:
    """Whether a buffered HTTP reply arrived whole. False is what turns a cut-off job into a 5xx.

    Framing decides when there is any (Content-Length, chunked); an unframed reply ended by EOF
    must parse as JSON, and a streamed one must have reached `data: [DONE]`.
    """
    head, sep, body = raw.partition(b"\r\n\r\n")
    if not sep:
        return False
    lines = head.split(b"\r\n")
    status = lines[0].split()
    if len(status) < 2 or not status[0].startswith(b"HTTP/") or not status[1].isdigit():
        return False
    headers = _headers(lines[1:])
    has_framing, framed = _framed_body(headers, body)
    if framed is None:
        return False
    if status[1] != b"200":
        return True  # an error is an error however short; the client must not see a 200 for it
    if b"event-stream" in headers.get(b"content-type", b"").lower():
        return _stream_ended(framed)
    if has_framing:
        return True
    try:
        json.loads(framed)
    except ValueError:
        return False
    return True


def _headers(lines: Iterable[bytes]) -> dict[bytes, bytes]:
    """Lowercased header name -> value, from a head's lines after its start line."""
    return {k.strip().lower(): v.strip() for k, _, v in (line.partition(b":") for line in lines)}


def _stream_ended(body: bytes) -> bool:
    """Whether an event stream reached its last event: OpenAI's [DONE], Anthropic's message_stop,
    or the Responses API's response.completed (the latter two never send [DONE])."""
    last = body.replace(b"\r\n", b"\n").rstrip().rsplit(b"\n\n", 1)[-1]
    return last.endswith(b"data: [DONE]") or last.startswith((b"event: message_stop", b"event: response.completed"))


def _framed_body(headers: dict[bytes, bytes], body: bytes) -> tuple[bool, Optional[bytes]]:
    """(whether the reply declared its length, its body or None if shorter than declared)."""
    if b"chunked" in headers.get(b"transfer-encoding", b"").lower():
        return True, _dechunk_strict(body)
    if b"content-length" in headers:
        length = headers[b"content-length"]
        return True, body if length.isdigit() and len(body) == int(length) else None
    return False, body


def _dechunk_strict(body: bytes) -> Optional[bytes]:
    """A chunked body, or None if it stops before its terminating zero-size chunk."""
    out = bytearray()
    while True:
        size_line, sep, rest = body.partition(b"\r\n")
        if not sep:
            return None
        try:
            size = int(size_line.split(b";")[0].strip() or b"x", 16)
        except ValueError:
            return None
        if size == 0:
            return bytes(out)
        if len(rest) < size + 2:
            return None
        out += rest[:size]
        body = rest[size + 2:]


async def read_request(reader: asyncio.StreamReader) -> Optional[tuple[list[bytes], bytes]]:
    try:
        head = await asyncio.wait_for(reader.readuntil(b"\r\n\r\n"), READ_TIMEOUT_S)
    except (asyncio.IncompleteReadError, asyncio.LimitOverrunError, asyncio.TimeoutError, ConnectionError):
        return None
    lines = head[:-4].split(b"\r\n")
    headers = _headers(lines[1:])
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


async def _send(
    backend: Backend, request: bytes,
) -> Optional[tuple[asyncio.StreamReader, asyncio.StreamWriter, bytes, bytes]]:
    """Connect and read the status line, the last point a request can still go elsewhere.

    -> (reader, writer, status line, header bytes already read), or None when the box could not
    answer at all. Only that marks it down: an HTTP error the engine itself sent (Strata's 502 for
    a reply that failed its schema) is that request's failure, passed on with its status.
    """
    writer = None
    try:
        reader, writer = await asyncio.wait_for(asyncio.open_connection(backend.host, backend.port), CONNECT_TIMEOUT_S)
        writer.write(request)
        await writer.drain()
        status = await asyncio.wait_for(reader.readline(), READ_TIMEOUT_S)
        parts = status.split()
        if len(parts) < 2:  # closed or reset before a status line: the box, not the request
            log.warning("%s answered %r", backend.name, status[:60])
            writer.close()
            return None
        if parts[1] not in RETRY_STATUSES:
            return reader, writer, status, b""
        head = await _engine_error_head(backend, reader, status)
    except asyncio.CancelledError:  # the client left: closing this is what stops the generation
        if writer:
            writer.close()
        raise
    except (OSError, asyncio.TimeoutError) as exc:
        log.warning("%s unreachable: %r", backend.name, exc)
        if writer:
            writer.close()
        return None
    if head is None:
        writer.close()
        return None
    return reader, writer, status, head


async def _engine_error_head(backend: Backend, reader: asyncio.StreamReader, status: bytes) -> Optional[bytes]:
    """A 502/503/504's headers if the engine sent it (pass it on), None if the box has no engine."""
    try:
        head = await asyncio.wait_for(reader.readuntil(b"\r\n\r\n"), READ_TIMEOUT_S)
    except (OSError, asyncio.TimeoutError, asyncio.IncompleteReadError, asyncio.LimitOverrunError):
        return None
    if proxy_failure(head):
        log.warning("%s has no engine behind it: %r", backend.name, status[:60])
        return None
    log.info("%s answered %r for this request; passed on", backend.name, status[:60])
    return head


def proxy_failure(head: bytes) -> bool:
    """Whether a 502/503/504 came from a proxy in front of the box rather than from its engine.

    r9700-lb.conf marks its own errors with X-Proxy-Error (error_page applies only to responses
    nginx generates, never to upstream ones). An HTML error page counts too: neither Strata nor
    kcpp answers in HTML, and it covers a proxy whose conf predates the header.
    """
    headers = _headers(head.split(b"\r\n"))
    return b"x-proxy-error" in headers or headers.get(b"content-type", b"").lower().startswith(b"text/html")


async def _client_closed(reader: asyncio.StreamReader) -> None:
    """Returns when the client hangs up. Clients send nothing further once the request is in."""
    try:
        while await reader.read(4096):
            pass
    except ConnectionError:
        pass


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

    tasks = [asyncio.ensure_future(pump()), asyncio.ensure_future(_client_closed(client_reader))]
    done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
    for task in pending:
        task.cancel()
    await asyncio.gather(*pending, return_exceptions=True)
    for task in done:
        task.exception()  # a reset on either side only ends this request


class Server:
    def __init__(self, router: Router, background: bool = False) -> None:
        self.router = router
        self.background = background  # the listener Hindsight's pool reaches: preemptible, never held

    async def handle(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        try:
            request = await read_request(reader)
            if request:
                peer = writer.get_extra_info("peername")
                requester = requester_of(request[0], peer[0] if isinstance(peer, tuple) else None, self.background)
                await self._serve(request[0], request[1], reader, writer, requester)
        except (ValueError, asyncio.IncompleteReadError):
            _respond(writer, b"400 Bad Request", {"error": "unreadable request"})
        except ConnectionError:
            pass
        finally:
            writer.close()

    async def _serve(
        self, lines: list[bytes], body: bytes, reader: asyncio.StreamReader, writer: asyncio.StreamWriter,
        requester: str = "unknown",
    ) -> None:
        method, target = (lines[0].split() + [b"", b""])[:2]
        if target == b"/router/status":
            _respond(writer, b"200 OK", self.router.status())
            return
        generation = _is_generation(method, target)
        if generation and self.background:
            await self._background(lines, body, reader, writer, requester)
            return
        job = Job(requester, "chat", time.monotonic(), len(body)) if generation else None
        if job is not None:
            held = await self._hold(body, reader, job)
            if held is None:  # the client left while waiting; no box ever saw it
                return
            order, chain = held
        else:  # status reads, model lists, control calls: the primary answers, as it did before the router
            chain = []
            order = [b for b in self.router.backends if b.healthy]
        if not order:
            _respond(writer, b"503 Service Unavailable", {"error": "no inference server is up"})
            return
        await self._forward(order, chain, rewrite_head(lines, len(body)) + body, job, target, reader, writer)

    async def _hold(
        self, body: bytes, reader: asyncio.StreamReader, job: Optional[Job] = None,
    ) -> Optional[tuple[list[Backend], list[tuple[str, int]]]]:
        """Wait here until the chosen box is free. Sent on, a request is stuck in that box's queue."""
        gone = asyncio.ensure_future(_client_closed(reader))
        self.router.waiting += 1
        if job is not None:
            self.router.held.append(job)
        try:
            while True:
                order, chain = self.router.route(body, time.monotonic())
                if not order or (order[0].idle and order[0].background is None):
                    return order, chain
                if order[0].idle:
                    self.router.preempt(order[0])  # its task closes the upstream, answers 503, then wakes us
                changed = asyncio.ensure_future(self.router.changed.wait())
                await asyncio.wait({gone, changed}, return_when=asyncio.FIRST_COMPLETED)
                changed.cancel()
                if gone.done():
                    return None
        finally:
            self.router.waiting -= 1
            if job is not None:
                self.router.held.remove(job)
            gone.cancel()

    async def _background(
        self, lines: list[bytes], body: bytes, reader: asyncio.StreamReader, writer: asyncio.StreamWriter,
        requester: str = BACKGROUND_REQUESTER,
    ) -> None:
        """Run a Hindsight job if a box is free of chat; refuse it at once otherwise.

        The reply is read whole before anything is sent, so the only outcomes are the box's own
        complete answer or a 5xx: a preempt or a cut-off reply can never reach Hindsight as a 200
        it would read as a document with no facts.
        """
        router = self.router
        backend = router.admit_background(time.monotonic())
        if backend is None:
            router.background_counts["refused"] += 1
            _respond(writer, b"503 Service Unavailable", {"error": "busy with chat; try another worker"})
            return
        router.background_counts["admitted"] += 1
        preempt = backend.background = asyncio.Event()
        job = Job(requester, "background", time.monotonic(), len(body))
        backend.jobs.append(job)
        payload = background_body(body)
        request = rewrite_head(with_host(lines, upstream_host(backend)), len(payload)) + payload
        upstream_writer = None
        tasks: list[asyncio.Future[Any]] = []
        try:
            upstream_reader, upstream_writer = await asyncio.wait_for(
                asyncio.open_connection(backend.host, backend.port), CONNECT_TIMEOUT_S)
            upstream_writer.write(request)
            await upstream_writer.drain()
            reply = asyncio.ensure_future(_read_all(upstream_reader))
            tasks = [reply, asyncio.ensure_future(preempt.wait()), asyncio.ensure_future(_client_closed(reader))]
            await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
            if reply.done() and not reply.exception():
                raw = reply.result()
                if response_complete(raw):
                    status, _, rest = raw.partition(b"\r\n")
                    writer.write(status + b"\r\nX-Inference-Backend: " + backend.name.encode() + b"\r\n" + rest)
                else:
                    router.background_counts["incomplete"] += 1
                    log.warning("%s background reply incomplete (%d bytes); answered 502", backend.name, len(raw))
                    _respond(writer, b"502 Bad Gateway", {"error": "the model server's reply was incomplete"})
            elif preempt.is_set():
                _respond(writer, b"503 Service Unavailable", {"error": "preempted by a chat request; retry"})
            elif reply.done():
                _respond(writer, b"502 Bad Gateway", {"error": "the model server's reply could not be read"})
            # else: Hindsight hung up; closing the upstream below cancels the generation
        except (OSError, asyncio.TimeoutError) as exc:
            log.warning("%s unreachable for background work: %r", backend.name, exc)
            _respond(writer, b"503 Service Unavailable", {"error": "no inference server answered"})
        finally:
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            if upstream_writer:
                upstream_writer.close()  # before the chat goes in: this is what stops the generation
            backend.background = None
            backend.jobs.remove(job)
            backend.background_ended = time.monotonic()
            router.wake()

    async def _forward(
        self, order: list[Backend], chain: list[tuple[str, int]], request: bytes, job: Optional[Job], target: bytes,
        reader: asyncio.StreamReader, writer: asyncio.StreamWriter,
    ) -> None:
        for backend in order:
            sent = await self._dispatch(backend, chain, request, job, target, reader, writer)
            if sent:
                return
            if job is not None:
                backend.healthy = False  # the next poll restores it
        _respond(writer, b"503 Service Unavailable", {"error": "no inference server answered"})

    async def _dispatch(
        self, backend: Backend, chain: list[tuple[str, int]], request: bytes, job: Optional[Job], target: bytes,
        reader: asyncio.StreamReader, writer: asyncio.StreamWriter,
    ) -> bool:
        """`job` is None for a request that does not occupy a box (status reads, model lists)."""
        matched = backend.matched(chain)
        counted = job is not None
        if job is not None:
            job.started = time.monotonic()  # from here it is running, not waiting
            backend.jobs.append(job)
            backend.inflight += 1
            backend.last_active = time.monotonic()
            self.router.preempt(backend)  # a fallback box: _hold only cleared the first choice
        upstream_writer = None
        try:
            sending = asyncio.ensure_future(_send(backend, request))
            gone = asyncio.ensure_future(_client_closed(reader))
            await asyncio.wait({sending, gone}, return_when=asyncio.FIRST_COMPLETED)
            gone_first = gone.done() and not sending.done()
            gone.cancel()
            if gone_first:  # a box can sit on a request for minutes before its first byte
                sending.cancel()
                await asyncio.gather(sending, return_exceptions=True)
                return True
            sent = sending.result()
            if sent is None:
                return False
            upstream, upstream_writer, status, head = sent
            if counted:
                backend.remember(chain)
                log.info("%s -> %s (cached %d of %d chars, %d in flight, %d waiting)", target.decode(), backend.name,
                         matched, chain[-1][1] if chain else 0, backend.inflight, self.router.waiting)
            writer.write(status + b"X-Inference-Backend: " + backend.name.encode() + b"\r\n" + head)
            await relay(upstream, reader, writer)
            return True
        finally:
            if upstream_writer:
                upstream_writer.close()
            if job is not None:
                backend.jobs.remove(job)
                backend.inflight -= 1
                backend.last_active = time.monotonic()
                self.router.wake()


def _is_generation(method: bytes, target: bytes) -> bool:
    return method == b"POST" and target.split(b"?", 1)[0] in GENERATION_PATHS


def _respond(writer: asyncio.StreamWriter, status: bytes, payload: dict[str, Any]) -> None:
    body = json.dumps(payload).encode()
    writer.write(b"HTTP/1.1 " + status + b"\r\nContent-Type: application/json\r\nConnection: close\r\n"
                 b"Content-Length: %d\r\n\r\n" % len(body) + body)


async def _get_json(backend: Backend, path: str) -> Optional[dict[str, Any]]:
    writer = None
    try:
        reader, writer = await asyncio.wait_for(asyncio.open_connection(backend.host, backend.port), POLL_TIMEOUT_S)
        writer.write(f"GET {path} HTTP/1.1\r\nHost: {upstream_host(backend)}\r\nConnection: close\r\n\r\n".encode())
        raw = await asyncio.wait_for(reader.read(), POLL_TIMEOUT_S)
        head, _, body = raw.partition(b"\r\n\r\n")
        if b" 200" not in head.split(b"\r\n", 1)[0]:
            return None
        if b"chunked" in head.lower():
            dechunked = _dechunk_strict(body)
            if dechunked is None:
                return None
            body = dechunked
        parsed = json.loads(body)
        return parsed if isinstance(parsed, dict) else None
    except (OSError, ValueError, asyncio.TimeoutError):
        return None
    finally:
        if writer:
            writer.close()


async def poll_forever(router: Router) -> None:
    while True:
        for backend in router.backends:
            models = await _get_json(backend, "/v1/models")
            metrics = await _get_json(backend, "/metrics") if models else None
            # kcpp has no JSON /metrics; its perf counters stand in for the statusline (DP-499).
            perf = await _get_json(backend, "/api/extra/perf") if models and not (metrics or {}).get("engine") else None
            was = backend.healthy
            router.note_poll(backend, models, metrics, perf=perf)
            if was != backend.healthy:
                log.info("%s is %s (%s)", backend.name, "up" if backend.healthy else "down", backend.model)
        router.wake()  # busy flags and quiet timers change without a request ending
        await asyncio.sleep(POLL_INTERVAL_S)


async def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    holdoff = os.environ.get("ROUTER_BACKGROUND_HOLDOFF_S")
    background_boxes = os.environ.get("ROUTER_BACKGROUND_BACKENDS")
    router = Router(
        parse_backends(os.environ.get("ROUTER_BACKENDS", "r9700=127.0.0.1:5006,dt21=10.0.0.69:5001")),
        move_chars=int(os.environ.get("ROUTER_MOVE_CHARS", "40000")),
        quiet_s=float(os.environ.get("ROUTER_QUIET_S", "120")),
        holdoff_s=float(holdoff) if holdoff else None,
        background_names=[n.strip() for n in background_boxes.split(",") if n.strip()] if background_boxes else None,
    )
    host = os.environ.get("ROUTER_HOST", "0.0.0.0")
    poller = asyncio.ensure_future(poll_forever(router))
    servers = [await asyncio.start_server(
        Server(router).handle, host, int(os.environ.get("ROUTER_PORT", "5001")), limit=MAX_HEAD_BYTES,
    )]
    background_port = int(os.environ.get("ROUTER_BACKGROUND_PORT", "0") or 0)  # 0: no background listener
    if background_port:
        servers.append(await asyncio.start_server(
            Server(router, background=True).handle, host, background_port, limit=MAX_HEAD_BYTES,
        ))
        log.info("background work on :%d -> %s, hold-off %.0f s", background_port,
                 ", ".join(sorted(router.background_names)), router.holdoff_s)
    log.info("routing to %s", ", ".join(b.name for b in router.backends))
    try:
        await asyncio.gather(*(server.serve_forever() for server in servers))
    finally:
        poller.cancel()
        for server in servers:
            server.close()


if __name__ == "__main__":
    asyncio.run(main())
