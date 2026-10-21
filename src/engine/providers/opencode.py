# src/engine/providers/opencode.py
"""opencode provider (DP-448): a persistent remote agent over HTTP.

`opencode*` models hand the turn to an `opencode serve` instance on a sandbox
host. Like cc it is an autonomous agent running its OWN tools, so the engine's
`tools` argument is ignored and the result is final text. Unlike cc/agy it is
persistent, and the state lives on the server, not here:

  - workspace: `<OPENCODE_WORKSPACES_ROOT>/<persona slug>`, passed as opencode's
    `directory` query param (one server hosts every persona's workspace);
  - session: the newest root session in that directory titled
    `derpr:<persona>`. Found by listing, so derpr stores nothing and a session
    deleted on the server is simply recreated.

Because the session holds the history, only the messages after the persona's
last reply are forwarded. The driver retries a failed or empty attempt, which
for a persistent session would post the same message twice and re-run the
whole agentic turn, so each forwarded message carries the driver's per-call
`request_id` in its part metadata. An attempt whose id is already the
session's last user message resumes that turn (waits for it to finish and
reads its reply) instead of posting again.
"""

import asyncio
import json
import logging
import time
import uuid
from typing import TYPE_CHECKING, Any, AsyncIterator, Dict, List, Optional, Set, Tuple

import aiohttp
from aiolimiter import AsyncLimiter

from config import global_config
from src.llm_errors import LLMCommunicationError

from .base import Provider
from ._subprocess import render_transcript_blocks, sanitize_workspace_name

if TYPE_CHECKING:
    from src.engine.driver import TextEngine

logger = logging.getLogger(__name__)

MODEL_PREFIX = "opencode"
SESSION_TITLE_PREFIX = "derpr:"
_KEY_FIELD = "derpr_key"
_REQUEST_TIMEOUT = aiohttp.ClientTimeout(total=30)
_POLL_SECONDS = 2.0
# A resumed turn that is idle yet has no finished reply gets this long to
# produce one before it is reported as lost (the server may not have marked
# the session busy yet when we first look).
_RESUME_GRACE_SECONDS = 15.0

# Headless: nobody can answer opencode's own "ask" prompts, and an unanswered
# one stalls the turn until the timeout. Resolve each one opencode ships as
# "ask" (1.19.3 build agent). The sandbox host is the boundary, so reads
# outside the workspace and of .env files are allowed. `question` is denied:
# it would wait for an answer that cannot come, so the agent asks in its reply
# instead. doom_loop (the same tool call repeated verbatim) is denied: there is
# no legitimate reason for it, and a long-running agent must not spin on it.
SESSION_PERMISSIONS: List[Dict[str, str]] = [
    {"permission": "external_directory", "pattern": "*", "action": "allow"},
    {"permission": "read", "pattern": "*", "action": "allow"},
    {"permission": "question", "pattern": "*", "action": "deny"},
    {"permission": "doom_loop", "pattern": "*", "action": "deny"},
]


def ensure_opencode_supported() -> None:
    if not global_config.OPENCODE_SERVER_URL:
        raise LLMCommunicationError(
            "The 'opencode' provider needs OPENCODE_SERVER_URL (the `opencode serve` "
            "address). See docs/user_guide.md (opencode)."
        )


def opencode_model(model_name: str) -> Optional[Dict[str, str]]:
    """`opencode` -> None (the server's default model);
    `opencode-<provider>/<model>` -> opencode's `{providerID, modelID}`."""
    rest = model_name[len(MODEL_PREFIX):]
    if not rest:
        return None
    provider_id, sep, model_id = rest.lstrip("-").partition("/")
    if not (rest.startswith("-") and sep and provider_id and model_id):
        raise LLMCommunicationError(
            f"Model '{model_name}' is not 'opencode' or 'opencode-<provider>/<model>'."
        )
    return {"providerID": provider_id, "modelID": model_id}


def workspace_slug(persona_name: Optional[str]) -> str:
    return sanitize_workspace_name(persona_name) or "global"


def new_messages(history: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """What the session has not seen: everything after the persona's last
    reply. With no reply in the window yet, the whole window."""
    for i in range(len(history) - 1, -1, -1):
        if history[i].get("role") == "assistant":
            return history[i + 1:] or history[-1:]
    return history


def reply_text(parts: List[Dict[str, Any]]) -> str:
    return "\n".join(
        p.get("text", "") for p in parts
        if p.get("type") == "text" and not p.get("synthetic") and not p.get("ignored")
    ).strip()


def raise_for_reply_error(info: Dict[str, Any]) -> None:
    """A reply that ended in an error is an error even when it has text: an
    aborted turn's partial prose is not its answer."""
    error = info.get("error")
    if error:
        detail = (error.get("data") or {}).get("message") or ""
        raise LLMCommunicationError(f"opencode turn failed: {error.get('name', 'error')} {detail}".strip())


def _role(message: Dict[str, Any]) -> Any:
    return (message.get("info") or {}).get("role")


def _has_key(message: Dict[str, Any], key: str) -> bool:
    return any((p.get("metadata") or {}).get(_KEY_FIELD) == key for p in message.get("parts", []))


class OpencodeClient:
    """The few opencode HTTP calls derpr makes, scoped to one directory."""

    def __init__(self, http: aiohttp.ClientSession, directory: str) -> None:
        self._http = http
        self._base = global_config.OPENCODE_SERVER_URL
        self.directory = directory

    async def request(self, method: str, path: str, *, json_body: Any = None,
                      params: Optional[Dict[str, str]] = None,
                      timeout: Optional[aiohttp.ClientTimeout] = _REQUEST_TIMEOUT,
                      directory: Optional[str] = None) -> Any:
        query = {"directory": directory or self.directory, **(params or {})}
        try:
            async with self._http.request(method, f"{self._base}{path}", json=json_body,
                                          params=query, timeout=timeout) as resp:
                body = await resp.text()
                if resp.status >= 400:
                    raise LLMCommunicationError(
                        f"opencode {method} {path} -> HTTP {resp.status}: {body[:300]}"
                    )
                try:
                    return json.loads(body) if body else None
                except ValueError as e:
                    # A proxy's HTML page, a truncated body: still a failed
                    # call, so the driver's retry policy sees it as one.
                    raise LLMCommunicationError(
                        f"opencode {method} {path} returned non-JSON: {body[:300]}"
                    ) from e
        except asyncio.TimeoutError:
            # Raw, so the caller can tell a timed-out turn (abort it) from a
            # failed request. Listed first: aiohttp's ServerTimeoutError is
            # both a TimeoutError and a ClientError.
            raise
        except aiohttp.ClientError as e:
            raise LLMCommunicationError(f"opencode {method} {path} failed: {e}") from e

    async def ensure_directory(self, root: str, slug: str) -> None:
        """Create the workspace on first use. opencode creates no directory for
        a session and every tool fails in a missing one, so make it with the
        server's shell endpoint from a throwaway session in the root (no LLM call)."""
        nodes = await self.request("GET", "/file", params={"path": "."}, directory=root)
        if any(n.get("name") == slug and n.get("type") == "directory" for n in nodes or []):
            return
        session = await self.request("POST", "/session", json_body={"title": "derpr:bootstrap"},
                                     directory=root)
        try:
            await self.request("POST", f"/session/{session['id']}/shell", directory=root,
                               json_body={"agent": "build", "command": f"mkdir -p -- '{slug}'"})
        finally:
            await self.request("DELETE", f"/session/{session['id']}", directory=root)
        # The shell endpoint reports a failed command as output, not as an HTTP
        # error, so confirm the directory exists before the caller caches it.
        nodes = await self.request("GET", "/file", params={"path": "."}, directory=root)
        if not any(n.get("name") == slug and n.get("type") == "directory" for n in nodes or []):
            raise LLMCommunicationError(f"opencode: could not create workspace {root}/{slug}.")
        logger.info("opencode: created workspace %s/%s", root, slug)

    async def find_or_create_session(self, title: str) -> str:
        sessions = await self.request("GET", "/session")
        mine = [s for s in sessions or [] if s.get("title") == title and not s.get("parentID")]
        if mine:
            return str(max(mine, key=lambda s: (s.get("time") or {}).get("updated", 0))["id"])
        created = await self.request("POST", "/session",
                                     json_body={"title": title, "permission": SESSION_PERMISSIONS})
        logger.info("opencode: new session %s for %s in %s", created["id"], title, self.directory)
        return str(created["id"])

    async def messages(self, session_id: str, limit: int = 50) -> List[Dict[str, Any]]:
        return await self.request("GET", f"/session/{session_id}/message",
                                  params={"limit": str(limit)}) or []

    async def is_busy(self, session_id: str) -> bool:
        status = await self.request("GET", "/session/status") or {}
        return bool((status.get(session_id) or {}).get("type", "idle") != "idle")

    async def abort(self, session_id: str) -> None:
        try:
            await self.request("POST", f"/session/{session_id}/abort")
        except (LLMCommunicationError, asyncio.TimeoutError) as e:
            logger.warning("opencode: abort of %s failed: %r", session_id, e)

    async def wait_idle(self, session_id: str, deadline: float) -> None:
        """Wait out a turn already running in the session (one derpr did not
        post, or one whose call was dropped). Posting into a busy session
        queues behind it and blurs which reply is whose."""
        while await self.is_busy(session_id):
            if time.monotonic() > deadline:
                raise LLMCommunicationError("opencode session stayed busy with another turn.")
            await asyncio.sleep(_POLL_SECONDS)


async def resume_turn(client: OpencodeClient, session_id: str, key: str, deadline: float) -> Optional[str]:
    """If a user message in the session carries `key`, this attempt is a retry
    of a turn already posted: wait for it and return its reply. None when the
    key is not there (the caller posts). Any user message, not only the last:
    the provider lock is released between driver attempts, so another call's
    message may already sit after ours."""
    msgs = await client.messages(session_id)
    if not any(_has_key(m, key) for m in msgs if _role(m) == "user"):
        return None
    idle_since: Optional[float] = None
    while True:
        busy = await client.is_busy(session_id)
        msgs = await client.messages(session_id)
        start = max((i for i, m in enumerate(msgs) if _has_key(m, key)), default=-1)
        # Our turn's replies end at the next user message (another call's turn).
        end = next((i for i in range(start + 1, len(msgs)) if _role(msgs[i]) == "user"), len(msgs))
        replies = [m for m in msgs[start + 1:end] if _role(m) == "assistant"]
        final = replies[-1] if replies else None
        if not busy and final and (final["info"].get("time") or {}).get("completed"):
            text = reply_text(final.get("parts", []))
            raise_for_reply_error(final["info"])
            return text
        now = time.monotonic()
        idle_since = None if busy else (idle_since or now)
        if idle_since is not None and now - idle_since > _RESUME_GRACE_SECONDS:
            raise LLMCommunicationError("opencode turn ended without a reply.")
        if now > deadline:
            await client.abort(session_id)
            raise LLMCommunicationError("opencode turn timed out; aborted on the server.")
        await asyncio.sleep(_POLL_SECONDS)


async def run_turn(client: OpencodeClient, session_id: str, key: str,
                   body: Dict[str, Any], timeout: float) -> Tuple[str, bool]:
    """Resume the turn `key` names, or post `body` as a new one. Returns
    (reply text, resumed)."""
    deadline = time.monotonic() + timeout
    # Whether a turn running in the session is this call's own (one it posted
    # or is resuming) rather than one it is waiting out.
    ours = True
    try:
        text = await resume_turn(client, session_id, key, deadline)
        if text is not None:
            return text, True
        ours = False
        await client.wait_idle(session_id, deadline)
        ours = True
        try:
            reply = await client.request(
                "POST", f"/session/{session_id}/message", json_body=body,
                timeout=aiohttp.ClientTimeout(total=timeout, sock_connect=10),
            )
        except asyncio.TimeoutError as e:
            await client.abort(session_id)
            raise LLMCommunicationError(
                f"opencode turn timed out after {timeout:.0f}s; aborted on the server."
            ) from e
        reply = reply or {}
        text = reply_text(reply.get("parts", []))
        raise_for_reply_error(reply.get("info") or {})
        return text, False
    except asyncio.CancelledError:
        # The caller went away (the portal's stop button, shutdown): stop the
        # agent rather than leave it running unobserved, as cc's process-tree
        # kill does for its turn.
        if ours:
            await client.abort(session_id)
        raise


class OpencodeProvider(Provider):
    """Remote opencode agent: one persistent session + workspace per persona."""

    route_method_name = "_stream_opencode_response"

    def __init__(self, engine: "TextEngine") -> None:
        self._engine = engine
        # One turn at a time per workspace: a second prompt to a busy session
        # would queue behind it on the server and blur which reply is whose.
        self._locks: Dict[str, asyncio.Lock] = {}
        self._ready_dirs: Set[str] = set()

    def matches(self, model_name: str) -> bool:
        return model_name.startswith(MODEL_PREFIX)

    def limiters_for(self, model_name: str) -> List[AsyncLimiter]:
        return [self._engine._opencode_limiter]

    def ensure_supported(self, model_name: str) -> None:
        ensure_opencode_supported()

    async def generate(self, config: Dict[str, Any], history_object: Dict[str, Any],
                       tools: Optional[List[Dict[str, Any]]] = None) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        system_prompt, history = self._engine._extract_system_prompt(history_object)
        model = opencode_model(config.get("model_name", ""))
        persona_name = config.get("persona_name")
        slug = workspace_slug(persona_name)
        root = global_config.OPENCODE_WORKSPACES_ROOT
        directory = f"{root}/{slug}"
        forwarded = new_messages(history)
        prompt = "\n\n".join(b.text for b in render_transcript_blocks(forwarded))
        # The driver's per-call id, stable across its retries (a user's
        # regenerate is a new call, so it gets a new turn, not the old reply).
        key = config.get("request_id") or uuid.uuid4().hex
        timeout = global_config.OPENCODE_CALL_TIMEOUT_SECONDS

        api_payload: Dict[str, Any] = {
            "model": config.get("model_name"),
            "opencode_model": model,
            "server": global_config.OPENCODE_SERVER_URL,
            "workspace": directory,
            "messages_forwarded": len(forwarded),
            "prompt_chars": len(prompt),
            "system_prompt_chars": len(system_prompt or ""),
            "tools_ignored": [t["function"]["name"] for t in tools or []
                              if "function" in t and "name" in t["function"]],
        }

        auth = {"Authorization": aiohttp.encode_basic_auth(
            global_config.OPENCODE_SERVER_USERNAME, global_config.OPENCODE_SERVER_PASSWORD)}
        lock = self._locks.setdefault(directory, asyncio.Lock())
        try:
            async with lock, aiohttp.ClientSession(headers=auth) as http:
                client = OpencodeClient(http, directory)
                if directory not in self._ready_dirs:
                    await client.ensure_directory(root, slug)
                    self._ready_dirs.add(directory)
                session_id = await client.find_or_create_session(f"{SESSION_TITLE_PREFIX}{persona_name or slug}")
                api_payload["session"] = session_id

                body: Dict[str, Any] = {
                    "parts": [{"type": "text", "text": prompt, "metadata": {_KEY_FIELD: key}}],
                }
                if system_prompt:
                    body["system"] = system_prompt
                if model:
                    body["model"] = model
                text, api_payload["resumed"] = await run_turn(client, session_id, key, body, timeout)
        except LLMCommunicationError as e:
            if e.api_payload is None:
                e.api_payload = api_payload
            raise
        except asyncio.TimeoutError as e:
            raise LLMCommunicationError("opencode request timed out.", api_payload=api_payload) from e

        return {"type": "text", "content": text}, api_payload

    async def stream(
        self,
        persona_config: Dict[str, Any],
        history_object: Dict[str, Any],
        tools: Optional[List[Dict[str, Any]]] = None,
        *,
        local_inference_config: Optional[Dict[str, Any]] = None,
    ) -> AsyncIterator[Dict[str, Any]]:
        async for ev in self._engine._stream_opencode_response(persona_config, history_object, tools):
            yield ev
