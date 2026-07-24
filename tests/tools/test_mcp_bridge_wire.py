# tests/tools/test_mcp_bridge_wire.py
"""DP-240 MCP bridge over the real wire.

``test_mcp_bridge.py`` drives the handlers directly — it proves the gating
logic but says nothing about whether a real MCP client can *reach* them. A
transport or schema mismatch (streamable-HTTP negotiation, tool schema shape,
auth header handling) passes every test in that file and fails on the first
live dispatch.

So these tests boot the bridge under uvicorn on a real port and connect with
the SDK's own client: initialize handshake, tools/list, tools/call, and the
401 path. No LLM involved, so this runs in CI on every change.

Not covered here: whether the ``--mcp-config`` payload ``Dispatcher`` builds is
a shape the ``claude`` CLI still accepts. That was verified once by hand
(2026-07-24, CLI 2.1.219 — connected and called a tool). It is deliberately not
a test: it needs the CLI, live API keys, and an llm_live run nobody performs on
the day the CLI schema changes, so it would document rather than gate. Re-run
the recipe in memory/project/tasks/DP-240.md if dispatch starts failing.
"""

import asyncio
import contextlib
import json
import socket
from typing import Any, AsyncIterator, Dict, List

import pytest
import uvicorn
from mcp import ClientSession
from mcp.client.streamable_http import create_mcp_http_client, streamable_http_client

from src.proposals.agent_call import AgentCallRunner
from src.tool_policy import ToolPolicy
from src.tools.mcp_bridge import PARKED_STATUS, McpBridge
from src.tools.tool_manager import ToolManager

pytestmark = pytest.mark.integration

READ_TOOL = {
    "type": "function",
    "is_write": False,
    "function": {
        "name": "safe_read",
        "description": "Read something harmless.",
        "parameters": {
            "type": "object",
            "properties": {"target": {"type": "string"}},
            "required": ["target"],
        },
    },
}
WRITE_TOOL = {
    "type": "function",
    "is_write": True,
    "function": {
        "name": "dangerous_write",
        "description": "Mutate something.",
        "parameters": {"type": "object", "properties": {"value": {"type": "string"}}},
    },
}


class _FixedToolManager(ToolManager):
    def __init__(self, definitions: List[Dict[str, Any]]) -> None:
        super().__init__()
        self._definitions = definitions

    def get_tool_definitions(self) -> List[Dict[str, Any]]:
        return list(self._definitions)


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


class _ServedBridge:
    """A live bridge: uvicorn on a real port, plus the records it wrote."""

    def __init__(self, bridge: McpBridge, url: str,
                 calls: List[Any], proposed: List[Any]) -> None:
        self.bridge = bridge
        self.url = url
        self.calls = calls
        self.proposed = proposed


@contextlib.asynccontextmanager
async def _serve() -> AsyncIterator[_ServedBridge]:
    manager = _FixedToolManager([READ_TOOL, WRITE_TOOL])
    calls: List[Any] = []
    proposed: List[Any] = []

    async def _read(**kwargs: Any) -> Dict[str, Any]:
        calls.append(("safe_read", kwargs))
        return {"content": f"read {kwargs.get('target')}"}

    async def _write(**kwargs: Any) -> str:
        calls.append(("dangerous_write", kwargs))
        return "should never run"

    manager.register("safe_read", _read)
    manager.register("dangerous_write", _write)

    runner = AgentCallRunner(
        lambda: manager,
        lambda: ToolPolicy(default="deny", allow=["safe_read", "dangerous_write"]),
    )

    async def _propose(agent_id: str, tool_name: str, tool_args: Dict[str, Any]) -> int:
        proposed.append((agent_id, tool_name, tool_args))
        return 7

    bridge = McpBridge(runner, _propose)
    port = _free_port()

    # Mounted at a path, matching how KoboldEngineAdapter mounts it — a bridge
    # that only works at the ASGI root would pass and then 404 in production.
    async def app(scope: Any, receive: Any, send: Any) -> None:
        if scope["type"] == "lifespan":
            while True:
                message = await receive()
                if message["type"] == "lifespan.startup":
                    await send({"type": "lifespan.startup.complete"})
                elif message["type"] == "lifespan.shutdown":
                    await send({"type": "lifespan.shutdown.complete"})
                    return
            return
        scope = dict(scope)
        path = scope.get("path", "")
        scope["path"] = path[len("/mcp"):] or "/"
        await bridge.handle_asgi(scope, receive, send)

    config = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning")
    server = uvicorn.Server(config)
    async with bridge.lifespan():
        task = asyncio.create_task(server.serve())
        try:
            for _ in range(100):
                if server.started:
                    break
                await asyncio.sleep(0.05)
            assert server.started, "uvicorn did not start"
            yield _ServedBridge(bridge, f"http://127.0.0.1:{port}/mcp", calls, proposed)
        finally:
            server.should_exit = True
            with contextlib.suppress(asyncio.TimeoutError):
                await asyncio.wait_for(task, timeout=10)


@contextlib.asynccontextmanager
async def _client(served: _ServedBridge, token: str) -> AsyncIterator[ClientSession]:
    headers = {"Authorization": f"Bearer {token}"}
    async with create_mcp_http_client(headers=headers) as http_client:
        async with streamable_http_client(served.url, http_client=http_client) as (read, write, _):
            async with ClientSession(read, write) as session:
                await session.initialize()
                yield session


def _flatten(exc: BaseException) -> List[BaseException]:
    """Leaf exceptions of a (possibly nested) group.

    The MCP client runs its transport in a task group, so a transport-level
    rejection surfaces as an ExceptionGroup. Asserting on the leaf is what
    makes the auth tests fail for the right reason instead of passing on any
    error at all — including a typo in this file's own helpers.
    """
    if isinstance(exc, BaseExceptionGroup):
        leaves: List[BaseException] = []
        for sub in exc.exceptions:
            leaves.extend(_flatten(sub))
        return leaves
    return [exc]


def _assert_rejected_401(exc: BaseException) -> None:
    messages = [str(e) for e in _flatten(exc)]
    assert any("401" in m for m in messages), f"expected a 401, got: {messages}"


def _payload(result: Any) -> Dict[str, Any]:
    """The tool result as the subagent would see it."""
    assert result.content, "tool result had no content blocks"
    return dict(json.loads(result.content[0].text))


@pytest.mark.asyncio
async def test_real_client_lists_tools_with_usable_schemas():
    """The initialize handshake completes and tools/list returns schemas an MCP
    client accepts — the failure this whole file exists to catch."""
    async with _serve() as served:
        token = served.bridge.tokens.mint("DP-999-1")
        async with _client(served, token) as session:
            tools = (await session.list_tools()).tools

    by_name = {t.name: t for t in tools}
    assert set(by_name) == {"safe_read", "dangerous_write"}
    assert by_name["safe_read"].inputSchema["required"] == ["target"]
    # The gated tool advertises its own gating, so the agent can plan around it.
    assert "requires human approval" in (by_name["dangerous_write"].description or "")


@pytest.mark.asyncio
async def test_real_client_executes_an_ungated_tool():
    async with _serve() as served:
        token = served.bridge.tokens.mint("DP-999-1")
        async with _client(served, token) as session:
            result = await session.call_tool("safe_read", {"target": "logs"})

    assert _payload(result)["result"] == {"content": "read logs"}
    assert served.calls == [("safe_read", {"target": "logs"})]
    assert served.proposed == []


@pytest.mark.asyncio
async def test_real_client_gets_parked_instead_of_a_write():
    """Over the wire, a gated call must return the park envelope and leave the
    handler unrun — the gate is the only boundary there is."""
    async with _serve() as served:
        token = served.bridge.tokens.mint("DP-999-1")
        async with _client(served, token) as session:
            result = await session.call_tool("dangerous_write", {"value": "x"})

    payload = _payload(result)
    assert payload["status"] == PARKED_STATUS
    assert payload["proposal_id"] == 7
    assert served.calls == []
    assert served.proposed == [("DP-999-1", "dangerous_write", {"value": "x"})]


@pytest.mark.asyncio
async def test_unknown_token_never_reaches_the_mcp_layer():
    """A bad token is rejected by the ASGI wrapper, so the client cannot even
    complete initialize — not merely denied at tool-call time."""
    async with _serve() as served:
        served.bridge.tokens.mint("DP-999-1")
        with pytest.raises(BaseException) as caught:
            async with _client(served, "not-a-real-token"):
                pass
        _assert_rejected_401(caught.value)
        assert served.calls == []


@pytest.mark.asyncio
async def test_revoked_token_stops_working():
    """Terminal-state revocation must actually close the door on a live URL."""
    async with _serve() as served:
        token = served.bridge.tokens.mint("DP-999-1")
        async with _client(served, token) as session:
            await session.call_tool("safe_read", {"target": "logs"})

        served.bridge.tokens.revoke("DP-999-1")
        with pytest.raises(BaseException) as caught:
            async with _client(served, token):
                pass
        _assert_rejected_401(caught.value)

    assert served.calls == [("safe_read", {"target": "logs"})]
