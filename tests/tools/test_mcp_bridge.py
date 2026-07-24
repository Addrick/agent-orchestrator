# tests/tools/test_mcp_bridge.py
"""DP-240 MCP bridge: exposure, gating, and token auth.

The gate is the ONLY authorization boundary for subagent tool calls (capable
dispatches keep --dangerously-skip-permissions), so the deny paths here are
load-bearing, not smoke tests.
"""

import pytest

from src.proposals.agent_call import AgentCallRunner
from src.tool_policy import ToolPolicy
from src.tools.mcp_bridge import PARKED_STATUS, BridgeTokenStore, McpBridge, _current_agent
from src.tools.tool_manager import ToolManager


READ_TOOL = {
    "type": "function",
    "is_write": False,
    "function": {
        "name": "safe_read",
        "description": "Read something harmless.",
        "parameters": {"type": "object", "properties": {}},
    },
}
WRITE_TOOL = {
    "type": "function",
    "is_write": True,
    "function": {
        "name": "dangerous_write",
        "description": "Mutate something.",
        "parameters": {"type": "object", "properties": {}},
    },
}


class _FakeToolManager(ToolManager):
    """ToolManager with a fixed definition list, so these tests don't depend on
    the real catalog's contents."""

    def __init__(self, definitions):
        super().__init__()
        self._definitions = definitions

    def get_tool_definitions(self):
        return list(self._definitions)


@pytest.fixture
def wiring():
    manager = _FakeToolManager([READ_TOOL, WRITE_TOOL])
    calls = []

    async def _read(**kwargs):
        calls.append(("safe_read", kwargs))
        return "read-result"

    async def _write(**kwargs):
        calls.append(("dangerous_write", kwargs))
        return "write-result"

    manager.register("safe_read", _read)
    manager.register("dangerous_write", _write)

    policy = ToolPolicy(default="deny", allow=["safe_read", "dangerous_write"])
    runner = AgentCallRunner(lambda: manager, lambda: policy)

    proposed = []

    async def _propose(agent_id, tool_name, tool_args):
        proposed.append((agent_id, tool_name, tool_args))
        return 42

    bridge = McpBridge(runner, _propose)
    return bridge, runner, calls, proposed, policy


@pytest.fixture
def as_agent():
    """Bind an authenticated agent for the duration of a test."""
    token = _current_agent.set("DP-999-1")
    yield
    _current_agent.reset(token)


# -- exposure ----------------------------------------------------------------

def test_only_policy_allowed_tools_are_exposed(wiring):
    bridge, runner, _, _, policy = wiring
    assert set(runner.exposed_tool_names()) == {"safe_read", "dangerous_write"}

    policy.allow = ["safe_read"]
    assert runner.exposed_tool_names() == ["safe_read"]


def test_unregistered_tool_is_not_exposed_even_if_allowed(wiring):
    """Policy allowing a name is not enough — a tool with no handler cannot be
    called, so listing it would advertise a capability that does not exist."""
    _, runner, _, _, policy = wiring
    policy.allow = ["safe_read", "never_registered"]
    assert runner.exposed_tool_names() == ["safe_read"]


def test_schema_round_trip_marks_gated_tools(wiring):
    bridge, _, _, _, _ = wiring
    read = bridge._to_mcp_tool(READ_TOOL)
    write = bridge._to_mcp_tool(WRITE_TOOL)

    assert read.name == "safe_read"
    assert read.inputSchema == {"type": "object", "properties": {}}
    assert "requires human approval" not in (read.description or "")
    # The subagent must be able to see that a call will park before it makes it.
    assert "requires human approval" in (write.description or "")


# -- gating ------------------------------------------------------------------

@pytest.mark.asyncio
async def test_read_tool_executes_immediately(wiring, as_agent):
    bridge, _, calls, proposed, _ = wiring
    result = await bridge._handle_call("safe_read", {"q": "x"})

    assert result == {"result": "read-result"}
    assert calls == [("safe_read", {"q": "x"})]
    assert proposed == []


@pytest.mark.asyncio
async def test_write_tool_parks_and_does_not_execute(wiring, as_agent):
    bridge, _, calls, proposed, _ = wiring
    result = await bridge._handle_call("dangerous_write", {"target": "prod"})

    assert result["status"] == PARKED_STATUS
    assert result["proposal_id"] == 42
    assert proposed == [("DP-999-1", "dangerous_write", {"target": "prod"})]
    # The whole point: nothing ran.
    assert calls == []


@pytest.mark.asyncio
async def test_unexposed_tool_is_refused(wiring, as_agent):
    bridge, _, calls, proposed, policy = wiring
    policy.allow = ["safe_read"]

    result = await bridge._handle_call("dangerous_write", {})

    assert "not exposed" in result["error"]
    assert calls == []
    assert proposed == []


@pytest.mark.asyncio
async def test_call_without_authenticated_agent_is_refused(wiring):
    """Backstop for the ASGI auth layer: an unattributable call cannot be gated
    or audited, so it must never execute."""
    bridge, _, calls, proposed, _ = wiring
    result = await bridge._handle_call("safe_read", {})

    assert result == {"error": "unauthenticated"}
    assert calls == []
    assert proposed == []


# -- tokens ------------------------------------------------------------------

def test_token_mint_resolve_revoke():
    store = BridgeTokenStore()
    token = store.mint("agent-1")

    assert store.resolve(token) == "agent-1"
    assert store.token_for("agent-1") == token

    store.revoke("agent-1")
    assert store.resolve(token) is None
    assert store.token_for("agent-1") is None


def test_unknown_and_empty_tokens_resolve_to_nothing():
    store = BridgeTokenStore()
    store.mint("agent-1")

    assert store.resolve("") is None
    assert store.resolve("not-a-real-token") is None


def test_reminting_invalidates_the_previous_token():
    """A re-dispatch must not leave the old credential usable."""
    store = BridgeTokenStore()
    first = store.mint("agent-1")
    second = store.mint("agent-1")

    assert first != second
    assert store.resolve(first) is None
    assert store.resolve(second) == "agent-1"


# -- ASGI auth layer ---------------------------------------------------------

@pytest.mark.asyncio
async def test_asgi_rejects_missing_and_bad_tokens(wiring):
    """The SDK must never be entered by an unauthenticated caller."""
    bridge, _, _, _, _ = wiring
    entered = []
    bridge._session_manager.handle_request = (
        lambda *a, **k: entered.append(True))  # type: ignore[assignment]

    for headers in ([], [(b"authorization", b"Bearer wrong")],
                    [(b"authorization", b"Basic xyz")]):
        sent = []

        async def _send(message):
            sent.append(message)

        await bridge.handle_asgi({"type": "http", "headers": headers}, None, _send)

        assert sent[0]["status"] == 401
        assert entered == []


@pytest.mark.asyncio
async def test_asgi_accepts_a_live_token_and_binds_the_agent(wiring):
    bridge, _, _, _, _ = wiring
    token = bridge.tokens.mint("DP-777-1")
    seen = {}

    async def _handle(scope, receive, send):
        seen["agent"] = _current_agent.get()

    bridge._session_manager.handle_request = _handle  # type: ignore[assignment]

    await bridge.handle_asgi(
        {"type": "http", "headers": [(b"authorization", f"Bearer {token}".encode())]},
        None, None,
    )

    assert seen["agent"] == "DP-777-1"
    # The binding must not leak out of the request.
    assert _current_agent.get() == ""


@pytest.mark.asyncio
async def test_asgi_rejects_a_revoked_token(wiring):
    """A terminal agent's credential must stop working immediately."""
    bridge, _, _, _, _ = wiring
    token = bridge.tokens.mint("DP-777-1")
    bridge.tokens.revoke("DP-777-1")
    sent = []

    async def _send(message):
        sent.append(message)

    await bridge.handle_asgi(
        {"type": "http", "headers": [(b"authorization", f"Bearer {token}".encode())]},
        None, _send,
    )

    assert sent[0]["status"] == 401


def test_tokens_are_per_agent():
    store = BridgeTokenStore()
    a = store.mint("agent-a")
    b = store.mint("agent-b")

    assert store.resolve(a) == "agent-a"
    assert store.resolve(b) == "agent-b"
    store.revoke("agent-a")
    assert store.resolve(b) == "agent-b"
