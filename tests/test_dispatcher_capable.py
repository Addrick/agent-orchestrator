# tests/test_dispatcher_capable.py
"""DP-240: the capable dispatch tier (MCP bridge wiring).

The most important test here is the *negative* one: a default dispatch's argv
must not drift as the capable tier grows. Default fixr is the common path and it
is supposed to be unchanged by DP-240.
"""

import json

import pytest

from config import global_config
from src.self_edit.dispatcher import Dispatcher, DispatcherError, _host_of
from src.self_edit.registry import AgentRegistry
from src.tools.mcp_bridge import BridgeTokenStore


async def _noop_wake(record, event):
    return None


@pytest.fixture
def bridge_url(monkeypatch):
    url = "http://10.0.0.70:5003/mcp"
    monkeypatch.setattr(global_config, "MCP_BRIDGE_PUBLIC_URL", url)
    monkeypatch.setattr(global_config, "MCP_BRIDGE_PATH", "/mcp")
    return url


# -- default tier is unchanged ----------------------------------------------

def test_default_argv_has_no_mcp_wiring():
    d = Dispatcher(AgentRegistry(), on_wake=_noop_wake)
    argv = d._build_argv("fix it", "sys", None)

    assert "--mcp-config" not in argv


def test_default_argv_identical_with_and_without_a_token_store(bridge_url):
    """Merely wiring a token store must not change a normal dispatch — only
    passing a token does."""
    plain = Dispatcher(AgentRegistry(), on_wake=_noop_wake)
    wired = Dispatcher(AgentRegistry(), on_wake=_noop_wake,
                       token_store=BridgeTokenStore())

    assert plain._build_argv("p", "s", None) == wired._build_argv("p", "s", None)


def test_default_sandbox_domains_unchanged(monkeypatch, bridge_url):
    monkeypatch.setattr(global_config, "CC_SANDBOX", True)
    monkeypatch.setattr(global_config, "CC_SANDBOX_ALLOWED_DOMAINS", ["github.com"])

    settings = Dispatcher._sandbox_settings(capable=False)

    assert settings["sandbox"]["network"]["allowedDomains"] == ["github.com"]


# -- capable tier ------------------------------------------------------------

def test_capable_argv_carries_mcp_config(bridge_url):
    d = Dispatcher(AgentRegistry(), on_wake=_noop_wake, token_store=BridgeTokenStore())
    argv = d._build_argv("fix it", "sys", None, bridge_token="tok-123")

    assert "--mcp-config" in argv
    config = json.loads(argv[argv.index("--mcp-config") + 1])
    server = config["mcpServers"]["derpr"]

    assert server["type"] == "http"
    assert server["url"] == bridge_url
    # Token in the header, not the URL — keeps it out of process listings.
    assert server["headers"]["Authorization"] == "Bearer tok-123"
    assert "tok-123" not in server["url"]


def test_capable_sandbox_allows_the_bridge_host(monkeypatch, bridge_url):
    """Without this the sandbox blocks the MCP connection and the agent silently
    has no derpr tools."""
    monkeypatch.setattr(global_config, "CC_SANDBOX", True)
    monkeypatch.setattr(global_config, "CC_SANDBOX_ALLOWED_DOMAINS", ["github.com"])

    settings = Dispatcher._sandbox_settings(capable=True)

    assert settings["sandbox"]["network"]["allowedDomains"] == ["github.com", "10.0.0.70"]


def test_capable_sandbox_does_not_duplicate_an_existing_host(monkeypatch, bridge_url):
    monkeypatch.setattr(global_config, "CC_SANDBOX", True)
    monkeypatch.setattr(global_config, "CC_SANDBOX_ALLOWED_DOMAINS", ["10.0.0.70"])

    settings = Dispatcher._sandbox_settings(capable=True)

    assert settings["sandbox"]["network"]["allowedDomains"] == ["10.0.0.70"]


@pytest.mark.asyncio
async def test_capable_dispatch_without_a_bridge_is_refused(monkeypatch):
    """Fail loudly rather than silently degrading to a normal dispatch — the
    caller asked for tools the agent would not actually have."""
    d = Dispatcher(AgentRegistry(), on_wake=_noop_wake)
    monkeypatch.setattr("src.self_edit.clone_manager.create_worktree",
                        lambda *a, **k: (_ for _ in ()).throw(
                            AssertionError("must refuse before creating a worktree")))

    with pytest.raises(DispatcherError, match="MCP bridge is not wired"):
        await d.dispatch("DP-999", "desc", capable=True)


# -- token lifecycle ---------------------------------------------------------

def test_revoke_is_safe_for_agents_that_never_had_a_token():
    d = Dispatcher(AgentRegistry(), on_wake=_noop_wake, token_store=BridgeTokenStore())
    d._revoke_bridge_token("never-dispatched")  # must not raise


def test_revoke_is_a_noop_without_a_store():
    d = Dispatcher(AgentRegistry(), on_wake=_noop_wake)
    d._revoke_bridge_token("anything")  # must not raise


# -- helper ------------------------------------------------------------------

@pytest.mark.parametrize("url,expected", [
    ("http://10.0.0.70:5003/mcp", "10.0.0.70"),
    ("https://derpr.example.com/mcp", "derpr.example.com"),
    ("", ""),
    ("not-a-url", ""),
])
def test_host_of(url, expected):
    assert _host_of(url) == expected
