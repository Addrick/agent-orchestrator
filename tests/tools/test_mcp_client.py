# tests/tools/test_mcp_client.py
#
# DP-268 MCP client core: discovery/translation with restrictive defaults,
# live (de)registration, config persistence, degrade behavior, and the
# persona-revalidation trigger. The MCP transport is faked at the
# `_open_session` seam — no network, no real server.

import asyncio
import json
from contextlib import asynccontextmanager
from types import SimpleNamespace

import pytest

from mcp import types as mcp_types

from src.persona import Persona
from src.tools import definitions
from src.tools import mcp_client
from src.tools.definitions import ToolDefinitionRegistry, ALL_TOOL_DEFINITIONS
from src.tools.mcp_client import MCPClientManager
from src.tools.mcp_integration import MCPIntegration
from src.tools.tool_manager import ToolManager


def _mcp_tool(name="do_thing", description="Does the thing.", schema=None,
              annotations=None):
    return mcp_types.Tool(
        name=name,
        description=description,
        inputSchema=schema or {"type": "object", "properties": {}},
        annotations=annotations,
    )


def _text_result(text, is_error=False):
    return mcp_types.CallToolResult(
        content=[mcp_types.TextContent(type="text", text=text)],
        isError=is_error,
    )


class FakeSession:
    """Stands in for an initialized mcp.ClientSession."""

    def __init__(self, tools=None, call_result=None):
        self.tools = tools if tools is not None else [_mcp_tool()]
        self.call_result = call_result or _text_result("ok")
        self.calls = []
        self.message_handler = None  # captured by the fake transport

    async def list_tools(self):
        return SimpleNamespace(tools=self.tools)

    async def call_tool(self, name, arguments=None, read_timeout_seconds=None):
        self.calls.append((name, arguments))
        return self.call_result

    async def notify_tools_changed(self):
        """Deliver a tools/list_changed notification like a real server."""
        assert self.message_handler is not None
        await self.message_handler(mcp_types.ServerNotification(
            mcp_types.ToolListChangedNotification(
                method="notifications/tools/list_changed"
            )
        ))


@pytest.fixture
def fresh_registry(monkeypatch):
    """Isolate the live tool catalog so registrations don't leak across tests."""
    reg = ToolDefinitionRegistry(ALL_TOOL_DEFINITIONS)
    monkeypatch.setattr(definitions, "_REGISTRY", reg)
    return reg


@pytest.fixture
def fake_transport(monkeypatch):
    """Patch the `_open_session` seam; returns the dict of url -> FakeSession."""
    sessions = {}

    @asynccontextmanager
    async def fake_open_session(url, message_handler=None):
        session = sessions.get(url)
        if session is None:
            raise ConnectionError(f"no fake server at {url}")
        session.message_handler = message_handler
        yield session

    monkeypatch.setattr(mcp_client, "_open_session", fake_open_session)
    return sessions


def _make_manager(tmp_path, personas=None, enabled=True, reconnect_interval=60):
    manager = MCPClientManager(
        config_path=tmp_path / "mcp_servers.json",
        personas_provider=(lambda: personas) if personas is not None else None,
        enabled=enabled,
        reconnect_interval=reconnect_interval,
    )
    tool_manager = ToolManager()
    MCPIntegration(manager).register_tools(tool_manager)
    return manager, tool_manager


# --- add_server: discovery, translation, persistence ---------------------------

async def test_add_server_registers_tools_with_restrictive_defaults(
        tmp_path, fresh_registry, fake_transport):
    fake_transport["http://srv/mcp"] = FakeSession(tools=[_mcp_tool("do_thing")])
    manager, tool_manager = _make_manager(tmp_path)

    result = await manager.add_server("home", "http://srv/mcp")
    assert result["tools_registered"] == ["mcp__home__do_thing"]

    definition = definitions.get_tool_definition("mcp__home__do_thing")
    assert definition is not None
    assert definition["dynamic"] is True
    assert definition["is_write"] is True
    assert definition["service_binding"] == "mcp:home"
    assert definition["capabilities"] == {
        "produces_untrusted": True,
        "irreversible": True,
        "locality": "network",
        "sensitivity": "pii",
    }
    assert definitions.is_write_tool("mcp__home__do_thing") is True

    # Handler is live in the ToolManager and round-trips through the session.
    out = await tool_manager.execute_tool("mcp__home__do_thing")
    assert out == {"result": "ok"}

    # Config persisted.
    config = json.loads((tmp_path / "mcp_servers.json").read_text())
    assert config["servers"]["home"]["url"] == "http://srv/mcp"

    await manager.aclose()


async def test_add_server_rejects_bad_input(tmp_path, fresh_registry, fake_transport):
    manager, _ = _make_manager(tmp_path)
    with pytest.raises(ValueError, match="Invalid MCP server name"):
        await manager.add_server("Bad__Name", "http://srv/mcp")
    with pytest.raises(ValueError, match="must be http"):
        await manager.add_server("ok", "ftp://srv/mcp")


async def test_add_server_duplicate_rejected(tmp_path, fresh_registry, fake_transport):
    fake_transport["http://srv/mcp"] = FakeSession()
    manager, _ = _make_manager(tmp_path)
    await manager.add_server("home", "http://srv/mcp")
    with pytest.raises(ValueError, match="already configured"):
        await manager.add_server("home", "http://srv/mcp")
    await manager.aclose()


async def test_add_server_connect_failure_leaves_nothing_behind(
        tmp_path, fresh_registry, fake_transport):
    manager, _ = _make_manager(tmp_path)
    with pytest.raises(RuntimeError, match="connection failed"):
        await manager.add_server("dead", "http://dead/mcp")
    assert not (tmp_path / "mcp_servers.json").exists()
    assert await manager.list_servers() == []
    before = {t["function"]["name"] for t in ALL_TOOL_DEFINITIONS if t.get("type") == "function"}
    now = {
        t["function"]["name"]
        for t in definitions.get_all_tool_definitions() if t.get("type") == "function"
    }
    assert now == before


async def test_tool_overrides_relax_metadata(tmp_path, fresh_registry, fake_transport):
    """Operator per-tool config downgrades the restrictive defaults."""
    config_path = tmp_path / "mcp_servers.json"
    config_path.write_text(json.dumps({
        "servers": {
            "home": {
                "url": "http://srv/mcp",
                "enabled": True,
                "tool_overrides": {
                    "get_state": {
                        "is_write": False,
                        "capabilities": {"sensitivity": "internal", "irreversible": False},
                    },
                },
            },
        },
    }))
    fake_transport["http://srv/mcp"] = FakeSession(
        tools=[_mcp_tool("get_state"), _mcp_tool("set_state")]
    )
    manager, _ = _make_manager(tmp_path)
    await manager.start()

    relaxed = definitions.get_tool_definition("mcp__home__get_state")
    assert relaxed["is_write"] is False
    assert relaxed["capabilities"]["sensitivity"] == "internal"
    assert relaxed["capabilities"]["irreversible"] is False
    # Un-overridden keys keep the restrictive default.
    assert relaxed["capabilities"]["produces_untrusted"] is True

    untouched = definitions.get_tool_definition("mcp__home__set_state")
    assert untouched["is_write"] is True
    assert untouched["capabilities"]["sensitivity"] == "pii"
    await manager.aclose()


async def test_server_annotations_do_not_drive_policy(
        tmp_path, fresh_registry, fake_transport):
    """A server claiming its tool is read-only/harmless still gets the
    restrictive defaults — annotations are hints from an untrusted party."""
    friendly = _mcp_tool(
        "totally_safe",
        annotations=mcp_types.ToolAnnotations(readOnlyHint=True, destructiveHint=False),
    )
    fake_transport["http://srv/mcp"] = FakeSession(tools=[friendly])
    manager, _ = _make_manager(tmp_path)
    await manager.add_server("home", "http://srv/mcp")

    definition = definitions.get_tool_definition("mcp__home__totally_safe")
    assert definition["is_write"] is True
    assert definition["capabilities"]["produces_untrusted"] is True
    await manager.aclose()


async def test_long_description_truncated(tmp_path, fresh_registry, fake_transport):
    fake_transport["http://srv/mcp"] = FakeSession(
        tools=[_mcp_tool("wordy", description="x" * 5000)]
    )
    manager, _ = _make_manager(tmp_path)
    await manager.add_server("home", "http://srv/mcp")
    definition = definitions.get_tool_definition("mcp__home__wordy")
    assert len(definition["function"]["description"]) <= mcp_client._DESCRIPTION_MAX_CHARS + 1
    await manager.aclose()


async def test_invalid_tool_name_skipped(tmp_path, fresh_registry, fake_transport):
    fake_transport["http://srv/mcp"] = FakeSession(
        tools=[_mcp_tool("bad name!"), _mcp_tool("good_name")]
    )
    manager, _ = _make_manager(tmp_path)
    result = await manager.add_server("home", "http://srv/mcp")
    assert result["tools_registered"] == ["mcp__home__good_name"]
    await manager.aclose()


async def test_namespaced_name_over_provider_limit_skipped(
        tmp_path, fresh_registry, fake_transport):
    """A raw tool name can pass the 64-char check while the mcp__<server>__
    prefix pushes the NAMESPACED name past the provider limit — the tool must
    be skipped, not registered (one oversized def 400s every request)."""
    long_name = "t" * 60  # valid raw (≤64); mcp__home__ + 60 = 71 chars
    fake_transport["http://srv/mcp"] = FakeSession(
        tools=[_mcp_tool(long_name), _mcp_tool("fits")]
    )
    manager, _ = _make_manager(tmp_path)
    result = await manager.add_server("home", "http://srv/mcp")
    assert result["tools_registered"] == ["mcp__home__fits"]
    assert definitions.get_tool_definition(f"mcp__home__{long_name}") is None
    await manager.aclose()


async def test_add_server_save_failure_rolls_back_registration(
        tmp_path, fresh_registry, fake_transport):
    """A failed config persist must not leave a live-but-unpersisted server
    (tools callable yet invisible to list_mcp_servers, gone on restart)."""
    fake_transport["http://srv/mcp"] = FakeSession()
    manager, tool_manager = _make_manager(tmp_path)

    original_save = manager._save_config

    def broken_save(config):
        raise OSError("disk full")
    manager._save_config = broken_save  # type: ignore[method-assign]

    with pytest.raises(OSError, match="disk full"):
        await manager.add_server("home", "http://srv/mcp")

    assert definitions.get_tool_definition("mcp__home__do_thing") is None
    assert "home" not in manager._connections
    out = await tool_manager.execute_tool("mcp__home__do_thing")
    assert "not found" in out["error"]

    # The add is cleanly retryable once persistence works again.
    manager._save_config = original_save  # type: ignore[method-assign]
    result = await manager.add_server("home", "http://srv/mcp")
    assert result["tools_registered"] == ["mcp__home__do_thing"]
    await manager.aclose()


async def test_add_server_rejects_malformed_servers_shape(
        tmp_path, fresh_registry, fake_transport):
    """{"servers": []} passes json.load but must fail validation up front,
    not TypeError after the server is already connected and registered."""
    (tmp_path / "mcp_servers.json").write_text(json.dumps({"servers": []}))
    fake_transport["http://srv/mcp"] = FakeSession()
    manager, _ = _make_manager(tmp_path)
    with pytest.raises(RuntimeError, match="unreadable"):
        await manager.add_server("home", "http://srv/mcp")
    assert definitions.get_tool_definition("mcp__home__do_thing") is None
    assert "home" not in manager._connections


# --- call path ------------------------------------------------------------------

async def test_call_tool_prefers_structured_content(
        tmp_path, fresh_registry, fake_transport):
    result = mcp_types.CallToolResult(
        content=[mcp_types.TextContent(type="text", text="raw")],
        structuredContent={"temp": 21},
        isError=False,
    )
    fake_transport["http://srv/mcp"] = FakeSession(call_result=result)
    manager, _ = _make_manager(tmp_path)
    await manager.add_server("home", "http://srv/mcp")
    assert await manager.call_tool("home", "do_thing", {}) == {"temp": 21}
    await manager.aclose()


async def test_call_tool_error_result_raises(tmp_path, fresh_registry, fake_transport):
    fake_transport["http://srv/mcp"] = FakeSession(
        call_result=_text_result("boom", is_error=True)
    )
    manager, tool_manager = _make_manager(tmp_path)
    await manager.add_server("home", "http://srv/mcp")
    # Through the ToolManager the raise degrades to an {"error": ...} dict.
    out = await tool_manager.execute_tool("mcp__home__do_thing")
    assert "boom" in out["error"]
    await manager.aclose()


async def test_call_tool_on_disconnected_server_errors(
        tmp_path, fresh_registry, fake_transport):
    manager, _ = _make_manager(tmp_path)
    with pytest.raises(RuntimeError, match="not connected"):
        await manager.call_tool("ghost", "do_thing", {})


async def test_call_passes_arguments_through(tmp_path, fresh_registry, fake_transport):
    session = FakeSession(tools=[_mcp_tool("with_args")])
    fake_transport["http://srv/mcp"] = session
    manager, tool_manager = _make_manager(tmp_path)
    await manager.add_server("home", "http://srv/mcp")
    await tool_manager.execute_tool("mcp__home__with_args", state="on", level=3)
    assert session.calls == [("with_args", {"state": "on", "level": 3})]
    await manager.aclose()


# --- remove_server / list_servers ------------------------------------------------

async def test_save_config_is_atomic_no_tmp_left_behind(
        tmp_path, fresh_registry, fake_transport):
    fake_transport["http://srv/mcp"] = FakeSession()
    manager, _ = _make_manager(tmp_path)
    await manager.add_server("home", "http://srv/mcp")
    assert (tmp_path / "mcp_servers.json").exists()
    assert not (tmp_path / "mcp_servers.json.tmp").exists()
    await manager.aclose()


async def test_session_task_ends_cancelled_on_external_cancel(
        tmp_path, fresh_registry, fake_transport):
    """_run must re-raise CancelledError so the task ends cancelled instead
    of 'completed normally' (suppressing a requested cancellation)."""
    fake_transport["http://srv/mcp"] = FakeSession()
    manager, _ = _make_manager(tmp_path)
    await manager.add_server("home", "http://srv/mcp")
    task = manager._connections["home"]._task
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert task.cancelled()


async def test_remove_server_unregisters_and_persists(
        tmp_path, fresh_registry, fake_transport):
    fake_transport["http://srv/mcp"] = FakeSession()
    manager, tool_manager = _make_manager(tmp_path)
    await manager.add_server("home", "http://srv/mcp")

    result = await manager.remove_server("home")
    assert result["tools_unregistered"] == ["mcp__home__do_thing"]
    assert definitions.get_tool_definition("mcp__home__do_thing") is None
    out = await tool_manager.execute_tool("mcp__home__do_thing")
    assert "not found" in out["error"]
    config = json.loads((tmp_path / "mcp_servers.json").read_text())
    assert config["servers"] == {}


async def test_remove_unknown_server_raises(tmp_path, fresh_registry, fake_transport):
    manager, _ = _make_manager(tmp_path)
    with pytest.raises(ValueError, match="not configured"):
        await manager.remove_server("ghost")


async def test_list_servers_reports_status(tmp_path, fresh_registry, fake_transport):
    fake_transport["http://srv/mcp"] = FakeSession()
    manager, _ = _make_manager(tmp_path)
    await manager.add_server("home", "http://srv/mcp")
    listing = await manager.list_servers()
    assert listing == [{
        "name": "home",
        "url": "http://srv/mcp",
        "enabled": True,
        "connected": True,
        "tools": ["mcp__home__do_thing"],
    }]
    await manager.aclose()


# --- feature flag / startup -------------------------------------------------------

async def test_disabled_manager_short_circuits(tmp_path, fresh_registry, fake_transport):
    manager, tool_manager = _make_manager(tmp_path, enabled=False)
    await manager.start()  # no-op, no raise
    for call in (manager.add_server("h", "http://srv/mcp"),
                 manager.remove_server("h"),
                 manager.list_servers()):
        with pytest.raises(RuntimeError, match="disabled"):
            await call
    # The management tools still surface the error through the ToolManager.
    out = await tool_manager.execute_tool("list_mcp_servers")
    assert "disabled" in out["error"]


async def test_startup_connects_enabled_servers_only(
        tmp_path, fresh_registry, fake_transport):
    (tmp_path / "mcp_servers.json").write_text(json.dumps({
        "servers": {
            "on": {"url": "http://on/mcp", "enabled": True},
            "off": {"url": "http://off/mcp", "enabled": False},
        },
    }))
    fake_transport["http://on/mcp"] = FakeSession(tools=[_mcp_tool("a")])
    fake_transport["http://off/mcp"] = FakeSession(tools=[_mcp_tool("b")])
    manager, _ = _make_manager(tmp_path)
    await manager.start()
    assert definitions.get_tool_definition("mcp__on__a") is not None
    assert definitions.get_tool_definition("mcp__off__b") is None
    await manager.aclose()


async def test_startup_dead_server_does_not_break_others(
        tmp_path, fresh_registry, fake_transport):
    (tmp_path / "mcp_servers.json").write_text(json.dumps({
        "servers": {
            "dead": {"url": "http://dead/mcp", "enabled": True},
            "alive": {"url": "http://alive/mcp", "enabled": True},
        },
    }))
    fake_transport["http://alive/mcp"] = FakeSession(tools=[_mcp_tool("a")])
    manager, _ = _make_manager(tmp_path)
    await manager.start()  # no raise
    assert definitions.get_tool_definition("mcp__alive__a") is not None
    listing = await manager.list_servers()
    by_name = {s["name"]: s for s in listing}
    assert by_name["dead"]["connected"] is False
    assert by_name["alive"]["connected"] is True
    await manager.aclose()


# --- persona revalidation on live (de)registration -------------------------------

def _persona(name, allow):
    return Persona(
        persona_name=name, model_name="gemini-2.5-flash", prompt="p",
        tool_policy={"default": "deny", "allow": allow},
        service_bindings=["mcp", "mcp:home"],
    )


async def test_registration_revalidates_personas_no_wildcard_cascade(
        tmp_path, fresh_registry, fake_transport):
    """Installing a server re-validates every persona; one not referencing
    the new tools stays clean, and one explicitly combining an MCP tool with
    a foreign-domain read trips quarantine — live, without a restart."""
    clean = _persona("clean", allow=["get_ticket_details"])
    risky = _persona("risky", allow=["web_search", "mcp__home__do_thing"])
    personas = {"clean": clean, "risky": risky}

    fake_transport["http://srv/mcp"] = FakeSession()
    manager, _ = _make_manager(tmp_path, personas=personas)

    # Before the server exists, risky's MCP tool name resolves to nothing.
    assert not risky.is_security_blocked()

    await manager.add_server("home", "http://srv/mcp")
    assert not clean.is_security_blocked()
    # untrusted open-domain read (web_search) + untrusted/pii network write on
    # the new mcp:home domain → Rules 2/3 trip now that the tool exists.
    assert risky.is_security_blocked()

    await manager.remove_server("home")
    assert not risky.is_security_blocked()


# --- hot reload (phase 3): reconnect + tools/list_changed re-discovery -----------

async def _kill_session(manager, name):
    """Simulate a server death: the session task unwinds, session goes None."""
    conn = manager._connections[name]
    session = conn.session
    conn._stop.set()
    await conn._task
    conn._stop = asyncio.Event()  # irrelevant post-mortem, keeps state tidy
    assert conn.session is None
    return session


async def test_maintenance_reconnects_dead_server(
        tmp_path, fresh_registry, fake_transport):
    fake_transport["http://srv/mcp"] = FakeSession(tools=[_mcp_tool("a")])
    manager, tool_manager = _make_manager(tmp_path)
    await manager.add_server("home", "http://srv/mcp")
    await _kill_session(manager, "home")

    # While dead: tools stay registered but degrade to per-call errors.
    assert definitions.get_tool_definition("mcp__home__a") is not None
    out = await tool_manager.execute_tool("mcp__home__a")
    assert "not connected" in out["error"]

    await manager._maintain()
    listing = await manager.list_servers()
    assert listing[0]["connected"] is True
    assert (await tool_manager.execute_tool("mcp__home__a")) == {"result": "ok"}
    await manager.aclose()


async def test_failed_reconnect_keeps_existing_registrations(
        tmp_path, fresh_registry, fake_transport):
    """A server that is merely down must not lose its registered tools on a
    failed reconnect attempt — persona policies stay stable across outages."""
    fake_transport["http://srv/mcp"] = FakeSession(tools=[_mcp_tool("a")])
    manager, _ = _make_manager(tmp_path)
    await manager.add_server("home", "http://srv/mcp")
    await _kill_session(manager, "home")
    del fake_transport["http://srv/mcp"]  # server unreachable now

    await manager._maintain()  # reconnect fails, logged, no raise
    assert definitions.get_tool_definition("mcp__home__a") is not None
    listing = await manager.list_servers()
    assert listing[0]["connected"] is False
    assert listing[0]["tools"] == ["mcp__home__a"]

    # Server comes back → next pass restores it.
    fake_transport["http://srv/mcp"] = FakeSession(tools=[_mcp_tool("a")])
    await manager._maintain()
    assert (await manager.list_servers())[0]["connected"] is True
    await manager.aclose()


async def test_startup_failed_server_retried_by_maintenance(
        tmp_path, fresh_registry, fake_transport):
    (tmp_path / "mcp_servers.json").write_text(json.dumps({
        "servers": {"late": {"url": "http://late/mcp", "enabled": True}},
    }))
    manager, _ = _make_manager(tmp_path)
    await manager.start()  # connect fails, logged
    assert definitions.get_tool_definition("mcp__late__do_thing") is None

    fake_transport["http://late/mcp"] = FakeSession()
    await manager._maintain()
    assert definitions.get_tool_definition("mcp__late__do_thing") is not None
    assert (await manager.list_servers())[0]["connected"] is True
    await manager.aclose()


async def test_tools_list_changed_triggers_rediscovery(
        tmp_path, fresh_registry, fake_transport):
    session = FakeSession(tools=[_mcp_tool("old_tool")])
    fake_transport["http://srv/mcp"] = session
    manager, _ = _make_manager(tmp_path)
    await manager.add_server("home", "http://srv/mcp")
    assert definitions.get_tool_definition("mcp__home__old_tool") is not None

    # Server swaps its toolset and notifies, like a real MCP server would.
    session.tools = [_mcp_tool("new_tool")]
    await session.notify_tools_changed()
    assert "home" in manager._tools_changed
    assert manager._wake.is_set()

    await manager._maintain()
    assert definitions.get_tool_definition("mcp__home__old_tool") is None
    assert definitions.get_tool_definition("mcp__home__new_tool") is not None
    assert (await manager.list_servers())[0]["tools"] == ["mcp__home__new_tool"]
    await manager.aclose()


async def test_rediscovery_failure_keeps_old_toolset_and_retries(
        tmp_path, fresh_registry, fake_transport):
    session = FakeSession(tools=[_mcp_tool("a")])
    fake_transport["http://srv/mcp"] = session
    manager, _ = _make_manager(tmp_path)
    await manager.add_server("home", "http://srv/mcp")

    original_list_tools = session.list_tools

    async def broken_list_tools():
        raise ConnectionError("mid-flight death")
    session.list_tools = broken_list_tools
    session.tools = [_mcp_tool("b")]
    await session.notify_tools_changed()

    manager._wake.clear()  # the loop clears the wake before each pass
    await manager._maintain()  # logged, no raise
    assert definitions.get_tool_definition("mcp__home__a") is not None
    assert (await manager.list_servers())[0]["tools"] == ["mcp__home__a"]
    # The change signal survives the transient failure (no wake — the next
    # periodic tick retries instead of spinning).
    assert "home" in manager._tools_changed
    assert not manager._wake.is_set()

    # Listing works again → the next pass completes the swap.
    session.list_tools = original_list_tools
    await manager._maintain()
    assert definitions.get_tool_definition("mcp__home__a") is None
    assert definitions.get_tool_definition("mcp__home__b") is not None
    await manager.aclose()


async def test_registration_failure_flags_retry_not_stranded(
        tmp_path, fresh_registry, fake_transport):
    """If the re-registration half of a swap fails, the server must be
    re-flagged for the next tick — not stranded connected-but-toolless."""
    session = FakeSession(tools=[_mcp_tool("a")])
    fake_transport["http://srv/mcp"] = session
    manager, _ = _make_manager(tmp_path)
    await manager.add_server("home", "http://srv/mcp")

    session.tools = [_mcp_tool("b")]
    await session.notify_tools_changed()
    original_register = manager._register_definitions
    calls = {"n": 0}

    def flaky_register(name, defs):
        calls["n"] += 1
        if calls["n"] == 1:
            raise ValueError("bad definition")
        return original_register(name, defs)
    manager._register_definitions = flaky_register  # type: ignore[method-assign]

    await manager._maintain()  # swap fails; zero tools this pass
    assert (await manager.list_servers())[0]["tools"] == []
    assert "home" in manager._tools_changed

    await manager._maintain()  # retried and recovered
    assert definitions.get_tool_definition("mcp__home__b") is not None
    assert (await manager.list_servers())[0]["tools"] == ["mcp__home__b"]
    await manager.aclose()


async def test_reconnect_identical_toolset_skips_revalidation_churn(
        tmp_path, fresh_registry, fake_transport):
    """A flapping server whose toolset is unchanged must not unregister/
    re-register (window where tools vanish) or re-run persona validation."""
    revalidations = []

    def personas_provider():
        revalidations.append(1)
        return {}

    fake_transport["http://srv/mcp"] = FakeSession(tools=[_mcp_tool("a")])
    manager = MCPClientManager(
        config_path=tmp_path / "mcp_servers.json",
        personas_provider=personas_provider,
        enabled=True,
        reconnect_interval=60,
    )
    tool_manager = ToolManager()
    MCPIntegration(manager).register_tools(tool_manager)
    await manager.add_server("home", "http://srv/mcp")
    baseline = len(revalidations)

    await _kill_session(manager, "home")
    await manager._maintain()  # reconnects; same toolset
    assert (await manager.list_servers())[0]["connected"] is True
    assert (await manager.list_servers())[0]["tools"] == ["mcp__home__a"]
    assert len(revalidations) == baseline
    await manager.aclose()


async def test_stop_after_external_cancel_does_not_poison_aclose(
        tmp_path, fresh_registry, fake_transport):
    """A session task that ended cancelled must not make stop()/aclose()
    raise CancelledError (the old code caught this; a regression aborts
    shutdown mid-loop)."""
    fake_transport["http://a/mcp"] = FakeSession(tools=[_mcp_tool("a")])
    fake_transport["http://b/mcp"] = FakeSession(tools=[_mcp_tool("b")])
    manager, _ = _make_manager(tmp_path)
    await manager.add_server("srv-a", "http://a/mcp")
    await manager.add_server("srv-b", "http://b/mcp")

    task_a = manager._connections["srv-a"]._task
    task_a.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task_a

    await manager.aclose()  # must not raise, must stop srv-b too
    assert manager._connections == {}
    assert manager._connections.get("srv-b") is None


async def test_cancelled_connect_reaps_session_task(
        tmp_path, fresh_registry, fake_transport, monkeypatch):
    """Cancelling add_server mid-connect must not orphan the freshly spawned
    session task (it is not in _connections yet — nobody else stops it)."""
    started = asyncio.Event()

    @asynccontextmanager
    async def hanging_open_session(url, message_handler=None):
        started.set()
        await asyncio.Event().wait()  # never connects
        yield None  # pragma: no cover

    monkeypatch.setattr(mcp_client, "_open_session", hanging_open_session)
    manager, _ = _make_manager(tmp_path)

    add_task = asyncio.create_task(manager.add_server("home", "http://srv/mcp"))
    await started.wait()
    add_task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await add_task

    orphans = [
        t for t in asyncio.all_tasks()
        if t.get_name().startswith("mcp-session-") and not t.done()
    ]
    assert orphans == []


async def test_rediscovery_revalidates_personas(
        tmp_path, fresh_registry, fake_transport):
    """A toolset swap must re-run composition for every persona — a tool
    disappearing (or its metadata changing) can flip a quarantine verdict."""
    risky = _persona("risky", allow=["web_search", "mcp__home__do_thing"])
    session = FakeSession(tools=[_mcp_tool("do_thing")])
    fake_transport["http://srv/mcp"] = session
    manager, _ = _make_manager(tmp_path, personas={"risky": risky})
    await manager.add_server("home", "http://srv/mcp")
    assert risky.is_security_blocked()

    # The offending tool vanishes server-side → re-discovery clears the block.
    session.tools = [_mcp_tool("harmless_other")]
    await session.notify_tools_changed()
    await manager._maintain()
    assert not risky.is_security_blocked()
    await manager.aclose()


async def test_maintenance_loop_lifecycle(tmp_path, fresh_registry, fake_transport):
    """start() spawns the loop (when enabled, interval > 0); aclose() stops it.
    Interval <= 0 or disabled → no loop."""
    manager, _ = _make_manager(tmp_path)
    await manager.start()
    assert manager._maintenance_task is not None
    task = manager._maintenance_task
    await manager.aclose()
    assert task.done()
    assert manager._maintenance_task is None

    no_loop, _ = _make_manager(tmp_path, reconnect_interval=0)
    await no_loop.start()
    assert no_loop._maintenance_task is None

    disabled, _ = _make_manager(tmp_path, enabled=False)
    await disabled.start()
    assert disabled._maintenance_task is None


async def test_maintenance_loop_wakes_on_notification(
        tmp_path, fresh_registry, fake_transport):
    """End-to-end through the real background loop: a notification (not the
    periodic tick — interval is far too long) drives the re-discovery."""
    session = FakeSession(tools=[_mcp_tool("old_tool")])
    fake_transport["http://srv/mcp"] = session
    manager, _ = _make_manager(tmp_path, reconnect_interval=3600)
    await manager.start()
    await manager.add_server("home", "http://srv/mcp")

    session.tools = [_mcp_tool("new_tool")]
    await session.notify_tools_changed()
    for _ in range(200):  # let the loop task run; bounded wait, no sleep math
        if definitions.get_tool_definition("mcp__home__new_tool") is not None:
            break
        await asyncio.sleep(0.01)
    assert definitions.get_tool_definition("mcp__home__new_tool") is not None
    assert definitions.get_tool_definition("mcp__home__old_tool") is None
    await manager.aclose()


async def test_wildcard_persona_unaffected_by_server_install(
        tmp_path, fresh_registry, fake_transport):
    """The quarantine-cascade scenario: a wildcard persona's verdict must be
    identical before and after an MCP server registers its tools."""
    wildcard = Persona(
        persona_name="wild", model_name="gemini-2.5-flash", prompt="p",
        tool_policy={"default": "allow", "allow": ["*"]},
    )
    from src.tools.composition import revalidate_persona_security
    before = revalidate_persona_security(wildcard)
    reasons_before = list(wildcard.get_security_block_reasons() or [])

    fake_transport["http://srv/mcp"] = FakeSession()
    manager, _ = _make_manager(tmp_path, personas={"wild": wildcard})
    await manager.add_server("home", "http://srv/mcp")

    assert wildcard.is_security_blocked() == before
    assert list(wildcard.get_security_block_reasons() or []) == reasons_before
    await manager.aclose()
