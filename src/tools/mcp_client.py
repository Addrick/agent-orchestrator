"""MCP client core (DP-268): consume external MCP tool servers.

``MCPClientManager`` owns server sessions and their lifecycle (voice
precedent: ``ServiceIntegration`` is registration-only, so ``main.py`` owns
the manager and ``MCPIntegration`` only registers the management tools).
Discovered tools are translated into derpr tool definitions and registered
into the live catalog (``ToolDefinitionRegistry``) plus the ``ToolManager``
as closures over ``session.call_tool``, inheriting the full security model
(parking, taint, composition rules, per-persona policy) with no MCP-specific
paths downstream.

Security model:
- Server-provided annotations (``readOnlyHint``/``destructiveHint``/...) are
  hints from an untrusted party: logged for the operator, NEVER driving
  policy. Operator config drives everything.
- Every discovered tool defaults to the most restrictive metadata
  (``is_write: True``; untrusted, irreversible, network, pii) unless the
  operator downgrades it via per-tool ``tool_overrides`` in the config file.
- ``service_binding: "mcp:<server>"`` makes each server its own egress
  domain, so composition Rules 2/3 treat one server's read+write as a closed
  loop and re-arm exactly when combined with foreign domains.
- Definitions carry ``dynamic: True`` — the marker that excludes them from
  ``['*']`` policy expansion (a new server must never silently widen or
  quarantine-cascade wildcard personas).
- Tool descriptions are server-authored text that enters the system prompt
  (prompt-injection surface) → length-capped at translation.

Dead servers degrade to per-call errors (phase 3 reconnect deferred): the
session task exits, ``call_tool`` raises, ``ToolManager`` wraps it as
``{"error": ...}``.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from contextlib import asynccontextmanager
from datetime import timedelta
from pathlib import Path
from typing import (
    TYPE_CHECKING, Any, AsyncIterator, Callable, Coroutine, Dict, List, Optional,
)

from mcp import ClientSession
from mcp import types as mcp_types
from mcp.client.streamable_http import streamablehttp_client

from config.global_config import (
    MCP_CALL_TIMEOUT,
    MCP_CONNECT_TIMEOUT,
    MCP_ENABLED,
    MCP_SERVERS_FILE,
)
from src.tools.composition import revalidate_persona_security
from src.tools.definitions import (
    register_tool_definition,
    unregister_tool_definition,
)

if TYPE_CHECKING:
    from src.persona import Persona
    from src.tools.tool_manager import ToolManager

logger = logging.getLogger(__name__)

# Server names become the mcp__<server>__ tool prefix and the mcp:<server>
# binding; the double-underscore separator must stay unambiguous.
_SERVER_NAME_RE = re.compile(r"^[a-z0-9][a-z0-9-]{0,31}$")
# Provider function-calling APIs restrict tool names to [A-Za-z0-9_-], 64 chars.
_TOOL_NAME_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
_DESCRIPTION_MAX_CHARS = 1024

# Most-restrictive defaults for a discovered tool. Operator ``tool_overrides``
# (per tool, in the config file) may relax individual keys; server annotations
# never do.
_DEFAULT_TOOL_META: Dict[str, Any] = {
    "is_write": True,
    "capabilities": {
        "produces_untrusted": True,
        "irreversible": True,
        "locality": "network",
        "sensitivity": "pii",
    },
}


@asynccontextmanager
async def _open_session(url: str) -> AsyncIterator[ClientSession]:
    """Open and initialize a streamable-HTTP MCP session. Test seam."""
    async with streamablehttp_client(url, timeout=MCP_CONNECT_TIMEOUT) as (
        read_stream, write_stream, _get_session_id,
    ):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()
            yield session


class _ServerConnection:
    """One live server session, owned by a dedicated task.

    The MCP transport contexts are anyio-scoped and must be entered and
    exited by the same task, so the connection task holds them open and only
    unwinds when ``stop()`` is requested (or the transport dies — after which
    ``session`` is None and calls fail fast).
    """

    def __init__(self, name: str, url: str) -> None:
        self.name = name
        self.url = url
        self.session: Optional[ClientSession] = None
        self.error: Optional[BaseException] = None
        self._ready = asyncio.Event()
        self._stop = asyncio.Event()
        self._task: Optional[asyncio.Task[None]] = None

    async def start(self, timeout: float) -> None:
        """Spawn the session task and wait until connected (or fail)."""
        self._task = asyncio.create_task(self._run(), name=f"mcp-session-{self.name}")
        try:
            await asyncio.wait_for(self._ready.wait(), timeout)
        except asyncio.TimeoutError:
            await self.stop()
            raise RuntimeError(
                f"MCP server '{self.name}' did not connect within {timeout}s"
            )
        if self.session is None:
            raise RuntimeError(
                f"MCP server '{self.name}' connection failed: {self.error}"
            )

    async def _run(self) -> None:
        try:
            async with _open_session(self.url) as session:
                self.session = session
                self._ready.set()
                await self._stop.wait()
        except BaseException as e:  # noqa: BLE001 — anyio group errors included
            self.error = e
            if self._ready.is_set():
                logger.error(f"MCP server '{self.name}' session died: {e}")
        finally:
            self.session = None
            self._ready.set()

    async def stop(self) -> None:
        self._stop.set()
        if self._task is not None:
            try:
                await asyncio.wait_for(self._task, timeout=10)
            except (asyncio.TimeoutError, asyncio.CancelledError):
                self._task.cancel()
            except Exception:  # already-logged session errors
                pass


class MCPClientManager:
    """Owns MCP server config, sessions, discovery, and live (de)registration.

    ``attach_tool_manager`` is called by ``MCPIntegration.register_tools``;
    after that, ``start()`` connects every enabled configured server. Any
    live (de)registration triggers ``revalidate_persona_security`` across all
    personas (via ``personas_provider``) so an installed server can never
    leave a stale quarantine verdict standing.
    """

    def __init__(
        self,
        config_path: Path = MCP_SERVERS_FILE,
        personas_provider: Optional[Callable[[], Dict[str, "Persona"]]] = None,
        enabled: bool = MCP_ENABLED,
    ) -> None:
        self._config_path = Path(config_path)
        self._personas_provider = personas_provider
        self._enabled = enabled
        self._tool_manager: Optional["ToolManager"] = None
        self._connections: Dict[str, _ServerConnection] = {}
        # server name -> namespaced tool names registered for it
        self._registered_tools: Dict[str, List[str]] = {}
        self._lock = asyncio.Lock()

    # ------------------------------------------------------------------ wiring

    def attach_tool_manager(self, tool_manager: "ToolManager") -> None:
        self._tool_manager = tool_manager

    async def start(self) -> None:
        """Connect all enabled configured servers. Per-server failures are
        logged and skipped — a dead server must not break startup."""
        if not self._enabled:
            logger.info("MCP client disabled (MCP_ENABLED=false); no servers connected.")
            return
        config = self._load_config(strict=False)
        for name, server_cfg in config.get("servers", {}).items():
            if not server_cfg.get("enabled", True):
                logger.info(f"MCP server '{name}' disabled in config; skipping.")
                continue
            try:
                async with self._lock:
                    await self._connect_and_register(name, server_cfg)
            except Exception as e:
                logger.error(f"MCP server '{name}' startup connect failed: {e}")

    async def aclose(self) -> None:
        for conn in list(self._connections.values()):
            await conn.stop()
        self._connections.clear()

    # ----------------------------------------------------------- tool handlers

    def _require_enabled(self) -> None:
        if not self._enabled:
            raise RuntimeError(
                "MCP client is disabled. Set MCP_ENABLED=true to manage MCP servers."
            )

    async def add_server(self, name: str, url: str) -> Dict[str, Any]:
        """Connect + discover + register a new server live, then persist it.

        Config is only persisted after a successful connect+discovery so a
        failed add leaves nothing half-installed.
        """
        self._require_enabled()
        if not _SERVER_NAME_RE.match(name or ""):
            raise ValueError(
                f"Invalid MCP server name '{name}': need lowercase letters/"
                "digits/hyphens, starting alphanumeric, max 32 chars."
            )
        if not str(url).startswith(("http://", "https://")):
            raise ValueError(f"Invalid MCP server url '{url}': must be http(s).")

        async with self._lock:
            config = self._load_config()
            if name in config.get("servers", {}):
                raise ValueError(f"MCP server '{name}' is already configured.")
            server_cfg = {"url": url, "enabled": True, "tool_overrides": {}}
            registered = await self._connect_and_register(name, server_cfg)
            config.setdefault("servers", {})[name] = server_cfg
            self._save_config(config)

        logger.info(
            f"MCP server '{name}' added ({url}); registered tools: {registered}"
        )
        return {
            "server": name,
            "url": url,
            "tools_registered": registered,
            "note": (
                "Tools carry restrictive default security metadata (write/"
                "untrusted/irreversible/pii). Personas must list them "
                f"explicitly (no wildcard) and bind 'mcp:{name}'. Operator can "
                "relax per-tool metadata via tool_overrides in "
                f"{self._config_path.name}."
            ),
        }

    async def remove_server(self, name: str) -> Dict[str, Any]:
        """Disconnect, unregister the server's tools, delete it from config."""
        self._require_enabled()
        async with self._lock:
            config = self._load_config()
            known_in_config = name in config.get("servers", {})
            if not known_in_config and name not in self._connections:
                raise ValueError(f"MCP server '{name}' is not configured.")

            conn = self._connections.pop(name, None)
            if conn is not None:
                await conn.stop()
            removed = self._unregister_server_tools(name)
            if known_in_config:
                del config["servers"][name]
                self._save_config(config)
            self._revalidate_personas()

        logger.info(f"MCP server '{name}' removed; unregistered tools: {removed}")
        return {"server": name, "tools_unregistered": removed}

    async def list_servers(self) -> List[Dict[str, Any]]:
        self._require_enabled()
        config = self._load_config(strict=False)
        result = []
        for name, server_cfg in config.get("servers", {}).items():
            conn = self._connections.get(name)
            result.append({
                "name": name,
                "url": server_cfg.get("url"),
                "enabled": server_cfg.get("enabled", True),
                "connected": bool(conn and conn.session is not None),
                "tools": list(self._registered_tools.get(name, [])),
            })
        return result

    async def call_tool(
        self, server: str, tool_name: str, arguments: Dict[str, Any]
    ) -> Any:
        """Invoke a discovered tool on its server session."""
        conn = self._connections.get(server)
        if conn is None or conn.session is None:
            raise RuntimeError(
                f"MCP server '{server}' is not connected. "
                "Restart or re-add the server to reconnect."
            )
        result = await asyncio.wait_for(
            conn.session.call_tool(
                tool_name,
                arguments or {},
                read_timeout_seconds=timedelta(seconds=MCP_CALL_TIMEOUT),
            ),
            timeout=MCP_CALL_TIMEOUT + 5,
        )
        texts = [
            c.text for c in result.content if isinstance(c, mcp_types.TextContent)
        ]
        if result.isError:
            raise RuntimeError(
                "; ".join(texts) or f"MCP tool '{tool_name}' reported an error."
            )
        if result.structuredContent is not None:
            return result.structuredContent
        if texts:
            return "\n".join(texts)
        return [c.model_dump(mode="json") for c in result.content]

    # -------------------------------------------------------------- internals

    async def _connect_and_register(
        self, name: str, server_cfg: Dict[str, Any]
    ) -> List[str]:
        """Connect one server, discover its tools, register defs + handlers.

        Caller holds ``self._lock``. On any failure the connection is torn
        down and nothing stays registered.
        """
        if self._tool_manager is None:
            raise RuntimeError("MCPClientManager has no ToolManager attached yet.")
        if name in self._connections:
            raise ValueError(f"MCP server '{name}' is already connected.")

        conn = _ServerConnection(name, str(server_cfg["url"]))
        await conn.start(MCP_CONNECT_TIMEOUT)
        registered: List[str] = []
        try:
            assert conn.session is not None
            listing = await asyncio.wait_for(
                conn.session.list_tools(), timeout=MCP_CONNECT_TIMEOUT
            )
            overrides = server_cfg.get("tool_overrides") or {}
            for tool in listing.tools:
                if not _TOOL_NAME_RE.match(tool.name or ""):
                    logger.warning(
                        f"MCP server '{name}' tool '{tool.name}' has an invalid "
                        "name; skipped."
                    )
                    continue
                definition = self._translate_tool(name, tool, overrides.get(tool.name))
                namespaced = definition["function"]["name"]
                register_tool_definition(definition)
                self._tool_manager.register(
                    namespaced, self._make_handler(name, tool.name)
                )
                registered.append(namespaced)
        except BaseException:
            # Roll back whatever landed before the failure, then disconnect.
            for tool_name in registered:
                unregister_tool_definition(tool_name)
                self._tool_manager.unregister(tool_name)
            await conn.stop()
            raise

        self._connections[name] = conn
        self._registered_tools[name] = registered
        self._revalidate_personas()
        logger.info(
            f"MCP server '{name}' connected; {len(registered)} tool(s) registered."
        )
        return registered

    def _translate_tool(
        self, server: str, tool: Any, override: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """MCP tool → derpr definition with restrictive defaults + overrides.

        Server annotations are logged only — an untrusted party's opinion of
        its own tool never drives policy.
        """
        if tool.annotations is not None:
            logger.info(
                f"MCP server '{server}' tool '{tool.name}' annotations "
                f"(hints only, not policy): {tool.annotations.model_dump(exclude_none=True)}"
            )
        description = tool.description or f"MCP tool '{tool.name}' on server '{server}'."
        if len(description) > _DESCRIPTION_MAX_CHARS:
            logger.warning(
                f"MCP server '{server}' tool '{tool.name}' description truncated "
                f"({len(description)} > {_DESCRIPTION_MAX_CHARS} chars)."
            )
            description = description[:_DESCRIPTION_MAX_CHARS] + "…"

        override = override or {}
        capabilities = dict(_DEFAULT_TOOL_META["capabilities"])
        capabilities.update(override.get("capabilities") or {})
        is_write = override.get("is_write", _DEFAULT_TOOL_META["is_write"])

        return {
            "type": "function",
            "dynamic": True,
            "is_write": bool(is_write),
            "service_binding": f"mcp:{server}",
            "capabilities": capabilities,
            "function": {
                "name": f"mcp__{server}__{tool.name}",
                "description": description,
                "parameters": tool.inputSchema
                or {"type": "object", "properties": {}, "required": []},
            },
        }

    def _make_handler(
        self, server: str, tool_name: str
    ) -> Callable[..., Coroutine[Any, Any, Any]]:
        async def handler(**kwargs: Any) -> Any:
            return await self.call_tool(server, tool_name, kwargs)
        return handler

    def _unregister_server_tools(self, name: str) -> List[str]:
        removed = self._registered_tools.pop(name, [])
        for tool_name in removed:
            unregister_tool_definition(tool_name)
            if self._tool_manager is not None:
                self._tool_manager.unregister(tool_name)
        return removed

    def _revalidate_personas(self) -> None:
        """Re-run composition validation for every persona after any live
        (de)registration — the toolset just changed under their policies."""
        if self._personas_provider is None:
            return
        for persona in self._personas_provider().values():
            revalidate_persona_security(persona)

    # ----------------------------------------------------------------- config

    def _load_config(self, strict: bool = True) -> Dict[str, Any]:
        """Load the persisted server config. ``strict`` (the mutation paths)
        raises on a corrupt file so a subsequent save can never clobber
        operator config; non-strict (startup/list) degrades to empty."""
        if not self._config_path.exists():
            return {"servers": {}}
        try:
            with open(self._config_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if not isinstance(data, dict):
                raise ValueError("top-level JSON must be an object")
            data.setdefault("servers", {})
            return data
        except (json.JSONDecodeError, ValueError, OSError) as e:
            logger.error(f"Failed to load MCP config {self._config_path}: {e}")
            if strict:
                raise RuntimeError(
                    f"MCP config {self._config_path} is unreadable ({e}); "
                    "fix it before adding/removing servers."
                ) from e
            return {"servers": {}}

    def _save_config(self, config: Dict[str, Any]) -> None:
        self._config_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self._config_path, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2)
