# src/tools/mcp_bridge.py
"""derpr-hosted MCP server exposing ToolManager tools to dispatched subagents (DP-240).

This is the *server* half of derpr's MCP story; ``mcp_client.py`` (DP-268) is the
client half that consumes third-party servers. They share the SDK and nothing else.

Trust model — the part that must not be refactored away:

- The server runs **inside the trusted derpr process**. The subagent reaches it
  across the sandbox boundary over HTTP (one ``CC_SANDBOX_ALLOWED_DOMAINS``
  entry). It is deliberately NOT a stdio server spawned as a child of the
  subagent's ``claude``: a child is subvertible by its parent, and the parent is
  the thing being gated.
- Capable dispatches keep ``--dangerously-skip-permissions`` (decided
  2026-07-23), so Claude Code's own permission layer is NOT a second line of
  defense. Everything in this module is the only boundary there is.
- A gated call is never executed here. It becomes a ``call_derpr_tool`` proposal
  row and the subagent is told to stop and wait. Execution happens later, from
  the approved row, via ``AgentCallRunner`` — which re-checks the policy at that
  point. See ``memory/project/decisions/2026-07-23-mcp-gate-is-the-proposal-queue.md``.

Per-dispatch tokens: each capable dispatch mints one, the subagent gets it in its
environment, and it is revoked when the agent reaches a terminal state. A token
identifies *which* agent is calling, which is what lands in the proposal row's
``agent_id`` so an approving human knows whose request they are looking at.
"""

from __future__ import annotations

import contextlib
import logging
import secrets
from contextvars import ContextVar
from typing import Any, AsyncIterator, Callable, Dict, List, Optional

import mcp.types as mcp_types
from mcp.server.lowlevel import Server
from mcp.server.streamable_http_manager import StreamableHTTPSessionManager

from src.proposals.agent_call import AgentCallRunner
from src.tools.definitions import get_tool_capabilities

logger = logging.getLogger(__name__)

# The authenticated agent for the request currently being served. A contextvar
# rather than a parameter because the MCP SDK owns the call path between the
# ASGI layer (where the token is verified) and the tool handlers.
_current_agent: ContextVar[str] = ContextVar("_current_agent", default="")

#: Returned to the subagent when its call was queued for human review.
PARKED_STATUS = "parked_for_approval"


class BridgeTokenStore:
    """Per-dispatch bearer tokens, agent_id ↔ token.

    In-process and deliberately not persisted: a token must not outlive the
    derpr process that minted it, because the agent it authenticates does not
    either (the dispatcher's subprocess handles are process-local).
    """

    def __init__(self) -> None:
        self._by_token: Dict[str, str] = {}

    def mint(self, agent_id: str) -> str:
        """Issue a fresh token for ``agent_id``, replacing any prior one."""
        self.revoke(agent_id)
        token = secrets.token_urlsafe(32)
        self._by_token[token] = agent_id
        return token

    def token_for(self, agent_id: str) -> Optional[str]:
        """The live token for ``agent_id``, or None.

        Needed because a resumed agent (``answer_agent``) must be handed the
        SAME token it was dispatched with — minting a fresh one on resume would
        invalidate the config the still-running session already holds.
        """
        for token, owner in self._by_token.items():
            if owner == agent_id:
                return token
        return None

    def revoke(self, agent_id: str) -> None:
        for token, owner in list(self._by_token.items()):
            if owner == agent_id:
                del self._by_token[token]

    def resolve(self, supplied: str) -> Optional[str]:
        """The agent a token belongs to, or None.

        Constant-time compared against every live token — a plain dict lookup
        would leak token content through timing. The set is small (one entry per
        in-flight capable dispatch), so the linear scan is free in practice.
        """
        if not supplied:
            return None
        match: Optional[str] = None
        for token, agent_id in self._by_token.items():
            if secrets.compare_digest(supplied, token):
                match = agent_id
        return match


class McpBridge:
    """MCP server fronting ToolManager, mountable as an ASGI app."""

    def __init__(
        self,
        runner: AgentCallRunner,
        propose_call: Callable[[str, str, Dict[str, Any]], Any],
        token_store: Optional[BridgeTokenStore] = None,
        server_name: str = "derpr",
    ) -> None:
        # propose_call(agent_id, tool_name, tool_args) -> awaitable[int]
        # Injected rather than importing the proposal store directly: this module
        # sits in the tools layer and must not reach into agents/persistence.
        self._runner = runner
        self._propose_call = propose_call
        self.tokens = token_store or BridgeTokenStore()
        self._server: Server = Server(server_name)
        self._session_manager = StreamableHTTPSessionManager(
            app=self._server, stateless=True, json_response=True
        )
        self._register_handlers()

    # -- MCP handlers --------------------------------------------------------

    def _register_handlers(self) -> None:
        # The MCP SDK's registration decorators are untyped, so mypy cannot see
        # through them; the handler bodies themselves stay typed.
        @self._server.list_tools()  # type: ignore[no-untyped-call, untyped-decorator]
        async def list_tools() -> List[mcp_types.Tool]:
            return [self._to_mcp_tool(d) for d in self._runner.exposed_tool_definitions()]

        @self._server.call_tool()  # type: ignore[untyped-decorator]
        async def call_tool(name: str, arguments: Dict[str, Any]) -> Any:
            return await self._handle_call(name, arguments or {})

    @staticmethod
    def _to_mcp_tool(definition: Dict[str, Any]) -> mcp_types.Tool:
        """Translate a derpr tool definition into an MCP Tool.

        The description is annotated when the tool is gated, so the subagent can
        plan around the park instead of being surprised by it mid-task.
        """
        fn = definition.get("function", {})
        name = str(fn.get("name", ""))
        description = str(fn.get("description", ""))
        if definition.get("is_write"):
            description += (
                "\n\nNOTE: this tool requires human approval. Calling it queues a "
                "request and returns immediately without executing; you will be "
                "resumed with the result once a human approves."
            )
        return mcp_types.Tool(
            name=name,
            description=description,
            inputSchema=fn.get("parameters") or {"type": "object", "properties": {}},
        )

    def _is_gated(self, definition: Dict[str, Any]) -> bool:
        """Whether a call to this tool must be queued rather than executed.

        Write tools are gated. So is anything the definition marks irreversible
        even if it somehow is not flagged as a write — the two are independently
        sourced and this path fails closed on either.
        """
        if definition.get("is_write"):
            return True
        caps = get_tool_capabilities(str(definition.get("function", {}).get("name", "")))
        return bool(caps.get("irreversible"))

    async def _handle_call(self, name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        agent_id = _current_agent.get()
        if not agent_id:
            # Unreachable through the ASGI wrapper, which rejects before the SDK
            # is entered. Kept as a fail-closed backstop: an unattributable call
            # must never execute, because it cannot be gated or audited.
            logger.error("MCP call for '%s' arrived with no authenticated agent", name)
            return {"error": "unauthenticated"}

        exposed = {
            d.get("function", {}).get("name"): d
            for d in self._runner.exposed_tool_definitions()
        }
        definition = exposed.get(name)
        if definition is None:
            return {"error": f"tool '{name}' is not exposed by the bridge"}

        if self._is_gated(definition):
            proposal_id = await self._propose_call(agent_id, name, arguments)
            logger.info("Agent %s queued gated call %s as proposal %s",
                        agent_id, name, proposal_id)
            return {
                "status": PARKED_STATUS,
                "proposal_id": proposal_id,
                "message": (
                    f"'{name}' requires human approval and has been queued as "
                    f"proposal {proposal_id}. It has NOT run. Stop and wait — you "
                    "will be resumed with the result once a human decides."
                ),
            }

        logger.info("Agent %s calling ungated tool %s", agent_id, name)
        return await self._runner.execute_ungated(name, arguments)

    # -- ASGI ----------------------------------------------------------------

    @contextlib.asynccontextmanager
    async def lifespan(self) -> AsyncIterator[None]:
        """Must wrap the serving app's lifetime — the session manager's task
        group is created here and ``handle_request`` fails without it."""
        async with self._session_manager.run():
            yield

    async def handle_asgi(self, scope: Any, receive: Any, send: Any) -> None:
        """ASGI entrypoint: authenticate, bind the agent identity, then delegate.

        Authentication happens *before* the MCP SDK sees the request, so an
        unauthenticated caller never reaches a tool handler at all.
        """
        supplied = _bearer_from_scope(scope)
        agent_id = self.tokens.resolve(supplied)
        if agent_id is None:
            await _send_401(send)
            return
        token = _current_agent.set(agent_id)
        try:
            await self._session_manager.handle_request(scope, receive, send)
        finally:
            _current_agent.reset(token)


def _bearer_from_scope(scope: Any) -> str:
    """Extract the bridge token from an ASGI scope's headers."""
    for raw_name, raw_value in scope.get("headers", []):
        if raw_name.lower() == b"authorization":
            value = str(raw_value.decode("latin-1"))
            if value.lower().startswith("bearer "):
                return value[7:].strip()
    return ""


async def _send_401(send: Any) -> None:
    await send({
        "type": "http.response.start",
        "status": 401,
        "headers": [(b"content-type", b"application/json")],
    })
    await send({
        "type": "http.response.body",
        "body": b'{"error":"bridge token required"}',
    })
