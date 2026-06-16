# src/interfaces/kobold_engine_adapter.py
#
# SECURITY: CodeQL py/stack-trace-exposure is suppressed for this file because
# the kobold web UI is internal-only and tracebacks in responses aid debugging.
# See memory/project/decisions/kobold_stack_trace_exposure.md for rationale.
# If this UI ever becomes externally accessible, re-enable the rule (un-dismiss
# the alerts in GitHub code scanning) and scrub tracebacks from all responses.

import json
import logging
import mimetypes
import os

from src.request_builder import AssembledRequest

# On Windows, mimetypes seeds from the registry, where `.js` is frequently
# mapped to `text/plain`. Starlette's StaticFiles uses mimetypes.guess_type, so
# the bespoke UI bundle would be served as text/plain and browsers refuse to
# execute the ES module (strict MIME checking) → blank /derpr page. Force the
# correct types at import so the SPA mounts regardless of the host registry.
mimetypes.add_type("application/javascript", ".js")
mimetypes.add_type("application/javascript", ".mjs")
mimetypes.add_type("text/css", ".css")
from typing import (
    TYPE_CHECKING, Any, AsyncGenerator, AsyncIterator, Awaitable, Callable,
    Dict, Optional, List, Set, Tuple,
)
from datetime import datetime, timezone

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
import httpx
import uvicorn
import asyncio

from config import global_config
from src.chat_system import (
    ChatSystem, DoneEvent, ErrorEvent, GenerationEvent,
    PendingConfirmationEvent, ResponseType, TokenEvent, ToolCallResultEvent,
    ToolCallStartEvent,
)
from src.interfaces.kobold_export import build_kobold_savefile, build_transcript, _parse_tool_context
from src.stream_engine import CHAT_TEMPLATES
from src.interfaces.portal_render import render_portal_html
from src.personas.store import save_personas_to_file
from src.persona_fields import apply_patch_fields
from src.interfaces._persona_patch import (
    _KNOWN_PATCH_KEYS_ENGINE as _KNOWN_PATCH_KEYS,
    _apply_kobold_sampler_extras,
    get_kobold_extras_for_get,
)

if TYPE_CHECKING:
    from src.confirmations import ConfirmationManager
    from src.memory.memory_manager import MemoryManager
    from src.message_handler import BotLogic
    from src.persona import Persona
    from src.tools.tool_manager import ToolManager

logger = logging.getLogger(__name__)


def _kobold_base_url() -> str:
    """KoboldCPP base URL without trailing /v1."""
    raw = os.environ.get("LOCAL_LLM_URL", global_config.LOCAL_LLM_URL).rstrip("/")
    if raw.endswith("/v1"):
        raw = raw[:-3]
    return raw


def _channel_source(channel: str) -> str:
    """Derive a coarse source tag from a `channel` string (DP-136 / handoff §10).

    `channel` is a source-agnostic tag on the engine side; the bespoke portal
    groups channels by their originating interface. Web UI channels are tagged
    `web_ui` (or `web_ui:<name>`); Discord/Zammad/Gmail channels carry their own
    conventions. This is a best-effort prefix match for grouping/badge styling
    only — it never gates behavior.
    """
    c = (channel or "").lower()
    if c.startswith("web_ui") or c.startswith("web"):
        return "web"
    if c.startswith("discord") or c.startswith("dsc"):
        return "dsc"
    if c.startswith("zammad") or c.startswith("ticket") or c.startswith("zmd"):
        return "zmd"
    if c.startswith("gmail") or c.startswith("email") or c.startswith("gml"):
        return "gml"
    return "web"


class KoboldEngineAdapter:
    """HTTP boundary between kobold-lite and the DERPR engine.

    Phase D split (2026-04-28): the OAI route (`/v1/chat/completions`) is a
    thin SSE transcoder over `chat_system.stream_response` — engine rebuilds
    history from DB, only `derpr_user_text` (or last-user fallback) drives
    the user turn. The native route (`/api/extra/generate/stream`) and
    `/api/v1/generate` remain verbatim passthrough to KoboldCPP because the
    pre-rendered kobold prompt cannot be safely reconstructed from DB.
    See decisions/2026-04-28-portal-engine-as-source-of-truth.md.
    """

    def __init__(self, chat_system: ChatSystem, host: str = "0.0.0.0", port: int = 5003):
        self.chat_system = chat_system
        self.host = host
        self.port = port
        self.active_persona: Optional[str] = None
        self.app: FastAPI = FastAPI(title="DERPR Kobold Engine Adapter")

        # CORS open — required for lite.koboldai.net to reach a local instance.
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

        self._http = httpx.AsyncClient(timeout=None)
        self._setup_routes()
        self._setup_portal()

    # ------------------------------------------------------------------
    # Engine dependency surface (DP-205).
    #
    # Everything the adapter uses from the engine, enumerated in one place —
    # the routes below address these named seams, never `self.chat_system.*`
    # directly (pinned by test_adapter_engine_surface_is_enumerated). These
    # are read-through properties rather than eager destructuring because the
    # engine REBINDS several of them after construction (`models_available`
    # on every `update_models`, `personas`/`tool_manager` in admin and test
    # paths) — a copy bound at __init__ would go stale. memory_manager's
    # DP-130 transcript/version methods stay direct per the standing decision
    # (no facade over storage).
    # ------------------------------------------------------------------

    @property
    def _personas(self) -> Dict[str, "Persona"]:
        """All routable personas (system personas included) — name → Persona."""
        return self.chat_system.personas

    def _visible_personas(self) -> Dict[str, "Persona"]:
        """User-selectable personas (system personas excluded) for listings."""
        return self.chat_system.visible_personas()

    @property
    def _system_persona_names(self) -> Set[str]:
        """Names persisted as system personas (save_personas_to_file split)."""
        return self.chat_system.system_persona_names

    @property
    def _models_available(self) -> Dict[str, Any]:
        """Live LLM catalog (`what models` source) — rebound by update_models."""
        return self.chat_system.models_available

    @property
    def _memory_manager(self) -> "MemoryManager":
        """Storage: DP-130 transcript, version, suppression + logging methods."""
        return self.chat_system.memory_manager

    @property
    def _tool_manager(self) -> "ToolManager":
        """Tool registry — drives the /tools/catalog route."""
        return self.chat_system.tool_manager

    @property
    def _bot_logic(self) -> "BotLogic":
        """Command layer — dev_command preprocessing (`set`/`what` commands)."""
        return self.chat_system.bot_logic

    @property
    def _confirmations(self) -> "ConfirmationManager":
        """CONFIRM-mode park store — pending map keyed by (user, persona)."""
        return self.chat_system.confirmations

    @property
    def _stream_response(self) -> Callable[..., AsyncIterator[GenerationEvent]]:
        """Live generation kernel — the /v1/chat/completions event source."""
        return self.chat_system.stream_response

    @property
    def _stream_resume_confirmation(
        self,
    ) -> Callable[..., AsyncGenerator[GenerationEvent, None]]:
        """Approve/deny continuation stream for a parked CONFIRM write."""
        return self.chat_system.stream_resume_confirmation

    @property
    def _assemble_request(self) -> Callable[..., Awaitable[Optional[AssembledRequest]]]:
        """S5 dry-run assembler (parity inspector) — shares the live builder."""
        return self.chat_system.assemble_request

    @property
    def _get_view_history(self) -> Callable[..., Tuple[List[Dict[str, Any]], str]]:
        """History exactly as the engine would see it (DP-136 transcript seam)."""
        return self.chat_system.get_view_history

    @property
    def _get_session_memory_block(self) -> Callable[..., Awaitable[Optional[str]]]:
        """Public LTM seam for the client-side ltm_block route."""
        return self.chat_system.get_session_memory_block

    def _setup_portal(self) -> None:
        @self.app.get("/portal")
        async def get_portal() -> HTMLResponse:
            return HTMLResponse(render_portal_html("engine"))

        @self.app.get("/")
        async def root_redirect() -> HTMLResponse:
            return HTMLResponse(render_portal_html("engine"))

        # --- DP-132: bespoke "DERPR Portal" web UI (React/Vite build) ---------
        # Additive only. The existing /portal (Kobold-Lite PoC) is untouched.
        # The Vite app is built with base="/derpr/" so its asset URLs resolve
        # under the StaticFiles mount below. GET /derpr returns the SPA entry
        # (index.html); the mount serves the hashed JS/CSS assets. If the build
        # output is absent (UI not yet built), /derpr returns a short hint so
        # the engine still boots without the front-end artifacts present.
        derpr_dist = os.path.join(
            os.path.dirname(__file__), "web_assets", "derpr_ui", "dist"
        )
        derpr_index = os.path.join(derpr_dist, "index.html")

        @self.app.get("/derpr")
        async def get_derpr_portal() -> HTMLResponse:
            try:
                with open(derpr_index, "r", encoding="utf-8") as fh:
                    return HTMLResponse(fh.read())
            except FileNotFoundError:
                return HTMLResponse(
                    "<h1>DERPR Portal not built</h1>"
                    "<p>Run <code>npm run build</code> in "
                    "<code>src/interfaces/web_assets/derpr_ui</code>.</p>",
                    status_code=503,
                )

        if os.path.isdir(derpr_dist):
            # html=True so the mount serves index.html for /derpr/ and falls
            # back gracefully; assets live under /derpr/assets/*.
            self.app.mount(
                "/derpr",
                StaticFiles(directory=derpr_dist, html=True),
                name="derpr_portal",
            )

    def _setup_routes(self) -> None:
        @self.app.get("/api/v1/model")
        async def get_model() -> Any:
            return {"result": self._get_current_persona_name()}

        @self.app.put("/api/v1/model")
        async def set_model(request: Request) -> Any:
            data = await request.json()
            new_persona = data.get("model") or data.get("result")
            if new_persona in self._personas:
                self.active_persona = new_persona
                logger.info(f"Switched active persona to: {new_persona}")
                return {"result": self.active_persona}
            return {"error": f"Persona '{new_persona}' not found", "available": list(self._visible_personas().keys())}

        @self.app.get("/v1/models")
        async def list_models() -> Any:
            models = [
                {"id": name, "object": "model", "owned_by": "derpr", "permission": []}
                for name in self._visible_personas().keys()
            ]
            return {"object": "list", "data": models}

        @self.app.get("/api/v1/tools/catalog")
        async def get_tools_catalog() -> Any:
            defs = self._tool_manager.get_tool_definitions()
            tools = []
            for t in defs:
                func = t.get("function", {})
                caps = t.get("capabilities", {})
                tools.append({
                    "name": func.get("name"),
                    "description": func.get("description"),
                    "is_write": bool(t.get("is_write", False)),
                    # null for built-in/local tools (no external service); the UI
                    # groups the catalog by this into collapsible categories.
                    "service_binding": t.get("service_binding"),
                    "capabilities": {
                        "locality": caps.get("locality"),
                        "sensitivity": caps.get("sensitivity"),
                        "produces_untrusted": bool(caps.get("produces_untrusted", False)),
                    }
                })
            return {"tools": tools}

        @self.app.get("/api/v1/persona/{name}")
        async def get_persona(name: str) -> Any:
            if name not in self._personas:
                return JSONResponse(status_code=404, content={"error": f"Persona '{name}' not found"})
            p = self._personas[name]
            return {
                "name": p.get_name(),
                "display_name": p.get_name().title(),
                "prompt": p.get_prompt(),
                "model_name": p.get_model_name(),
                "temperature": p.get_temperature(),
                "top_p": p.get_top_p(),
                "top_k": p.get_top_k(),
                "max_tokens": p.get_response_token_limit(),
                "history_messages": p.get_base_history_messages(),
                "thinking_level": p.get_thinking_level(),
                "memory_mode": p.get_memory_mode().name,
                "max_context_tokens": p.get_max_context_tokens(),
                "long_term_memory": p.get_long_term_memory(),
                "inject_timestamp": p.get_inject_timestamp(),
                "self_edit": p.get_self_edit(),
                "chat_template": p.get_chat_template(),
                "instruct_tags": p.get_provider_extra("kobold", "instruct_tags"),
                "kobold_extras": get_kobold_extras_for_get(p),
                "enabled_tools": p.get_enabled_tools(),
                "tool_policy": p.get_tool_policy().to_dict(),
                "service_bindings": p.get_service_bindings(),
                "security_blocked": p.is_security_blocked(),
                "security_block_reasons": p.get_security_block_reasons(),
            }

        @self.app.get("/api/v1/channels")
        async def list_channels(persona: Optional[str] = None) -> Any:
            """List the distinct channels seen in history (DP-136 / handoff §10).

            Drives the bespoke portal's channel list. Each entry carries the raw
            `channel` tag, a derived `source` prefix (web/dsc/zmd/gml) for
            grouping + badge styling, the most recent `last_ts`, and a row
            `count`. Scoped to `persona` when given so the list reflects the
            channels the active persona has been used in. The portal always
            includes a synthetic `web_ui` row so a fresh persona with no history
            still offers a default channel to talk in.
            """
            rows = await asyncio.to_thread(
                self._memory_manager.get_distinct_channels, persona
            )
            channels = [
                {
                    "channel": r.get("channel"),
                    "server_id": r.get("server_id"),
                    "source": _channel_source(r.get("channel") or ""),
                    "count": r.get("count", 0),
                    "last_ts": r.get("last_ts"),
                }
                for r in rows
                if r.get("channel")
            ]
            if not any(c["channel"] == "web_ui" for c in channels):
                channels.append({
                    "channel": "web_ui", "server_id": None, "source": "web",
                    "count": 0, "last_ts": None,
                })
            return {"channels": channels}

        @self.app.get("/api/v1/session/{persona}/ltm_block")
        async def ltm_block(
            persona: str,
            query: str = "",
            channel: str = "web_ui",
            user_identifier: str = "portal",
            server_id: Optional[str] = None,
        ) -> Any:
            """Retrieve an LTM memory block for the given persona and query text.

            Returns {"block": "<memory>...</memory>"} or {"block": null} when
            no relevant memories exist or LTM is disabled for the persona.
            Phase 2.2: called client-side before each submit when LTM is on;
            the block is written into kobold-lite's current_anote so kobold
            places it at its normal author's-note position in the prompt.
            DP-136: `channel`/`user_identifier`/`server_id` are accepted so the
            recalled block is scoped to the active channel (defaults preserve the
            single web_ui/portal behavior).
            """
            if persona not in self._personas:
                return JSONResponse(status_code=404, content={"error": f"Persona '{persona}' not found"})
            block = await self._get_session_memory_block(
                persona_name=persona,
                user_identifier=user_identifier,
                channel=channel,
                server_id=server_id,
                query=query,
            )
            return {"block": block}

        @self.app.get("/api/v1/session/{persona}/assemble")
        async def assemble(
            persona: str,
            message: str = "",
            channel: str = "web_ui",
            ltm: bool = True,
            retry: bool = False,
        ) -> Any:
            """Dry-run request assembler — the Raw-req parity inspector source.

            Runs the SAME assembly path as a live `/v1/chat/completions` submit
            (history rebuild from DB + LTM injection + token truncation + param
            resolution + system-prompt prepend) with inference OFF, and returns
            exactly what would be sent: {parity, route, model_name, params,
            messages[]} per API_CONTRACTS.md §9. Because `assemble_request`
            shares `prepare_request`, `resolve_generation_params`, and
            `build_wire_messages` with the live kernel, the messages/params it
            returns cannot drift from a live submit *with the same inputs*.

            SCOPE: the parity guarantee holds for the bespoke portal client,
            which is the only caller — it submits as `user_identifier="portal"`
            with persona-default samplers and no `server_id`, exactly what this
            endpoint hardcodes. It is NOT a parity claim for arbitrary clients: a
            different user/server (PERSONAL/SERVER-mode personas) or a
            sampler-overriding client (e.g. kobold-lite) assembles a different
            history window / merges different params, which this dry-run does not
            model. (DP-132 #12 — broadening to accept those inputs is deferred.)

            `ltm` is accepted for API-shape compatibility but assembly follows the
            persona's own `long_term_memory` setting (the engine path has no
            client LTM toggle), so the dry-run mirrors true engine behavior.
            """
            if persona not in self._personas:
                return JSONResponse(status_code=404, content={"error": f"Persona '{persona}' not found"})
            assembled = await self._assemble_request(
                persona_name=persona,
                user_identifier="portal",
                channel=channel,
                message=message,
                is_retry=retry,
            )
            if assembled is None:
                return JSONResponse(status_code=404, content={"error": f"Persona '{persona}' not found"})
            return JSONResponse(content=self._assembled_to_dict(assembled))

        @self.app.get("/api/v1/models/list")
        async def list_all_models() -> Dict[str, Any]:
            # Source from the same in-memory list the `what models` command
            # reads (chat_system.models_available), so the dropdown and the
            # command never diverge. update_models refreshes both.
            avail = self._models_available or {}
            all_m = []
            for sub in avail.values():
                if isinstance(sub, list):
                    all_m.extend(sub)
                else:
                    all_m.append(sub)
            return {"models": sorted(list(set(all_m)))}

        @self.app.get("/api/v1/chat_templates")
        async def list_chat_templates() -> Dict[str, Any]:
            # The instruct templates the local renderer understands
            # (StreamEngine.CHAT_TEMPLATES keys). Single source of truth for the
            # persona inspector's chat_template dropdown so the UI never drifts
            # from what the engine can actually render. Empty/None on a persona
            # = fall back to the env/global default at render time.
            return {"templates": sorted(CHAT_TEMPLATES.keys())}

        @self.app.post("/api/v1/persona/{name}/reset")
        async def reset_persona_history(name: str) -> Any:
            if name not in self._personas:
                return {"error": "Persona not found"}
            p = self._personas[name]
            p.start_new_conversation()
            return {"result": f"History for {name} reset successfully"}

        @self.app.post("/api/v1/persona/{name}/dev_command")
        async def dev_command(name: str, request: Request) -> Any:
            if name not in self._personas:
                return JSONResponse(status_code=404,
                                    content={"error": f"Persona '{name}' not found"})
            body = await request.json()
            command = body.get("command", "")
            try:
                result = await self._bot_logic.preprocess_message(name, "portal", command)
            except Exception as e:
                return {"response": str(e), "mutated": False}
            if result is None:
                return JSONResponse(
                    status_code=400,
                    content={"response": "Not a dev command", "mutated": False},
                )
            if result.get("mutated"):
                save_personas_to_file(self._personas, self._system_persona_names)
            return {"response": result.get("response", ""), "mutated": bool(result.get("mutated"))}

        @self.app.post("/api/v1/persona/{name}/confirm")
        async def confirm_pending(name: str, request: Request) -> Any:
            """Approve or deny a CONFIRM-mode write parked for persona `name`.

            The portal calls this after a `derpr-confirm` SSE frame surfaces a
            parked write. The continuation (write execution on approve, denial
            results on deny, then the model's follow-up turn) streams back as
            SSE using the same wire protocol as /v1/chat/completions — so any
            chained confirmation re-surfaces as another `derpr-confirm` frame.

            Body: {"approved": bool, "token": "<token from the frame>"}. The
            token guards against resuming a stale park (model proposed different
            writes since); omit or send empty to skip the check. The portal user
            is always "portal", matching the park key (user, persona).
            """
            if name not in self._personas:
                return JSONResponse(status_code=404,
                                    content={"error": f"Persona '{name}' not found"})
            body = await request.json()
            approved = bool(body.get("approved"))
            token = body.get("token") or None

            async def relay() -> AsyncIterator[bytes]:
                async for ev in self._stream_resume_confirmation(
                    user_identifier="portal",
                    persona_name=name,
                    approved=approved,
                    expected_token=token,
                ):
                    if await request.is_disconnected():
                        return
                    for _label, frame in self._event_to_sse(ev):
                        yield frame

            return StreamingResponse(
                relay(),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "X-Accel-Buffering": "no",
                    "Connection": "keep-alive",
                },
            )

        @self.app.get("/api/v1/session/{persona}/kobold_export")
        async def kobold_export(persona: str, max_turns: Optional[int] = None) -> Any:
            """Build a kobold-lite savefile from DERPR's global history for `persona`.

            Phase 2.1 always pulls global history (all channels) — the portal has
            no channel concept. max_turns defaults to the persona's configured
            sliding-window size (`get_base_history_messages`); no new config key.
            """
            if persona not in self._personas:
                return JSONResponse(status_code=404, content={"error": f"Persona '{persona}' not found"})

            p = self._personas[persona]
            limit = max_turns if isinstance(max_turns, int) and max_turns > 0 else p.get_base_history_messages()
            raw_history = await asyncio.to_thread(
                self._memory_manager.get_global_history, persona, limit
            )
            savefile, skipped = build_kobold_savefile(raw_history)
            logger.warning(
                f"kobold_export persona={persona} limit={limit} "
                f"rows={len(raw_history)} actions={len(savefile.get('actions', []))} "
                f"ids={len(savefile.get('interaction_ids', []))} skipped={skipped}"
            )
            return JSONResponse(content=savefile)

        @self.app.get("/api/v1/session/{persona}/transcript")
        async def session_transcript(
            persona: str,
            max_turns: Optional[int] = None,
            channel: Optional[str] = None,
            user_identifier: str = "portal",
            server_id: Optional[str] = None,
        ) -> Any:
            """DP-130 history contract — the authoritative transcript projection.

            Returns `{"chunks": [...]}` where every chunk carries a
            server-authored `interaction_id` (or `ephemeral=true` for a live
            parked confirmation). This is the single re-sync source for the Lite
            stopgap (DP-131) and the render source for the bespoke UI (DP-132+):
            consumers address chunks by identity, never by story position, so
            the id array can no longer drift on parked/tool-only/abort turns.

            `max_turns` defaults to the persona's sliding-window size. Suppressed
            rows are filtered upstream (invariant C5).

            DP-136 channel scoping: when `channel` is supplied, history is fetched
            through the SAME memory-mode dispatch the live `/v1/chat/completions`
            path uses (`ChatSystem.get_view_history`), so the rendered transcript
            mirrors exactly what the engine would feed the model for that
            (persona, channel) — a CHANNEL-mode persona isolates per channel, a
            GLOBAL-mode persona merges all channels. When `channel` is omitted the
            legacy global-history behavior is preserved (back-compat for the
            single-channel Lite/DP-132 callers).
            """
            if persona not in self._personas:
                return JSONResponse(status_code=404, content={"error": f"Persona '{persona}' not found"})

            p = self._personas[persona]
            limit = max_turns if isinstance(max_turns, int) and max_turns > 0 else p.get_base_history_messages()
            raw_history, mode_used = await asyncio.to_thread(
                self._get_view_history,
                persona, user_identifier, channel, server_id, limit,
            )
            ids = [
                m["interaction_id"] for m in raw_history
                if isinstance(m.get("interaction_id"), int)
            ]
            ids_with_versions = await asyncio.to_thread(
                self._memory_manager.get_ids_with_versions, ids
            )
            # Surface a live parked confirmation (portal session) as a trailing
            # ephemeral chunk so a fresh load renders the awaiting-approval text.
            # Park key is (user_identifier, persona) — honor the requested user.
            pending_map = self._confirmations.pending
            pending_obj = pending_map.get((user_identifier, persona))
            pending: Optional[Dict[str, Any]] = None
            if pending_obj is not None:
                tool_msgs = pending_obj.conversation_history[pending_obj.tool_context_start:] if pending_obj.conversation_history else []
                pending = {
                    "ephemeral_chunk_id": pending_obj.token,
                    "content": pending_obj.confirmation_text,
                    "tool_context": _parse_tool_context(tool_msgs) if tool_msgs else None,
                }
            transcript = build_transcript(
                raw_history,
                ids_with_versions=ids_with_versions,
                pending=pending,
            )
            logger.debug(
                "transcript persona=%s channel=%s mode=%s limit=%s rows=%d "
                "chunks=%d pending=%s",
                persona, channel, mode_used, limit, len(raw_history),
                len(transcript["chunks"]), pending is not None,
            )
            return JSONResponse(content=transcript)

        @self.app.get("/api/v1/interaction/{interaction_id}/versions")
        async def list_interaction_versions(interaction_id: int) -> Any:
            """List all stored versions for an interaction, canonical last.

            Portal hydrates `retry_prev_text` / `redo_prev_text` stacks from
            this after seeing `assistant_id` in the stream's derpr event.
            """
            try:
                versions = await asyncio.to_thread(
                    self._memory_manager.list_interaction_versions,
                    interaction_id,
                )
            except Exception as e:
                logger.error(f"list_interaction_versions({interaction_id}) failed: {e}")
                return JSONResponse(status_code=500, content={"error": str(e)})
            if not versions:
                return JSONResponse(
                    status_code=404,
                    content={"error": f"interaction {interaction_id} not found"},
                )

            for v in versions:
                reasoning = v.get("reasoning_content")
                if reasoning:
                    v["content"] = f"<think>\n{reasoning}\n</think>\n{v['content']}"

            return {"interaction_id": interaction_id, "versions": versions}

        @self.app.post("/api/v1/interaction/{interaction_id}/select_version/{k}")
        async def select_interaction_version(interaction_id: int, k: int) -> Any:
            """Swap archive position `k` with canonical (0-indexed pre-swap).

            Returns new canonical + refreshed version list so portal can
            re-sync its chevron stacks in one round-trip.
            """
            try:
                result = await asyncio.to_thread(
                    self._memory_manager.swap_interaction_version,
                    interaction_id,
                    k,
                )
            except ValueError as e:
                return JSONResponse(status_code=404, content={"error": str(e)})
            except IndexError as e:
                return JSONResponse(status_code=400, content={"error": str(e)})
            except Exception as e:
                logger.error(
                    f"swap_interaction_version({interaction_id}, {k}) failed: {e}"
                )
                return JSONResponse(status_code=500, content={"error": str(e)})
            logger.debug(
                "select_version id=%s archive_k=%d -> total_versions=%s",
                interaction_id, k, result.get("total_versions"),
            )
            versions = await asyncio.to_thread(
                self._memory_manager.list_interaction_versions,
                interaction_id,
            )
            for v in versions:
                reasoning = v.get("reasoning_content")
                if reasoning:
                    v["content"] = f"<think>\n{reasoning}\n</think>\n{v['content']}"

            if result.get("reasoning_content"):
                result["current_content"] = f"<think>\n{result['reasoning_content']}\n</think>\n{result['current_content']}"

            return {**result, "versions": versions}

        @self.app.patch("/api/v1/interaction/{interaction_id}")
        async def patch_interaction(interaction_id: int, request: Request) -> Any:
            """Update the content of an existing interaction (e.g. on manual edit)."""
            data = await request.json()
            content = data.get("content")
            if content is None:
                return JSONResponse(status_code=400, content={"error": "missing 'content' field"})
            try:
                await asyncio.to_thread(
                    self._memory_manager.update_interaction_content,
                    interaction_id,
                    content,
                )
                return {"result": "success", "interaction_id": interaction_id}
            except Exception as e:
                logger.error(f"patch_interaction({interaction_id}) failed: {e}")
                return JSONResponse(status_code=500, content={"error": str(e)})

        @self.app.delete("/api/v1/interaction/{interaction_id}")
        async def delete_interaction(interaction_id: int) -> Any:
            """Soft-suppress an interaction (portal empty-edit / delete flow).

            Idempotent: a second DELETE returns success with `already_suppressed: true`.
            Reply chains stay intact; suppressed rows are filtered from history,
            retrieval, and `kobold_export` via `_suppression_filter`.
            """
            try:
                inserted = await asyncio.to_thread(
                    self._memory_manager.suppress_interaction,
                    interaction_id,
                )
            except Exception as e:
                logger.error(f"delete_interaction({interaction_id}) failed: {e}")
                return JSONResponse(status_code=500, content={"error": str(e)})
            return {
                "result": "success",
                "interaction_id": interaction_id,
                "already_suppressed": not inserted,
            }

        @self.app.patch("/api/v1/persona/{name}")
        async def patch_persona(name: str, request: Request) -> Any:
            if name not in self._personas:
                return JSONResponse(status_code=404, content={"error": "Persona not found"})
            try:
                data = await request.json()
            except Exception as e:
                return JSONResponse(status_code=400, content={"error": f"invalid JSON: {e}"})
            p = self._personas[name]

            # Numeric setters silently coerce bad input to None / defaults and
            # return the resolved value. Capture rejections so the portal can
            # surface them instead of pretending the save was clean.
            rejected: List[str] = []

            # Core persona fields apply via the shared registry — one source
            # of truth with the dev-command surface (DP-200 slice D).
            apply_patch_fields(p, data, rejected)
            if "history_messages" in data:
                p.set_history_messages(data["history_messages"])
            elif "context_length" in data:
                p.set_history_messages(data["context_length"])
            if "instruct_tags" in data:
                tags = data["instruct_tags"]
                if isinstance(tags, dict) and any(tags.values()):
                    p.set_provider_extra("kobold", "instruct_tags", tags)
                else:
                    p.clear_provider_extra("kobold", "instruct_tags")

            _apply_kobold_sampler_extras(p, data, rejected)

            unknown = sorted(set(data.keys()) - _KNOWN_PATCH_KEYS)
            if unknown:
                logger.warning(f"PATCH /persona/{name}: unknown fields ignored: {unknown}")

            try:
                save_personas_to_file(self._personas, self._system_persona_names)
            except Exception as e:
                logger.error(f"Persona save failed for {name}: {e}")
                return JSONResponse(
                    status_code=500,
                    content={"error": "save_failed", "detail": str(e), "rejected_fields": rejected, "unknown_fields": unknown},
                )
            logger.info(f"Updated and saved persona settings for {name} (rejected={rejected}, unknown={unknown})")
            return {"result": "success", "rejected_fields": rejected, "unknown_fields": unknown}

        @self.app.get("/api/v1/info/version")
        async def get_info_version() -> Any:
            return await self._forward_get("/api/v1/info/version", {"version": "1.70", "lib_version": "1.70"})

        @self.app.get("/api/extra/version")
        async def get_extra_version() -> Any:
            # Forward verbatim so portal can detect KCPP version + jinja/mcp/etc.
            # Fallback only on upstream failure. Without real version portal
            # falls back to legacy prompt-field format and instruct tags break.
            return await self._forward_get("/api/extra/version", {"version": "1.70", "platform": "DERPR"})

        @self.app.get("/api/v1/config/soft_prompts")
        async def get_soft_prompts() -> Any:
            return await self._forward_get("/api/v1/config/soft_prompts", {"results": []})

        @self.app.get("/api/v1/config/max_context_length")
        async def get_max_history_messages() -> Any:
            return await self._forward_get("/api/v1/config/max_context_length", {"result": global_config.DEFAULT_MAX_CONTEXT_TOKENS})

        @self.app.get("/api/extra/true_max_context_length")
        async def get_true_max_ctx() -> Any:
            return await self._forward_get("/api/extra/true_max_context_length", {"value": global_config.DEFAULT_MAX_CONTEXT_TOKENS})

        @self.app.get("/api/extra/perf")
        async def get_perf() -> Any:
            return await self._forward_get("/api/extra/perf", {})

        @self.app.post("/api/extra/tokencount")
        async def tokencount(request: Request) -> Any:
            return await self._forward_post("/api/extra/tokencount", await request.json())

        @self.app.post("/api/v1/generate")
        async def kobold_generate(request: Request) -> Any:
            """Non-streaming KoboldCPP generation with DB logging."""
            data = await request.json()
            persona_name = self._get_current_persona_name()
            prompt = data.get("prompt", "")
            user_interaction_id: Optional[int] = None
            if prompt and prompt.strip():
                clean_prompt = self._extract_last_user_turn(prompt)
                user_interaction_id = self._log_interaction(persona_name, "user", clean_prompt)
            url = f"{_kobold_base_url()}/api/v1/generate"
            try:
                r = await self._http.post(url, json=data)
                resp = r.json() if r.content else {}
                if r.status_code == 200:
                    results = resp.get("results", [])
                    if results:
                        ai_text = results[0].get("text", "")
                        if ai_text:
                            self._commit_assistant(persona_name, ai_text, user_interaction_id, None)
                return JSONResponse(status_code=r.status_code, content=resp)
            except httpx.RequestError as e:
                logger.error(f"/api/v1/generate upstream failed: {e}")
                return JSONResponse(status_code=502, content={"error": str(e)})

        @self.app.post("/api/extra/generate/stream")
        async def kobold_generate_stream(request: Request) -> StreamingResponse:
            """Streaming KoboldCPP SSE generation with DB logging.

            Logs the user turn from `prompt` on entry, then collects all SSE
            token deltas and commits the assembled assistant turn on [DONE].
            Persona is selected by adapter.active_persona — uniform with the
            OAI path; per-request `model` override is rejected.
            """
            data = await request.json()
            persona_name = self._get_current_persona_name()

            prompt: str = data.get("prompt") or ""
            user_interaction_id: Optional[int] = None
            if prompt.strip():
                clean_prompt = self._extract_last_user_turn(prompt)
                user_interaction_id = self._log_interaction(persona_name, "user", clean_prompt)

            forward_body = {k: v for k, v in data.items() if k != "model"}
            url = f"{_kobold_base_url()}/api/extra/generate/stream"

            async def relay_stream() -> AsyncIterator[bytes]:
                full_response: List[str] = []
                committed = False
                try:
                    async with self._http.stream("POST", url, json=forward_body) as upstream:
                        async for chunk in upstream.aiter_raw():
                            if await request.is_disconnected():
                                return
                            if not chunk:
                                continue
                            try:
                                decoded = chunk.decode("utf-8")
                                for line in decoded.splitlines():
                                    if line.startswith("data: "):
                                        raw = line[6:].strip()
                                        if raw and raw != "[DONE]":
                                            try:
                                                tok_data = json.loads(raw)
                                                token = tok_data.get("token")
                                                if token:
                                                    full_response.append(token)
                                            except Exception:
                                                pass
                            except Exception:
                                pass
                            yield chunk

                except httpx.RequestError as e:
                    logger.error(f"/api/extra/generate/stream upstream failed: {e}")
                    err_payload = json.dumps({"error": str(e)})
                    yield f"data: {err_payload}\n\ndata: [DONE]\n\n".encode("utf-8")
                except asyncio.CancelledError:
                    if full_response and not committed:
                        committed = True
                        self._commit_assistant(
                            persona_name, "".join(full_response),
                            user_interaction_id, None,
                        )
                    raise
                finally:
                    if full_response and not committed:
                        committed = True
                        self._commit_assistant(
                            persona_name, "".join(full_response),
                            user_interaction_id, None,
                        )

            return StreamingResponse(
                relay_stream(),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "X-Accel-Buffering": "no",
                    "Connection": "keep-alive",
                },
            )

        @self.app.get("/api/extra/generate/check")
        @self.app.post("/api/extra/generate/check")
        async def generate_check(request: Request) -> Any:
            body = await request.json() if request.method == "POST" else {}
            return await self._forward_post("/api/extra/generate/check", body) if request.method == "POST" \
                else await self._forward_get("/api/extra/generate/check", {})

        @self.app.post("/api/v1/abort")
        @self.app.post("/api/extra/abort")
        async def abort_generation() -> Any:
            url = f"{_kobold_base_url()}/api/extra/abort"
            try:
                r = await self._http.post(url, json={})
                return JSONResponse(r.json() if r.content else {"result": "aborted"})
            except Exception as e:
                logger.warning(f"Abort forward failed: {e}")
                return {"result": "abort_failed", "error": str(e)}

        @self.app.post("/chat/completions")
        @self.app.post("/v1/chat/completions")
        async def oai_chat_completions(request: Request) -> Any:
            """Thin SSE transcoder over `chat_system.stream_response`.

            Phase D (2026-04-28): the engine is the single source of truth.
            Client `data["messages"]` is discarded — history is rebuilt from
            DB. Only `derpr_user_text` (or fallback last-user scan for
            non-portal clients) drives the user turn. Retry/edit/delete
            already round-trip via PATCH/DELETE/version routes, so DB == UI.
            See decisions/2026-04-28-portal-engine-as-source-of-truth.md.
            """
            data = await request.json()
            is_retry = bool(data.get("derpr_retry"))
            sidecar_user = data.get("derpr_user_text")
            is_stream = bool(data.get("stream"))
            persona_name = self._get_current_persona_name()
            # DP-136 channel scoping: the portal may target a specific channel
            # (and optionally user_identifier/server_id). Defaults preserve the
            # single web_ui/portal behavior. Creating a "new channel" is just
            # submitting the first turn with a fresh `channel` string — the
            # engine's memory_mode then governs isolation (see get_view_history).
            channel = data.get("channel") or "web_ui"
            user_identifier = data.get("user_identifier") or "portal"
            server_id = data.get("server_id") or None

            if isinstance(sidecar_user, str) and sidecar_user.strip():
                user_text: str = sidecar_user
            else:
                user_text = self._find_last_user_content(data.get("messages") or []) or ""

            # --- DEBUG: dump incoming messages so we can diagnose empty user_text ---
            msgs_in = data.get("messages") or []
            logger.warning(
                "OAI /chat/completions DEBUG — "
                "derpr_user_text=%r  user_text=%r  msg_count=%d  "
                "roles=%s  continue_assistant_turn=%r",
                sidecar_user,
                user_text[:120] if user_text else "",
                len(msgs_in),
                [m.get("role") for m in msgs_in],
                data.get("continue_assistant_turn"),
            )
            if msgs_in:
                last = msgs_in[-1]
                logger.warning(
                    "OAI DEBUG last message — role=%r  content_type=%s  content_preview=%r",
                    last.get("role"),
                    type(last.get("content")).__name__,
                    str(last.get("content"))[:200] if last.get("content") else None,
                )
            # --- END DEBUG ---

            # Extract sampling parameters from the OAI request for the local engine
            local_inference_config = {
                "temperature": data.get("temperature"),
                "top_p": data.get("top_p"),
                "top_k": data.get("top_k"),
                "max_tokens": data.get("max_tokens") or data.get("max_completion_tokens"),
                "stop_sequence": data.get("stop"),
            }
            # Kobold-specific extras (from kobold-lite's UI)
            for k in ("rep_pen", "rep_pen_range", "rep_pen_slope",
                      "min_p", "typical", "tfs", "max_context_length"):
                if data.get(k) is not None:
                    local_inference_config[k] = data[k]

            logger.info(
                f"OAI chat -> stream_response (stream={is_stream}, "
                f"retry={is_retry}, persona={persona_name}, "
                f"user_text_len={len(user_text)})"
            )

            if not is_stream:
                full_text = ""
                assistant_id: Optional[int] = None
                async for ev in self._stream_response(
                    persona_name=persona_name,
                    user_identifier=user_identifier,
                    channel=channel,
                    server_id=server_id,
                    message=user_text,
                    is_retry=is_retry,
                    local_inference_config=local_inference_config,
                    client_messages=msgs_in or None,
                ):
                    if isinstance(ev, DoneEvent):
                        full_text = ev.text
                        assistant_id = ev.assistant_id
                    elif isinstance(ev, ErrorEvent):
                        return JSONResponse(
                            status_code=502,
                            content={"error": {"message": ev.message}},
                        )
                return JSONResponse(content={
                    "object": "chat.completion",
                    "choices": [{
                        "index": 0,
                        "message": {"role": "assistant", "content": full_text},
                        "finish_reason": "stop",
                    }],
                    "derpr_assistant_id": assistant_id,
                })

            # Optional wire-level dump: set DERPR_DEBUG_SSE_DUMP=1 (or a dir path)
            # to capture the incoming request JSON + every outgoing SSE chunk to
            # a per-request file under <dir>/derpr_sse_<ts>_<persona>.log. Use
            # when the portal swallows a turn so we can replay exactly what hit
            # the wire. Default dir: ./logs/sse_dumps. Zero overhead when unset.
            dump_setting = os.environ.get("DERPR_DEBUG_SSE_DUMP", "").strip()
            dump_fh = None
            if dump_setting:
                dump_dir = dump_setting if dump_setting not in ("1", "true", "TRUE") else "logs/sse_dumps"
                try:
                    os.makedirs(dump_dir, exist_ok=True)
                    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%f")
                    dump_path = os.path.join(
                        dump_dir, f"derpr_sse_{ts}_{persona_name or 'nopersona'}.log"
                    )
                    dump_fh = open(dump_path, "w", encoding="utf-8")
                    dump_fh.write("=== REQUEST ===\n")
                    dump_fh.write(json.dumps({
                        "ts": ts,
                        "persona": persona_name,
                        "is_retry": is_retry,
                        "is_stream": is_stream,
                        "user_text": user_text,
                        "derpr_user_text_sidecar": sidecar_user,
                        "msg_count": len(msgs_in),
                        "roles": [m.get("role") for m in msgs_in],
                        "local_inference_config": local_inference_config,
                        "raw_body": data,
                    }, ensure_ascii=False, indent=2, default=str))
                    dump_fh.write("\n\n=== SSE STREAM ===\n")
                    dump_fh.flush()
                    logger.warning("SSE dump enabled — writing to %s", dump_path)
                except Exception as e:
                    logger.error("SSE dump init failed: %s", e)
                    dump_fh = None

            def _dump_write(label: str, payload: bytes) -> None:
                if dump_fh is None:
                    return
                try:
                    dump_fh.write(f"--- {label} ---\n")
                    dump_fh.write(payload.decode("utf-8", errors="replace"))
                    if not payload.endswith(b"\n"):
                        dump_fh.write("\n")
                    dump_fh.flush()
                except Exception:
                    pass

            async def relay() -> AsyncIterator[bytes]:
                try:
                    # Pass through the request to the central ChatSystem.
                    # In the Engine Adapter, we DISCARD the client-side message array
                    # (msgs_in) to ensure the engine's DB is the source of truth for
                    # history. This prevents the 'hollow history' issue where the
                    # UI might be empty but the DB has rich context.
                    async for ev in self._stream_response(
                        persona_name=persona_name,
                        user_identifier=user_identifier,
                        channel=channel,
                        server_id=server_id,
                        message=user_text,
                        is_retry=is_retry,
                        local_inference_config=local_inference_config,
                        client_messages=None,  # Explicitly None to force DB history rebuild
                    ):

                        if await request.is_disconnected():
                            return
                        # Rich diagnostic dump for DoneEvent (dump-only, not a
                        # wire frame) — preserved alongside the shared transcoder.
                        if dump_fh is not None and isinstance(ev, DoneEvent):
                            _dump_write(
                                "DoneEvent",
                                json.dumps({
                                    "assistant_id": ev.assistant_id,
                                    "user_interaction_id": ev.user_interaction_id,
                                    "response_type": getattr(ev.response_type, "name", str(ev.response_type)),
                                    "final_text_len": len(ev.text or ""),
                                    "final_text_preview": (ev.text or "")[:500],
                                }, ensure_ascii=False).encode("utf-8"),
                            )
                        for label, frame in self._event_to_sse(ev):
                            _dump_write(label, frame)
                            yield frame
                except asyncio.CancelledError:
                    # Engine kernel flushes the partial assistant row on cancel.
                    _dump_write("CANCELLED", b"client disconnected / asyncio.CancelledError")
                    raise
                finally:
                    if dump_fh is not None:
                        try:
                            dump_fh.write("\n=== END ===\n")
                            dump_fh.close()
                        except Exception:
                            pass

            return StreamingResponse(
                relay(),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "X-Accel-Buffering": "no",
                    "Connection": "keep-alive",
                },
            )

    @staticmethod
    def _event_to_sse(ev: GenerationEvent) -> List[Tuple[str, bytes]]:
        """Transcode one generation event into labelled SSE wire frames.

        Returns (debug_label, frame_bytes) pairs — usually one, but DoneEvent
        and ErrorEvent emit several (optional content delta, id-frame, [DONE]).
        Shared by the chat-completions stream and the confirm-resume stream so
        both speak the same wire protocol; the label feeds the optional SSE dump.
        """
        frames: List[Tuple[str, bytes]] = []
        if isinstance(ev, TokenEvent):
            chunk = json.dumps({
                "object": "chat.completion.chunk",
                "choices": [{"index": 0, "delta": {"content": ev.delta}}],
            })
            frames.append(("TokenEvent", f"data: {chunk}\n\n".encode("utf-8")))
        elif isinstance(ev, ToolCallStartEvent):
            payload_obj: Dict[str, Any] = {
                "tool_name": ev.tool_name,
                "arguments": ev.arguments,
                "call_id": ev.call_id,
            }
            if ev.group_id is not None:
                payload_obj["group_id"] = ev.group_id
            frames.append((
                "ToolCallStartEvent",
                f"event: derpr-tool-start\ndata: {json.dumps(payload_obj)}\n\n".encode("utf-8"),
            ))
        elif isinstance(ev, ToolCallResultEvent):
            payload_obj = {
                "call_id": ev.call_id,
                "tool_name": ev.tool_name,
                "result": ev.result,
                "error": ev.error,
            }
            if ev.group_id is not None:
                payload_obj["group_id"] = ev.group_id
            frames.append((
                "ToolCallResultEvent",
                f"event: derpr-tool-result\ndata: {json.dumps(payload_obj)}\n\n".encode("utf-8"),
            ))
        elif isinstance(ev, PendingConfirmationEvent):
            # A write parked for CONFIRM-mode approval. Carries the structured
            # calls + a resume token so the portal can render approve/deny and
            # POST back to /api/v1/persona/{name}/confirm. The terminal DoneEvent
            # that follows ends the stream without echoing this text into the
            # chat bubble — the modal is the canonical surface.
            payload = json.dumps({
                "text": ev.text,
                "persona": ev.persona_name,
                "token": ev.token,
                "calls": [
                    {
                        "name": c.get("name"),
                        "arguments": c.get("arguments", {}),
                        "id": c.get("id"),
                    }
                    for c in ev.write_calls
                ],
                "audit_info": ev.audit_info,
            })
            frames.append((
                "PendingConfirmationEvent",
                f"event: derpr-confirm\ndata: {payload}\n\n".encode("utf-8"),
            ))
        elif isinstance(ev, DoneEvent):
            # Dev-command responses carry their full text on the DoneEvent and
            # emit no TokenEvents (no LLM call), so unlike LLM turns the text was
            # never streamed. Transcode it as a content delta here or the portal
            # shows a blank reply even though the command ran + persisted.
            if ev.response_type == ResponseType.DEV_COMMAND and ev.text:
                cmd_chunk = json.dumps({
                    "object": "chat.completion.chunk",
                    "choices": [{"index": 0, "delta": {"content": ev.text}}],
                })
                frames.append(("DevCommandText", f"data: {cmd_chunk}\n\n".encode("utf-8")))
            # DP-130 history contract (C3): emit the `event: derpr` id-frame on
            # EVERY terminal turn — including parked writes (assistant_id=None),
            # tool-only, and aborts. Carrying user_id, assistant_id (null when
            # parked), response_type, and a stable ephemeral_chunk_id for the
            # unpersisted confirmation chunk means the client never has to advance
            # an id array positionally, so it can no longer drift vs the visible
            # story. The transcript endpoint is the authoritative re-sync source.
            rtype = getattr(ev.response_type, "name", str(ev.response_type))
            frame = (
                f"event: derpr\n"
                f"data: {json.dumps({'user_id': ev.user_interaction_id, 'assistant_id': ev.assistant_id, 'response_type': rtype, 'ephemeral_chunk_id': ev.ephemeral_chunk_id})}\n\n"
            )
            frames.append(("derpr-id-frame", frame.encode("utf-8")))
            frames.append(("DONE", b"data: [DONE]\n\n"))
        elif isinstance(ev, ErrorEvent):
            err = json.dumps({"error": {"message": ev.message}})
            frames.append(("ErrorEvent", f"data: {err}\n\n".encode("utf-8")))
            frames.append(("DONE", b"data: [DONE]\n\n"))
        return frames

    @staticmethod
    def _find_last_user_content(messages: List[Dict[str, Any]]) -> Optional[str]:
        """Scan messages in reverse for the last role=='user' entry with string content."""
        for msg in reversed(messages):
            if msg.get("role") != "user":
                continue
            content = msg.get("content")
            if isinstance(content, str) and content.strip():
                return content
            # OAI vision/tool payloads use content=[{type:'text', text:...}, ...]
            if isinstance(content, list):
                parts = [p.get("text", "") for p in content if isinstance(p, dict) and p.get("type") == "text"]
                joined = "".join(parts).strip()
                if joined:
                    return joined
        return None

    def _extract_last_user_turn(self, prompt: str) -> str:
        """Extract only the last user turn from a raw Kobold prompt string.

        Supports standard instruct templates (Alpaca, ChatML, Llama-3, etc.).
        Avoids nested or mangled wrappers by filtering out candidate tag matches.
        """
        if not prompt:
            return ""

        prompt_stripped = prompt.rstrip()

        # Standard candidate tags (user_tag, assistant_tag)
        candidates = [
            ("### Instruction:", "### Response:"),
            ("<|im_start|>user", "<|im_start|>assistant"),
            ("<|start_header_id|>user<|end_header_id|>", "<|start_header_id|>assistant<|end_header_id|>"),
            ("{{[INPUT]}}", "{{[OUTPUT]}}"),
            ("[INST]", "[/INST]"),
            ("USER:", "ASSISTANT:"),
            ("User:", "Assistant:"),
            ("Input:", "Output:"),
            ("<|user|>", "<|assistant|>"),
        ]

        all_tags = []
        for ut, at in candidates:
            all_tags.append(ut)
            all_tags.append(at)

        best_message = None
        best_assistant_idx = -1

        for user_tag, assistant_tag in candidates:
            idx_assistant = prompt_stripped.rfind(assistant_tag)
            if idx_assistant == -1:
                continue

            # Find the user_tag before this assistant_tag
            idx_user = prompt_stripped[:idx_assistant].rfind(user_tag)
            if idx_user == -1:
                continue

            candidate_message = prompt_stripped[idx_user + len(user_tag) : idx_assistant].strip()

            # Check if this candidate message contains any other tags (to avoid nested/mangled wrappers)
            has_other_tags = any(tag in candidate_message for tag in all_tags)

            if not has_other_tags:
                if idx_assistant > best_assistant_idx:
                    best_assistant_idx = idx_assistant
                    best_message = candidate_message

        if best_message is not None:
            return best_message

        # Fallback 1: No clean matching tags with assistant suffix, look for last user_tag with nothing after it
        max_user_idx = -1
        best_user_tag = None
        for user_tag, _ in candidates:
            idx_user = prompt_stripped.rfind(user_tag)
            if idx_user > max_user_idx:
                max_user_idx = idx_user
                best_user_tag = user_tag

        if best_user_tag is not None:
            candidate_message = prompt_stripped[max_user_idx + len(best_user_tag):].strip()
            if not any(tag in candidate_message for tag in all_tags):
                return candidate_message

        # Fallback 2: Retrieve the one with the highest assistant index even if it contains tags
        best_assistant_idx = -1
        best_message = None
        for user_tag, assistant_tag in candidates:
            idx_assistant = prompt_stripped.rfind(assistant_tag)
            if idx_assistant != -1 and idx_assistant > best_assistant_idx:
                idx_user = prompt_stripped[:idx_assistant].rfind(user_tag)
                if idx_user != -1:
                    best_assistant_idx = idx_assistant
                    best_message = prompt_stripped[idx_user + len(user_tag) : idx_assistant].strip()

        if best_message is not None:
            return best_message

        # Ultimate fallback: return the entire prompt (stripped)
        return prompt_stripped

    def _log_interaction(self, persona_name: str, role: str, content: str) -> Optional[int]:
        """Log an interaction synchronously and return its interaction_id.

        Synchronous call — MemoryManager.log_message is a fast SQLite insert
        under a thread lock. Return value lets callers thread reply_to_id.
        """
        if not content or not content.strip():
            return None
        try:
            res = self._memory_manager.log_message(
                user_identifier="portal",
                persona_name=persona_name,
                channel="web_ui",
                author_role=role,
                author_name=None,
                content=content,
                timestamp=datetime.now(timezone.utc),
            )
            return int(res) if res is not None else None
        except Exception as e:
            logger.error(f"Interaction logging failed (role={role}): {e}")
            return None

    def _commit_assistant(self, persona_name: str, content: str, user_interaction_id: Optional[int],
                          retry_assistant_id: Optional[int], reasoning_content: Optional[str] = None) -> Optional[int]:
        """Helper to append the full assistant stream into history."""
        if retry_assistant_id is not None:
            try:
                self._memory_manager.update_interaction_content(
                    retry_assistant_id, content, reasoning_content=reasoning_content
                )
                return retry_assistant_id
            except Exception as e:
                logger.error(f"Failed to patch assistant response for retry_id {retry_assistant_id}: {e}")
                return None
        else:
            try:
                res = self._memory_manager.log_message(
                    user_identifier="portal", persona_name=persona_name,
                    channel="web_ui", author_role='assistant',
                    author_name=persona_name, content=content,
                    timestamp=datetime.now(timezone.utc),
                    reply_to_id=user_interaction_id,
                    reasoning_content=reasoning_content
                )
                return int(res) if res is not None else None
            except Exception as e:
                logger.error(f"Assistant log failed: {e}")
                return None

    async def _forward_get(self, path: str, fallback: Dict[str, Any]) -> JSONResponse:
        url = f"{_kobold_base_url()}{path}"
        try:
            r = await self._http.get(url)
            return JSONResponse(status_code=r.status_code, content=r.json() if r.content else fallback)
        except Exception as e:
            logger.warning(f"Forward GET {path} failed: {e}; returning fallback")
            return JSONResponse(content=fallback)

    async def _forward_post(self, path: str, body: Dict[str, Any]) -> JSONResponse:
        url = f"{_kobold_base_url()}{path}"
        try:
            r = await self._http.post(url, json=body)
            return JSONResponse(status_code=r.status_code, content=r.json() if r.content else {})
        except Exception as e:
            logger.warning(f"Forward POST {path} failed: {e}")
            return JSONResponse(status_code=502, content={"error": str(e)})

    @staticmethod
    def _assembled_to_dict(assembled: "AssembledRequest") -> Dict[str, Any]:
        """Project an AssembledRequest into the API_CONTRACTS.md §9 wire shape.

        Flattens the resolved GenerationParams (universal fields + kobold sampler
        extras the route forwards) into the `params` block, and zips each wire
        message with its provenance `src` tag. `source: engine.dry_run` +
        `matches_live: true` because the assembler shares the live builder.
        """
        p = assembled.params
        flat: Dict[str, Any] = {
            "temperature": p.temperature,
            "top_p": p.top_p,
            "top_k": p.top_k,
            "max_tokens": p.max_tokens,
            "stop": p.stop_sequences or None,
            "seed": p.seed,
        }
        flat.update(p.get_provider_extras("kobold"))

        messages: List[Dict[str, Any]] = []
        for m, src in zip(assembled.messages, assembled.sources):
            content = m.get("content")
            if content is None and m.get("tool_calls"):
                # Replayed assistant tool-call rows carry no prose content.
                content = json.dumps(m["tool_calls"])
            messages.append({
                "role": m.get("role"),
                "content": content if content is not None else "",
                "src": src,
            })

        return {
            "parity": {
                "source": "engine.dry_run",
                "builder": "chat_system.stream_response",
                "matches_live": True,
            },
            "route": assembled.route,
            "model_name": assembled.model_name,
            "params": flat,
            "messages": messages,
        }

    def _get_current_persona_name(self) -> str:
        if self.active_persona and self.active_persona in self._personas:
            return self.active_persona
        default = getattr(global_config, "KOBOLD_DEFAULT_PERSONA", None)
        if default and default in self._personas:
            return str(default)
        return str(next(iter(self._personas.keys()), "assistant"))

    async def start(self) -> None:
        logger.info(f"Starting Kobold Engine Adapter on http://{self.host}:{self.port}")
        config = uvicorn.Config(self.app, host=self.host, port=self.port, log_level="warning")
        server = uvicorn.Server(config)
        try:
            await server.serve()
        finally:
            await self._http.aclose()


def create_kobold_engine_adapter(chat_system: ChatSystem) -> KoboldEngineAdapter:
    return KoboldEngineAdapter(chat_system)
