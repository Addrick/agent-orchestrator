# src/interfaces/kobold_adapter.py

import json
import logging
import os
from typing import Any, AsyncIterator, Dict, Optional

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, FileResponse, JSONResponse
import httpx
import uvicorn
import asyncio

from config import global_config
from src.chat_system import ChatSystem
from src.interfaces.kobold_export import build_kobold_savefile
from src.utils.model_utils import get_model_list
from src.utils.save_utils import save_personas_to_file

logger = logging.getLogger(__name__)


def _kobold_base_url() -> str:
    """KoboldCPP base URL without trailing /v1."""
    raw = os.environ.get("LOCAL_LLM_URL", global_config.LOCAL_LLM_URL).rstrip("/")
    if raw.endswith("/v1"):
        raw = raw[:-3]
    return raw


class KoboldAdapter:
    """Verbatim-passthrough adapter between kobold-lite and local KoboldCPP.

    Stage 1 scope: forward kobold-lite's rendered prompt + params directly to
    KoboldCPP, relay SSE back unchanged. Persona sampling defaults are pushed
    into lite's UI sliders by the frontend on persona switch, so server-side
    merging is not needed. History Override / DB-driven prompt rebuild is
    deferred pending the local-model tag-schema system.
    See memory/project/decisions/2026-04-19-kobold-portal-passthrough.md.
    """

    def __init__(self, chat_system: ChatSystem, host: str = "0.0.0.0", port: int = 5002):
        self.chat_system = chat_system
        self.host = host
        self.port = port
        self.active_persona: Optional[str] = None
        self.app = FastAPI(title="DERPR Kobold Adapter")

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

    def _setup_portal(self):
        portal_path = os.path.join(os.path.dirname(__file__), "web_assets", "portal.html")

        @self.app.get("/portal")
        async def get_portal():
            return FileResponse(portal_path)

        @self.app.get("/")
        async def root_redirect():
            return FileResponse(portal_path)

    def _setup_routes(self):
        @self.app.get("/api/v1/model")
        async def get_model():
            return {"result": self._get_current_persona_name()}

        @self.app.put("/api/v1/model")
        async def set_model(request: Request):
            data = await request.json()
            new_persona = data.get("model") or data.get("result")
            if new_persona in self.chat_system.personas:
                self.active_persona = new_persona
                logger.info(f"Switched active persona to: {new_persona}")
                return {"result": self.active_persona}
            return {"error": f"Persona '{new_persona}' not found", "available": list(self.chat_system.personas.keys())}

        @self.app.get("/v1/models")
        async def list_models():
            models = [
                {"id": name, "object": "model", "owned_by": "derpr", "permission": []}
                for name in self.chat_system.personas.keys()
            ]
            return {"object": "list", "data": models}

        @self.app.get("/api/v1/persona/{name}")
        async def get_persona_detail(name: str):
            if name not in self.chat_system.personas:
                return {"error": f"Persona '{name}' not found"}
            p = self.chat_system.personas[name]
            return {
                "name": p.get_name(),
                "display_name": p.get_name().title(),
                "prompt": p.get_prompt(),
                "model_name": p.get_model_name(),
                "temperature": p.get_temperature(),
                "top_p": p.get_top_p(),
                "top_k": p.get_top_k(),
                "max_tokens": p.get_response_token_limit(),
                "context_length": p.get_base_context_length(),
                "thinking_level": p.get_thinking_level(),
            }

        @self.app.get("/api/v1/models/list")
        async def list_all_models():
            avail = get_model_list() or {}
            all_m = []
            for sub in avail.values():
                if isinstance(sub, list):
                    all_m.extend(sub)
                else:
                    all_m.append(sub)
            return {"models": sorted(list(set(all_m)))}

        @self.app.post("/api/v1/persona/{name}/reset")
        async def reset_persona_context(name: str):
            if name not in self.chat_system.personas:
                return {"error": "Persona not found"}
            p = self.chat_system.personas[name]
            p.start_new_conversation()
            return {"result": f"Context for {name} reset sync successfully"}

        @self.app.get("/api/v1/session/{persona}/kobold_export")
        async def kobold_export(persona: str, max_turns: Optional[int] = None):
            """Build a kobold-lite savefile from DERPR's global history for `persona`.

            Phase 2.1 always pulls global history (all channels) — the portal has
            no channel concept. max_turns defaults to the persona's configured
            sliding-window size (`get_base_context_length`); no new config key.
            """
            if persona not in self.chat_system.personas:
                return JSONResponse(status_code=404, content={"error": f"Persona '{persona}' not found"})

            p = self.chat_system.personas[persona]
            limit = max_turns if isinstance(max_turns, int) and max_turns > 0 else p.get_base_context_length()
            raw_history = await asyncio.to_thread(
                self.chat_system.memory_manager.get_global_history, persona, limit
            )
            savefile, skipped = build_kobold_savefile(raw_history)
            logger.info(
                f"kobold_export persona={persona} limit={limit} "
                f"rows={len(raw_history)} skipped={skipped}"
            )
            return JSONResponse(content=savefile)

        @self.app.patch("/api/v1/persona/{name}")
        async def update_persona(name: str, request: Request):
            if name not in self.chat_system.personas:
                return {"error": "Persona not found"}
            data = await request.json()
            p = self.chat_system.personas[name]

            if "prompt" in data: p.set_prompt(data["prompt"])
            if "model_name" in data: p.set_model_name(data["model_name"])
            if "temperature" in data: p.set_temperature(data["temperature"])
            if "top_p" in data: p.set_top_p(data["top_p"])
            if "top_k" in data: p.set_top_k(data["top_k"])
            if "max_tokens" in data: p.set_response_token_limit(data["max_tokens"])
            if "context_length" in data: p.set_context_length(data["context_length"])

            save_personas_to_file(self.chat_system.personas)
            logger.info(f"Updated and saved persona settings for {name}")
            return {"result": "success"}

        @self.app.get("/api/v1/info/version")
        async def get_info_version():
            return await self._forward_get("/api/v1/info/version", {"version": "1.70", "lib_version": "1.70"})

        @self.app.get("/api/extra/version")
        async def get_extra_version():
            # Forward verbatim so portal can detect KCPP version + jinja/mcp/etc.
            # Fallback only on upstream failure. Without real version portal
            # falls back to legacy prompt-field format and instruct tags break.
            return await self._forward_get("/api/extra/version", {"version": "1.70", "platform": "DERPR"})

        @self.app.get("/api/v1/config/soft_prompts")
        async def get_soft_prompts():
            return await self._forward_get("/api/v1/config/soft_prompts", {"results": []})

        @self.app.get("/api/v1/config/max_context_length")
        async def get_max_context_length():
            return await self._forward_get("/api/v1/config/max_context_length", {"result": 8192})

        @self.app.get("/api/extra/true_max_context_length")
        async def get_true_max_ctx():
            return await self._forward_get("/api/extra/true_max_context_length", {"value": 8192})

        @self.app.get("/api/extra/perf")
        async def get_perf():
            return await self._forward_get("/api/extra/perf", {})

        @self.app.post("/api/extra/tokencount")
        async def tokencount(request: Request):
            return await self._forward_post("/api/extra/tokencount", await request.json())

        @self.app.get("/api/extra/generate/check")
        @self.app.post("/api/extra/generate/check")
        async def generate_check(request: Request):
            body = await request.json() if request.method == "POST" else {}
            return await self._forward_post("/api/extra/generate/check", body) if request.method == "POST" \
                else await self._forward_get("/api/extra/generate/check", {})

        @self.app.post("/api/v1/abort")
        @self.app.post("/api/extra/abort")
        async def abort_generation():
            url = f"{_kobold_base_url()}/api/extra/abort"
            try:
                r = await self._http.post(url, json={})
                return JSONResponse(r.json() if r.content else {"result": "aborted"})
            except Exception as e:
                logger.warning(f"Abort forward failed: {e}")
                return {"result": "abort_failed", "error": str(e)}

        @self.app.post("/api/v1/generate")
        async def generate(request: Request):
            data = await request.json()
            body = self._strip_envelope(data)
            url = f"{_kobold_base_url()}/api/v1/generate"
            logger.info(f"Kobold passthrough sync -> {url}")
            try:
                r = await self._http.post(url, json=body)
                return JSONResponse(status_code=r.status_code, content=r.json())
            except httpx.RequestError as e:
                logger.error(f"Upstream sync generate failed: {e}")
                return JSONResponse(status_code=502, content={"error": str(e)})

        @self.app.post("/chat/completions")
        @self.app.post("/v1/chat/completions")
        async def oai_chat_completions(request: Request):
            data = await request.json()
            body = self._strip_envelope(data)
            url = f"{_kobold_base_url()}/v1/chat/completions"
            is_stream = bool(body.get("stream"))
            logger.info(
                f"OAI chat passthrough -> {url} "
                f"(stream={is_stream}, msgs={len(body.get('messages', []))})"
            )

            if not is_stream:
                try:
                    r = await self._http.post(url, json=body)
                    return JSONResponse(status_code=r.status_code, content=r.json() if r.content else {})
                except httpx.RequestError as e:
                    logger.error(f"OAI sync upstream failed: {e}")
                    return JSONResponse(status_code=502, content={"error": str(e)})

            async def relay() -> AsyncIterator[bytes]:
                try:
                    async with self._http.stream("POST", url, json=body) as upstream:
                        async for chunk in upstream.aiter_raw():
                            if await request.is_disconnected():
                                return
                            if chunk:
                                yield chunk
                except httpx.RequestError as e:
                    logger.error(f"OAI stream upstream failed: {e}")
                    err = json.dumps({"error": {"message": str(e)}})
                    yield f"data: {err}\n\ndata: [DONE]\n\n".encode("utf-8")
                except asyncio.CancelledError:
                    try:
                        await self._http.post(f"{_kobold_base_url()}/api/extra/abort", json={})
                    except Exception:
                        pass
                    raise

            return StreamingResponse(
                relay(),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "X-Accel-Buffering": "no",
                    "Connection": "keep-alive",
                },
            )

        @self.app.post("/api/extra/generate/stream")
        async def generate_stream(request: Request):
            data = await request.json()
            body = self._strip_envelope(data)

            # Stub for deferred History Override mode. Toggle currently hidden
            # in UI; if a client still sends it, log and fall through to
            # passthrough. Real impl lands with the tag-schema system.
            if data.get("params", {}).get("history_override") or data.get("history_override"):
                logger.warning(
                    "history_override=true received but Override mode is "
                    "stubbed — forwarding as passthrough."
                )

            url = f"{_kobold_base_url()}/api/extra/generate/stream"
            logger.info(
                f"Kobold passthrough stream -> {url} "
                f"(prompt_chars={len(body.get('prompt', ''))})"
            )

            async def relay() -> AsyncIterator[bytes]:
                try:
                    async with self._http.stream("POST", url, json=body) as upstream:
                        async for chunk in upstream.aiter_raw():
                            if await request.is_disconnected():
                                return
                            if chunk:
                                yield chunk
                except httpx.RequestError as e:
                    logger.error(f"Upstream stream failed: {e}")
                    err = json.dumps({"token": f"\n[Upstream error] {e}", "finish_reason": "error"})
                    yield f"event: message\ndata: {err}\n\n".encode("utf-8")
                except asyncio.CancelledError:
                    # Client disconnected; forward abort upstream.
                    try:
                        await self._http.post(f"{_kobold_base_url()}/api/extra/abort", json={})
                    except Exception:
                        pass
                    raise

            return StreamingResponse(
                relay(),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "X-Accel-Buffering": "no",
                    "Connection": "keep-alive",
                },
            )

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
    def _strip_envelope(data: Dict[str, Any]) -> Dict[str, Any]:
        """Remove DERPR-only routing fields before forwarding to KoboldCPP."""
        out = dict(data)
        out.pop("model", None)  # our persona selector, not kobold's
        # history_override is a DERPR-only flag; KoboldCPP would ignore it
        # but drop defensively in case stricter upstream versions reject unknowns.
        if isinstance(out.get("params"), dict):
            out["params"] = {k: v for k, v in out["params"].items() if k != "history_override"}
        out.pop("history_override", None)
        return out

    def _get_current_persona_name(self) -> str:
        if self.active_persona and self.active_persona in self.chat_system.personas:
            return self.active_persona
        default = getattr(global_config, "KOBOLD_DEFAULT_PERSONA", None)
        if default and default in self.chat_system.personas:
            return default
        return next(iter(self.chat_system.personas.keys()), "assistant")

    async def start(self):
        logger.info(f"Starting Kobold Adapter on http://{self.host}:{self.port}")
        config = uvicorn.Config(self.app, host=self.host, port=self.port, log_level="warning")
        server = uvicorn.Server(config)
        try:
            await server.serve()
        finally:
            await self._http.aclose()


def create_kobold_adapter(chat_system: ChatSystem) -> KoboldAdapter:
    return KoboldAdapter(chat_system)
