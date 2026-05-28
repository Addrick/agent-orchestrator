# src/interfaces/kobold_engine_adapter.py
#
# SECURITY: CodeQL py/stack-trace-exposure is suppressed for this file because
# the kobold web UI is internal-only and tracebacks in responses aid debugging.
# See memory/project/decisions/kobold_stack_trace_exposure.md for rationale.
# If this UI ever becomes externally accessible, re-enable the rule (un-dismiss
# the alerts in GitHub code scanning) and scrub tracebacks from all responses.

import json
import logging
import os
from typing import Any, AsyncIterator, Dict, Optional, List
from datetime import datetime, timezone

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, FileResponse, JSONResponse
import httpx
import uvicorn
import asyncio

from config import global_config
from src.chat_system import (
    ChatSystem, DoneEvent, ErrorEvent, TokenEvent,
    ToolCallResultEvent, ToolCallStartEvent,
)
from src.interfaces.kobold_export import build_kobold_savefile
from src.utils.model_utils import get_model_list
from src.utils.save_utils import save_personas_to_file
from src.interfaces._persona_patch import (
    _KNOWN_PATCH_KEYS_ENGINE as _KNOWN_PATCH_KEYS,
    _apply_kobold_sampler_extras,
    get_kobold_extras_for_get,
)

logger = logging.getLogger(__name__)


def _kobold_base_url() -> str:
    """KoboldCPP base URL without trailing /v1."""
    raw = os.environ.get("LOCAL_LLM_URL", global_config.LOCAL_LLM_URL).rstrip("/")
    if raw.endswith("/v1"):
        raw = raw[:-3]
    return raw


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

    def _setup_portal(self) -> None:
        portal_path = os.path.join(os.path.dirname(__file__), "web_assets", "portal.html")

        @self.app.get("/portal")
        async def get_portal() -> FileResponse:
            return FileResponse(portal_path)

        @self.app.get("/")
        async def root_redirect() -> FileResponse:
            return FileResponse(portal_path)

    def _setup_routes(self) -> None:
        @self.app.get("/api/v1/model")
        async def get_model() -> Any:
            return {"result": self._get_current_persona_name()}

        @self.app.put("/api/v1/model")
        async def set_model(request: Request) -> Any:
            data = await request.json()
            new_persona = data.get("model") or data.get("result")
            if new_persona in self.chat_system.personas:
                self.active_persona = new_persona
                logger.info(f"Switched active persona to: {new_persona}")
                return {"result": self.active_persona}
            return {"error": f"Persona '{new_persona}' not found", "available": list(self.chat_system.visible_personas().keys())}

        @self.app.get("/v1/models")
        async def list_models() -> Any:
            models = [
                {"id": name, "object": "model", "owned_by": "derpr", "permission": []}
                for name in self.chat_system.visible_personas().keys()
            ]
            return {"object": "list", "data": models}

        @self.app.get("/api/v1/tools/catalog")
        async def get_tools_catalog() -> Any:
            defs = self.chat_system.tool_manager.get_tool_definitions()
            tools = []
            for t in defs:
                func = t.get("function", {})
                caps = t.get("capabilities", {})
                tools.append({
                    "name": func.get("name"),
                    "description": func.get("description"),
                    "is_write": bool(t.get("is_write", False)),
                    "capabilities": {
                        "locality": caps.get("locality"),
                        "sensitivity": caps.get("sensitivity"),
                        "produces_untrusted": bool(caps.get("produces_untrusted", False)),
                    }
                })
            return {"tools": tools}

        @self.app.get("/api/v1/persona/{name}")
        async def get_persona(name: str) -> Any:
            if name not in self.chat_system.personas:
                return JSONResponse(status_code=404, content={"error": f"Persona '{name}' not found"})
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
                "history_messages": p.get_base_history_messages(),
                "thinking_level": p.get_thinking_level(),
                "memory_mode": p.get_memory_mode().name,
                "max_context_tokens": p.get_max_context_tokens(),
                "chat_template": p.get_chat_template(),
                "instruct_tags": p.get_provider_extra("kobold", "instruct_tags"),
                "kobold_extras": get_kobold_extras_for_get(p),
                "enabled_tools": p.get_enabled_tools(),
                "tool_policy": p.get_tool_policy().to_dict(),
            }

        @self.app.get("/api/v1/session/{persona}/ltm_block")
        async def ltm_block(persona: str, query: str = "") -> Any:
            """Retrieve an LTM memory block for the given persona and query text.

            Returns {"block": "<memory>...</memory>"} or {"block": null} when
            no relevant memories exist or LTM is disabled for the persona.
            Phase 2.2: called client-side before each submit when LTM is on;
            the block is written into kobold-lite's current_anote so kobold
            places it at its normal author's-note position in the prompt.
            """
            if persona not in self.chat_system.personas:
                return JSONResponse(status_code=404, content={"error": f"Persona '{persona}' not found"})
            block = await self.chat_system.get_session_memory_block(
                persona_name=persona,
                user_identifier="portal",
                channel="web_ui",
                server_id=None,
                query=query,
            )
            return {"block": block}

        @self.app.get("/api/v1/models/list")
        async def list_all_models() -> Dict[str, Any]:
            avail = get_model_list() or {}
            all_m = []
            for sub in avail.values():
                if isinstance(sub, list):
                    all_m.extend(sub)
                else:
                    all_m.append(sub)
            return {"models": sorted(list(set(all_m)))}

        @self.app.post("/api/v1/persona/{name}/reset")
        async def reset_persona_history(name: str) -> Any:
            if name not in self.chat_system.personas:
                return {"error": "Persona not found"}
            p = self.chat_system.personas[name]
            p.start_new_conversation()
            return {"result": f"History for {name} reset successfully"}

        @self.app.post("/api/v1/persona/{name}/dev_command")
        async def dev_command(name: str, request: Request) -> Any:
            if name not in self.chat_system.personas:
                return JSONResponse(status_code=404,
                                    content={"error": f"Persona '{name}' not found"})
            body = await request.json()
            command = body.get("command", "")
            try:
                result = await self.chat_system.bot_logic.preprocess_message(name, "portal", command)
            except Exception as e:
                return {"response": str(e), "mutated": False}
            if result is None:
                return JSONResponse(
                    status_code=400,
                    content={"response": "Not a dev command", "mutated": False},
                )
            if result.get("mutated"):
                save_personas_to_file(self.chat_system.personas)
            return {"response": result.get("response", ""), "mutated": bool(result.get("mutated"))}

        @self.app.get("/api/v1/session/{persona}/kobold_export")
        async def kobold_export(persona: str, max_turns: Optional[int] = None) -> Any:
            """Build a kobold-lite savefile from DERPR's global history for `persona`.

            Phase 2.1 always pulls global history (all channels) — the portal has
            no channel concept. max_turns defaults to the persona's configured
            sliding-window size (`get_base_history_messages`); no new config key.
            """
            if persona not in self.chat_system.personas:
                return JSONResponse(status_code=404, content={"error": f"Persona '{persona}' not found"})

            p = self.chat_system.personas[persona]
            limit = max_turns if isinstance(max_turns, int) and max_turns > 0 else p.get_base_history_messages()
            raw_history = await asyncio.to_thread(
                self.chat_system.memory_manager.get_global_history, persona, limit
            )
            savefile, skipped = build_kobold_savefile(raw_history)
            logger.warning(
                f"kobold_export persona={persona} limit={limit} "
                f"rows={len(raw_history)} actions={len(savefile.get('actions', []))} "
                f"ids={len(savefile.get('interaction_ids', []))} skipped={skipped}"
            )
            return JSONResponse(content=savefile)

        @self.app.get("/api/v1/interaction/{interaction_id}/versions")
        async def list_interaction_versions(interaction_id: int) -> Any:
            """List all stored versions for an interaction, canonical last.

            Portal hydrates `retry_prev_text` / `redo_prev_text` stacks from
            this after seeing `assistant_id` in the stream's derpr event.
            """
            try:
                versions = await asyncio.to_thread(
                    self.chat_system.memory_manager.list_interaction_versions,
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
                    self.chat_system.memory_manager.swap_interaction_version,
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
            versions = await asyncio.to_thread(
                self.chat_system.memory_manager.list_interaction_versions,
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
                    self.chat_system.memory_manager.update_interaction_content,
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
                    self.chat_system.memory_manager.suppress_interaction,
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
            if name not in self.chat_system.personas:
                return JSONResponse(status_code=404, content={"error": "Persona not found"})
            try:
                data = await request.json()
            except Exception as e:
                return JSONResponse(status_code=400, content={"error": f"invalid JSON: {e}"})
            p = self.chat_system.personas[name]

            # Numeric setters silently coerce bad input to None / defaults and
            # return the resolved value. Capture rejections so the portal can
            # surface them instead of pretending the save was clean.
            rejected: List[str] = []

            if "prompt" in data: p.set_prompt(data["prompt"])
            if "model_name" in data: p.set_model_name(data["model_name"])
            if "temperature" in data and p.set_temperature(data["temperature"]) is None and data["temperature"] is not None:
                rejected.append("temperature")
            if "top_p" in data and p.set_top_p(data["top_p"]) is None and data["top_p"] is not None:
                rejected.append("top_p")
            if "top_k" in data and p.set_top_k(data["top_k"]) is None and data["top_k"] is not None:
                rejected.append("top_k")
            if "max_tokens" in data:
                p.set_response_token_limit(data["max_tokens"])
            if "history_messages" in data:
                p.set_history_messages(data["history_messages"])
            elif "context_length" in data:
                p.set_history_messages(data["context_length"])
            if "memory_mode" in data:
                before = p.get_memory_mode()
                p.set_memory_mode(data["memory_mode"])
                if p.get_memory_mode() == before and data["memory_mode"] not in (None, before.name):
                    rejected.append("memory_mode")
            if "max_context_tokens" in data:
                p.set_max_context_tokens(data["max_context_tokens"])
            if "chat_template" in data:
                p.set_chat_template(data["chat_template"])
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
                save_personas_to_file(self.chat_system.personas)
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
                user_interaction_id = self._log_interaction(persona_name, "user", prompt)
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
                user_interaction_id = self._log_interaction(persona_name, "user", prompt)

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
                async for ev in self.chat_system.stream_response(
                    persona_name=persona_name,
                    user_identifier="portal",
                    channel="web_ui",
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
                    async for ev in self.chat_system.stream_response(
                        persona_name=persona_name,
                        user_identifier="portal",
                        channel="web_ui",
                        message=user_text,
                        is_retry=is_retry,
                        local_inference_config=local_inference_config,
                        client_messages=None,  # Explicitly None to force DB history rebuild
                    ):

                        if await request.is_disconnected():
                            return
                        if isinstance(ev, TokenEvent):
                            chunk = json.dumps({
                                "object": "chat.completion.chunk",
                                "choices": [{
                                    "index": 0,
                                    "delta": {"content": ev.delta},
                                }],
                            })
                            out = f"data: {chunk}\n\n".encode("utf-8")
                            _dump_write("TokenEvent", out)
                            yield out
                        elif isinstance(ev, ToolCallStartEvent):
                            payload_obj: Dict[str, Any] = {
                                "tool_name": ev.tool_name,
                                "arguments": ev.arguments,
                                "call_id": ev.call_id,
                            }
                            if ev.group_id is not None:
                                payload_obj["group_id"] = ev.group_id
                            payload = json.dumps(payload_obj)
                            out = f"event: derpr-tool-start\ndata: {payload}\n\n".encode("utf-8")
                            _dump_write("ToolCallStartEvent", out)
                            yield out
                        elif isinstance(ev, ToolCallResultEvent):
                            payload_obj = {
                                "call_id": ev.call_id,
                                "tool_name": ev.tool_name,
                                "result": ev.result,
                                "error": ev.error,
                            }
                            if ev.group_id is not None:
                                payload_obj["group_id"] = ev.group_id
                            payload = json.dumps(payload_obj)
                            out = f"event: derpr-tool-result\ndata: {payload}\n\n".encode("utf-8")
                            _dump_write("ToolCallResultEvent", out)
                            yield out
                        elif isinstance(ev, DoneEvent):
                            if dump_fh is not None:
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
                            if ev.assistant_id is not None:
                                frame = (
                                    f"event: derpr\n"
                                    f"data: {json.dumps({'assistant_id': ev.assistant_id, 'user_id': ev.user_interaction_id})}\n\n"
                                )
                                out = frame.encode("utf-8")
                                _dump_write("derpr-id-frame", out)
                                yield out
                            out = b"data: [DONE]\n\n"
                            _dump_write("DONE", out)
                            yield out
                        elif isinstance(ev, ErrorEvent):
                            err = json.dumps({"error": {"message": ev.message}})
                            out = f"data: {err}\n\n".encode("utf-8")
                            _dump_write("ErrorEvent", out)
                            yield out
                            yield b"data: [DONE]\n\n"
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

    def _log_interaction(self, persona_name: str, role: str, content: str) -> Optional[int]:
        """Log an interaction synchronously and return its interaction_id.

        Synchronous call — MemoryManager.log_message is a fast SQLite insert
        under a thread lock. Return value lets callers thread reply_to_id.
        """
        if not content or not content.strip():
            return None
        try:
            res = self.chat_system.memory_manager.log_message(
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
                self.chat_system.memory_manager.update_interaction_content(
                    retry_assistant_id, content, reasoning_content=reasoning_content
                )
                return retry_assistant_id
            except Exception as e:
                logger.error(f"Failed to patch assistant response for retry_id {retry_assistant_id}: {e}")
                return None
        else:
            try:
                res = self.chat_system.memory_manager.log_message(
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

    def _get_current_persona_name(self) -> str:
        if self.active_persona and self.active_persona in self.chat_system.personas:
            return self.active_persona
        default = getattr(global_config, "KOBOLD_DEFAULT_PERSONA", None)
        if default and default in self.chat_system.personas:
            return str(default)
        return str(next(iter(self.chat_system.personas.keys()), "assistant"))

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
