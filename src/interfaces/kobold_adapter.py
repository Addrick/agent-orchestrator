# src/interfaces/kobold_adapter.py

import json
import logging
import os
from typing import Any, AsyncIterator, Dict, Optional

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, FileResponse
import uvicorn
import asyncio

from config import global_config
from src.chat_system import ChatSystem
from src.utils.model_utils import get_model_list
from src.utils.save_utils import save_personas_to_file

logger = logging.getLogger(__name__)


# Known markers kobold-lite uses to delimit user turns in its flat prompt.
# We only use these to locate the *last* user message in the prompt we receive;
# rendering of outgoing prompts is owned by the persona's chat_template.
_USER_MARKERS = (
    "\n### Instruction:\n", "\nUser:", "\nHuman:", "\nInput:",
    "<|im_start|>user\n", "<start_of_turn>user\n", "<|turn>user\n",
    "<|user|>\n", "<|start_of_role|>user<|end_of_role|>",
    "<|im_user|>user<|im_middle|>", "<\uff5cUser\uff5c>",
    "<|START_OF_TURN_TOKEN|><|USER_TOKEN|>",
    "<|start_header_id|>user<|end_header_id|>",
    "<|header_start|>user<|header_end|>", "[INST]", "{{[INPUT]}}",
)
_BOUNDARY_MARKERS = (
    "\n### Response:", "\nAssistant:", "\nAI:", "\nBot:", "\nASSISTANT:",
    "<|im_start|>assistant", "<start_of_turn>model", "<end_of_turn>",
    "<|turn>model", "<turn|>", "<|assistant|>", "<|im_assistant|>assistant",
    "<\uff5cAssistant\uff5c>", "<|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|>",
    "<|start_header_id|>assistant<|end_header_id|>",
    "<|header_start|>assistant<|header_end|>",
    "<|start_of_role|>assistant<|end_of_role|>", "[/INST]", "{{[OUTPUT]}}",
    "{{[INPUT_END]}}", "<|im_end|>", "<\uff5cend\u2581of\u2581sentence\uff5c>",
    "<|end_of_text|>",
)


def _extract_user_parts(prompt: str) -> Dict[str, str]:
    """Pulls the last user turn and any trailing assistant prefill out of
    kobold-lite's flat prompt blob.

    Returns {"message", "assistant_prefill"}.

    - `message`: the newest user turn's text. The persona's chat_template
      renders this downstream — we do not propagate the user-side markers.
    - `assistant_prefill`: any content lite placed after the *final* assistant
      marker. Lite uses this to signal reasoning mode — e.g. it ends the
      prompt with `<|channel>thought\\n` so the model continues directly into
      a thought block. Without forwarding this, the model generates its own
      opener and lite fails to recognize the boundary, so the thinking tags
      leak into visible output.
    """
    last_pos = -1
    user_marker = ""
    for m in _USER_MARKERS:
        pos = prompt.rfind(m)
        if pos > last_pos:
            last_pos = pos
            user_marker = m

    if last_pos == -1:
        return {"message": prompt.strip()[-1000:], "assistant_prefill": ""}

    segment = prompt[last_pos + len(user_marker):]
    cut = len(segment)
    boundary_end = len(segment)
    for m in _BOUNDARY_MARKERS:
        p = segment.find(m)
        if 0 <= p < cut:
            cut = p
            boundary_end = p + len(m)
    message = segment[:cut].strip() or prompt.strip()[-1000:]

    # Prefill = anything after the final boundary/assistant marker in the tail.
    tail = segment[boundary_end:]
    last_marker_end = 0
    for m in _BOUNDARY_MARKERS:
        p = tail.rfind(m)
        if p != -1 and p + len(m) > last_marker_end:
            last_marker_end = p + len(m)
    prefill = tail[last_marker_end:].strip() if last_marker_end else ""

    return {"message": message, "assistant_prefill": prefill}


def _extract_params(data: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize kobold-lite's wire format.

    The kobold-lite frontend flattens `params` into the top-level body before
    POSTing to `custom_kobold_endpoint + /api/extra/generate/stream`
    (see `submit_payload = submit_payload.params` in the frontend). So the
    request body is `{prompt, temperature, history_override, ...}`, not
    `{prompt, params: {...}}`. Accept either shape: nested `params` takes
    precedence; otherwise treat the whole body minus known envelope keys as
    the params dict.
    """
    nested = data.get("params")
    if isinstance(nested, dict) and nested:
        return nested
    envelope_keys = {"prompt", "model", "models", "workers"}
    return {k: v for k, v in data.items() if k not in envelope_keys}


def _build_inference_config(params: Dict[str, Any], assistant_prefill: str = "") -> Dict[str, Any]:
    """Flatten kobold-lite's params payload into the fields stream_engine uses.

    Sampling params, context cap, stop sequences, a verbatim assistant prefill
    (for reasoning-mode triggers), and the `history_override` flag are
    forwarded. Template markers are *not* propagated — the persona's
    chat_template owns prompt rendering.
    """
    cfg: Dict[str, Any] = {
        "stop_sequence": params.get("stop_sequence") or [],
        "max_context_length": params.get("max_context_length"),
        "assistant_prefill": assistant_prefill,
        "history_override": bool(params.get("history_override")),
    }
    for p in ("temperature", "top_p", "top_k", "rep_pen", "rep_pen_range",
              "rep_pen_slope", "min_p", "typical", "tfs"):
        if params.get(p) is not None:
            cfg[p] = params[p]
    # Keep stop_sequence + assistant_prefill + history_override even when empty/false; drop other None.
    return {k: v for k, v in cfg.items()
            if v is not None or k in ("stop_sequence", "assistant_prefill", "history_override")}


class KoboldAdapter:
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

        self._active_tasks: Dict[str, asyncio.Task] = {}
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
                "thinking_level": p.get_thinking_level()
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
            return {"version": "1.70", "lib_version": "1.70"}

        @self.app.get("/api/extra/version")
        async def get_extra_version():
            # Must be >= 1.40 so kobold-lite enables SSE streaming.
            return {"version": "1.70", "platform": "DERPR"}

        @self.app.get("/api/v1/config/soft_prompts")
        async def get_soft_prompts():
            return {"results": []}

        @self.app.get("/api/v1/config/max_context_length")
        async def get_max_context_length():
            # Report an advertised token budget for kobold-lite's UI slider.
            # Must NOT be persona.context_length — that's a turn-count for the
            # history window, not a token budget. The actual per-request cap is
            # whatever lite sends back in params.max_context_length.
            return {"result": 8192}

        @self.app.post("/api/v1/abort")
        async def abort_generation(request: Request):
            # Try to identify user by hostname or fixed ID
            # For simplicity in local portal, we'll use a single 'web_user' key
            # unless we start passing user identifiers from the frontend.
            user_id = "web_user"
            task = self._active_tasks.get(user_id)
            if task and not task.done():
                task.cancel()
                logger.info(f"Aborted generation for {user_id}")
                return {"result": "aborted"}
            return {"result": "not_running"}

        @self.app.post("/api/v1/generate")
        async def generate(request: Request, persona: Optional[str] = None):
            data = await request.json()
            prompt = data.get("prompt", "")
            params = _extract_params(data)
            target_persona = persona or data.get("model") or self._get_current_persona_name()
            if target_persona not in self.chat_system.personas:
                target_persona = self._get_current_persona_name()

            parts = _extract_user_parts(prompt)
            user_msg = parts["message"]
            assistant_prefill = parts["assistant_prefill"]

            inference_config = _build_inference_config(params, assistant_prefill)

            logger.info(
                f"Kobold sync request for persona '{target_persona}'. "
                f"history_override={inference_config.get('history_override')}, "
                f"Extracted user msg ({len(user_msg)} chars), "
                f"prefill={assistant_prefill!r}."
            )

            final_text = ""
            async for ev in self.chat_system.stream_response(
                persona_name=target_persona,
                user_identifier="web_user",
                channel="web_interface",
                message=user_msg,
                user_display_name="WebUser",
                local_inference_config=inference_config,
            ):
                if ev.get("type") == "done":
                    final_text = ev.get("full_text", final_text)
                elif ev.get("type") == "error":
                    final_text = ev.get("text", "")

            return {"results": [{"text": final_text}]}

        @self.app.post("/api/extra/generate/stream")
        async def generate_stream(request: Request, persona: Optional[str] = None):
            data = await request.json()
            prompt = data.get("prompt", "")
            params = _extract_params(data)
            target_persona = persona or data.get("model") or self._get_current_persona_name()
            if target_persona not in self.chat_system.personas:
                target_persona = self._get_current_persona_name()

            parts = _extract_user_parts(prompt)
            user_msg = parts["message"]
            assistant_prefill = parts["assistant_prefill"]

            inference_config = _build_inference_config(params, assistant_prefill)

            logger.info(
                f"Kobold stream request for persona '{target_persona}'. "
                f"history_override={inference_config.get('history_override')}, "
                f"Extracted user msg ({len(user_msg)} chars), "
                f"prefill={assistant_prefill!r}."
            )

            async def event_source() -> AsyncIterator[bytes]:
                token_count = 0
                try:
                    async for ev in self.chat_system.stream_response(
                        persona_name=target_persona,
                        user_identifier="web_user",
                        channel="web_interface",
                        message=user_msg,
                        user_display_name="WebUser",
                        local_inference_config=inference_config,
                    ):
                        if await request.is_disconnected():
                            return

                        etype = ev.get("type")
                        if etype == "token":
                            token_count += 1
                            payload = json.dumps({"token": ev.get("text", "")})
                            yield f"event: message\ndata: {payload}\n\n".encode("utf-8")
                        elif etype == "error":
                            payload = json.dumps({
                                "token": f"\n[Error] {ev.get('text', '')}",
                                "finish_reason": ev.get("finish_reason", "error"),
                            })
                            yield f"event: message\ndata: {payload}\n\n".encode("utf-8")
                        elif etype == "done":
                            payload = json.dumps({
                                "token": "",
                                "finish_reason": ev.get("finish_reason", "stop"),
                            })
                            yield f"event: message\ndata: {payload}\n\n".encode("utf-8")
                            return
                except asyncio.CancelledError:
                    raise

            user_id = "web_user"

            # Clear any old task
            if user_id in self._active_tasks and not self._active_tasks[user_id].done():
                self._active_tasks[user_id].cancel()

            async def wrapped_event_source() -> AsyncIterator[bytes]:
                try:
                    async for chunk in event_source():
                        yield chunk
                finally:
                    if user_id in self._active_tasks and self._active_tasks[user_id] == asyncio.current_task():
                        del self._active_tasks[user_id]

            self._active_tasks[user_id] = asyncio.current_task()

            return StreamingResponse(
                wrapped_event_source(),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "X-Accel-Buffering": "no",
                    "Connection": "keep-alive",
                },
            )

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
        await server.serve()


def create_kobold_adapter(chat_system: ChatSystem) -> KoboldAdapter:
    return KoboldAdapter(chat_system)
