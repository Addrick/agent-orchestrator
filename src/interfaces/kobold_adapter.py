# src/interfaces/kobold_adapter.py

import json
import logging
from typing import AsyncIterator, Optional
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, FileResponse
import uvicorn
import asyncio

from src.chat_system import ChatSystem, ResponseType

logger = logging.getLogger(__name__)

class KoboldAdapter:
    def __init__(self, chat_system: ChatSystem, host: str = "0.0.0.0", port: int = 5002):
        self.chat_system = chat_system
        self.host = host
        self.port = port
        self.active_persona: Optional[str] = None
        self.app = FastAPI(title="DERPR Kobold Adapter")
        
        # Setup CORS to allow any origin (required for lite.koboldai.net)
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        
        self._setup_routes()
        self._setup_portal()

    def _setup_portal(self):
        import os
        portal_path = os.path.join(os.path.dirname(__file__), "web_assets", "portal.html")
        
        @self.app.get("/portal")
        async def get_portal():
            """Serves the Antigravity Agent Portal UI."""
            return FileResponse(portal_path)

        @self.app.get("/")
        async def root_redirect():
            """Redirect root to portal."""
            return FileResponse(portal_path)

    def _setup_routes(self):
        @self.app.get("/api/v1/model")
        async def get_model():
            """Returns the current active persona name."""
            return {"result": self._get_current_persona_name()}

        @self.app.put("/api/v1/model")
        async def set_model(request: Request):
            """Sets the active persona."""
            data = await request.json()
            new_persona = data.get("model") or data.get("result")
            if new_persona in self.chat_system.personas:
                self.active_persona = new_persona
                logger.info(f"Switched active persona to: {new_persona}")
                return {"result": self.active_persona}
            return {"error": f"Persona '{new_persona}' not found", "available": list(self.chat_system.personas.keys())}

        @self.app.get("/v1/models")
        async def list_models():
            """OpenAI-compatible models list (returns all personas)."""
            models = []
            for name in self.chat_system.personas.keys():
                models.append({
                    "id": name,
                    "object": "model",
                    "owned_by": "derpr",
                    "permission": []
                })
            return {"object": "list", "data": models}

        @self.app.get("/api/v1/info/version")
        async def get_info_version():
            """Returns a mock version to satisfy connection checks."""
            return {"version": "1.70", "lib_version": "1.70"}

        @self.app.get("/api/extra/version")
        async def get_extra_version():
            """Mock KoboldCPP version. Must be >= 1.40 so lite enables SSE."""
            return {"version": "1.70", "platform": "DERPR"}

        @self.app.get("/api/v1/config/soft_prompts")
        async def get_soft_prompts():
            return {"results": []}

        @self.app.get("/api/v1/config/max_context_length")
        async def get_max_context_length():
            """Returns the context length of the active persona."""
            persona_name = self._get_current_persona_name()
            persona = self.chat_system.personas.get(persona_name)
            limit = persona.get_context_length() if persona else 8192
            return {"result": limit}

        @self.app.post("/api/v1/generate")
        async def generate(request: Request, persona: Optional[str] = None):
            """
            Kobold-Lite sync endpoint. Routes through ChatSystem.stream_response.
            """
            data = await request.json()
            prompt = data.get("prompt", "")
            params = data.get("params", {})
            target_persona = persona or data.get("model") or self._get_current_persona_name()
            if target_persona not in self.chat_system.personas:
                target_persona = self._get_current_persona_name()

            msg_data = self._extract_user_message_with_markers(prompt)
            user_msg = msg_data["message"]
            
            # Detect inference settings to preserve parity
            inference_config = {
                "user_marker": msg_data["user_marker"],
                "assistant_marker": msg_data["assistant_marker"],
                "thinking_trigger": msg_data["thinking_trigger"],
                "stop_sequence": params.get("stop_sequence") or [],
                "temperature": params.get("temperature"),
                "top_p": params.get("top_p"),
                "top_k": params.get("top_k"),
                "rep_pen": params.get("rep_pen"),
                "rep_pen_range": params.get("rep_pen_range"),
                "rep_pen_slope": params.get("rep_pen_slope"),
                "min_p": params.get("min_p"),
                "typical": params.get("typical"),
                "tfs": params.get("tfs"),
            }

            logger.info(f"Kobold API received prompt for persona '{target_persona}'. "
                        f"Detected User Marker: {msg_data['user_marker']!r}, "
                        f"Assistant Marker: {msg_data['assistant_marker']!r}")

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
            """SSE token streaming endpoint."""
            data = await request.json()
            prompt = data.get("prompt", "")
            params = data.get("params", {})
            target_persona = persona or data.get("model") or self._get_current_persona_name()
            if target_persona not in self.chat_system.personas:
                target_persona = self._get_current_persona_name()

            msg_data = self._extract_user_message_with_markers(prompt)
            user_msg = msg_data["message"]

            inference_config = {
                "user_marker": msg_data["user_marker"],
                "assistant_marker": msg_data["assistant_marker"],
                "thinking_trigger": msg_data["thinking_trigger"],
                "stop_sequence": params.get("stop_sequence") or [],
                "temperature": params.get("temperature"),
                "top_p": params.get("top_p"),
                "top_k": params.get("top_k"),
                "rep_pen": params.get("rep_pen"),
                "rep_pen_range": params.get("rep_pen_range"),
                "rep_pen_slope": params.get("rep_pen_slope"),
                "min_p": params.get("min_p"),
                "typical": params.get("typical"),
                "tfs": params.get("tfs"),
            }

            logger.info(f"Kobold stream request for persona '{target_persona}'. "
                        f"Detected User Marker: {msg_data['user_marker']!r}, "
                        f"Assistant Marker: {msg_data['assistant_marker']!r}")

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
                            payload = json.dumps({"token": f"\n[Error] {ev.get('text', '')}"})
                            yield f"event: message\ndata: {payload}\n\n".encode("utf-8")
                        elif etype == "done":
                            payload = json.dumps({"token": "", "finish_reason": "stop"})
                            yield f"event: message\ndata: {payload}\n\n".encode("utf-8")
                            return
                except asyncio.CancelledError:
                    raise

            return StreamingResponse(
                event_source(),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "X-Accel-Buffering": "no",
                    "Connection": "keep-alive",
                },
            )

    # Known markers across kobold-lite templates.
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

    def _extract_user_message_with_markers(self, prompt: str) -> Dict[str, str]:
        """Extracts message and the markers used for user/assistant turns."""
        last_pos = -1
        user_marker = ""
        for m in self._USER_MARKERS:
            pos = prompt.rfind(m)
            if pos > last_pos:
                last_pos = pos
                user_marker = m

        if last_pos == -1:
            return {
                "message": prompt.strip()[-1000:],
                "user_marker": "",
                "assistant_marker": "",
                "thinking_trigger": ""
            }

        segment = prompt[last_pos + len(user_marker):]

        cut = len(segment)
        assistant_marker = ""
        for m in self._BOUNDARY_MARKERS:
            p = segment.find(m)
            if 0 <= p < cut:
                cut = p
                assistant_marker = m
        
        # If assistant_marker wasn't found in the tail segment, check the whole prompt for the last one
        if not assistant_marker:
            for m in self._BOUNDARY_MARKERS:
                if m in prompt:
                    assistant_marker = m # Assume consistent tagging

        message = segment[:cut].strip()
        # Capture anything after the assistant marker as a trigger (e.g. <thought>)
        suffix = segment[cut + len(assistant_marker):].strip() if assistant_marker else ""
        
        return {
            "message": message or prompt.strip()[-1000:],
            "user_marker": user_marker,
            "assistant_marker": assistant_marker,
            "thinking_trigger": suffix
        }

    def _extract_user_message(self, prompt: str) -> str:
        return self._extract_user_message_with_markers(prompt)["message"]

    def _get_current_persona_name(self) -> str:
        """Helper to get the active persona name or a sane default."""
        if self.active_persona and self.active_persona in self.chat_system.personas:
            return self.active_persona
        return next(iter(self.chat_system.personas.keys()), "assistant")

    async def start(self):
        """Runs the server. Must be called as a coroutine."""
        logger.info(f"Starting Kobold Adapter on http://{self.host}:{self.port}")
        config = uvicorn.Config(self.app, host=self.host, port=self.port, log_level="warning")
        server = uvicorn.Server(config)
        await server.serve()

def create_kobold_adapter(chat_system: ChatSystem) -> KoboldAdapter:
    return KoboldAdapter(chat_system)
