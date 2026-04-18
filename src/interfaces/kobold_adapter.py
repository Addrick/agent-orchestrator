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
            Kobold-Lite sync endpoint. Routes through ChatSystem.stream_response so
            streaming-capable models fall back to a single chunk transparently,
            keeping the flow consistent with /api/extra/generate/stream.
            """
            data = await request.json()
            prompt = data.get("prompt", "")
            target_persona = persona or data.get("model") or self._get_current_persona_name()
            if target_persona not in self.chat_system.personas:
                target_persona = self._get_current_persona_name()

            user_msg = self._extract_user_message(prompt)
            logger.info(f"Kobold API received prompt for persona '{target_persona}'. "
                        f"Extracted message: '{user_msg[:50]}...'")

            final_text = ""
            async for ev in self.chat_system.stream_response(
                persona_name=target_persona,
                user_identifier="web_user",
                channel="web_interface",
                message=user_msg,
                user_display_name="WebUser",
            ):
                if ev.get("type") == "done":
                    final_text = ev.get("full_text", final_text)
                elif ev.get("type") == "error":
                    final_text = ev.get("text", "")

            return {"results": [{"text": final_text}]}

        @self.app.post("/api/extra/generate/stream")
        async def generate_stream(request: Request, persona: Optional[str] = None):
            """SSE token streaming endpoint — KoboldCPP /api/extra/generate/stream spec.

            Wire format per event:
                event: message
                data: {"token": "..."}\n\n

            Final event signals completion; the client also detects end via connection close.
            """
            data = await request.json()
            prompt = data.get("prompt", "")
            target_persona = persona or data.get("model") or self._get_current_persona_name()
            if target_persona not in self.chat_system.personas:
                target_persona = self._get_current_persona_name()

            user_msg = self._extract_user_message(prompt)
            logger.info(f"Kobold stream request for persona '{target_persona}'. "
                        f"Extracted message: '{user_msg[:50]}...'")

            async def event_source() -> AsyncIterator[bytes]:
                token_count = 0
                try:
                    async for ev in self.chat_system.stream_response(
                        persona_name=target_persona,
                        user_identifier="web_user",
                        channel="web_interface",
                        message=user_msg,
                        user_display_name="WebUser",
                    ):
                        # Bail early if client went away.
                        if await request.is_disconnected():
                            logger.info("Client disconnected; stopping stream.")
                            return

                        etype = ev.get("type")
                        if etype == "token":
                            token_count += 1
                            if token_count <= 3 or token_count % 25 == 0:
                                logger.debug(f"SSE token #{token_count}: {ev.get('text', '')!r}")
                            payload = json.dumps({"token": ev.get("text", "")})
                            yield f"event: message\ndata: {payload}\n\n".encode("utf-8")
                        elif etype == "error":
                            payload = json.dumps({"token": f"\n[Error] {ev.get('text', '')}"})
                            yield f"event: message\ndata: {payload}\n\n".encode("utf-8")
                        elif etype == "done":
                            logger.info(f"SSE stream complete: {token_count} token events emitted")
                            payload = json.dumps({"token": "", "finish_reason": "stop"})
                            yield f"event: message\ndata: {payload}\n\n".encode("utf-8")
                            return
                        # Ignore "status" for now — no markers per spec.
                except asyncio.CancelledError:
                    logger.info("Stream cancelled (client disconnected).")
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

    # Known user_start markers across kobold-lite instruct templates.
    _USER_MARKERS = (
        "\n### Instruction:\n",
        "\nUser:", "\nHuman:", "\nInput:",
        "<|im_start|>user\n",
        "<start_of_turn>user\n",
        "<|turn>user\n",
        "<|user|>\n",
        "<|start_of_role|>user<|end_of_role|>",
        "<|im_user|>user<|im_middle|>",
        "<\uff5cUser\uff5c>",
        "<|START_OF_TURN_TOKEN|><|USER_TOKEN|>",
        "<|start_header_id|>user<|end_header_id|>",
        "<|header_start|>user<|header_end|>",
        "[INST]",
        "{{[INPUT]}}",
    )
    # Assistant_start / user_end markers. Content is bounded by these.
    _BOUNDARY_MARKERS = (
        "\n### Response:", "\nAssistant:", "\nAI:", "\nBot:",
        "\nASSISTANT:",
        "<|im_start|>assistant",
        "<start_of_turn>model", "<end_of_turn>",
        "<|turn>model", "<turn|>",
        "<|assistant|>",
        "<|im_assistant|>assistant",
        "<\uff5cAssistant\uff5c>",
        "<|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|>",
        "<|start_header_id|>assistant<|end_header_id|>",
        "<|header_start|>assistant<|header_end|>",
        "<|start_of_role|>assistant<|end_of_role|>",
        "[/INST]",
        "{{[OUTPUT]}}", "{{[INPUT_END]}}",
        "<|im_end|>", "<\uff5cend\u2581of\u2581sentence\uff5c>",
        "<|end_of_text|>",
    )

    def _extract_user_message(self, prompt: str) -> str:
        """Extract latest user turn from a fully-rendered instruct/chat prompt.

        Lite sends templates that vary by model (### Instruction / ChatML /
        Gemma / Llama3 / etc.). Strategy: locate the last user_start marker,
        take text up to the next assistant/end marker.
        """
        last_pos = -1
        last_marker = ""
        for m in self._USER_MARKERS:
            pos = prompt.rfind(m)
            if pos > last_pos:
                last_pos = pos
                last_marker = m

        if last_pos == -1:
            return prompt.strip()[-1000:]

        segment = prompt[last_pos + len(last_marker):]

        cut = len(segment)
        for m in self._BOUNDARY_MARKERS:
            p = segment.find(m)
            if 0 <= p < cut:
                cut = p
        segment = segment[:cut].strip()

        return segment or prompt.strip()[-1000:]

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
