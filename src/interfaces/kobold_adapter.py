# src/interfaces/kobold_adapter.py

import logging
import re
from typing import Any, Dict, List, Optional
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import asyncio

from src.chat_system import ChatSystem, ResponseType

logger = logging.getLogger(__name__)

class KoboldAdapter:
    def __init__(self, chat_system: ChatSystem, host: str = "0.0.0.0", port: int = 5002):
        self.chat_system = chat_system
        self.host = host
        self.port = port
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

    def _setup_routes(self):
        @self.app.get("/api/v1/model")
        async def get_model():
            """Returns the current persona name as the 'model'."""
            persona_name = next(iter(self.chat_system.personas.keys()), "assistant")
            return {"result": persona_name}

        @self.app.get("/api/v1/info/version")
        async def get_info_version():
            """Returns a mock version to satisfy connection checks."""
            return {"version": "1.2.3", "lib_version": "1.2.3"}

        @self.app.get("/api/extra/version")
        async def get_extra_version():
            """Returns a mock extra version."""
            return {"version": "1.2.3", "platform": "DERPR"}

        @self.app.get("/api/v1/config/soft_prompts")
        async def get_soft_prompts():
            return {"results": []}

        @self.app.get("/api/v1/config/max_context_length")
        async def get_max_context_length():
            """Returns the context length of the first persona."""
            persona_name = next(iter(self.chat_system.personas.keys()), "assistant")
            persona = self.chat_system.personas.get(persona_name)
            limit = persona.get_context_length() if persona else 8192
            return {"result": limit}

        @self.app.post("/api/v1/generate")
        async def generate(request: Request):
            """
            Handles text generation requests from Kobold Lite.
            Extracts the last user message from the prompt and routes it through ChatSystem.
            """
            data = await request.json()
            prompt = data.get("prompt", "")
            
            # Simple heuristic to extract the last user message from the multi-turn prompt
            user_msg = self._extract_user_message(prompt)
            
            # Use the first available persona
            persona_name = next(iter(self.chat_system.personas.keys()), "assistant")

            logger.info(f"Kobold API received prompt. Extracted message: '{user_msg[:50]}...'")

            # Route to ChatSystem
            response_text, response_type, _, _ = await self.chat_system.generate_response(
                persona_name=persona_name,
                user_identifier="web_user",
                channel="web_interface",
                message=user_msg,
                user_display_name="WebUser"
            )
            
            # Return in Kobold API format
            return {
                "results": [
                    {
                        "text": response_text
                    }
                ]
            }

    def _extract_user_message(self, prompt: str) -> str:
        """
        Attempts to find the actual 'latest' message in a chat-like prompt blocks.
        Common patterns: 'User: ...', '### Instruction: ...', etc.
        """
        # Try finding last occurrence of common chat prefixes
        prefixes = [
            (r'User:\s*(.*)', 1),
            (r'### Instruction:\s*(.*)', 1),
            (r'Human:\s*(.*)', 1),
            (r'Input:\s*(.*)', 1)
        ]
        
        # We split by newlines and search backwards for these patterns
        lines = prompt.strip().split('\n')
        for i in range(len(lines) - 1, -1, -1):
            line = lines[i]
            for pattern, group in prefixes:
                match = re.search(pattern, line, re.IGNORECASE)
                if match:
                    # Found the start of the last user turn.
                    # We take everything from here forward, but stop if we hit an Assistant marker.
                    # Actually, usually there's only one "user" turn in the very last segment.
                    full_segment = "\n".join(lines[i:])
                    assistant_markers = ["Assistant:", "### Response:", "AI:", "Bot:"]
                    for marker in assistant_markers:
                        if marker in full_segment:
                            # Strip the assistant part
                            full_segment = full_segment.split(marker)[0]
                    
                    # Strip the prefix itself
                    clean_msg = re.sub(pattern, r'\1', full_segment, flags=re.IGNORECASE).strip()
                    if clean_msg:
                        return clean_msg

        # Fallback: Just take the last 500 chars if no markers found, or the whole thing if it's short
        return prompt.strip()[-1000:]

    async def start(self):
        """Runs the server. Must be called as a coroutine."""
        logger.info(f"Starting Kobold Adapter on http://{self.host}:{self.port}")
        config = uvicorn.Config(self.app, host=self.host, port=self.port, log_level="warning")
        server = uvicorn.Server(config)
        await server.serve()

def create_kobold_adapter(chat_system: ChatSystem) -> KoboldAdapter:
    return KoboldAdapter(chat_system)
