# src/engine/providers/_shared.py
"""Provider-agnostic helpers, hoisted off ``TextEngine`` (DP-244).

Free functions (no inheritance coupling) shared by the per-provider streams and
the driver. Promote to a mixin only if 3+ providers grow the same wiring.
"""

import json
import logging
from typing import Any, Dict, List, Tuple

import aiohttp

logger = logging.getLogger(__name__)


def extract_system_prompt(history_object: Dict[str, Any]) -> Tuple[str, List[Dict[str, Any]]]:
    """Returns (merged_system_prompt, remaining_history). A leading system turn
    in the history is folded into the persona prompt."""
    system_prompt = history_object["persona_prompt"]
    history = history_object.get("message_history", history_object.get("history", []))
    if history and history[0]["role"] == "system":
        system_prompt = f"{system_prompt}\n\n{history[0]['content']}"
        history = history[1:]
    return system_prompt, history


async def download_image(image_url: str) -> Tuple[bytes, str]:
    """Downloads image, returns (raw_bytes, mime_type).
    Raises aiohttp.ClientError on failure."""
    async with aiohttp.ClientSession() as session:
        async with session.get(image_url) as resp:
            resp.raise_for_status()
            image_bytes = await resp.read()
            mime_type = resp.content_type
    return image_bytes, mime_type


def parse_openai_tool_calls(raw_calls: List[Any]) -> List[Dict[str, Any]]:
    """Parses OpenAI-style tool call objects into standardized dicts. Reusable by
    any OpenAI-compatible provider (e.g. the local kobold path)."""
    tool_calls: List[Dict[str, Any]] = []
    for call in raw_calls:
        try:
            arguments = json.loads(call.function.arguments)
            tool_calls.append({"id": call.id, "name": call.function.name, "arguments": arguments})
        except json.JSONDecodeError:
            logger.error(f"Failed to parse tool call arguments: {call.function.arguments}")
            continue
    return tool_calls
