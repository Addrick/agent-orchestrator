# src/engine/providers/_shared.py
"""Provider-agnostic helpers, hoisted off ``TextEngine`` (DP-244).

Free functions (no inheritance coupling) shared by the per-provider streams and
the driver. Promote to a mixin only if 3+ providers grow the same wiring.
"""

import json
import logging
from typing import Any, Dict, List, Tuple

import aiohttp

# DP-317: moved down to the `utils` leaf so `src.stream_engine` — which sits
# below `src.engine` in the layer order — can share it without an upward
# import. Re-exported here because every provider already imports it from
# `_shared`, and this is its provider-facing home.
# The redundant-looking alias is the explicit-re-export form mypy requires
# under `no_implicit_reexport`.
from src.utils.history_shape import (  # noqa: F401
    extract_system_prompt as extract_system_prompt,
)

logger = logging.getLogger(__name__)


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
