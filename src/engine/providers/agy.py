# src/engine/providers/agy.py
"""Antigravity (agy) provider (DP-244).

agy is a TUI CLI invoked as a subprocess (POSIX-only — see `ensure_agy_supported`)
whose entire response arrives at process exit; there is no token stream to make
canonical, so the route stays one-shot and is adapted into the unified event
shape via `_events_from_one_shot`. agy CLAMPS tools off and round-trips derpr's
`<tool_call>` text protocol (contrast cc.py, which runs its own tools).

The logic bodies live here; ``TextEngine`` keeps thin delegators for every method
(the seams the driver routes through and the existing tests call/patch directly).
Cross-method calls go back through the engine seams (e.g. `engine._run_agy_cli`)
so a test's instance-level monkeypatch still intercepts.
"""

import logging
import os
import re
import shutil
import tempfile
import uuid
from typing import TYPE_CHECKING, Any, AsyncIterator, Dict, List, Optional, Tuple

import asyncio

from aiolimiter import AsyncLimiter

from config import global_config
from src.llm_errors import LLMCommunicationError
from src.text_tool_protocol import (
    TOOL_CALL_OPEN,
    TOOL_CALL_CLOSE,
    decode_tool_call_payload,
    extract_first_tool_call_block,
    render_tool_descriptions,
)

from .base import Provider

if TYPE_CHECKING:
    from src.engine.driver import TextEngine

logger = logging.getLogger(__name__)

AGY_CALL_TIMEOUT_SECONDS = 120.0


def render_agy_tool_protocol(tools: Optional[List[Dict[str, Any]]]) -> str:
    if not tools:
        return ""

    protocol_desc = (
        "You may request a tool by emitting EXACTLY "
        f"{TOOL_CALL_OPEN}{{\"name\": \"<tool_name>\", \"arguments\": "
        f"{{<json args>}}}}{TOOL_CALL_CLOSE} "
        "as the last thing. Answer in plain text otherwise, and use no other tools/files/shell/web."
    )

    # Shared renderer keeps the agy and streaming paths from drifting on how a
    # tool's name/description/parameters are formatted.
    lines = [protocol_desc, *render_tool_descriptions(tools)]
    return "\n".join(lines)


def parse_agy_tool_call(text: str) -> Optional[List[Dict[str, Any]]]:
    if not text:
        return None
    cleaned = re.sub(r"<SYSTEM_MESSAGE>.*?</SYSTEM_MESSAGE>", "", text, flags=re.DOTALL)
    inner = extract_first_tool_call_block(cleaned)
    if inner is None:
        return None
    parsed = decode_tool_call_payload(inner)
    if parsed is None:
        return None
    # agy policy: both keys must be present; id is a fresh uuid.
    if "name" not in parsed or "arguments" not in parsed:
        return None
    call_id = f"agy_{uuid.uuid4().hex}"
    return [{
        "id": call_id,
        "name": parsed["name"],
        "arguments": parsed["arguments"]
    }]


def ensure_agy_supported() -> None:
    """agy is a TUI CLI that only emits its response to a TTY. DERPR captures
    stdout via a pipe — fine on POSIX, but on native Windows agy renders to the
    console and writes *nothing* to a non-TTY stdout/file, so the route silently
    returns empty. Fail loudly instead and point at the docs; run the engine on
    the POSIX host (Linux/macOS/WSL/Docker) to use agy.
    """
    if os.name != "posix":
        raise LLMCommunicationError(
            "The 'agy' provider is unsupported on native Windows: agy only "
            "writes its response to a TTY, but DERPR captures stdout via a "
            "pipe, so the response is always empty. Run the engine on the "
            "POSIX host (Linux/macOS/WSL/Docker) to use agy. "
            "See docs/user_guide.md (Antigravity / agy provider)."
        )


def resolve_agy_workspace(engine: "TextEngine", persona_name: Optional[str]) -> Optional[str]:
    """Returns the persistent workspace dir for this call, or None when
    persistence is disabled (caller uses a throwaway temp dir). Does not create
    the directory."""
    if not global_config.AGY_PERSISTENT_WORKSPACES:
        return None
    workspaces_dir = global_config.AGY_WORKSPACES_DIR
    slug = engine._sanitize_agy_workspace_name(persona_name)
    if global_config.AGY_WORKSPACE_MODE == "persona" and slug:
        return os.path.abspath(workspaces_dir / f"agy_{slug}")
    return os.path.abspath(workspaces_dir / "agy_global")


def remove_agy_cli_link_targets(workspace_dir: str) -> None:
    cli_dir = os.path.join(workspace_dir, ".antigravitycli")
    if not os.path.isdir(cli_dir):
        return
    for f in os.listdir(cli_dir):
        p = os.path.join(cli_dir, f)
        if os.path.islink(p):
            try:
                target = os.readlink(p)
                if os.path.exists(target):
                    os.remove(target)
            except Exception:
                pass


async def run_agy_cli(engine: "TextEngine", prompt: str, timeout: float = AGY_CALL_TIMEOUT_SECONDS,
                      persona_name: Optional[str] = None) -> str:
    engine._ensure_agy_supported()

    binary = os.environ.get("ANTIGRAVITY_HARNESS_PATH") or shutil.which("agy")
    if not binary:
        raise LLMCommunicationError("Antigravity harness/agy binary not found.")

    timeout_sec_str = f"{int(timeout) + 30}s"
    args = ["--print-timeout", timeout_sec_str, "-p", prompt]
    if global_config.AGY_SANDBOX:
        args = ["--sandbox", *args]

    workspace_dir = engine._resolve_agy_workspace(persona_name)
    if workspace_dir is None:
        temp_dir = tempfile.mkdtemp()
        try:
            return await engine._exec_agy(binary, args, temp_dir, timeout)
        finally:
            # The CLI leaves symlinks under .antigravitycli pointing at files
            # outside the temp dir; remove the targets so rmtree doesn't strand
            # them. Persistent workspaces keep this state on purpose — that
            # cache is the point of persistence.
            engine._remove_agy_cli_link_targets(temp_dir)
            shutil.rmtree(temp_dir, ignore_errors=True)

    os.makedirs(workspace_dir, exist_ok=True)
    lock = engine._agy_workspace_locks.setdefault(workspace_dir, asyncio.Lock())
    async with lock:
        return await engine._exec_agy(binary, args, workspace_dir, timeout)


async def generate_agy(
    engine: "TextEngine", config: Dict[str, Any], history_object: Dict[str, Any],
    tools: Optional[List[Dict[str, Any]]] = None,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """One-shot agy path. DP-206 decision: agy stays one-shot-only — it is a TUI
    CLI invoked as a subprocess (POSIX-only) whose entire response arrives at
    process exit; there is no token stream to make canonical. Streaming consumers
    get it via `stream_messages`' generate_response wrap (single text_delta)."""
    system_prompt, history = engine._extract_system_prompt(history_object)

    prompt_parts = []
    if system_prompt:
        prompt_parts.append(system_prompt)

    if tools:
        rendered_tools = engine._render_agy_tool_protocol(tools)
        if rendered_tools:
            prompt_parts.append(rendered_tools)

    rendered_history = engine._render_agy_prompt(history)
    if rendered_history:
        prompt_parts.append(rendered_history)

    prompt = "\n\n".join(prompt_parts)

    tool_names = []
    if tools:
        tool_names = [t["function"]["name"] for t in tools if "function" in t and "name" in t["function"]]

    persona_name = config.get("persona_name")
    workspace_dir = engine._resolve_agy_workspace(persona_name)
    api_payload = {
        "model": config.get("model_name"),
        "prompt_chars": len(prompt),
        "tools": tool_names,
        "isolation": {
            "stdin": "devnull",
            "skip_permissions": False,
            "workspace": workspace_dir if workspace_dir else "temp-dir-per-call",
        }
    }

    try:
        raw = await engine._run_agy_cli(prompt, persona_name=persona_name)
    except LLMCommunicationError as e:
        if e.api_payload is None:
            e.api_payload = api_payload
        raise

    calls = engine._parse_agy_tool_call(raw)
    if calls:
        return {"type": "tool_calls", "calls": calls}, api_payload
    else:
        cleaned_content = re.sub(r"<SYSTEM_MESSAGE>.*?</SYSTEM_MESSAGE>", "", raw, flags=re.DOTALL).strip()
        return {"type": "text", "content": cleaned_content}, api_payload


async def stream_agy(
    engine: "TextEngine", config: Dict[str, Any], history_object: Dict[str, Any],
    tools: Optional[List[Dict[str, Any]]] = None,
) -> AsyncIterator[Dict[str, Any]]:
    """agy adapter into the unified event shape. agy stays one-shot by decision
    (subprocess TUI CLI — the entire response arrives at process exit, there is
    no token stream to make canonical); streaming consumers get the full text as
    a single text_delta."""
    result, api_payload = await engine._generate_agy_response(config, history_object, tools)
    async for ev in engine._events_from_one_shot(result, api_payload):
        yield ev


class AgyProvider(Provider):
    """Antigravity (agy-*) provider. POSIX-only subprocess CLI, one-shot,
    dedicated rate limiter, clamps tools to the <tool_call> text protocol."""

    def __init__(self, engine: "TextEngine") -> None:
        self._engine = engine

    #: name of the engine seam method (back-compat for `_get_provider_route`).
    route_method_name = "_stream_agy_response"

    def matches(self, model_name: str) -> bool:
        return model_name.startswith("agy")

    def limiters_for(self, model_name: str) -> List[AsyncLimiter]:
        return [self._engine._agy_limiter]

    def ensure_supported(self, model_name: str) -> None:
        # Looked up live so per-test monkeypatches of the guard take effect.
        self._engine._ensure_agy_supported()

    async def stream(
        self,
        persona_config: Dict[str, Any],
        history_object: Dict[str, Any],
        tools: Optional[List[Dict[str, Any]]] = None,
        *,
        local_inference_config: Optional[Dict[str, Any]] = None,
    ) -> AsyncIterator[Dict[str, Any]]:
        async for ev in self._engine._stream_agy_response(persona_config, history_object, tools):
            yield ev
