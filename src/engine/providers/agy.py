# src/engine/providers/agy.py
"""Antigravity (agy) provider (DP-244).

agy is a TUI CLI invoked as a subprocess whose entire response arrives at process
exit; there is no token stream to make canonical, so the route stays one-shot and
is adapted into the unified event shape via `_events_from_one_shot`. agy CLAMPS
tools off and round-trips derpr's `<tool_call>` text protocol (contrast cc.py,
which runs its own tools).

DP-324: the route used to be POSIX-only. `agy --print` wrote its response only to
a TTY on native Windows, so derpr's piped capture came back empty and the engine
refused the route outright. agy >= 1.1.9 writes to a pipe on Windows too
(verified against 1.1.9 on Windows 10: `-p` and `--sandbox --p` both return the
text on a piped stdout), so the guard is gone and the platform differences that
remain are handled in `_subprocess.py` (command-line budget, process teardown).

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
from src.utils.claude_cli_env import build_agy_cli_env
from src.text_tool_protocol import (
    TOOL_CALL_OPEN,
    TOOL_CALL_CLOSE,
    decode_tool_call_payload,
    extract_tool_call_blocks,
    render_tool_descriptions,
    strip_tool_call_blocks,
)

from .base import Provider
from ._subprocess import fit_cli_prompt, render_transcript_blocks

if TYPE_CHECKING:
    from src.engine.driver import TextEngine

logger = logging.getLogger(__name__)

AGY_CALL_TIMEOUT_SECONDS = 120.0


def render_agy_tool_protocol(tools: Optional[List[Dict[str, Any]]]) -> str:
    if not tools:
        return ""

    # DP-338: "EXACTLY one block, as the last thing" used to be the wording, and
    # it was wrong in both directions — the parser only honoured the first block
    # while personas ask for independent reads in one message, so a compliant
    # model's 2nd and 3rd calls were dropped on the floor. Calls in one message
    # are now dispatched together, which is also the only place batching's
    # latency win can come from.
    protocol_desc = (
        "You may request tools by emitting one or more blocks of EXACTLY "
        f"{TOOL_CALL_OPEN}{{\"name\": \"<tool_name>\", \"arguments\": "
        f"{{<json args>}}}}{TOOL_CALL_CLOSE} "
        "as the last thing, one block per call. Reads that do not depend on "
        "each other belong in the same message — they run at the same time. "
        "Answer in plain text otherwise, and use no other tools/files/shell/web."
    )

    # Shared renderer keeps the agy and streaming paths from drifting on how a
    # tool's name/description/parameters are formatted.
    lines = [protocol_desc, *render_tool_descriptions(tools)]
    return "\n".join(lines)


def parse_agy_tool_call(text: str) -> Optional[List[Dict[str, Any]]]:
    """Every well-formed `<tool_call>` block in a complete agy response.

    Returns None when the response contains no usable call at all, which is the
    signal the driver's one-shot retry keys off; a response whose blocks are
    ALL malformed must stay indistinguishable from one that made no call.

    A malformed block sitting among well-formed ones is skipped rather than
    failing the response (DP-338). Dropping the whole batch over one bad block
    would discard calls the model got right and put it back in the loop this
    ticket exists to remove; the skipped call simply goes unanswered, and the
    model can re-ask for it — which is the one case where re-asking makes
    progress, because the calls beside it did land.
    """
    if not text:
        return None
    cleaned = re.sub(r"<SYSTEM_MESSAGE>.*?</SYSTEM_MESSAGE>", "", text, flags=re.DOTALL)
    calls: List[Dict[str, Any]] = []
    for inner in extract_tool_call_blocks(cleaned):
        parsed = decode_tool_call_payload(inner)
        if parsed is None:
            continue
        # agy policy: both keys must be present; id is a fresh uuid.
        if "name" not in parsed or "arguments" not in parsed:
            continue
        calls.append({
            "id": f"agy_{uuid.uuid4().hex}",
            "name": parsed["name"],
            "arguments": parsed["arguments"],
        })
    return calls or None


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
    binary = os.environ.get("ANTIGRAVITY_HARNESS_PATH") or shutil.which("agy")
    if not binary:
        raise LLMCommunicationError("Antigravity harness/agy binary not found.")

    timeout_sec_str = f"{int(timeout) + 30}s"
    args = ["--print-timeout", timeout_sec_str, "-p", prompt]
    if global_config.AGY_SANDBOX:
        args = ["--sandbox", *args]

    # DP-277: strip derpr's machine secrets from the child env — agy runs
    # untrusted content and authenticates with its own harness OAuth, not our
    # provider/portal creds.
    env = build_agy_cli_env()

    workspace_dir = engine._resolve_agy_workspace(persona_name)
    if workspace_dir is None:
        temp_dir = tempfile.mkdtemp()
        try:
            return await engine._exec_agy(binary, args, temp_dir, timeout, env=env)
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
        return await engine._exec_agy(binary, args, workspace_dir, timeout, env=env)


async def generate_agy(
    engine: "TextEngine", config: Dict[str, Any], history_object: Dict[str, Any],
    tools: Optional[List[Dict[str, Any]]] = None,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """One-shot agy path. DP-206 decision: agy stays one-shot-only — it is a TUI
    CLI invoked as a subprocess whose entire response arrives at process exit;
    there is no token stream to make canonical. Streaming consumers
    get it via `stream_messages`' generate_response wrap (single text_delta)."""
    system_prompt, history = engine._extract_system_prompt(history_object)

    prompt_parts = []
    if system_prompt:
        prompt_parts.append(system_prompt)

    if tools:
        rendered_tools = engine._render_agy_tool_protocol(tools)
        if rendered_tools:
            prompt_parts.append(rendered_tools)

    # The whole prompt travels as ONE argv entry (`agy --print <prompt>` — the CLI
    # has no stdin/prompt-file transport), and the OS caps that: 128 KiB per
    # argument under execve, 32767 chars for the entire command line under
    # CreateProcess. The engine bounds history by tokens (~131k ≈ 0.5 MB of
    # text), so an unclamped prompt fails the *spawn* before agy ever runs. Keep
    # the system prompt + tool protocol whole and elide the oldest history
    # messages to fit this host's budget.
    prompt, elided = fit_cli_prompt(
        "\n\n".join(prompt_parts), render_transcript_blocks(history),
    )

    tool_names = []
    if tools:
        tool_names = [t["function"]["name"] for t in tools if "function" in t and "name" in t["function"]]

    persona_name = config.get("persona_name")
    workspace_dir = engine._resolve_agy_workspace(persona_name)
    api_payload = {
        "model": config.get("model_name"),
        "prompt_chars": len(prompt),
        "history_messages_elided": elided,
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

    # Only parse the protocol we actually rendered. The `if tools:` guards above
    # suppress *sending* the tool protocol but not parsing it back, so a
    # toolless call whose prompt merely CONTAINS `<tool_call>` spans — which is
    # exactly what DP-335's exhaustion wrap-up sends, a transcript of the
    # turn's own tool calls plus a persona prompt naming tools by hand — came
    # back classified as `tool_calls`. `_events_from_one_shot` then reports
    # `full_text: ""` and the prose is discarded, dropping the caller back to
    # its no-text fallback after paying for the subprocess.
    calls = engine._parse_agy_tool_call(raw) if tools else None
    cleaned_content = re.sub(
        r"<SYSTEM_MESSAGE>.*?</SYSTEM_MESSAGE>", "", raw, flags=re.DOTALL,
    ).strip()
    if calls:
        # DP-338: the prose beside the blocks travels with them. It is the
        # model's stated plan for the batch ("checking the node, the card and
        # the unit list before proposing the swap"), and dropping it meant the
        # next iteration re-read a transcript in which the model appeared to
        # have called a tool for no stated reason — so it re-derived the plan
        # from scratch, every iteration, off an identical history.
        return {
            "type": "tool_calls",
            "calls": calls,
            "content": strip_tool_call_blocks(cleaned_content),
        }, api_payload
    else:
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
    """Antigravity (agy-*) provider. Subprocess CLI, one-shot, dedicated rate
    limiter, clamps tools to the <tool_call> text protocol. Runs on any platform
    the `agy` CLI itself supports (DP-324); `ensure_supported` stays the base
    class's no-op."""

    def __init__(self, engine: "TextEngine") -> None:
        self._engine = engine

    #: name of the engine seam method (back-compat for `_get_provider_route`).
    route_method_name = "_stream_agy_response"

    def matches(self, model_name: str) -> bool:
        return model_name.startswith("agy")

    def limiters_for(self, model_name: str) -> List[AsyncLimiter]:
        return [self._engine._agy_limiter]

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
