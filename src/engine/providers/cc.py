# src/engine/providers/cc.py
"""Claude Code (cc-*) provider (DP-222, extracted DP-244).

Structural parity with the agy route (subprocess-per-call, one-shot, POSIX-only,
persistent per-persona workspace, dedicated rate limiter), but with a deliberate
behavioural divergence: the agy route CLAMPS tools off and round-trips derpr's
`<tool_call>` text protocol, whereas Claude Code runs its OWN sandboxed tools
autonomously (`--dangerously-skip-permissions` bounded by the built-in OS
sandbox). So the cc route ignores the engine's `tools` argument and returns
Claude Code's final text; derpr's tool loop does not wrap it. Bridging derpr
tools -> Claude Code is future MCP work, where approval routing will also live.

Logic bodies live here; ``TextEngine`` keeps thin delegators for every method
(the seams tests call/patch directly). Cross-method calls go back through the
engine seams so instance-level monkeypatches still intercept.
"""

import json
import logging
import os
import shutil
import tempfile
from typing import TYPE_CHECKING, Any, AsyncIterator, Dict, List, Optional, Tuple

import asyncio

from aiolimiter import AsyncLimiter

from config import global_config
from src.llm_errors import LLMCommunicationError
from src.utils.claude_cli_env import build_claude_cli_env

from .base import Provider

if TYPE_CHECKING:
    from src.engine.driver import TextEngine

logger = logging.getLogger(__name__)

CC_CALL_TIMEOUT_SECONDS = 600.0


def ensure_cc_supported() -> None:
    """Claude Code's OS sandbox (Seatbelt/bubblewrap) only runs on
    macOS/Linux/WSL2 — never native Windows. Since this provider runs
    `--dangerously-skip-permissions` (yolo), the sandbox is the safety boundary,
    so refuse the route on non-POSIX hosts when the sandbox is enabled. Run the
    engine on the POSIX host (Linux/macOS/WSL/Docker).
    """
    if global_config.CC_SANDBOX and os.name != "posix":
        raise LLMCommunicationError(
            "The 'cc-*' (Claude Code) provider runs yolo bounded by Claude "
            "Code's OS sandbox, which is unavailable on native Windows. Run "
            "the engine on the POSIX host (Linux/macOS/WSL/Docker), or set "
            "CC_SANDBOX=False to run unsandboxed (no yolo; tools gated to "
            "CC_ALLOWED_TOOLS). "
            "See docs/user_guide.md (Claude Code / cc provider)."
        )


def cc_model_arg(model_name: str) -> str:
    """Map a `cc-<alias>` model name onto Claude Code's `--model` value
    (e.g. `cc-sonnet` -> `sonnet`). A bare `cc-` falls back to `sonnet`."""
    alias = model_name[len("cc-"):] if model_name.startswith("cc-") else model_name
    return alias or "sonnet"


def resolve_cc_workspace(
    engine: "TextEngine", persona_name: Optional[str], workspace_override: Optional[str] = None
) -> Optional[str]:
    """Returns the working dir for this call, or None when persistence is
    disabled (caller uses a throwaway temp dir). Precedence: per-call
    workspace_override (e.g. the DP-227 fixr clone, set by the orchestration
    layer) > explicit CC_WORKSPACE_DIR (the derpr checkout) > per-persona dir >
    global dir. Does not create the directory."""
    if workspace_override:
        return os.path.abspath(workspace_override)
    if global_config.CC_WORKSPACE_DIR:
        return os.path.abspath(global_config.CC_WORKSPACE_DIR)
    if not global_config.CC_PERSISTENT_WORKSPACES:
        return None
    workspaces_dir = global_config.CC_WORKSPACES_DIR
    slug = engine._sanitize_agy_workspace_name(persona_name)
    if global_config.CC_WORKSPACE_MODE == "persona" and slug:
        return os.path.abspath(workspaces_dir / f"cc_{slug}")
    return os.path.abspath(workspaces_dir / "cc_global")


def build_cc_sandbox_settings() -> Optional[Dict[str, Any]]:
    """Build the `--settings` sandbox block, or None when CC_SANDBOX is off.
    Auto-allows sandboxed Bash so a headless run never blocks on a prompt; the
    OS sandbox confines it to the workspace + allowed domains."""
    if not global_config.CC_SANDBOX:
        return None
    sandbox: Dict[str, Any] = {
        "enabled": True,
        "autoAllowBashIfSandboxed": True,
    }
    if global_config.CC_SANDBOX_WEAKER_NESTED:
        sandbox["enableWeakerNestedSandbox"] = True
    if global_config.CC_SANDBOX_ALLOWED_DOMAINS:
        sandbox["network"] = {"allowedDomains": list(global_config.CC_SANDBOX_ALLOWED_DOMAINS)}
    return {"sandbox": sandbox}


def build_cc_args(engine: "TextEngine", prompt: str, system_prompt: str, model_arg: str) -> List[str]:
    """Assemble the `claude -p` argv (without the binary)."""
    args = ["-p", prompt, "--output-format", "text", "--model", model_arg]
    if system_prompt:
        args += ["--system-prompt", system_prompt]
    if global_config.CC_SANDBOX:
        # yolo: skip per-tool approval prompts. The OS sandbox is the safety
        # boundary; root's skip-permissions check is waived inside it.
        args += ["--dangerously-skip-permissions"]
    elif global_config.CC_ALLOWED_TOOLS:
        # Unsandboxed (e.g. native Windows smoke): NEVER bare yolo. Use Claude
        # Code's OS-independent permission system — only the explicitly
        # allowlisted tools may run; everything else is refused (headless cannot
        # answer an approval prompt).
        args += ["--allowedTools", *global_config.CC_ALLOWED_TOOLS]
    if global_config.CC_MAX_TURNS > 0:
        args += ["--max-turns", str(global_config.CC_MAX_TURNS)]
    sandbox_settings = engine._build_cc_sandbox_settings()
    if sandbox_settings is not None:
        args += ["--settings", json.dumps(sandbox_settings)]
    return args


async def run_cc_cli(
    engine: "TextEngine",
    prompt: str,
    system_prompt: str,
    model_arg: str,
    timeout: float = CC_CALL_TIMEOUT_SECONDS,
    persona_name: Optional[str] = None,
    workspace_override: Optional[str] = None,
) -> str:
    engine._ensure_cc_supported()

    binary = os.environ.get("CLAUDE_CLI_PATH") or shutil.which("claude")
    if not binary:
        raise LLMCommunicationError("Claude Code 'claude' binary not found on PATH.")

    args = engine._build_cc_args(prompt, system_prompt, model_arg)
    # cc-* must use the Claude subscription, not the metered API: strip the
    # inherited ANTHROPIC_API_KEY so `-p` mode falls through to the OAuth token.
    cc_env = build_claude_cli_env()

    workspace_dir = engine._resolve_cc_workspace(persona_name, workspace_override)
    if workspace_dir is None:
        temp_dir = tempfile.mkdtemp()
        try:
            return await engine._exec_agy(binary, args, temp_dir, timeout, label="Claude Code", env=cc_env)
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    os.makedirs(workspace_dir, exist_ok=True)
    lock = engine._cc_workspace_locks.setdefault(workspace_dir, asyncio.Lock())
    async with lock:
        return await engine._exec_agy(binary, args, workspace_dir, timeout, label="Claude Code", env=cc_env)


async def generate_cc(
    engine: "TextEngine", config: Dict[str, Any], history_object: Dict[str, Any],
    tools: Optional[List[Dict[str, Any]]] = None,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """One-shot Claude Code path. The persona prompt is delivered via
    `--system-prompt` (replace); the rendered history transcript is the `-p`
    prompt. `tools` is intentionally ignored — Claude Code uses its own sandboxed
    tools and returns final text."""
    system_prompt, history = engine._extract_system_prompt(history_object)
    prompt = engine._render_agy_prompt(history)
    model_arg = engine._cc_model_arg(config.get("model_name", ""))
    persona_name = config.get("persona_name")
    # DP-227: the orchestration layer may inject a per-run workspace (the fixr
    # self-edit clone). It takes precedence over CC_WORKSPACE_DIR.
    workspace_override = config.get("cc_workspace_override")
    workspace_dir = engine._resolve_cc_workspace(persona_name, workspace_override)

    if tools:
        logger.debug(
            "cc provider ignoring %d derpr tool(s) — Claude Code uses its own tools.",
            len(tools),
        )

    api_payload = {
        "model": config.get("model_name"),
        "cc_model": model_arg,
        "prompt_chars": len(prompt),
        "system_prompt_chars": len(system_prompt or ""),
        "tools_ignored": [
            t["function"]["name"] for t in tools or []
            if "function" in t and "name" in t["function"]
        ],
        "isolation": {
            "stdin": "devnull",
            "skip_permissions": True,
            "sandbox": global_config.CC_SANDBOX,
            "max_turns": global_config.CC_MAX_TURNS or None,
            "workspace": workspace_dir if workspace_dir else "temp-dir-per-call",
        },
    }

    try:
        raw = await engine._run_cc_cli(
            prompt, system_prompt or "", model_arg,
            persona_name=persona_name, workspace_override=workspace_override,
        )
    except LLMCommunicationError as e:
        if e.api_payload is None:
            e.api_payload = api_payload
        raise

    return {"type": "text", "content": raw.strip()}, api_payload


async def stream_cc(
    engine: "TextEngine", config: Dict[str, Any], history_object: Dict[str, Any],
    tools: Optional[List[Dict[str, Any]]] = None,
) -> AsyncIterator[Dict[str, Any]]:
    """Claude Code adapter into the unified event shape. One-shot by nature (the
    headless `claude -p` agentic run's full result arrives at process exit);
    streaming consumers get the final text as a single text_delta."""
    result, api_payload = await engine._generate_cc_response(config, history_object, tools)
    async for ev in engine._events_from_one_shot(result, api_payload):
        yield ev


class CcProvider(Provider):
    """Claude Code (cc-*) provider. POSIX-only subprocess CLI, one-shot,
    dedicated rate limiter; runs Claude Code's own sandboxed tools (ignores the
    engine's tools argument)."""

    def __init__(self, engine: "TextEngine") -> None:
        self._engine = engine

    #: name of the engine seam method (back-compat for `_get_provider_route`).
    route_method_name = "_stream_cc_response"

    def matches(self, model_name: str) -> bool:
        return model_name.startswith("cc-")

    def limiters_for(self, model_name: str) -> List[AsyncLimiter]:
        return [self._engine._cc_limiter]

    def ensure_supported(self, model_name: str) -> None:
        # Looked up live so per-test monkeypatches of the guard take effect.
        self._engine._ensure_cc_supported()

    async def stream(
        self,
        persona_config: Dict[str, Any],
        history_object: Dict[str, Any],
        tools: Optional[List[Dict[str, Any]]] = None,
        *,
        local_inference_config: Optional[Dict[str, Any]] = None,
    ) -> AsyncIterator[Dict[str, Any]]:
        async for ev in self._engine._stream_cc_response(persona_config, history_object, tools):
            yield ev
