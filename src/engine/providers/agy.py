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

import contextlib
import json
import logging
import os
import pathlib
import re
import shutil
import tempfile
import uuid
from typing import TYPE_CHECKING, Any, AsyncIterator, Dict, List, Optional, Tuple

import asyncio

import aiohttp
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

# agy wraps its own out-of-band notices in `<SYSTEM_MESSAGE>` spans. The calls
# and the prose are both derived from the scrubbed text, so the pattern lives
# in one place — two hand-copied literals could drift into disagreeing about
# what was removed, and a tool call inside a span would then be executed while
# the prose claimed it was stripped.
_SYSTEM_MESSAGE_RE = re.compile(
    r"<SYSTEM_MESSAGE>.*?</SYSTEM_MESSAGE>", flags=re.DOTALL,
)


def strip_system_messages(text: str) -> str:
    """`text` with every `<SYSTEM_MESSAGE>…</SYSTEM_MESSAGE>` span removed."""
    return _SYSTEM_MESSAGE_RE.sub("", text)


# DP-382: agy has no image transport — no CLI flag, and `--input-format
# stream-json` rejects every content block but "text" (1.2.8). What it does have
# is its own `view_file` tool, which reads images. Headless agy denies that read
# (it cannot prompt) unless agy's settings.json carries an allow rule, and it
# does NOT auto-allow reads in its own cwd — only under %TEMP% (measured on
# 1.2.8). So derpr needs ONE static rule, `read_file(<AGY_WORKSPACES_DIR>/
# _image_calls)`, and no --dangerously-skip-permissions: commands, writes and
# every other read stay denied.
#
# An image turn writes the image alone into a fresh dir under _image_calls and
# runs agy there, one image call at a time, so the only file the rule can ever
# reach is the image being viewed. Not tempfile.mkdtemp(): %TEMP% is readable
# by every agy call, so concurrent calls could read each other's images.
_AGY_IMAGE_EXTENSIONS = {
    "image/png": ".png",
    "image/jpeg": ".jpg",
    "image/webp": ".webp",
    "image/gif": ".gif",
}
AGY_IMAGE_CALLS_DIRNAME = "_image_calls"
# agy's own settings file (not derpr config) — where the allow rule must live.
AGY_SETTINGS_PATH = pathlib.Path.home() / ".gemini" / "antigravity-cli" / "settings.json"


def agy_image_calls_root() -> str:
    return os.path.abspath(global_config.AGY_WORKSPACES_DIR / AGY_IMAGE_CALLS_DIRNAME)


def agy_image_read_rule() -> str:
    """The exact allow rule an operator adds to agy's settings.json."""
    return f"read_file({agy_image_calls_root()})"


def agy_image_read_allowed() -> bool:
    """True when agy's settings.json allows reads of exactly the image-calls
    dir. Read per call, so adding the rule takes effect without a restart.
    Exact match only: an ancestor rule would work but grant more than the one
    staged image, which is the whole point of the dir."""
    try:
        settings = json.loads(AGY_SETTINGS_PATH.read_text(encoding="utf-8"))
        allow = settings.get("permissions", {}).get("allow", [])
    except (OSError, ValueError, AttributeError):
        return False
    root = os.path.normcase(agy_image_calls_root())
    for rule in allow if isinstance(allow, list) else []:
        m = re.fullmatch(r"read_file\((.+)\)", rule) if isinstance(rule, str) else None
        if m and os.path.normcase(os.path.abspath(m.group(1))) == root:
            return True
    return False


async def load_agy_image(engine: "TextEngine", image_url: str) -> Optional[Tuple[bytes, str]]:
    """(bytes, file extension) for an image agy will be allowed to read, else
    None. A missing allow rule, failed download or unsupported type degrades to
    the "cannot see" note rather than failing the turn — the same contract as
    the API providers."""
    if not agy_image_read_allowed():
        logger.warning(
            f"agy image input needs the allow rule {agy_image_read_rule()!r} under "
            f"permissions.allow in {AGY_SETTINGS_PATH}; replying without the image."
        )
        return None
    try:
        image_bytes, mime_type = await engine._download_image(image_url)
    except aiohttp.ClientError as e:
        logger.error(f"Failed to download image from {image_url}: {e}")
        return None
    ext = _AGY_IMAGE_EXTENSIONS.get(mime_type)
    if ext is None:
        logger.warning(f"Unsupported image MIME type '{mime_type}' for agy. Skipping image.")
        return None
    return image_bytes, ext


def new_agy_image_path(ext: str) -> str:
    """Absolute path for a staged image, inside its own not-yet-created call dir."""
    call_dir = global_config.AGY_WORKSPACES_DIR / AGY_IMAGE_CALLS_DIRNAME / uuid.uuid4().hex
    return os.path.abspath(call_dir / f"image{ext}")


def render_agy_image_notice(image_path: Optional[str]) -> str:
    """The prompt line telling the model where its image is. The path is
    absolute and the tool is named on purpose: given a relative path the model
    first ran a shell command to locate the file, which headless agy denies —
    and ONE denied tool call ends the run with no output at all."""
    if image_path is None:
        return (
            "[System note: The user has attached an image that you cannot see."
            " Please inform them of this fact in your response.]"
        )
    return (
        f"[System note: The user attached an image to their latest message. It is"
        f" saved at {image_path} — use the view_file tool on exactly that path to"
        f" see it. That is the only file or tool you may use.]"
    )


@contextlib.asynccontextmanager
async def staged_agy_image(
    engine: "TextEngine", image_path: Optional[str], image_bytes: Optional[bytes],
) -> AsyncIterator[Optional[str]]:
    """Writes the image alone into its call dir and yields that dir (None when
    there is no image). Image calls are serialized: the allow rule covers all of
    _image_calls, so one call at a time keeps it holding only the image being
    viewed. The dir is always removed on exit, including agy's .antigravitycli
    link targets (same teardown as the stateless temp dir)."""
    if image_path is None or image_bytes is None:
        yield None
        return
    call_dir = os.path.dirname(image_path)
    lock = engine._agy_workspace_locks.setdefault(agy_image_calls_root(), asyncio.Lock())
    async with lock:
        try:
            os.makedirs(call_dir)
            with open(image_path, "wb") as f:
                f.write(image_bytes)
            yield call_dir
        finally:
            remove_agy_cli_link_targets(call_dir)
            shutil.rmtree(call_dir, ignore_errors=True)


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
    cleaned = strip_system_messages(text)
    calls: List[Dict[str, Any]] = []
    for inner in extract_tool_call_blocks(cleaned):
        parsed = decode_tool_call_payload(inner)
        if parsed is None:
            # Log it, or the drop is invisible in exactly the way this ticket
            # exists to fix: `strip_tool_call_blocks` removes the malformed
            # block from the prose too, so a call the model made vanishes from
            # `calls`, from `content` and from the transcript with nothing
            # anywhere to say it existed. The streaming twin has always logged
            # both of these (stream_engine._commit_call).
            logger.warning("Discarding malformed <tool_call> block: %r", inner[:200])
            continue
        # Same field policy as `_ToolCallStreamParser._commit_call`, which is
        # the point of sharing the extraction: `name` is required, `arguments`
        # defaults to {}, and a stringified args object (a common small-model
        # slip) is decoded rather than passed through. Handing a str to
        # `execute_tool(name, **args)` raises TypeError inside the loop, which
        # costs a budget slot and returns "Tool execution failed" — and the
        # divergence scaled with the batch size.
        name = parsed.get("name")
        if not name:
            logger.warning("<tool_call> block missing 'name': %r", inner[:200])
            continue
        args = parsed.get("arguments", {})
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except json.JSONDecodeError:
                args = {}
        calls.append({
            # id is agy's own: a fresh uuid, not the streaming path's
            # positional `call_<name>_<n>`, because these are minted per
            # response rather than per stream.
            "id": f"agy_{uuid.uuid4().hex}",
            "name": name,
            "arguments": args if isinstance(args, dict) else {},
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
                      persona_name: Optional[str] = None, call_dir: Optional[str] = None) -> str:
    """`call_dir`, when given, is a caller-owned single-use cwd (a DP-382 image
    call): it replaces the persona/stateless workspace, and the caller creates
    and removes it."""
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

    if call_dir is not None:
        return await engine._exec_agy(binary, args, call_dir, timeout, env=env)

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

    # DP-382: the notice rides the preamble, which fit_cli_prompt keeps whole —
    # history elision must never drop the model's only pointer to its image.
    image_url = history_object.get("current_message", {}).get("image_url")
    image = await load_agy_image(engine, image_url) if image_url else None
    image_path = new_agy_image_path(image[1]) if image else None
    if image_url:
        prompt_parts.append(render_agy_image_notice(image_path))

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
            "workspace": ("image-dir-per-call" if image_path
                          else workspace_dir if workspace_dir else "temp-dir-per-call"),
        }
    }
    if image_url:
        api_payload["image_attached"] = image is not None

    try:
        async with staged_agy_image(engine, image_path, image[0] if image else None) as call_dir:
            raw = await engine._run_agy_cli(prompt, persona_name=persona_name, call_dir=call_dir)
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
    cleaned_content = strip_system_messages(raw).strip()
    if calls:
        # DP-338: the prose beside the blocks travels with them. It is the
        # model's stated plan for the batch ("checking the node, the card and
        # the unit list before proposing the swap"), and dropping it meant the
        # next iteration re-read a transcript in which the model appeared to
        # have called a tool for no stated reason — so it re-derived the plan
        # from scratch, every iteration, off an identical history.
        result: Dict[str, Any] = {"type": "tool_calls", "calls": calls}
        prose = strip_tool_call_blocks(cleaned_content)
        if prose:
            # Key omitted, not empty — `collect_stream` omits it on a call-only
            # response, so emitting `"content": ""` here made the one-shot
            # result something the event round trip could not reproduce.
            result["content"] = prose
        return result, api_payload
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
