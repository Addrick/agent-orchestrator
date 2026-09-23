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
from ._shared import IMAGE_UNSEEN_NOTE
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
# The rule is host-wide — every agy call can use it, not just image calls — so
# the dir must hold nothing when no image call is running. An image call wipes
# _image_calls, writes the image alone into a fresh dir there and runs agy in
# it, while `AgyImageGate` keeps every other agy call in this process from
# running at the same time; the dir is emptied again on exit. Not
# tempfile.mkdtemp(): %TEMP% is readable by every agy call, always.
_AGY_IMAGE_EXTENSIONS = {
    "image/png": ".png",
    "image/jpeg": ".jpg",
    "image/webp": ".webp",
    "image/gif": ".gif",
}
AGY_IMAGE_CALLS_DIRNAME = "_image_calls"
# agy's own settings file (not derpr config) — where the allow rule must live.
AGY_SETTINGS_PATH = pathlib.Path.home() / ".gemini" / "antigravity-cli" / "settings.json"
_READ_RULE_RE = re.compile(r"read_file\((.+)\)")


def agy_image_calls_root() -> str:
    return os.path.abspath(global_config.AGY_WORKSPACES_DIR / AGY_IMAGE_CALLS_DIRNAME)


def agy_image_read_rule() -> str:
    """The exact allow rule an operator adds to agy's settings.json."""
    return f"read_file({agy_image_calls_root()})"


def _rules_covering(rules: Any, root: str) -> List[str]:
    """The `read_file(<abs path>)` rules in `rules` whose path is `root` or an
    ancestor of it. A relative path is skipped: agy resolves it against its
    own cwd (the call dir), not derpr's, so it can't be judged from here."""
    covering = []
    for rule in rules if isinstance(rules, list) else []:
        m = _READ_RULE_RE.fullmatch(rule) if isinstance(rule, str) else None
        if not m or not os.path.isabs(m.group(1)):
            continue
        path = os.path.normcase(os.path.normpath(m.group(1)))
        with contextlib.suppress(ValueError):  # paths on different drives
            if os.path.commonpath([path, root]) == path:
                covering.append(rule)
    return covering


def agy_image_read_allowed() -> bool:
    """True when agy's settings.json lets agy read the image-calls dir: an
    absolute `read_file` allow rule on it or an ancestor (an ancestor already
    grants the dir to every agy run, so refusing it would narrow nothing) and
    no deny rule over it (agy applies deny before allow). Read per call, so
    adding the rule takes effect without a restart. Logs why when False."""
    try:
        # utf-8-sig: PowerShell 5.1's `-Encoding utf8` writes a BOM.
        settings = json.loads(AGY_SETTINGS_PATH.read_text(encoding="utf-8-sig"))
        permissions = settings.get("permissions", {})
        allow, deny = permissions.get("allow", []), permissions.get("deny", [])
    except FileNotFoundError:
        allow, deny = [], []
    except (OSError, ValueError, AttributeError) as e:
        logger.warning(f"agy image input: cannot read {AGY_SETTINGS_PATH} ({e}); replying without the image.")
        return False
    root = os.path.normcase(os.path.normpath(agy_image_calls_root()))
    denied = _rules_covering(deny, root)
    if denied:
        logger.warning(f"agy image input: {AGY_SETTINGS_PATH} denies {denied}; replying without the image.")
        return False
    if not _rules_covering(allow, root):
        logger.warning(
            f"agy image input needs the allow rule {agy_image_read_rule()!r} under "
            f"permissions.allow in {AGY_SETTINGS_PATH}; replying without the image."
        )
        return False
    return True


async def load_agy_image(engine: "TextEngine", image_url: str) -> Optional[Tuple[bytes, str]]:
    """(bytes, file extension) for an image agy can view, else None. A failed
    download or unsupported type degrades to the "cannot see" note rather than
    failing the turn — the same contract as the API providers. Whether agy may
    read the file at all is the driver's gate (`model_supports_images`)."""
    try:
        image_bytes, mime_type = await engine._download_image(image_url)
    except (aiohttp.ClientError, asyncio.TimeoutError) as e:
        # aiohttp's total timeout raises a bare TimeoutError, not a ClientError.
        logger.error(f"Failed to download image from {image_url}: {e!r}")
        return None
    ext = _AGY_IMAGE_EXTENSIONS.get(mime_type)
    if ext is None:
        logger.warning(f"Unsupported image MIME type '{mime_type}' for agy. Skipping image.")
        return None
    return image_bytes, ext


def render_agy_image_notice(image_path: Optional[str]) -> str:
    """The prompt line telling the model where its image is. The path is
    absolute and the tool is named on purpose: given a relative path the model
    first ran a shell command to locate the file, which headless agy denies —
    and ONE denied tool call ends the run with no output at all."""
    if image_path is None:
        return IMAGE_UNSEEN_NOTE
    # Worded as the one exception to the tool protocol's "no other
    # tools/files", so it neither contradicts derpr's own <tool_call> tools nor
    # invites any other agy tool.
    return (
        f"[System note: The user attached an image to their latest message. It is"
        f" saved at {image_path} — open it with the view_file tool on exactly that"
        f" path. That one file is the only exception to using no other files:"
        f" open nothing else and run no commands.]"
    )


class AgyImageGate:
    """Exclusive for image calls, shared for every other agy call. The allow
    rule is host-wide, so an image may only sit in _image_calls while no other
    agy call can look. A waiting image call blocks new shared entries, so a
    busy persona can't starve it. Per process only: a second engine on the same
    data dir is covered by the wipe-before-stage in `_stage_image`, which ends
    its in-flight image call (that call then falls back to the note)."""

    def __init__(self) -> None:
        self._cond = asyncio.Condition()
        self._shared = 0
        self._exclusive = False
        self._exclusive_waiting = 0

    @contextlib.asynccontextmanager
    async def shared(self) -> AsyncIterator[None]:
        async with self._cond:
            await self._cond.wait_for(lambda: not self._exclusive and not self._exclusive_waiting)
            self._shared += 1
        try:
            yield
        finally:
            async with self._cond:
                self._shared -= 1
                self._cond.notify_all()

    @contextlib.asynccontextmanager
    async def exclusive(self) -> AsyncIterator[None]:
        async with self._cond:
            self._exclusive_waiting += 1
            try:
                await self._cond.wait_for(lambda: not self._exclusive and not self._shared)
            finally:
                self._exclusive_waiting -= 1
                self._cond.notify_all()
            self._exclusive = True
        try:
            yield
        finally:
            async with self._cond:
                self._exclusive = False
                self._cond.notify_all()


def _clear_image_calls(engine: "TextEngine") -> bool:
    """Removes everything under _image_calls — this call's dir on exit, and on
    entry whatever a killed process or a failed delete left behind. True when
    the dir ends up empty. Blocking; run off the event loop."""
    root = agy_image_calls_root()
    if not os.path.isdir(root):
        return True
    for name in os.listdir(root):
        path = os.path.join(root, name)
        engine._remove_agy_cli_link_targets(path)
        try:
            if os.path.isdir(path) and not os.path.islink(path):
                shutil.rmtree(path)
            else:
                os.remove(path)
        except OSError as e:
            logger.error(f"Could not remove agy image dir {path}: {e}")
    return not os.listdir(root)


def _stage_image(engine: "TextEngine", image: Tuple[bytes, str]) -> Optional[str]:
    """Writes the image alone into a fresh call dir and returns its absolute
    path, or None (logged) when _image_calls can't be emptied first or the
    write fails. Blocking; run off the event loop."""
    if not _clear_image_calls(engine):
        logger.error("agy image input: stale files remain in _image_calls; replying without the image.")
        return None
    image_bytes, ext = image
    image_path = os.path.join(agy_image_calls_root(), uuid.uuid4().hex, f"image{ext}")
    try:
        os.makedirs(os.path.dirname(image_path))
        with open(image_path, "wb") as f:
            f.write(image_bytes)
    except OSError as e:
        logger.error(f"agy image input: could not stage the image at {image_path}: {e}")
        return None
    return image_path


@contextlib.asynccontextmanager
async def staged_agy_image(
    engine: "TextEngine", image: Optional[Tuple[bytes, str]],
) -> AsyncIterator[Optional[str]]:
    """Yields the staged image's absolute path — its dir is the call's cwd —
    holding the image gate exclusively, and empties _image_calls on exit,
    including agy's .antigravitycli link targets (same teardown as the
    stateless temp dir). Yields None when there is no image or it could not be
    staged, and then OUTSIDE the gate: the caller's no-image agy call takes the
    gate shared, which would deadlock inside it."""
    if image is not None:
        async with engine._agy_image_gate.exclusive():
            try:
                image_path = await asyncio.to_thread(_stage_image, engine, image)
                if image_path is not None:
                    yield image_path
                    return
            finally:
                await asyncio.to_thread(_clear_image_calls, engine)
    yield None


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
        # The caller holds the image gate exclusively (staged_agy_image).
        return await engine._exec_agy(binary, args, call_dir, timeout, env=env)

    # DP-382: every other agy call can read _image_calls too, so none runs
    # while an image is staged there.
    async with engine._agy_image_gate.shared():
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


def build_agy_prompt(
    engine: "TextEngine", config: Dict[str, Any], history_object: Dict[str, Any],
    tools: Optional[List[Dict[str, Any]]], image_notice: Optional[str],
    image_attached: bool,
) -> Tuple[str, Dict[str, Any]]:
    """(prompt, api_payload) for one agy call. `image_notice` is None when the
    turn has no image."""
    system_prompt, history = engine._extract_system_prompt(history_object)

    # DP-382: the image notice leads the preamble. fit_cli_prompt keeps the
    # preamble whole and, when even that is over budget, keeps its HEAD — so
    # first is the only place neither history elision nor that truncation can
    # drop the model's one pointer to its image.
    prompt_parts = []
    if image_notice:
        prompt_parts.append(image_notice)
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

    workspace_dir = engine._resolve_agy_workspace(config.get("persona_name"))
    if image_attached:
        workspace = "image-dir-per-call"
    else:
        workspace = workspace_dir or "temp-dir-per-call"
    api_payload: Dict[str, Any] = {
        "model": config.get("model_name"),
        "prompt_chars": len(prompt),
        "history_messages_elided": elided,
        "tools": tool_names,
        "isolation": {
            "stdin": "devnull",
            "skip_permissions": False,
            "workspace": workspace,
        }
    }
    if image_notice:
        api_payload["image_attached"] = image_attached
    return prompt, api_payload


async def _run_agy_with_image(
    engine: "TextEngine", config: Dict[str, Any], history_object: Dict[str, Any],
    tools: Optional[List[Dict[str, Any]]],
) -> Tuple[Optional[str], Optional[Dict[str, Any]]]:
    """(raw, api_payload) from an agy call that can see the turn's image, or
    (None, None) when there is no image or it could not be delivered — the
    caller then runs the call without it."""
    image_url = history_object.get("current_message", {}).get("image_url")
    image = await load_agy_image(engine, image_url) if image_url else None
    async with staged_agy_image(engine, image) as image_path:
        if image_path is None:
            return None, None
        prompt, api_payload = build_agy_prompt(
            engine, config, history_object, tools, render_agy_image_notice(image_path), True,
        )
        try:
            raw = await engine._run_agy_cli(
                prompt, persona_name=config.get("persona_name"), call_dir=os.path.dirname(image_path),
            )
        except LLMCommunicationError as e:
            # Most often agy denying the read: the driver's gate cannot judge
            # every rule the way agy does (or the platform), and a denied tool
            # call exits with no output. Answer without the image rather than
            # fail the turn — and rather than let the driver retry an identical,
            # deterministic denial.
            logger.warning(f"agy image call failed ({e}); replying without the image.")
            return None, None
    return raw, api_payload


async def generate_agy(
    engine: "TextEngine", config: Dict[str, Any], history_object: Dict[str, Any],
    tools: Optional[List[Dict[str, Any]]] = None,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """One-shot agy path. DP-206 decision: agy stays one-shot-only — it is a TUI
    CLI invoked as a subprocess whose entire response arrives at process exit;
    there is no token stream to make canonical. Streaming consumers
    get it via `stream_messages`' generate_response wrap (single text_delta)."""
    raw, api_payload = await _run_agy_with_image(engine, config, history_object, tools)
    if raw is None or api_payload is None:
        has_image = bool(history_object.get("current_message", {}).get("image_url"))
        prompt, api_payload = build_agy_prompt(
            engine, config, history_object, tools,
            IMAGE_UNSEEN_NOTE if has_image else None, False,
        )
        try:
            raw = await engine._run_agy_cli(prompt, persona_name=config.get("persona_name"))
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
