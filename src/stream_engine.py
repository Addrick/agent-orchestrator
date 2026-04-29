# src/stream_engine.py

import json
import logging
import os
import random
import re
from typing import Any, AsyncIterator, Dict, List, Optional, Tuple

import httpx

from config import global_config
from src.engine import LLMCommunicationError
from src.generation_params import GenerationParams

logger = logging.getLogger(__name__)


# Chat templates used when flattening OpenAI-style `messages` into a raw prompt
# for KoboldCPP's native `/api/extra/generate/stream` endpoint.
#
# The OAI endpoint in koboldcpp batches streamed deltas into a single chunk,
# defeating token-by-token rendering. The native endpoint emits per-token SSE
# events but requires a pre-formatted prompt string — so we render here.
CHAT_TEMPLATES: Dict[str, Dict[str, Any]] = {
    "chatml": {
        "system": "<|im_start|>system\n{content}<|im_end|>\n",
        "user": "<|im_start|>user\n{content}<|im_end|>\n",
        "assistant": "<|im_start|>assistant\n{content}<|im_end|>\n",
        "assistant_start": "<|im_start|>assistant\n",
        "stop": ["<|im_end|>", "<|im_start|>"],
    },
    "gemma": {
        "system": "",  # gemma has no system role — merge into first user turn
        "user": "<start_of_turn>user\n{content}<end_of_turn>\n",
        "assistant": "<start_of_turn>model\n{content}<end_of_turn>\n",
        "assistant_start": "<start_of_turn>model\n",
        "stop": ["<end_of_turn>", "<start_of_turn>"],
    },
    "llama3": {
        "system": "<|start_header_id|>system<|end_header_id|>\n\n{content}<|eot_id|>",
        "user": "<|start_header_id|>user<|end_header_id|>\n\n{content}<|eot_id|>",
        "assistant": "<|start_header_id|>assistant<|end_header_id|>\n\n{content}<|eot_id|>",
        "assistant_start": "<|start_header_id|>assistant<|end_header_id|>\n\n",
        "stop": ["<|eot_id|>"],
    },
    "alpaca": {
        "system": "{content}\n\n",
        "user": "### Instruction:\n{content}\n\n",
        "assistant": "### Response:\n{content}\n\n",
        "assistant_start": "### Response:\n",
        "stop": ["### Instruction:"],
    },
}


def _template_from_instruct_tags(tags: Dict[str, Any]) -> Dict[str, Any]:
    """Build a CHAT_TEMPLATES-shape dict from raw kobold-lite instruct tag
    strings (system_start/end, user_start/end, assistant_start/end). Stops
    fall back to whatever closing tags were provided."""
    sys_s = tags.get("system_start") or ""
    sys_e = tags.get("system_end") or ""
    usr_s = tags.get("user_start") or ""
    usr_e = tags.get("user_end") or ""
    ast_s = tags.get("assistant_start") or ""
    ast_e = tags.get("assistant_end") or ""
    stops = [s for s in (usr_s, ast_e, usr_e) if s]
    return {
        "system": (sys_s + "{content}" + sys_e) if (sys_s or sys_e) else "",
        "user": usr_s + "{content}" + usr_e,
        "assistant": ast_s + "{content}" + ast_e,
        "assistant_start": ast_s,
        "stop": stops,
    }


def _render_prompt(
    messages: List[Dict[str, Any]],
    template_name: str,
    inference_config: Optional[Dict[str, Any]] = None
) -> Tuple[str, List[str]]:
    instruct_tags = (inference_config or {}).get("instruct_tags")
    if isinstance(instruct_tags, dict) and any(instruct_tags.values()):
        tpl = _template_from_instruct_tags(instruct_tags)
    else:
        tpl = CHAT_TEMPLATES.get(template_name, CHAT_TEMPLATES["chatml"])

    parts: List[str] = []
    deferred_system = ""
    for msg in messages:
        role = msg.get("role", "")
        content = msg.get("content", "") or ""
        if role == "system":
            if tpl["system"]:
                parts.append(tpl["system"].format(content=content))
            else:
                deferred_system = content + "\n\n"
        elif role == "user":
            text = (deferred_system + content) if deferred_system else content
            parts.append(tpl["user"].format(content=text))
            deferred_system = ""
        elif role == "assistant":
            tool_calls = [tc for tc in (msg.get("tool_calls") or []) if tc.get("name")]
            if tool_calls:
                blocks = [
                    f"<tool_call>{json.dumps({'name': tc['name'], 'arguments': tc.get('arguments', {})})}</tool_call>"
                    for tc in tool_calls
                ]
                serialized = (content + "\n" if content else "") + "\n".join(blocks)
                parts.append(tpl["assistant"].format(content=serialized))
            else:
                parts.append(tpl["assistant"].format(content=content))
        elif role == "tool":
            name = msg.get("name", "tool")
            parts.append(tpl["user"].format(
                content=f"<tool_result name=\"{name}\">{content}</tool_result>"
            ))

    final_prompt = "".join(parts) + tpl["assistant_start"]

    stop_seqs: List[str] = []
    if inference_config and inference_config.get("stop_sequence"):
        stop_seqs.extend(inference_config["stop_sequence"])
    for s in tpl["stop"]:
        if s not in stop_seqs:
            stop_seqs.append(s)

    return final_prompt, stop_seqs


def _format_tools_instruction(tools: List[Dict[str, Any]]) -> str:
    """Build a system-prompt addendum describing available tools and the
    `<tool_call>` JSON syntax the model must emit to invoke them."""
    lines = [
        "",
        "# Tool Use",
        "You have access to the tools listed below. To call a tool, emit exactly:",
        '<tool_call>{"name": "TOOL_NAME", "arguments": {"arg1": "value", ...}}</tool_call>',
        "Emit one `<tool_call>` block per call. Emit them verbatim with no",
        "surrounding code fences. After a tool returns, continue the conversation",
        "normally. If no tool is needed, just answer the user.",
        "",
        "## Available tools",
    ]
    for t in tools:
        fn = t.get("function", t)
        name = fn.get("name", "unknown")
        desc = fn.get("description", "").strip()
        params = fn.get("parameters") or {}
        props = params.get("properties") or {}
        required = set(params.get("required") or [])
        lines.append(f"- **{name}**: {desc}")
        for pname, pspec in props.items():
            ptype = pspec.get("type", "any")
            pdesc = pspec.get("description", "").strip()
            req = " (required)" if pname in required else ""
            lines.append(f"    - `{pname}` ({ptype}){req}: {pdesc}")
    return "\n".join(lines)


class StreamEngine:
    """Streaming counterpart to TextEngine.

    Targets KoboldCPP's native `/api/extra/generate/stream` endpoint so that
    tokens arrive one at a time. The OpenAI-compat endpoint on koboldcpp
    batches content into one delta, which breaks UI streaming.
    """

    def __init__(self) -> None:
        self._http_client: Optional[httpx.AsyncClient] = None

    def supports(self, model_name: str) -> bool:
        """True if the given model can be streamed by this engine."""
        return model_name == "local"

    async def _get_http_client(self) -> httpx.AsyncClient:
        if self._http_client is None:
            self._http_client = httpx.AsyncClient(timeout=httpx.Timeout(600.0, connect=15.0))
        return self._http_client

    async def aclose(self) -> None:
        if self._http_client is not None:
            await self._http_client.aclose()
            self._http_client = None

    @staticmethod
    def _kobold_base_url() -> str:
        """Returns the koboldcpp base URL without the trailing /v1 suffix."""
        raw = os.environ.get("LOCAL_LLM_URL", global_config.LOCAL_LLM_URL)
        raw = raw.rstrip("/")
        if raw.endswith("/v1"):
            raw = raw[:-3]
        return raw

    @staticmethod
    def _build_messages(history_object: Dict[str, Any]) -> List[Dict[str, Any]]:
        messages: List[Dict[str, Any]] = []
        history = history_object.get("message_history", history_object.get("history", []))
        if history and history[0].get("role") == "system":
            messages.append(history[0])
            history = history[1:]
        else:
            messages.append({"role": "system", "content": history_object["persona_prompt"]})
        messages.extend(history)
        return messages

    @staticmethod
    def _resolve_template_name(persona_config: Dict[str, Any]) -> str:
        return (
            persona_config.get("chat_template")
            or os.environ.get("KOBOLD_CHAT_TEMPLATE")
            or getattr(global_config, "KOBOLD_CHAT_TEMPLATE", "chatml")
            or "chatml"
        )

    @staticmethod
    def _params_from_legacy_dicts(
        persona_config: Dict[str, Any],
        local_inference_config: Optional[Dict[str, Any]],
    ) -> GenerationParams:
        """Bridge legacy (persona_config dict + local_inference_config) callers
        into a GenerationParams. local_inference_config keys override the
        persona_config defaults — same precedence as the old payload builder.
        Kobold-only knobs land in `provider_extras['kobold']`."""
        lic = local_inference_config or {}
        extras: Dict[str, Any] = {}
        for k in ("rep_pen", "rep_pen_range", "rep_pen_slope",
                  "min_p", "typical", "tfs", "max_context_length"):
            if lic.get(k) is not None:
                extras[k] = lic[k]
        if lic.get("stop_sequence"):
            extras["stop_sequence"] = list(lic["stop_sequence"])

        persona_kobold_extras = (persona_config.get("provider_extras") or {}).get("kobold") or {}
        persona_tags = persona_kobold_extras.get("instruct_tags")
        req_tags = lic.get("instruct_tags")
        if isinstance(req_tags, dict) and any(req_tags.values()):
            extras["instruct_tags"] = req_tags
        elif isinstance(persona_tags, dict) and any(persona_tags.values()):
            extras["instruct_tags"] = persona_tags

        def _override(key: str, fallback_key: Optional[str] = None) -> Any:
            val = lic.get(key)
            if val is not None:
                return val
            return persona_config.get(fallback_key or key)

        return GenerationParams(
            temperature=_override("temperature"),
            top_p=_override("top_p"),
            top_k=_override("top_k"),
            max_tokens=_override("max_tokens", "max_output_tokens"),
            provider_extras={"kobold": extras} if extras else {},
        )

    def _build_kobold_payload(
        self,
        *,
        persona_config: Dict[str, Any],
        prompt: str,
        stop_seqs: List[str],
        params: GenerationParams,
        template_name: str,
        tools_advertised: List[str],
    ) -> Tuple[Dict[str, Any], Dict[str, Any], str]:
        """Assemble the kobold native /generate/stream POST body and a
        log-safe dump (prompt summarized, tools listed by name)."""
        genkey = f"KCPP{random.randint(1000, 9999)}"
        kobold_extras = params.get_provider_extras("kobold")
        # Token budget comes from kobold-lite's UI slider
        # (params.max_context_length). Persona.context_length is a *turn count*
        # for the history window — a different concept — so do not use it here.
        ctx_len = kobold_extras.get("max_context_length")
        payload: Dict[str, Any] = {
            "prompt": prompt,
            "max_context_length": ctx_len,
            "max_length": params.max_tokens
                or persona_config.get("max_output_tokens")
                or global_config.DEFAULT_TOKEN_LIMIT,
            "temperature": params.temperature,
            "top_p": params.top_p,
            "top_k": params.top_k,
            "stop_sequence": stop_seqs,
            "trim_stop": True,
            "genkey": genkey,
        }
        for p in ("rep_pen", "rep_pen_range", "rep_pen_slope",
                  "min_p", "typical", "tfs"):
            if kobold_extras.get(p) is not None:
                payload[p] = kobold_extras[p]
        payload = {k: v for k, v in payload.items() if v is not None}
        dump_payload: Dict[str, Any] = {
            **payload,
            "prompt": f"<{len(prompt)} chars, template={template_name}>",
            "tools_advertised": list(tools_advertised),
        }
        return payload, dump_payload, genkey

    async def _kobold_stream(
        self,
        payload: Dict[str, Any],
        dump_payload: Dict[str, Any],
        genkey: str,
    ) -> AsyncIterator[Dict[str, Any]]:
        """Run the kobold native SSE stream and emit the unified event shape."""
        yield {"type": "api_payload", "payload": dump_payload}

        url = f"{self._kobold_base_url()}/api/extra/generate/stream"
        client = await self._get_http_client()
        accumulated_text = ""
        event_count = 0
        tool_parser = _ToolCallStreamParser()
        finished_cleanly = False

        try:
            async with client.stream("POST", url, json=payload) as resp:
                if resp.status_code != 200:
                    body = await resp.aread()
                    raise LLMCommunicationError(
                        f"Kobold native stream returned {resp.status_code}: "
                        f"{body.decode('utf-8', errors='replace')[:500]}",
                        api_payload=dump_payload,
                    )
                buf = ""
                done = False
                async for chunk in resp.aiter_text():
                    if not chunk:
                        continue
                    buf += chunk
                    while not done:
                        sep = buf.find("\n\n")
                        if sep == -1:
                            break
                        raw = buf[:sep]
                        buf = buf[sep + 2:]
                        parsed = _parse_sse_event(raw)
                        if parsed is None:
                            if raw.strip():
                                logger.debug("Failed to parse SSE event: %r", raw)
                            continue
                        event_count += 1
                        etype, data = parsed
                        if etype != "message":
                            continue
                        token = data.get("token", "")
                        if token:
                            accumulated_text += token
                            visible = tool_parser.feed(token)
                            if visible:
                                yield {"type": "text_delta", "text": visible}
                        elif data.get("finish_reason") != "stop":
                            # Log empty tokens that aren't terminal
                            logger.debug("Received empty token in message event: %r", data)
                        if data.get("finish_reason") == "stop":
                            done = True
                    if done:
                        break
                finished_cleanly = True
        except httpx.HTTPError as e:
            logger.error(f"Kobold stream transport error: {e}", exc_info=True)
            raise LLMCommunicationError(
                f"Kobold native stream transport error: {e}",
                api_payload=dump_payload,
            ) from e
        except LLMCommunicationError:
            raise
        except Exception as e:
            logger.error(f"Unexpected kobold stream error: {e}", exc_info=True)
            raise LLMCommunicationError(
                "An unexpected error occurred with the Kobold native stream.",
                api_payload=dump_payload,
            ) from e
        finally:
            if not finished_cleanly:
                # Client disconnect or error — tell koboldcpp to stop burning
                # tokens on this genkey. Best-effort; swallow any errors.
                try:
                    await client.post(
                        f"{self._kobold_base_url()}/api/extra/abort",
                        json={"genkey": genkey},
                        timeout=5.0,
                    )
                except Exception:
                    pass

        tail = tool_parser.flush()
        if tail:
            yield {"type": "text_delta", "text": tail}

        calls = tool_parser.finalize()
        logger.info(
            "kobold stream: %d SSE events, %d chars total, %d tool call(s) parsed",
            event_count, len(accumulated_text), len(calls),
        )
        if calls:
            yield {"type": "tool_calls", "calls": calls}

        yield {"type": "done", "full_text": tool_parser.visible_text}

    def stream_messages(
        self,
        persona_config: Dict[str, Any],
        messages: List[Dict[str, Any]],
        params: GenerationParams,
        tools: Optional[List[Dict[str, Any]]] = None,
    ) -> AsyncIterator[Dict[str, Any]]:
        """Phase B entry — render OAI-style messages via the persona's chat
        template and stream from kobold native. Tool list folds into the
        system prompt as a `<tool_call>` instruction block."""
        # Returns the underlying _kobold_stream generator directly — wrapping
        # in another `async def` would block aclose() from reaching the
        # native stream's finally (the abort POST). See test
        # test_stream_local_aborts_upstream_when_caller_breaks_early.
        tool_list = [t for t in (tools or []) if t.get("function") or t.get("name")]
        rendered_messages = list(messages)
        if tool_list and rendered_messages and rendered_messages[0].get("role") == "system":
            rendered_messages[0] = {
                **rendered_messages[0],
                "content": (rendered_messages[0].get("content") or "")
                    + _format_tools_instruction(tool_list),
            }
        template_name = self._resolve_template_name(persona_config)
        kobold_extras = params.get_provider_extras("kobold")
        prompt, stop_seqs = _render_prompt(rendered_messages, template_name, kobold_extras)

        payload, dump_payload, genkey = self._build_kobold_payload(
            persona_config=persona_config,
            prompt=prompt,
            stop_seqs=stop_seqs,
            params=params,
            template_name=template_name,
            tools_advertised=[(t.get("function") or t).get("name", "unknown")
                              for t in tool_list],
        )
        return self._kobold_stream(payload, dump_payload, genkey)

    def stream_prompt(
        self,
        persona_config: Dict[str, Any],
        rendered_prompt: str,
        params: GenerationParams,
        *,
        stop_sequences: Optional[List[str]] = None,
        tools_advertised: Optional[List[str]] = None,
    ) -> AsyncIterator[Dict[str, Any]]:
        """Phase B entry — stream from a caller-rendered prompt. Used by the
        portal where kobold-lite owns templating; the engine never rewraps."""
        payload, dump_payload, genkey = self._build_kobold_payload(
            persona_config=persona_config,
            prompt=rendered_prompt,
            stop_seqs=list(stop_sequences or []),
            params=params,
            template_name="<caller>",
            tools_advertised=list(tools_advertised or []),
        )
        return self._kobold_stream(payload, dump_payload, genkey)

    def stream_local(
        self,
        persona_config: Dict[str, Any],
        history_object: Dict[str, Any],
        tools: Optional[List[Dict[str, Any]]] = None,
        local_inference_config: Optional[Dict[str, Any]] = None,
    ) -> AsyncIterator[Dict[str, Any]]:
        """Legacy entry preserved for backwards compatibility — bridges
        history_object + local_inference_config callers into stream_messages.
        New callers should construct a GenerationParams and call
        `stream_messages` / `stream_prompt` directly."""
        messages = self._build_messages(history_object)
        params = self._params_from_legacy_dicts(persona_config, local_inference_config)
        return self.stream_messages(persona_config, messages, params, tools)


_EVENT_RE = re.compile(r"^event:\s*(.*)$", re.MULTILINE)
_DATA_RE = re.compile(r"^data:\s*(.*)$", re.MULTILINE)


def _parse_sse_event(raw: str) -> Optional[Tuple[str, Dict[str, Any]]]:
    etype_m = _EVENT_RE.search(raw)
    data_m = _DATA_RE.search(raw)
    if not etype_m or not data_m:
        return None
    try:
        data = json.loads(data_m.group(1))
    except json.JSONDecodeError:
        return None
    return etype_m.group(1).strip(), data


_TOOL_OPEN = "<tool_call>"
_TOOL_CLOSE = "</tool_call>"


class _ToolCallStreamParser:
    """Incremental parser that splits streamed text into visible content and
    `<tool_call>{json}</tool_call>` blocks. Holds enough lookahead to avoid
    leaking a partially-arrived opening tag to the consumer."""

    def __init__(self) -> None:
        self._buffer = ""           # holds tail bytes that may start a tag
        self._inside = False        # currently between <tool_call> and </tool_call>
        self._inner = ""            # accumulated content inside a tool_call block
        self.visible_text = ""      # full de-tag-ed text shown to user
        self.calls: List[Dict[str, Any]] = []

    def feed(self, token: str) -> str:
        """Consume a new token chunk; return whatever is safe to show now."""
        self._buffer += token
        out_parts: List[str] = []
        while self._buffer:
            if self._inside:
                close_at = self._buffer.find(_TOOL_CLOSE)
                if close_at == -1:
                    self._inner += self._buffer
                    self._buffer = ""
                    break
                self._inner += self._buffer[:close_at]
                self._buffer = self._buffer[close_at + len(_TOOL_CLOSE):]
                self._commit_call(self._inner)
                self._inner = ""
                self._inside = False
            else:
                open_at = self._buffer.find(_TOOL_OPEN)
                if open_at == -1:
                    # Keep a short tail in case an opening tag is partially buffered.
                    safe_len = max(0, len(self._buffer) - (len(_TOOL_OPEN) - 1))
                    out_parts.append(self._buffer[:safe_len])
                    self._buffer = self._buffer[safe_len:]
                    break
                out_parts.append(self._buffer[:open_at])
                self._buffer = self._buffer[open_at + len(_TOOL_OPEN):]
                self._inside = True
        chunk = "".join(out_parts)
        self.visible_text += chunk
        return chunk

    def flush(self) -> str:
        """Emit remaining buffered text once the stream has ended."""
        if self._inside:
            # Unterminated tool_call — treat as a parse failure, keep it visible.
            leftover = _TOOL_OPEN + self._inner + self._buffer
            self._inner = ""
            self._buffer = ""
            self._inside = False
            self.visible_text += leftover
            return leftover
        chunk = self._buffer
        self._buffer = ""
        self.visible_text += chunk
        return chunk

    def finalize(self) -> List[Dict[str, Any]]:
        return list(self.calls)

    def _commit_call(self, raw_json: str) -> None:
        try:
            parsed = json.loads(raw_json.strip())
        except json.JSONDecodeError:
            logger.warning("Discarding malformed <tool_call> block: %r", raw_json[:200])
            return
        name = parsed.get("name")
        if not name:
            logger.warning("<tool_call> missing 'name' field: %r", raw_json[:200])
            return
        args = parsed.get("arguments", {})
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except json.JSONDecodeError:
                args = {}
        self.calls.append({
            "id": f"call_{name}_{len(self.calls)}",
            "name": name,
            "arguments": args if isinstance(args, dict) else {},
        })
