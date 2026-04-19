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


def _render_prompt(
    messages: List[Dict[str, Any]],
    template_name: str,
    inference_config: Optional[Dict[str, Any]] = None
) -> Tuple[str, List[str]]:
    # 1. Select base template
    base_tpl = CHAT_TEMPLATES.get(template_name, CHAT_TEMPLATES["chatml"])
    
    # 2. Build the working template (Either from inference_config or base)
    if inference_config and (inference_config.get("user_marker") or inference_config.get("assistant_marker")):
        u_marker = inference_config.get("user_marker", base_tpl["user"].split("{content}")[0])
        a_marker = inference_config.get("assistant_marker", base_tpl["assistant_start"])
        
        # Build a clean, minimal template based on detected markers
        tpl = {
            "system": base_tpl["system"], # Keep base system if not specified
            "user": f"{u_marker}{{content}}",
            "assistant": f"{a_marker}{{content}}",
            "assistant_start": a_marker,
            "stop": list(base_tpl["stop"])
        }
        # If user marker doesn't look like it has a suffix in the base, add a newline-ish suffix
        if "{content}" in base_tpl["user"]:
            suffix = base_tpl["user"].split("{content}")[1]
            if suffix and suffix not in tpl["user"]:
                tpl["user"] += suffix
        if "{content}" in base_tpl["assistant"]:
             suffix = base_tpl["assistant"].split("{content}")[1]
             if suffix and suffix not in tpl["assistant"]:
                tpl["assistant"] += suffix
    else:
        tpl = base_tpl.copy()

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
            tool_calls = msg.get("tool_calls") or []
            if tool_calls:
                blocks = [
                    f"<tool_call>{json.dumps({'name': tc.get('name'), 'arguments': tc.get('arguments', {})})}</tool_call>"
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
    
    # 3. Append thinking trigger if requested
    if inference_config and inference_config.get("thinking_trigger"):
        # Ensure we don't double up or miss a newline
        trigger = inference_config["thinking_trigger"]
        if not final_prompt.endswith(("\n", " ")) and not trigger.startswith(("\n", " ")):
            final_prompt += "\n"
        final_prompt += trigger

    # 4. Combine stop sequences (Inference config overrides take priority)
    stop_seqs = []
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

    @staticmethod
    def _kobold_base_url() -> str:
        """Returns the koboldcpp base URL without the trailing /v1 suffix."""
        raw = os.environ.get("LOCAL_LLM_URL", global_config.LOCAL_LLM_URL)
        raw = raw.rstrip("/")
        if raw.endswith("/v1"):
            raw = raw[:-3]
        return raw

    @staticmethod
    def _build_messages(context: Dict[str, Any]) -> List[Dict[str, Any]]:
        messages: List[Dict[str, Any]] = []
        history = context.get("history") or []
        if history and history[0].get("role") == "system":
            messages.append(history[0])
            history = history[1:]
        else:
            messages.append({"role": "system", "content": context["persona_prompt"]})
        messages.extend(history)
        return messages

    async def stream_local(
        self,
        persona_config: Dict[str, Any],
        context_object: Dict[str, Any],
        tools: Optional[List[Dict[str, Any]]] = None,
        local_inference_config: Optional[Dict[str, Any]] = None,
    ) -> AsyncIterator[Dict[str, Any]]:
        """Streams from KoboldCPP's native generate/stream endpoint."""
        messages = self._build_messages(context_object)
        tool_list = [t for t in (tools or []) if t.get("function") or t.get("name")]
        if tool_list and messages and messages[0].get("role") == "system":
            messages = list(messages)
            messages[0] = {
                **messages[0],
                "content": (messages[0].get("content") or "") + _format_tools_instruction(tool_list),
            }

        template_name = (
            persona_config.get("chat_template")
            or os.environ.get("KOBOLD_CHAT_TEMPLATE")
            or getattr(global_config, "KOBOLD_CHAT_TEMPLATE", "chatml")
        )
        prompt, stop_seqs = _render_prompt(messages, template_name, local_inference_config)

        genkey = f"KCPP{random.randint(1000, 9999)}"
        payload: Dict[str, Any] = {
            "prompt": prompt,
            "max_context_length": persona_config.get("context_length") or 8192,
            "max_length": persona_config.get("max_output_tokens") or global_config.DEFAULT_TOKEN_LIMIT,
            "temperature": persona_config.get("temperature"),
            "top_p": persona_config.get("top_p"),
            "top_k": persona_config.get("top_k"),
            "stop_sequence": stop_seqs,
            "trim_stop": True,
            "genkey": genkey,
        }
        
        # Override sampling strings from the incoming request (Exact Parity)
        if local_inference_config:
            # Simple direct copy for overlapping standard params
            direct_params = [
                "temperature", "top_p", "top_k", "rep_pen", "rep_pen_range", 
                "rep_pen_slope", "min_p", "typical", "tfs"
            ]
            for p in direct_params:
                if local_inference_config.get(p) is not None:
                    payload[p] = local_inference_config[p]

        payload = {k: v for k, v in payload.items() if v is not None}

        dump_payload = {
            **payload,
            "prompt": f"<{len(prompt)} chars, template={template_name}>",
            "tools_advertised": [
                (t.get("function") or t).get("name", "unknown") for t in tool_list
            ],
        }
        yield {"type": "api_payload", "payload": dump_payload}

        url = f"{self._kobold_base_url()}/api/extra/generate/stream"
        client = await self._get_http_client()
        accumulated_text = ""
        event_count = 0
        tool_parser = _ToolCallStreamParser()

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
                async for chunk in resp.aiter_text():
                    if not chunk:
                        continue
                    buf += chunk
                    while True:
                        sep = buf.find("\n\n")
                        if sep == -1:
                            break
                        raw = buf[:sep]
                        buf = buf[sep + 2:]
                        parsed = _parse_sse_event(raw)
                        if parsed is None:
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
                        if data.get("finish_reason") == "stop":
                            break
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

        # Flush any buffered text sitting inside the parser's lookahead window.
        tail = tool_parser.flush()
        if tail:
            yield {"type": "text_delta", "text": tail}

        calls = tool_parser.finalize()
        logger.info(
            "stream_local: %d SSE events from kobold native, %d chars total, "
            "%d tool call(s) parsed",
            event_count, len(accumulated_text), len(calls),
        )
        if calls:
            yield {"type": "tool_calls", "calls": calls}

        yield {"type": "done", "full_text": tool_parser.visible_text}


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
