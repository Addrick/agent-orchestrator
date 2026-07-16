# src/stream_engine.py
#
# Kobold-native local transport: chat-template rendering, the `<tool_call>`
# text protocol, and per-token SSE streaming. The single driving layer lives
# in src/engine.py (TextEngine); this module is the `local` provider beneath
# it, exactly as the SDK clients are for openai/anthropic/google.

import json
import logging
import os
import random
import re
from typing import Any, AsyncIterator, Dict, List, Optional, Tuple

import httpx

from config import global_config
from src.llm_errors import LLMCommunicationError
from src.generation_params import GenerationParams
from src.text_tool_protocol import (
    TOOL_CALL_OPEN,
    TOOL_CALL_CLOSE,
    TOOL_CALL_SYNTAX,
    decode_tool_call_payload,
)
from src.utils.model_utils import get_chat_template_for_model, get_current_kobold_model

logger = logging.getLogger(__name__)


def _template_from_instruct_tags(tags: Dict[str, Any]) -> Dict[str, Any]:
    """Build a CHAT_TEMPLATES-shape dict from kobold-lite's raw `instruct_*`
    vocabulary. The live portal forwards `localsettings.instruct_*` verbatim,
    and the named-preset registry below is mapped onto the same vocabulary, so
    this is the single translation point for both paths.

    Field semantics (kobold-lite):
      - `instruct_starttag` / `instruct_starttag_end` — user-turn open / close
      - `instruct_endtag`   / `instruct_endtag_end`   — assistant-turn open / close
      - `instruct_systag`   / `instruct_systag_end`   — system-turn open / close
      - `instruct_gentag`   — gen-time assistant prefix; carries the `<think>`
        trigger for thinking presets (or the empty `<think></think>` for the
        non-thinking variants). Falls back to assistant-open when blank.

    `stop` here is the naive tag-derived guess; named presets override it with a
    curated list (see `_PRESET_STOPS`), so callers building from the registry
    replace it. The live-passthrough path keeps the derived value.
    """
    usr_s = tags.get("instruct_starttag") or ""
    usr_e = tags.get("instruct_starttag_end") or ""
    ast_s = tags.get("instruct_endtag") or ""
    ast_e = tags.get("instruct_endtag_end") or ""
    sys_s = tags.get("instruct_systag") or ""
    sys_e = tags.get("instruct_systag_end") or ""
    ast_gen = tags.get("instruct_gentag") or ast_s
    stops = [s for s in (usr_s, ast_e, usr_e) if s]
    return {
        "system": (sys_s + "{content}" + sys_e) if (sys_s or sys_e) else "",
        "user": usr_s + "{content}" + usr_e,
        "assistant": ast_s + "{content}" + ast_e,
        "assistant_start": ast_gen,
        "stop": stops,
    }


# Named instruct presets for KoboldCPP's native `/api/extra/generate/stream`
# endpoint (the OAI endpoint batches streamed deltas into one chunk, defeating
# token-by-token rendering; the native one needs a pre-formatted prompt string).
#
# Sourced verbatim from KoboldCpp Lite's `instructpresets` array
# (lite.koboldai.net index.html, fetched 2026-06-06) and mapped from its
# field names (`user_start`/`assistant_start`/`system_start`/`assistant_gen`,
# …) onto our `instruct_*` vocabulary. Lite stores `\n` escaped and unescapes
# at use (`replaceAll("\\n","\n")`); here they are real newlines. The thinking
# control lives entirely in the gen-time prefix (`instruct_gentag`):
#   - chatml          → no gentag → model thinks by default (qwen3)
#   - chatml-nothink  → gentag emits an empty `<think></think>` to suppress it
#   - gemma4-think    → gentag opens a thought channel
#   - gemma4-nothink  → gentag opens+closes an empty thought channel
#
# Deliberate divergences from a literal kobold transcription:
#   - `gemma` blanks the system tags so the system message defers/merges into
#     the first user turn (gemma's official behavior). Kobold's preset instead
#     gives system its own `<start_of_turn>user` block; we keep DERPR's merge.
_KOBOLD_INSTRUCT_PRESETS: Dict[str, Dict[str, str]] = {
    "chatml": {
        "instruct_systag": "<|im_start|>system\n", "instruct_systag_end": "<|im_end|>\n",
        "instruct_starttag": "<|im_start|>user\n", "instruct_starttag_end": "<|im_end|>\n",
        "instruct_endtag": "<|im_start|>assistant\n", "instruct_endtag_end": "<|im_end|>\n",
    },
    "chatml-nothink": {
        "instruct_systag": "<|im_start|>system\n", "instruct_systag_end": "<|im_end|>\n",
        "instruct_starttag": "<|im_start|>user\n", "instruct_starttag_end": "<|im_end|>\n",
        "instruct_endtag": "<|im_start|>assistant\n", "instruct_endtag_end": "<|im_end|>\n",
        "instruct_gentag": "<|im_start|>assistant\n<think>\n\n</think>\n",
    },
    "gemma": {  # Gemma 2 & 3 — system blanked → deferred-merge (see note above)
        "instruct_systag": "", "instruct_systag_end": "",
        "instruct_starttag": "<start_of_turn>user\n", "instruct_starttag_end": "<end_of_turn>\n",
        "instruct_endtag": "<start_of_turn>model\n", "instruct_endtag_end": "<end_of_turn>\n",
    },
    "gemma4-think": {  # Gemma 4 (26B/31B) thinking
        "instruct_systag": "<|turn>system\n", "instruct_systag_end": "<turn|>\n",
        "instruct_starttag": "<|turn>user\n", "instruct_starttag_end": "<turn|>\n",
        "instruct_endtag": "<|turn>model\n", "instruct_endtag_end": "<turn|>\n",
        "instruct_gentag": "<|turn>model\n<|think|><|channel>thought",
    },
    "gemma4-nothink": {  # Gemma 4 (26B/31B) — empty thought channel suppresses
        "instruct_systag": "<|turn>system\n", "instruct_systag_end": "<turn|>\n",
        "instruct_starttag": "<|turn>user\n", "instruct_starttag_end": "<turn|>\n",
        "instruct_endtag": "<|turn>model\n", "instruct_endtag_end": "<turn|>\n",
        "instruct_gentag": "<|turn>model\n<|channel>thought\n<channel|>",
    },
    "gemma4-e-nothink": {  # Gemma 4 E2B/E4B — no thinking
        "instruct_systag": "<|turn>system\n", "instruct_systag_end": "<turn|>\n",
        "instruct_starttag": "<|turn>user\n", "instruct_starttag_end": "<turn|>\n",
        "instruct_endtag": "<|turn>model\n", "instruct_endtag_end": "<turn|>\n",
    },
    "llama2": {
        "instruct_systag": "", "instruct_systag_end": "",
        "instruct_starttag": "[INST] ", "instruct_starttag_end": "",
        "instruct_endtag": " [/INST]", "instruct_endtag_end": "",
    },
    "llama3": {
        "instruct_systag": "<|start_header_id|>system<|end_header_id|>\n\n", "instruct_systag_end": "<|eot_id|>",
        "instruct_starttag": "<|start_header_id|>user<|end_header_id|>\n\n", "instruct_starttag_end": "<|eot_id|>",
        "instruct_endtag": "<|start_header_id|>assistant<|end_header_id|>\n\n", "instruct_endtag_end": "<|eot_id|>",
    },
    "llama4": {
        "instruct_systag": "<|header_start|>system<|header_end|>\n\n", "instruct_systag_end": "<|eot|>",
        "instruct_starttag": "<|header_start|>user<|header_end|>\n\n", "instruct_starttag_end": "<|eot|>",
        "instruct_endtag": "<|header_start|>assistant<|header_end|>\n\n", "instruct_endtag_end": "<|eot|>",
    },
}

# Curated stop sequences per preset. The tag-derived stops from
# `_template_from_instruct_tags` are deliberately NOT used: for chatml-family
# models a bare turn-end (`<|im_end|>`) is emitted *between channels* within one
# turn (thinking → tool_call → message), so stopping on it truncates tool calls
# mid-turn (harmony_channel_stop_seq.md). These curated lists guard only against
# the model rolling into a *new* role and leave normal turn-end to the EOS token.
_PRESET_STOPS: Dict[str, List[str]] = {
    "chatml": ["<|im_start|>user", "<|im_start|>system"],
    "chatml-nothink": ["<|im_start|>user", "<|im_start|>system"],
    "gemma": ["<end_of_turn>", "<start_of_turn>"],
    "gemma4-think": ["<turn|>", "<|turn>"],
    "gemma4-nothink": ["<turn|>", "<|turn>"],
    "gemma4-e-nothink": ["<turn|>", "<|turn>"],
    "llama2": ["</s>", "[INST]"],
    "llama3": ["<|eot_id|>"],
    "llama4": ["<|eot|>"],
}


def _build_chat_templates() -> Dict[str, Dict[str, Any]]:
    """Assemble the render-time template registry from the kobold-sourced
    presets (tag strings via the shared converter) + curated stops."""
    out: Dict[str, Dict[str, Any]] = {}
    for slug, tags in _KOBOLD_INSTRUCT_PRESETS.items():
        tpl = _template_from_instruct_tags(tags)
        tpl["stop"] = list(_PRESET_STOPS[slug])
        out[slug] = tpl
    # alpaca cannot be reproduced from kobold's preset (kobold puts `\n` before
    # each tag and has no system format; DERPR puts `\n\n` after content and
    # formats system). Kept verbatim so existing alpaca personas are unchanged.
    out["alpaca"] = {
        "system": "{content}\n\n",
        "user": "### Instruction:\n{content}\n\n",
        "assistant": "### Response:\n{content}\n\n",
        "assistant_start": "### Response:\n",
        "stop": ["### Instruction:"],
    }
    return out


CHAT_TEMPLATES: Dict[str, Dict[str, Any]] = _build_chat_templates()


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
        TOOL_CALL_SYNTAX,
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
    """Kobold-native local provider (DP-206b: an engine-owned component, not
    a peer engine — TextEngine constructs one and routes every
    `model_name == "local"` request here, streaming and one-shot alike).

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
        """Resolve the chat template name with fallbacks and model-aware detection.

        Priority (highest to lowest):
        1. Persona's explicit chat_template setting
        2. KOBOLD_CHAT_TEMPLATE environment variable
        3. KOBOLD_CHAT_TEMPLATE global config setting
        4. Auto-detection from currently loaded model (if model="local")
        5. Default "chatml" fallback
        """
        # Priority 1: explicit persona setting
        explicit = persona_config.get("chat_template")
        if explicit:
            return str(explicit)

        # Priority 2-3: environment or config
        env_or_config = (
            os.environ.get("KOBOLD_CHAT_TEMPLATE")
            or getattr(global_config, "KOBOLD_CHAT_TEMPLATE", None)
        )
        if env_or_config:
            return env_or_config

        # Priority 4: auto-detect from loaded model if this persona uses "local"
        if persona_config.get("model_name") == "local":
            current_model = get_current_kobold_model()
            if current_model:
                auto_template = get_chat_template_for_model(current_model)
                if auto_template:
                    logger.info(
                        f"Auto-detected chat template '{auto_template}' for model '{current_model}'"
                    )
                    return auto_template

        # Priority 5: default fallback
        return "chatml"

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
        ctx_len = (
            kobold_extras.get("max_context_length")
            or persona_config.get("max_context_tokens")
        )
        final_ctx_len = ctx_len or persona_config.get("max_context_tokens", 2048)
        # Hard-cap at global default (131k) to avoid artificial truncation.
        final_ctx_len = min(final_ctx_len, global_config.DEFAULT_MAX_CONTEXT_TOKENS)

        payload: Dict[str, Any] = {
            "prompt": prompt,
            "max_context_length": final_ctx_len,
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
                            # Diagnostic: log kcpp's stop verdict + tail of
                            # accumulated text so we can tell EOS vs stop-word
                            # vs limit. kcpp's terminal chunk carries
                            # `stopped_eos` / `stopped_word` / `stopped_limit`
                            # and `stopping_word` when matched on a string.
                            # Without this we can't tell why generation ended.
                            stop_info = {
                                "stopped_eos": data.get("stopped_eos"),
                                "stopped_word": data.get("stopped_word"),
                                "stopped_limit": data.get("stopped_limit"),
                                "stopping_word": data.get("stopping_word"),
                                "finish_reason": data.get("finish_reason"),
                                "tail": accumulated_text[-120:],
                                "total_chars": len(accumulated_text),
                            }
                            logger.warning("KCPP_STOP %s", stop_info)
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


# Aliased to the shared protocol constants so the streaming parser and the
# agy complete-response path can never disagree on the wire tags.
_TOOL_OPEN = TOOL_CALL_OPEN
_TOOL_CLOSE = TOOL_CALL_CLOSE
_HARMONY_OPEN = "<|"
_HARMONY_CLOSE = "|>"


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
        chunk, partial = self._strip_harmony(chunk)
        if partial:
            # Re-buffer the dangling "<|..." so the next feed() can complete it.
            self._buffer = partial + self._buffer
        self.visible_text += chunk
        return chunk

    @staticmethod
    def _strip_harmony(text: str) -> Tuple[str, str]:
        """Remove complete `<|...|>` channel markers (Qwen3/harmony tags
        like `<|tool|>`, `<|tool_call|>`, `<|channel|>`, `<|im_start|>`).
        Returns (cleaned_text, partial_tail). The partial tail is any
        incomplete `<|...` at the end of `text` that should be re-buffered
        until its closing `|>` arrives — without this, leaked markers
        contaminate visible_text and self-poison future model contexts."""
        out: List[str] = []
        i = 0
        while i < len(text):
            j = text.find(_HARMONY_OPEN, i)
            if j == -1:
                out.append(text[i:])
                return "".join(out), ""
            out.append(text[i:j])
            k = text.find(_HARMONY_CLOSE, j + len(_HARMONY_OPEN))
            if k == -1:
                return "".join(out), text[j:]
            i = k + len(_HARMONY_CLOSE)
        return "".join(out), ""

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
        chunk, _partial = self._strip_harmony(chunk)
        # Drop any unclosed `<|...` at stream end — leaking a half-marker
        # is worse than swallowing it.
        self.visible_text += chunk
        return chunk

    def finalize(self) -> List[Dict[str, Any]]:
        return list(self.calls)

    def _commit_call(self, raw_json: str) -> None:
        parsed = decode_tool_call_payload(raw_json)
        if parsed is None:
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
