import asyncio
import json
from typing import AsyncIterator, List, Optional
from unittest.mock import MagicMock

import httpx
import pytest

from src.stream_engine import StreamEngine, _render_prompt
from src.engine import LLMCommunicationError
from src.generation_params import GenerationParams


def test_render_prompt_default_chatml():
    # Default uses ChatML base
    prompt, stop = _render_prompt([
        {"role": "system", "content": "SysPrompt"},
        {"role": "user", "content": "UserMsg"}
    ], "chatml")

    assert "<|im_start|>system\nSysPrompt<|im_end|>" in prompt
    assert "<|im_start|>user\nUserMsg<|im_end|>" in prompt
    assert prompt.endswith("<|im_start|>assistant\n")
    assert "<|im_end|>" in stop


def test_render_prompt_template_selection():
    # Non-default template is picked up by name. Marker/thinking-trigger
    # overrides were intentionally dropped — the persona's chat_template owns
    # rendering and we pass through to kobold-lite otherwise.
    messages = [{"role": "user", "content": "Hello"}]
    prompt, _ = _render_prompt(messages, "gemma")
    assert "<start_of_turn>user\nHello<end_of_turn>" in prompt
    assert prompt.endswith("<start_of_turn>model\n")


def test_render_prompt_ignores_marker_overrides():
    # user_marker / assistant_marker / thinking_trigger in inference_config
    # are silently ignored; we never let runtime data reshape the template.
    messages = [{"role": "user", "content": "Hello"}]
    prompt, _ = _render_prompt(messages, "chatml", {
        "user_marker": "USER: ",
        "assistant_marker": "ASSISTANT: ",
        "thinking_trigger": "<|think|>",
    })
    assert "<|im_start|>user\nHello<|im_end|>" in prompt
    assert "USER:" not in prompt
    assert "<|think|>" not in prompt


def test_render_prompt_tool_call_serialization():
    messages = [
        {"role": "assistant", "content": "Checking...", "tool_calls": [
            {"name": "get_weather", "arguments": {"city": "Berlin"}}
        ]}
    ]
    prompt, _ = _render_prompt(messages, "chatml")

    assert "Checking..." in prompt
    assert "<tool_call>{\"name\": \"get_weather\", \"arguments\": {\"city\": \"Berlin\"}}</tool_call>" in prompt


def test_render_prompt_stop_sequence_merging():
    messages = [{"role": "user", "content": "test"}]
    inference_config = {
        "stop_sequence": ["OVERRIDE_STOP"]
    }
    _, stop = _render_prompt(messages, "chatml", inference_config)

    assert "OVERRIDE_STOP" in stop
    assert "<|im_start|>" in stop  # Should also contain base stops from ChatML
    assert stop[0] == "OVERRIDE_STOP"  # Priority


# --------------------------------------------------------------------------
# stream_local end-to-end
#
# Coverage-prep before portal_engine_reintegration Phase B. The kobold-native
# stream is the only async-iterator provider surface today and is the closest
# thing to the planned `TextProvider.stream_prompt`. These tests pin the
# event-stream contract so the migration to a unified provider ABC has a
# verifiable starting point. See memory/project/plans/portal_engine_reintegration.md.
# --------------------------------------------------------------------------


def _sse_token(token: str, *, finish_reason: Optional[str] = None) -> str:
    """Render one kobold-native SSE event: `event: message\\ndata: {...}\\n\\n`."""
    payload = {"token": token}
    if finish_reason is not None:
        payload["finish_reason"] = finish_reason
    return f"event: message\ndata: {json.dumps(payload)}\n\n"


class _FakeResp:
    """Stand-in for httpx.Response in stream context."""

    def __init__(self, *, status_code: int = 200,
                 chunks: Optional[List[str]] = None,
                 body: bytes = b"") -> None:
        self.status_code = status_code
        self._chunks = chunks or []
        self._body = body

    async def aread(self) -> bytes:
        return self._body

    async def aiter_text(self) -> AsyncIterator[str]:
        for c in self._chunks:
            yield c


class _FakeStreamCtx:
    def __init__(self, resp: _FakeResp,
                 captured: Optional[dict] = None,
                 raise_on_iter: Optional[Exception] = None) -> None:
        self._resp = resp
        self._captured = captured
        self._raise_on_iter = raise_on_iter

    async def __aenter__(self) -> _FakeResp:
        if self._raise_on_iter is not None:
            raise self._raise_on_iter
        return self._resp

    async def __aexit__(self, *a) -> bool:
        return False


class _FakeClient:
    """Minimal httpx.AsyncClient stand-in for StreamEngine."""

    def __init__(self, resp: _FakeResp,
                 raise_on_iter: Optional[Exception] = None) -> None:
        self._resp = resp
        self._raise_on_iter = raise_on_iter
        self.posts: List[dict] = []
        self.last_stream: Optional[dict] = None

    def stream(self, method: str, url: str, json=None, **kw):
        self.last_stream = {"method": method, "url": url, "json": json}
        return _FakeStreamCtx(self._resp, raise_on_iter=self._raise_on_iter)

    async def post(self, url: str, json=None, timeout=None, **kw):
        self.posts.append({"url": url, "json": json})
        return MagicMock()


def _make_engine(resp: _FakeResp,
                 raise_on_iter: Optional[Exception] = None) -> tuple[StreamEngine, _FakeClient]:
    engine = StreamEngine()
    client = _FakeClient(resp, raise_on_iter=raise_on_iter)
    engine._http_client = client  # bypass _get_http_client
    return engine, client


def _persona_config() -> dict:
    return {
        "model_name": "local",
        "max_output_tokens": 128,
        "temperature": 0.7,
        "top_p": 0.9,
        "top_k": 40,
        "chat_template": "chatml",
    }


def _history(user_text: str = "hi") -> dict:
    return {
        "persona_prompt": "you are test",
        "message_history": [{"role": "user", "content": user_text}],
        "current_message": {"text": user_text, "image_url": None},
    }


async def _drain(it) -> List[dict]:
    out = []
    async for ev in it:
        out.append(ev)
    return out


@pytest.mark.asyncio
async def test_stream_local_first_event_is_api_payload():
    # The dump payload event must lead so `_store_api_request` sees it before
    # any text deltas — same contract as TextEngine's non-streaming path.
    resp = _FakeResp(chunks=[_sse_token("hi"), _sse_token("", finish_reason="stop")])
    engine, _ = _make_engine(resp)

    events = await _drain(engine.stream_local(_persona_config(), _history()))
    assert events[0]["type"] == "api_payload"
    payload = events[0]["payload"]
    assert payload["temperature"] == 0.7
    assert payload["top_p"] == 0.9
    assert payload["top_k"] == 40
    # Prompt is summarized, not raw — protects logs from leaking content.
    assert isinstance(payload["prompt"], str)
    assert payload["prompt"].startswith("<") and "chars" in payload["prompt"]
    assert payload["tools_advertised"] == []


@pytest.mark.asyncio
async def test_stream_local_emits_text_deltas_in_order():
    resp = _FakeResp(chunks=[
        _sse_token("hello "),
        _sse_token("world"),
        _sse_token("", finish_reason="stop"),
    ])
    engine, _ = _make_engine(resp)

    events = await _drain(engine.stream_local(_persona_config(), _history()))
    deltas = [e["text"] for e in events if e["type"] == "text_delta"]
    assert "".join(deltas) == "hello world"


@pytest.mark.asyncio
async def test_stream_local_terminates_on_done_event():
    # The done event must come last and carry the full visible text.
    resp = _FakeResp(chunks=[
        _sse_token("foo"),
        _sse_token("bar"),
        _sse_token("", finish_reason="stop"),
    ])
    engine, _ = _make_engine(resp)

    events = await _drain(engine.stream_local(_persona_config(), _history()))
    assert events[-1]["type"] == "done"
    assert events[-1]["full_text"] == "foobar"


@pytest.mark.asyncio
async def test_stream_local_extracts_tool_call_from_inline_block():
    # `<tool_call>{...}</tool_call>` arriving mid-stream must surface as a
    # tool_calls event after the visible text drains, with the markup itself
    # excluded from the user-visible deltas.
    body = (
        '<tool_call>{"name": "get_weather", "arguments": {"city": "Berlin"}}</tool_call>'
    )
    resp = _FakeResp(chunks=[
        _sse_token("Looking up... "),
        _sse_token(body),
        _sse_token(" done."),
        _sse_token("", finish_reason="stop"),
    ])
    engine, _ = _make_engine(resp)

    events = await _drain(engine.stream_local(_persona_config(), _history()))
    visible = "".join(e["text"] for e in events if e["type"] == "text_delta")
    assert "<tool_call>" not in visible
    assert "</tool_call>" not in visible
    assert "Looking up..." in visible and "done." in visible

    tool_events = [e for e in events if e["type"] == "tool_calls"]
    assert len(tool_events) == 1
    calls = tool_events[0]["calls"]
    assert len(calls) == 1
    assert calls[0]["name"] == "get_weather"
    assert calls[0]["arguments"] == {"city": "Berlin"}


@pytest.mark.asyncio
async def test_stream_local_finish_reason_stops_processing():
    # Tokens after `finish_reason: "stop"` must be ignored.
    resp = _FakeResp(chunks=[
        _sse_token("kept"),
        _sse_token("", finish_reason="stop"),
        _sse_token("dropped"),
    ])
    engine, _ = _make_engine(resp)

    events = await _drain(engine.stream_local(_persona_config(), _history()))
    visible = "".join(e["text"] for e in events if e["type"] == "text_delta")
    assert visible == "kept"
    assert "dropped" not in visible


@pytest.mark.asyncio
async def test_stream_local_raises_on_non_200():
    resp = _FakeResp(status_code=500, body=b"upstream broken")
    engine, _ = _make_engine(resp)

    with pytest.raises(LLMCommunicationError) as ei:
        async for _ in engine.stream_local(_persona_config(), _history()):
            pass
    assert "500" in str(ei.value)
    assert "upstream broken" in str(ei.value)
    # api_payload preserved on the exception for the dump-last command path.
    assert ei.value.api_payload is not None


@pytest.mark.asyncio
async def test_stream_local_aborts_upstream_when_caller_breaks_early():
    # Caller exits the iterator before finish_reason arrives → finished_cleanly
    # stays False → finally block must POST to /api/extra/abort with the genkey.
    resp = _FakeResp(chunks=[
        _sse_token("partial1"),
        _sse_token("partial2"),
        _sse_token("partial3"),
        # No stop sentinel — caller will break before this exhausts.
    ])
    engine, client = _make_engine(resp)

    it = engine.stream_local(_persona_config(), _history())
    seen = 0
    async for _ in it:
        seen += 1
        if seen >= 2:
            break
    await it.aclose()

    abort_posts = [p for p in client.posts if p["url"].endswith("/api/extra/abort")]
    assert len(abort_posts) == 1
    # genkey threaded through so kcpp stops *this* generation, not all of them.
    assert "genkey" in abort_posts[0]["json"]
    assert abort_posts[0]["json"]["genkey"].startswith("KCPP")


@pytest.mark.asyncio
async def test_stream_local_transport_error_raises_llm_error():
    engine, _ = _make_engine(
        _FakeResp(),
        raise_on_iter=httpx.ConnectError("upstream down"),
    )
    with pytest.raises(LLMCommunicationError) as ei:
        async for _ in engine.stream_local(_persona_config(), _history()):
            pass
    assert "transport error" in str(ei.value).lower()
    assert ei.value.api_payload is not None


@pytest.mark.asyncio
async def test_stream_local_appends_tool_instructions_to_system_prompt():
    # Tool list must be folded into system prompt so the local model knows the
    # `<tool_call>` syntax. The forwarded payload's prompt length grows
    # measurably vs. the no-tools baseline.
    resp = _FakeResp(chunks=[_sse_token("", finish_reason="stop")])

    engine_a, client_a = _make_engine(resp)
    events_a = await _drain(engine_a.stream_local(_persona_config(), _history()))
    base_chars = events_a[0]["payload"]["prompt"]

    engine_b, client_b = _make_engine(_FakeResp(chunks=[_sse_token("", finish_reason="stop")]))
    tools = [{
        "function": {
            "name": "get_weather",
            "description": "fetch weather",
            "parameters": {"type": "object", "properties": {"city": {"type": "string"}},
                           "required": ["city"]},
        }
    }]
    events_b = await _drain(engine_b.stream_local(_persona_config(), _history(), tools=tools))
    assert events_b[0]["payload"]["tools_advertised"] == ["get_weather"]
    # api_payload.prompt is the summary string `<N chars, template=...>`.
    # Pull the integer back out and confirm tool-instruction text grew it.
    base_n = int(base_chars.split()[0].lstrip("<"))
    with_tools_n = int(events_b[0]["payload"]["prompt"].split()[0].lstrip("<"))
    assert with_tools_n > base_n


@pytest.mark.asyncio
async def test_stream_local_no_tool_call_yields_no_tool_calls_event():
    resp = _FakeResp(chunks=[
        _sse_token("plain reply"),
        _sse_token("", finish_reason="stop"),
    ])
    engine, _ = _make_engine(resp)
    events = await _drain(engine.stream_local(_persona_config(), _history()))
    assert all(e["type"] != "tool_calls" for e in events)


# --------------------------------------------------------------------------
# Phase B — typed entries: stream_messages and stream_prompt
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_stream_messages_typed_renders_via_persona_template():
    # GenerationParams replaces the legacy persona_config + local_inference_config
    # cocktail. The forwarded payload still gets the same temperature, top_p,
    # top_k, prompt summary, and tool advertising as the legacy entry.
    resp = _FakeResp(chunks=[_sse_token("ok"), _sse_token("", finish_reason="stop")])
    engine, _ = _make_engine(resp)

    messages = [
        {"role": "system", "content": "you are test"},
        {"role": "user", "content": "hi"},
    ]
    params = GenerationParams(temperature=0.42, top_p=0.88, top_k=11, max_tokens=64)
    events = await _drain(engine.stream_messages(_persona_config(), messages, params))

    payload = events[0]["payload"]
    assert payload["temperature"] == 0.42
    assert payload["top_p"] == 0.88
    assert payload["top_k"] == 11
    assert payload["max_length"] == 64
    assert payload["prompt"].startswith("<") and "chars" in payload["prompt"]
    # Stop sequences come from the chatml template since none were provided.
    assert "<|im_end|>" in payload["stop_sequence"]


@pytest.mark.asyncio
async def test_stream_messages_kobold_extras_flow_through():
    # rep_pen / min_p / etc live in provider_extras["kobold"] and must reach
    # the kobold native payload unchanged.
    resp = _FakeResp(chunks=[_sse_token("", finish_reason="stop")])
    engine, _ = _make_engine(resp)

    params = GenerationParams(
        temperature=0.5,
        provider_extras={"kobold": {
            "rep_pen": 1.07, "min_p": 0.05, "max_context_length": 4096,
        }},
    )
    events = await _drain(engine.stream_messages(
        _persona_config(),
        [{"role": "user", "content": "hi"}],
        params,
    ))
    payload = events[0]["payload"]
    assert payload["rep_pen"] == 1.07
    assert payload["min_p"] == 0.05
    assert payload["max_context_length"] == 4096


@pytest.mark.asyncio
async def test_stream_prompt_skips_template_rendering():
    # Caller-supplied prompt is forwarded verbatim. No chat template is
    # applied, so the dump's `template=` field reads `<caller>` and the raw
    # prompt's character count matches what we sent.
    raw = "<<RAW>>USER: hi<<END>>"
    resp = _FakeResp(chunks=[_sse_token("", finish_reason="stop")])
    engine, client = _make_engine(resp)

    events = await _drain(engine.stream_prompt(
        _persona_config(),
        raw,
        GenerationParams(temperature=0.3),
        stop_sequences=["<<END>>"],
        tools_advertised=["get_x"],
    ))
    payload = events[0]["payload"]
    assert payload["temperature"] == 0.3
    assert payload["stop_sequence"] == ["<<END>>"]
    assert payload["tools_advertised"] == ["get_x"]
    assert payload["prompt"] == f"<{len(raw)} chars, template=<caller>>"
    # Real prompt was forwarded verbatim to kobold.
    assert client.last_stream["json"]["prompt"] == raw
