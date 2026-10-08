# tests/test_local_provider.py
#
# DP-417 — the local provider speaks the OpenAI chat-completions API to the
# local server (KoboldCPP or Strata). Replaces tests/test_stream_engine.py,
# which covered the retired kobold-native transport.

import json
from typing import Any, Dict, List
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.engine import TextEngine, LLMCommunicationError
from src.engine.providers.local import (
    build_local_params,
    local_base_url,
    params_from_legacy_dicts,
    thinking_controls,
)
from src.engine.providers.openai import to_openai_wire_message
from src.generation_params import CHAT_TEMPLATE_THINKING, GenerationParams
from tests.provider_stream_mocks import (
    AsyncIterList,
    openai_chunk,
    openai_tool_call_delta,
)

TOOL = {"type": "function", "function": {
    "name": "get_weather", "description": "Weather for a city.",
    "parameters": {"type": "object", "properties": {"city": {"type": "string"}}}}}


def _history(*turns: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "persona_prompt": "you are local",
        "message_history": list(turns) or [{"role": "user", "content": "hi"}],
        "current_message": {"text": "", "image_url": None},
    }


def _engine_with_stream(stream: AsyncIterList) -> TextEngine:
    engine = TextEngine()
    client = MagicMock()
    client.chat.completions.create = AsyncMock(return_value=stream)
    engine.local_client = client
    return engine


async def _drain(stream) -> List[Dict[str, Any]]:
    return [ev async for ev in stream]


# --- request shape -----------------------------------------------------------

def test_kobold_samplers_and_context_ride_in_extra_body():
    params = GenerationParams(
        temperature=0.7, top_p=0.9, top_k=40, max_tokens=256,
        provider_extras={"kobold": {"rep_pen": 1.07, "rep_pen_range": 320, "min_p": 0.05,
                                    "tfs": 1.0, "max_context_length": 8192}},
    )
    api = build_local_params({"model_name": "local"}, _history(), params)
    assert api["temperature"] == 0.7 and api["top_p"] == 0.9 and api["max_tokens"] == 256
    assert api["extra_body"] == {
        "rep_pen": 1.07, "rep_pen_range": 320, "min_p": 0.05, "tfs": 1.0,
        "top_k": 40, "max_context_length": 8192,
    }


def test_context_falls_back_to_persona_budget_and_is_capped(monkeypatch):
    params = GenerationParams()
    api = build_local_params({"max_context_tokens": 10_000_000}, _history(), params)
    from config import global_config
    assert api["extra_body"]["max_context_length"] == global_config.DEFAULT_MAX_CONTEXT_TOKENS
    api = build_local_params({"max_context_tokens": 16384}, _history(), params)
    assert api["extra_body"]["max_context_length"] == 16384


def test_model_name_is_a_placeholder_neither_server_routes_on():
    api = build_local_params({"model_name": "default"}, _history(), GenerationParams())
    assert api["model"] == "local"


def test_stop_sequences_and_seed_pass_through():
    params = GenerationParams(stop_sequences=["\nUser:"], seed=7)
    api = build_local_params({}, _history(), params)
    assert api["stop"] == ["\nUser:"] and api["seed"] == 7


def test_tools_sent_as_structured_tools_not_prompt_text():
    api = build_local_params({}, _history(), GenerationParams(), tools=[TOOL])
    assert api["tools"] == [TOOL] and api["tool_choice"] == "auto"
    assert "tool_call" not in api["messages"][0]["content"]


def test_legacy_dicts_override_persona_defaults():
    params = params_from_legacy_dicts(
        {"temperature": 0.2, "max_output_tokens": 100},
        {"temperature": 0.9, "rep_pen": 1.1, "stop_sequence": ["###"], "bogus": 1},
    )
    assert params.temperature == 0.9 and params.max_tokens == 100
    assert params.provider_extras == {"kobold": {"rep_pen": 1.1}}
    assert params.stop_sequences == ["###"]


@pytest.mark.parametrize("url", ["http://box:5001/v1", "http://box:5001/v1/", "http://box:5001"])
def test_base_url_strips_v1(monkeypatch, url):
    monkeypatch.setenv("LOCAL_LLM_URL", url)
    assert local_base_url() == "http://box:5001"


# --- thinking switch ---------------------------------------------------------

@pytest.mark.parametrize("preset,expected", [
    ("chatml-nothink", {"chat_template_kwargs": {"enable_thinking": False}}),
    ("gemma4-nothink", {"chat_template_kwargs": {"enable_thinking": False}}),
    ("gemma4-think", {"chat_template_kwargs": {"enable_thinking": True}}),
    ("chatml", {}),
    ("alpaca", {}),
    (None, {}),
])
def test_preset_sets_only_the_template_switch(monkeypatch, preset, expected):
    """KoboldCPP treats any reasoning_effort as a token budget and turns off
    batching for it, so a preset must not send one."""
    monkeypatch.delenv("KOBOLD_CHAT_TEMPLATE", raising=False)
    assert thinking_controls({"chat_template": preset}) == expected


def test_thinking_level_overrides_preset():
    assert thinking_controls({"chat_template": "chatml-nothink", "thinking_level": "high"}) == {
        "reasoning_effort": "high", "chat_template_kwargs": {"enable_thinking": True}}
    assert thinking_controls({"thinking_level": "none"}) == {
        "reasoning_effort": "none", "chat_template_kwargs": {"enable_thinking": False}}


def test_env_preset_applies_when_persona_has_none(monkeypatch):
    monkeypatch.setenv("KOBOLD_CHAT_TEMPLATE", "chatml-nothink")
    assert thinking_controls({}) == {"chat_template_kwargs": {"enable_thinking": False}}


def test_every_preset_name_is_still_accepted():
    """Stored personas name these; DP-417 must not invalidate any of them."""
    assert set(CHAT_TEMPLATE_THINKING) == {
        "alpaca", "chatml", "chatml-nothink", "gemma", "gemma4-e-nothink",
        "gemma4-nothink", "gemma4-think", "llama2", "llama3", "llama4",
    }


# --- history wire shape ------------------------------------------------------

def test_tool_turns_are_translated_to_wire_shape():
    """The tool loop stores calls as {id, name, arguments: dict}; the wire
    wants {id, type, function: {name, arguments: str}} and no name on a tool
    turn. Before DP-417 only the kobold path replayed tool turns, as prompt text."""
    stored_call = {"role": "assistant", "content": "checking",
                   "tool_calls": [{"id": "c1", "name": "get_weather", "arguments": {"city": "Oslo"}}]}
    stored_result = {"role": "tool", "tool_call_id": "c1", "name": "get_weather",
                     "content": json.dumps({"temp_c": 4})}
    assert to_openai_wire_message(stored_call) == {
        "role": "assistant", "content": "checking",
        "tool_calls": [{"id": "c1", "type": "function",
                        "function": {"name": "get_weather", "arguments": '{"city": "Oslo"}'}}],
    }
    assert to_openai_wire_message(stored_result) == {
        "role": "tool", "tool_call_id": "c1", "content": '{"temp_c": 4}'}
    api = build_local_params({}, _history(
        {"role": "user", "content": "weather?"}, stored_call, stored_result), GenerationParams())
    assert [m["role"] for m in api["messages"]] == ["system", "user", "assistant", "tool"]


def test_plain_turns_pass_through_unchanged():
    msg = {"role": "user", "content": "hello"}
    assert to_openai_wire_message(msg) is msg


# --- streaming ---------------------------------------------------------------

@pytest.mark.asyncio
async def test_stream_emits_payload_deltas_done():
    engine = _engine_with_stream(AsyncIterList([
        openai_chunk(content="Hel"), openai_chunk(content="lo"), openai_chunk(finish_reason="stop")]))
    events = await _drain(engine._stream_local_response({"model_name": "local"}, _history()))
    assert [e["type"] for e in events] == ["api_payload", "text_delta", "text_delta", "done"]
    assert events[-1]["full_text"] == "Hello"
    assert events[0]["payload"]["model"] == "local"


@pytest.mark.asyncio
async def test_reasoning_is_folded_into_one_think_block():
    """Downstream (portal render, DP-409 retain strip) already handles inline
    <think>, which is what the kobold-native stream produced."""
    engine = _engine_with_stream(AsyncIterList([
        openai_chunk(reasoning="Let me "), openai_chunk(reasoning="add."),
        openai_chunk(content="\n\n391"), openai_chunk(finish_reason="stop")]))
    events = await _drain(engine._stream_local_response({}, _history()))
    assert events[-1]["full_text"] == "<think>\nLet me add.\n</think>\n391"


@pytest.mark.asyncio
async def test_reasoning_before_tool_calls_is_closed_and_ids_are_filled():
    engine = _engine_with_stream(AsyncIterList([
        openai_chunk(reasoning="need weather"),
        openai_chunk(tool_call_deltas=[openai_tool_call_delta(
            index=0, id=None, name="get_weather", arguments='{"city": "Oslo"}')]),
        openai_chunk(finish_reason="tool_calls")]))
    events = await _drain(engine._stream_local_response({}, _history(), tools=[TOOL]))
    text = "".join(e["text"] for e in events if e["type"] == "text_delta")
    assert text == "<think>\nneed weather\n</think>\n"
    (calls,) = [e["calls"] for e in events if e["type"] == "tool_calls"]
    assert calls == [{"id": "call_get_weather_0", "name": "get_weather", "arguments": {"city": "Oslo"}}]


@pytest.mark.asyncio
async def test_closing_the_stream_early_closes_the_http_response():
    """Cancelling is closing the stream: both local servers abort the
    generation on disconnect. The kobold path posted /api/extra/abort here."""
    upstream = AsyncIterList([openai_chunk(content=str(i)) for i in range(10)])
    engine = _engine_with_stream(upstream)
    gen = engine._stream_local_response({}, _history())
    async for ev in gen:
        if ev["type"] == "text_delta":
            break
    await gen.aclose()
    assert upstream.closed


@pytest.mark.asyncio
async def test_closing_the_typed_stream_early_also_cancels():
    """The portal path: `stream_messages` is what the stop button closes.
    `async for` does not close an inner async generator, so every layer must
    hand the close down explicitly."""
    upstream = AsyncIterList([openai_chunk(content=str(i)) for i in range(10)])
    engine = _engine_with_stream(upstream)
    gen = engine.stream_messages({"model_name": "local"}, [{"role": "user", "content": "hi"}],
                                 GenerationParams())
    async for ev in gen:
        if ev["type"] == "text_delta":
            break
    await gen.aclose()
    assert upstream.closed


@pytest.mark.asyncio
async def test_server_error_raises_llm_error_naming_the_local_model():
    from openai import APIStatusError
    engine = TextEngine()
    client = MagicMock()
    err = APIStatusError("boom", response=MagicMock(status_code=500), body=None)
    client.chat.completions.create = AsyncMock(side_effect=err)
    engine.local_client = client
    with pytest.raises(LLMCommunicationError, match="Local model API returned an error"):
        await _drain(engine._stream_local_response({}, _history()))


@pytest.mark.asyncio
async def test_stream_messages_local_routes_to_the_local_server():
    """The typed entry carries GenerationParams (incl. kobold extras) straight
    through, with no image (local has no image support)."""
    engine = _engine_with_stream(AsyncIterList([openai_chunk(content="ok")]))
    params = GenerationParams(temperature=0.3, provider_extras={"kobold": {"rep_pen": 1.2}})
    events = await _drain(engine.stream_messages(
        {"model_name": "local"},
        [{"role": "system", "content": "sys"}, {"role": "user", "content": "hi"}],
        params, image_url="http://x/img.png"))
    kwargs = engine.local_client.chat.completions.create.call_args.kwargs
    assert kwargs["temperature"] == 0.3 and kwargs["extra_body"]["rep_pen"] == 1.2
    assert kwargs["messages"][0] == {"role": "system", "content": "sys"}
    assert kwargs["messages"][1] == {"role": "user", "content": "hi"}
    assert events[-1] == {"type": "done", "full_text": "ok"}


@pytest.mark.asyncio
async def test_aclose_releases_the_client_and_is_idempotent():
    engine = TextEngine()
    client = MagicMock()
    client.close = AsyncMock()
    engine.local_client = client
    await engine.aclose()
    await engine.aclose()
    client.close.assert_awaited_once()
    assert engine.local_client is None
