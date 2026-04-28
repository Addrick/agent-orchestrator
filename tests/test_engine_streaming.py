# tests/test_engine_streaming.py
#
# Phase B coverage — provider streaming surface on TextEngine + collect-stream
# wrapper. See memory/project/plans/portal_engine_reintegration.md.

from typing import AsyncIterator, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.engine import TextEngine, LLMCommunicationError
from src.generation_params import GenerationParams


def _drain_factory():
    async def _drain(stream: AsyncIterator[Dict]) -> List[Dict]:
        out: List[Dict] = []
        async for ev in stream:
            out.append(ev)
        return out
    return _drain


@pytest.fixture
def drain():
    return _drain_factory()


@pytest.fixture
def text_engine():
    return TextEngine()


@pytest.fixture
def openai_config():
    return {"model_name": "gpt-4"}


@pytest.fixture
def anthropic_config():
    return {"model_name": "claude-3-opus-20240229"}


@pytest.fixture
def google_config():
    return {"model_name": "gemini-pro"}


@pytest.fixture
def local_config():
    return {"model_name": "local"}


@pytest.fixture
def messages():
    return [
        {"role": "system", "content": "you are a test bot"},
        {"role": "user", "content": "hello"},
    ]


# --------------------------------------------------------------------------
# stream_messages — non-local providers wrap generate_response into the
# unified event stream. Verifies event order, content, and tool_calls path.
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_stream_messages_text_path_yields_payload_delta_done(
    text_engine, openai_config, messages, drain
):
    with patch.object(
        text_engine, "generate_response", new_callable=AsyncMock,
        return_value=({"type": "text", "content": "hi back"}, {"forwarded": "ok"}),
    ):
        events = await drain(text_engine.stream_messages(
            openai_config, messages, GenerationParams(temperature=0.5),
        ))

    types = [e["type"] for e in events]
    assert types == ["api_payload", "text_delta", "done"]
    assert events[0]["payload"] == {"forwarded": "ok"}
    assert events[1]["text"] == "hi back"
    assert events[2]["full_text"] == "hi back"


@pytest.mark.asyncio
async def test_stream_messages_tool_calls_path(
    text_engine, openai_config, messages, drain
):
    calls = [{"id": "c1", "name": "get_x", "arguments": {"a": 1}}]
    with patch.object(
        text_engine, "generate_response", new_callable=AsyncMock,
        return_value=({"type": "tool_calls", "calls": calls}, {"p": 1}),
    ):
        events = await drain(text_engine.stream_messages(
            openai_config, messages, GenerationParams(),
        ))

    types = [e["type"] for e in events]
    assert types == ["api_payload", "tool_calls", "done"]
    assert events[1]["calls"] == calls
    assert events[2]["full_text"] == ""


@pytest.mark.asyncio
async def test_stream_messages_empty_text_skips_delta(
    text_engine, openai_config, messages, drain
):
    with patch.object(
        text_engine, "generate_response", new_callable=AsyncMock,
        return_value=({"type": "text", "content": ""}, {"p": 1}),
    ):
        events = await drain(text_engine.stream_messages(
            openai_config, messages, GenerationParams(),
        ))
    types = [e["type"] for e in events]
    assert types == ["api_payload", "done"]
    assert events[-1]["full_text"] == ""


@pytest.mark.asyncio
async def test_stream_messages_propagates_llm_error_with_payload(
    text_engine, openai_config, messages, drain
):
    err = LLMCommunicationError("boom", api_payload={"why": "broken"})
    with patch.object(
        text_engine, "generate_response", new_callable=AsyncMock, side_effect=err,
    ):
        with pytest.raises(LLMCommunicationError) as ei:
            await drain(text_engine.stream_messages(
                openai_config, messages, GenerationParams(),
            ))
    assert ei.value.api_payload == {"why": "broken"}


@pytest.mark.asyncio
async def test_stream_messages_overlays_params_onto_persona_config(
    text_engine, openai_config, messages, drain
):
    # GenerationParams.temperature must override whatever sat on persona_config
    # before reaching the provider handler.
    captured: Dict = {}

    async def fake_generate(persona_config, history_object, tools, lic):
        captured.update(persona_config)
        return {"type": "text", "content": "ok"}, {"p": 1}

    with patch.object(text_engine, "generate_response", new=fake_generate):
        await drain(text_engine.stream_messages(
            {**openai_config, "temperature": 0.1},
            messages,
            GenerationParams(temperature=0.9, top_p=0.5, top_k=20, max_tokens=512),
        ))
    assert captured["temperature"] == 0.9
    assert captured["top_p"] == 0.5
    assert captured["top_k"] == 20
    assert captured["max_output_tokens"] == 512


# --------------------------------------------------------------------------
# Local model dispatch — when StreamEngine is wired, stream_messages
# delegates to it; without one, it falls back to generate_response wrap.
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_stream_messages_local_delegates_to_stream_engine(
    local_config, messages, drain
):
    fake_events = [
        {"type": "api_payload", "payload": {"prompt": "<10 chars>"}},
        {"type": "text_delta", "text": "streamed"},
        {"type": "done", "full_text": "streamed"},
    ]

    async def _gen(*a, **kw):
        for e in fake_events:
            yield e

    fake_stream_engine = MagicMock()
    fake_stream_engine.stream_messages = MagicMock(side_effect=_gen)
    engine = TextEngine(stream_engine=fake_stream_engine)

    events = await drain(engine.stream_messages(
        local_config, messages, GenerationParams(temperature=0.7),
    ))
    assert events == fake_events
    fake_stream_engine.stream_messages.assert_called_once()


@pytest.mark.asyncio
async def test_stream_messages_local_without_stream_engine_falls_back(
    text_engine, local_config, messages, drain
):
    # No stream_engine wired: dispatch falls through to generate_response.
    with patch.object(
        text_engine, "generate_response", new_callable=AsyncMock,
        return_value=({"type": "text", "content": "fallback"}, {"p": 1}),
    ):
        events = await drain(text_engine.stream_messages(
            local_config, messages, GenerationParams(),
        ))
    assert events[-1]["full_text"] == "fallback"


# --------------------------------------------------------------------------
# stream_prompt — local-only entry, raises on other providers.
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_stream_prompt_delegates_to_stream_engine(local_config, drain):
    fake_events = [
        {"type": "api_payload", "payload": {"prompt": "<5 chars>"}},
        {"type": "text_delta", "text": "raw"},
        {"type": "done", "full_text": "raw"},
    ]

    async def _gen(*a, **kw):
        for e in fake_events:
            yield e

    fake_stream_engine = MagicMock()
    fake_stream_engine.stream_prompt = MagicMock(side_effect=_gen)
    engine = TextEngine(stream_engine=fake_stream_engine)

    events = await drain(engine.stream_prompt(
        local_config, "<|im_start|>user\nhi", GenerationParams(),
        stop_sequences=["<|im_end|>"], tools_advertised=["get_x"],
    ))
    assert events == fake_events
    call = fake_stream_engine.stream_prompt.call_args
    assert call.kwargs["stop_sequences"] == ["<|im_end|>"]
    assert call.kwargs["tools_advertised"] == ["get_x"]


@pytest.mark.asyncio
async def test_stream_prompt_rejects_non_local_models(text_engine, openai_config):
    with pytest.raises(LLMCommunicationError, match="only supports local"):
        # Note: stream_prompt returns the iterator; raise happens at call site
        # because there is no stream_engine on the test fixture either way.
        text_engine.stream_prompt(openai_config, "anything", GenerationParams())


@pytest.mark.asyncio
async def test_stream_prompt_requires_stream_engine_for_local(
    text_engine, local_config
):
    with pytest.raises(LLMCommunicationError, match="StreamEngine not configured"):
        text_engine.stream_prompt(local_config, "anything", GenerationParams())


# --------------------------------------------------------------------------
# collect_stream — drains the unified event stream into the same tuple shape
# that generate_response returns. Phase C uses this as the non-streaming seam.
# --------------------------------------------------------------------------


async def _aiter(events: List[Dict]) -> AsyncIterator[Dict]:
    for e in events:
        yield e


@pytest.mark.asyncio
async def test_collect_stream_text_concatenates_deltas():
    events = [
        {"type": "api_payload", "payload": {"x": 1}},
        {"type": "text_delta", "text": "hel"},
        {"type": "text_delta", "text": "lo"},
        {"type": "done", "full_text": "hello"},
    ]
    result, payload = await TextEngine.collect_stream(_aiter(events))
    assert result == {"type": "text", "content": "hello"}
    assert payload == {"x": 1}


@pytest.mark.asyncio
async def test_collect_stream_prefers_done_full_text_over_concat():
    # done's full_text is the source of truth (e.g. kobold parser strips
    # `<tool_call>` markup from visible text but text_deltas may have lagged).
    events = [
        {"type": "api_payload", "payload": {}},
        {"type": "text_delta", "text": "raw"},
        {"type": "done", "full_text": "clean"},
    ]
    result, _ = await TextEngine.collect_stream(_aiter(events))
    assert result["content"] == "clean"


@pytest.mark.asyncio
async def test_collect_stream_tool_calls_take_priority_over_text():
    events = [
        {"type": "api_payload", "payload": {}},
        {"type": "text_delta", "text": "thinking..."},
        {"type": "tool_calls", "calls": [
            {"id": "c1", "name": "x", "arguments": {}},
        ]},
        {"type": "done", "full_text": "thinking..."},
    ]
    result, _ = await TextEngine.collect_stream(_aiter(events))
    assert result["type"] == "tool_calls"
    assert result["calls"][0]["name"] == "x"


@pytest.mark.asyncio
async def test_collect_stream_handles_missing_payload():
    events = [
        {"type": "text_delta", "text": "hi"},
        {"type": "done", "full_text": "hi"},
    ]
    result, payload = await TextEngine.collect_stream(_aiter(events))
    assert payload is None
    assert result == {"type": "text", "content": "hi"}


# --------------------------------------------------------------------------
# Round-trip — collect_stream(stream_messages(...)) yields the exact tuple
# shape generate_response returns. Phase C reuses this invariant.
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_collect_round_trip_text(text_engine, openai_config, messages):
    expected = ({"type": "text", "content": "round-trip"}, {"payload": "x"})
    with patch.object(
        text_engine, "generate_response", new_callable=AsyncMock,
        return_value=expected,
    ):
        result, payload = await TextEngine.collect_stream(
            text_engine.stream_messages(openai_config, messages, GenerationParams())
        )
    assert (result, payload) == expected


@pytest.mark.asyncio
async def test_collect_round_trip_tool_calls(text_engine, openai_config, messages):
    expected = (
        {"type": "tool_calls", "calls": [
            {"id": "c1", "name": "get_x", "arguments": {"a": 1}},
        ]},
        {"payload": "y"},
    )
    with patch.object(
        text_engine, "generate_response", new_callable=AsyncMock,
        return_value=expected,
    ):
        result, payload = await TextEngine.collect_stream(
            text_engine.stream_messages(openai_config, messages, GenerationParams())
        )
    assert (result, payload) == expected
