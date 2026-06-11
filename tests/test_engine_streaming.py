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
# stream_messages — non-local providers dispatch through the `_stream_response`
# policy driver to the canonical per-provider streams (DP-206b cutover):
# true token deltas, with generate_response's retry/fallback policy.
# --------------------------------------------------------------------------


def _events_stream(events: List[Dict]):
    """An already-instantiated async generator over canned unified events."""
    async def _gen():
        for ev in events:
            yield ev
    return _gen()


@pytest.mark.asyncio
async def test_stream_messages_text_path_streams_true_deltas(
    text_engine, openai_config, messages, drain
):
    provider_events = [
        {"type": "api_payload", "payload": {"forwarded": "ok"}},
        {"type": "text_delta", "text": "hi "},
        {"type": "text_delta", "text": "back"},
        {"type": "done", "full_text": "hi back"},
    ]
    with patch.object(
        text_engine, "_stream_openai_response",
        MagicMock(side_effect=lambda *a, **k: _events_stream(provider_events)),
    ):
        events = await drain(text_engine.stream_messages(
            openai_config, messages, GenerationParams(temperature=0.5),
        ))

    # Token deltas pass through one-by-one — not collapsed into a single
    # text_delta the way the pre-cutover generate_response wrap did.
    types = [e["type"] for e in events]
    assert types == ["api_payload", "text_delta", "text_delta", "done"]
    assert events[0]["payload"] == {"forwarded": "ok"}
    assert events[1]["text"] == "hi "
    assert events[2]["text"] == "back"
    assert events[3]["full_text"] == "hi back"


@pytest.mark.asyncio
async def test_stream_messages_tool_calls_path(
    text_engine, openai_config, messages, drain
):
    calls = [{"id": "c1", "name": "get_x", "arguments": {"a": 1}}]
    provider_events = [
        {"type": "api_payload", "payload": {"p": 1}},
        {"type": "tool_calls", "calls": calls},
        {"type": "done", "full_text": ""},
    ]
    with patch.object(
        text_engine, "_stream_openai_response",
        MagicMock(side_effect=lambda *a, **k: _events_stream(provider_events)),
    ):
        events = await drain(text_engine.stream_messages(
            openai_config, messages, GenerationParams(),
        ))

    types = [e["type"] for e in events]
    assert types == ["api_payload", "tool_calls", "done"]
    assert events[1]["calls"] == calls
    assert events[2]["full_text"] == ""


@pytest.mark.asyncio
async def test_stream_messages_empty_text_retries_then_raises(
    text_engine, openai_config, messages, drain
):
    """An attempt with no real content emits nothing and is retried — the
    same policy generate_response applied pre-cutover; exhausting retries
    raises instead of yielding an empty stream."""
    empty_events = [
        {"type": "api_payload", "payload": {"p": 1}},
        {"type": "done", "full_text": ""},
    ]
    provider = MagicMock(side_effect=lambda *a, **k: _events_stream(list(empty_events)))
    with patch.object(text_engine, "_stream_openai_response", provider), \
            patch("src.engine.asyncio.sleep", new_callable=AsyncMock):
        with pytest.raises(LLMCommunicationError, match="empty or invalid response"):
            await drain(text_engine.stream_messages(
                openai_config, messages, GenerationParams(),
            ))
    assert provider.call_count > 1


@pytest.mark.asyncio
async def test_stream_messages_propagates_llm_error_with_payload(
    text_engine, openai_config, messages, drain
):
    err = LLMCommunicationError("boom", api_payload={"why": "broken"},
                                rate_limited=True)
    with patch.object(
        text_engine, "_stream_openai_response", MagicMock(side_effect=err),
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
    # before reaching the provider stream.
    captured: Dict = {}

    async def fake_stream(persona_config, history_object, tools=None):
        captured.update(persona_config)
        yield {"type": "api_payload", "payload": {"p": 1}}
        yield {"type": "text_delta", "text": "ok"}
        yield {"type": "done", "full_text": "ok"}

    with patch.object(text_engine, "_stream_openai_response", new=fake_stream):
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
# Local model dispatch — stream_messages delegates to the engine-owned
# kobold-native StreamEngine (params, incl. provider_extras, pass through).
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
async def test_stream_messages_local_always_streams_kobold_native(
    local_config, messages, drain
):
    """Facade collapse (DP-206b): there is no 'unwired' state — a default
    TextEngine owns a StreamEngine, so local always streams kobold-native."""
    fake_events = [
        {"type": "api_payload", "payload": {"prompt": "<10 chars>"}},
        {"type": "text_delta", "text": "native"},
        {"type": "done", "full_text": "native"},
    ]

    async def _gen(*a, **kw):
        for e in fake_events:
            yield e

    engine = TextEngine()
    engine.stream_engine = MagicMock()
    engine.stream_engine.stream_messages = MagicMock(side_effect=_gen)

    events = await drain(engine.stream_messages(
        local_config, messages, GenerationParams(),
    ))
    assert events == fake_events
    engine.stream_engine.stream_messages.assert_called_once()


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
        # Note: stream_prompt returns the iterator; the model check raises at
        # the call site, before any kobold transport is touched.
        text_engine.stream_prompt(openai_config, "anything", GenerationParams())


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
    provider_events = [
        {"type": "api_payload", "payload": {"payload": "x"}},
        {"type": "text_delta", "text": "round-"},
        {"type": "text_delta", "text": "trip"},
        {"type": "done", "full_text": "round-trip"},
    ]
    with patch.object(
        text_engine, "_stream_openai_response",
        MagicMock(side_effect=lambda *a, **k: _events_stream(provider_events)),
    ):
        result, payload = await TextEngine.collect_stream(
            text_engine.stream_messages(openai_config, messages, GenerationParams())
        )
    assert (result, payload) == (
        {"type": "text", "content": "round-trip"}, {"payload": "x"}
    )


@pytest.mark.asyncio
async def test_collect_round_trip_tool_calls(text_engine, openai_config, messages):
    calls = [{"id": "c1", "name": "get_x", "arguments": {"a": 1}}]
    provider_events = [
        {"type": "api_payload", "payload": {"payload": "y"}},
        {"type": "tool_calls", "calls": calls},
        {"type": "done", "full_text": ""},
    ]
    with patch.object(
        text_engine, "_stream_openai_response",
        MagicMock(side_effect=lambda *a, **k: _events_stream(provider_events)),
    ):
        result, payload = await TextEngine.collect_stream(
            text_engine.stream_messages(openai_config, messages, GenerationParams())
        )
    assert (result, payload) == (
        {"type": "tool_calls", "calls": calls}, {"payload": "y"}
    )
