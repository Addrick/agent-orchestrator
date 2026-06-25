# tests/test_engine_provider_streams.py
#
# DP-206 — unit coverage for the canonical per-provider streaming drivers
# (`_stream_<provider>_response`). The one-shot handlers are collect_stream
# wrappers over these generators; this file tests the streaming semantics the
# wrappers rely on (event ordering, delta accumulation, error payloads).

import pytest
from unittest.mock import patch, AsyncMock, MagicMock

from src.engine import TextEngine, LLMCommunicationError
from tests.provider_stream_mocks import (
    AsyncIterList,
    anthropic_stream,
    google_stream,
    openai_chunk,
    openai_text_stream,
    openai_tool_call_delta,
)


@pytest.fixture
def text_engine():
    return TextEngine()


@pytest.fixture
def base_context():
    return {
        "persona_prompt": "You are a test bot.",
        "history": [],
        "current_message": {"text": "Hello"},
    }


async def _drain(stream):
    return [ev async for ev in stream]


# ---------------------------------------------------------------------------
# OpenAI
# ---------------------------------------------------------------------------


@patch("src.engine.providers.openai.AsyncOpenAI")
class TestOpenAIStream:
    @pytest.mark.asyncio
    async def test_event_order_payload_deltas_done(self, mock_cls, text_engine,
                                                   base_context, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "dummy")
        inst = mock_cls.return_value
        inst.chat.completions.create = AsyncMock(return_value=AsyncIterList([
            openai_chunk(content="Hel"),
            openai_chunk(content="lo"),
            openai_chunk(finish_reason="stop"),
        ]))
        events = await _drain(text_engine._stream_openai_response(
            {"model_name": "gpt-4"}, base_context, None
        ))
        assert [e["type"] for e in events] == [
            "api_payload", "text_delta", "text_delta", "done"
        ]
        assert events[1]["text"] == "Hel" and events[2]["text"] == "lo"
        assert events[-1]["full_text"] == "Hello"

    @pytest.mark.asyncio
    async def test_tool_call_arguments_accumulate_across_chunks(
        self, mock_cls, text_engine, base_context, monkeypatch
    ):
        """OpenAI fragments tool-call arguments across deltas; the driver must
        reassemble them by index before parsing."""
        monkeypatch.setenv("OPENAI_API_KEY", "dummy")
        inst = mock_cls.return_value
        inst.chat.completions.create = AsyncMock(return_value=AsyncIterList([
            openai_chunk(tool_call_deltas=[
                openai_tool_call_delta(index=0, id="call_1", name="get_weather",
                                       arguments='{"loc'),
            ]),
            openai_chunk(tool_call_deltas=[
                openai_tool_call_delta(index=0, arguments='ation": "NYC"}'),
            ]),
            openai_chunk(finish_reason="tool_calls"),
        ]))
        events = await _drain(text_engine._stream_openai_response(
            {"model_name": "gpt-4"}, base_context,
            [{"type": "function", "function": {"name": "get_weather"}}],
        ))
        tool_ev = next(e for e in events if e["type"] == "tool_calls")
        assert tool_ev["calls"] == [
            {"id": "call_1", "name": "get_weather", "arguments": {"location": "NYC"}}
        ]
        assert events[-1] == {"type": "done", "full_text": ""}

    @pytest.mark.asyncio
    async def test_parallel_tool_calls_keep_index_order(self, mock_cls, text_engine,
                                                        base_context, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "dummy")
        inst = mock_cls.return_value
        inst.chat.completions.create = AsyncMock(return_value=AsyncIterList([
            openai_chunk(tool_call_deltas=[
                openai_tool_call_delta(index=1, id="call_b", name="tool_b", arguments="{}"),
                openai_tool_call_delta(index=0, id="call_a", name="tool_a", arguments="{}"),
            ]),
            openai_chunk(finish_reason="tool_calls"),
        ]))
        events = await _drain(text_engine._stream_openai_response(
            {"model_name": "gpt-4"}, base_context, None
        ))
        tool_ev = next(e for e in events if e["type"] == "tool_calls")
        assert [c["name"] for c in tool_ev["calls"]] == ["tool_a", "tool_b"]

    @pytest.mark.asyncio
    async def test_all_malformed_calls_yield_empty_tool_event_for_retry(
        self, mock_cls, text_engine, base_context, monkeypatch
    ):
        """Tool path with only unparseable calls must stay a tool_calls event
        (empty list) so collect_stream → generate_response retries, mirroring
        the pre-DP-206 one-shot behavior."""
        monkeypatch.setenv("OPENAI_API_KEY", "dummy")
        inst = mock_cls.return_value
        inst.chat.completions.create = AsyncMock(return_value=AsyncIterList([
            openai_chunk(tool_call_deltas=[
                openai_tool_call_delta(index=0, id="c1", name="broken", arguments="{nope"),
            ]),
            openai_chunk(finish_reason="tool_calls"),
        ]))
        result, _ = await TextEngine.collect_stream(text_engine._stream_openai_response(
            {"model_name": "gpt-4"}, base_context, None
        ))
        assert result == {"type": "tool_calls", "calls": []}

    @pytest.mark.asyncio
    async def test_transport_error_carries_full_api_payload(self, mock_cls, text_engine,
                                                            base_context, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "dummy")
        from openai import APIStatusError
        from unittest.mock import MagicMock
        inst = mock_cls.return_value
        inst.chat.completions.create = AsyncMock(
            side_effect=APIStatusError("boom", response=MagicMock(status_code=500), body=None)
        )
        with pytest.raises(LLMCommunicationError) as ei:
            await _drain(text_engine._stream_openai_response(
                {"model_name": "gpt-4"}, base_context,
                [{"type": "function", "function": {"name": "t"}}],
            ))
        # Error payloads keep the full tool definitions (pre-DP-206 contract).
        assert ei.value.api_payload["tools"][0]["function"]["name"] == "t"

    @pytest.mark.asyncio
    async def test_one_shot_is_collect_of_stream(self, mock_cls, text_engine,
                                                 base_context, monkeypatch):
        """generate_response('gpt-*') drains the canonical stream — same
        result tuple, with the dump payload (tools by name)."""
        monkeypatch.setenv("OPENAI_API_KEY", "dummy")
        inst = mock_cls.return_value
        inst.chat.completions.create = AsyncMock(return_value=openai_text_stream("hi"))
        result, payload = await text_engine.generate_response(
            {"model_name": "gpt-4"}, base_context,
            tools=[{"type": "function", "function": {"name": "t", "parameters": {}}}],
        )
        assert result == {"type": "text", "content": "hi"}
        assert payload["tools"] == ["t"]
        assert inst.chat.completions.create.call_args.kwargs["stream"] is True


# ---------------------------------------------------------------------------
# Anthropic
# ---------------------------------------------------------------------------


@patch("src.engine.anthropic.AsyncAnthropic")
class TestAnthropicStream:
    @pytest.mark.asyncio
    async def test_event_order_payload_deltas_done(self, mock_cls, text_engine,
                                                   base_context, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "dummy")
        from unittest.mock import MagicMock
        inst = mock_cls.return_value
        inst.messages.stream.return_value = anthropic_stream(MagicMock(
            content=[MagicMock(text="Hello")], stop_reason="end_turn"
        ), ["Hel", "lo"])
        events = await _drain(text_engine._stream_anthropic_response(
            {"model_name": "claude-3-opus-20240229"}, base_context, None
        ))
        assert [e["type"] for e in events] == [
            "api_payload", "text_delta", "text_delta", "done"
        ]
        assert events[1]["text"] == "Hel" and events[2]["text"] == "lo"
        assert events[-1]["full_text"] == "Hello"

    @pytest.mark.asyncio
    async def test_tool_use_yields_tool_calls_then_empty_done(
        self, mock_cls, text_engine, base_context, monkeypatch
    ):
        """tool_use stop reason: the accumulated final message carries the
        complete tool_use blocks; deltas streamed before it stay text_delta."""
        monkeypatch.setenv("ANTHROPIC_API_KEY", "dummy")
        from unittest.mock import MagicMock
        inst = mock_cls.return_value
        tool_block = MagicMock(type="tool_use", id="tu_1", input={"q": "x"})
        tool_block.name = "search"
        inst.messages.stream.return_value = anthropic_stream(MagicMock(
            content=[tool_block], stop_reason="tool_use"
        ), ["thinking..."])
        events = await _drain(text_engine._stream_anthropic_response(
            {"model_name": "claude-3-opus-20240229"}, base_context,
            [{"type": "function", "function": {"name": "search", "parameters": {}}}],
        ))
        tool_ev = next(e for e in events if e["type"] == "tool_calls")
        assert tool_ev["calls"] == [
            {"id": "tu_1", "name": "search", "arguments": {"q": "x"}}
        ]
        assert events[-1] == {"type": "done", "full_text": ""}

    @pytest.mark.asyncio
    async def test_transport_error_carries_full_api_payload(self, mock_cls, text_engine,
                                                            base_context, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "dummy")
        import anthropic as anthropic_sdk
        from unittest.mock import MagicMock
        inst = mock_cls.return_value
        inst.messages.stream.side_effect = anthropic_sdk.APIStatusError(
            "boom", response=MagicMock(status_code=500), body=None
        )
        with pytest.raises(LLMCommunicationError) as ei:
            await _drain(text_engine._stream_anthropic_response(
                {"model_name": "claude-3-opus-20240229"}, base_context,
                [{"type": "function", "function": {"name": "t", "parameters": {}}}],
            ))
        # Error payloads keep the full tool definitions (pre-DP-206 contract).
        assert ei.value.api_payload["tools"][0]["name"] == "t"

    @pytest.mark.asyncio
    async def test_one_shot_is_collect_of_stream(self, mock_cls, text_engine,
                                                 base_context, monkeypatch):
        """generate_response('claude-*') drains the canonical stream — same
        result tuple, with the dump payload (tools by name)."""
        monkeypatch.setenv("ANTHROPIC_API_KEY", "dummy")
        from unittest.mock import MagicMock
        inst = mock_cls.return_value
        inst.messages.stream.return_value = anthropic_stream(MagicMock(
            content=[MagicMock(text="hi")], stop_reason="end_turn"
        ), ["hi"])
        result, payload = await text_engine.generate_response(
            {"model_name": "claude-3-opus-20240229"}, base_context,
            tools=[{"type": "function", "function": {"name": "t", "parameters": {}}}],
        )
        assert result == {"type": "text", "content": "hi"}
        assert payload["tools"] == ["t"]


# ---------------------------------------------------------------------------
# Google
# ---------------------------------------------------------------------------


def _google_text_chunk(text):
    part = MagicMock(text=text, function_call=None)
    part.thought_signature = None
    cand = MagicMock(content=MagicMock(parts=[part]), grounding_metadata=None)
    return MagicMock(prompt_feedback=None, candidates=[cand])


def _google_fc_chunk(name, args):
    fcall = MagicMock()
    fcall.name = name
    fcall.args = args
    part = MagicMock(text=None, function_call=fcall)
    part.thought_signature = None
    cand = MagicMock(content=MagicMock(parts=[part]), grounding_metadata=None)
    return MagicMock(prompt_feedback=None, candidates=[cand])


@patch("src.engine.genai.client.AsyncClient")
class TestGoogleStream:
    @pytest.mark.asyncio
    async def test_event_order_payload_deltas_done(self, mock_cls, text_engine,
                                                   base_context, monkeypatch):
        """Text split across chunks streams as deltas; done joins them — the
        same full text the pre-DP-206 one-shot produced from a single
        response whose parts carried the complete text."""
        monkeypatch.setenv("GOOGLE_GENERATIVEAI_API_KEY", "dummy")
        inst = mock_cls.return_value
        inst.models.generate_content_stream = AsyncMock(return_value=google_stream(
            _google_text_chunk("Hel"), _google_text_chunk("lo"),
        ))
        events = await _drain(text_engine._stream_google_response(
            {"model_name": "gemini-pro"}, base_context, None
        ))
        assert [e["type"] for e in events] == [
            "api_payload", "text_delta", "text_delta", "done"
        ]
        assert events[1]["text"] == "Hel" and events[2]["text"] == "lo"
        assert events[-1]["full_text"] == "Hello"

    @pytest.mark.asyncio
    async def test_function_call_chunk_yields_tool_calls(self, mock_cls, text_engine,
                                                         base_context, monkeypatch):
        monkeypatch.setenv("GOOGLE_GENERATIVEAI_API_KEY", "dummy")
        from unittest.mock import MagicMock
        inst = mock_cls.return_value
        fcall = MagicMock()
        fcall.name = "search_web"
        fcall.args = {"query": "x"}
        part = MagicMock(text=None, function_call=fcall)
        part.thought_signature = None
        cand = MagicMock(content=MagicMock(parts=[part]), grounding_metadata=None)
        inst.models.generate_content_stream = AsyncMock(return_value=google_stream(
            MagicMock(prompt_feedback=None, candidates=[cand])
        ))
        events = await _drain(text_engine._stream_google_response(
            {"model_name": "gemini-pro"}, base_context,
            [{"type": "function", "function": {"name": "search_web"}}],
        ))
        tool_ev = next(e for e in events if e["type"] == "tool_calls")
        assert tool_ev["calls"][0]["name"] == "search_web"
        assert tool_ev["calls"][0]["arguments"] == {"query": "x"}
        assert events[-1] == {"type": "done", "full_text": ""}

    @pytest.mark.asyncio
    async def test_transport_error_carries_api_payload_and_rate_limit(
        self, mock_cls, text_engine, base_context, monkeypatch
    ):
        monkeypatch.setenv("GOOGLE_GENERATIVEAI_API_KEY", "dummy")
        inst = mock_cls.return_value
        inst.models.generate_content_stream = AsyncMock(
            side_effect=Exception("429 quota exceeded")
        )
        with pytest.raises(LLMCommunicationError) as ei:
            await _drain(text_engine._stream_google_response(
                {"model_name": "gemini-pro"}, base_context, None
            ))
        assert ei.value.rate_limited is True
        assert ei.value.api_payload["model"] == "gemini-pro"

    @pytest.mark.asyncio
    async def test_one_shot_is_collect_of_stream(self, mock_cls, text_engine,
                                                 base_context, monkeypatch):
        """generate_response('gemini-*') drains the canonical stream — same
        result tuple and serializable dump payload."""
        monkeypatch.setenv("GOOGLE_GENERATIVEAI_API_KEY", "dummy")
        inst = mock_cls.return_value
        inst.models.generate_content_stream = AsyncMock(return_value=google_stream(
            _google_text_chunk("hi")
        ))
        result, payload = await text_engine.generate_response(
            {"model_name": "gemini-pro"}, base_context
        )
        assert result == {"type": "text", "content": "hi"}
        assert payload["model"] == "gemini-pro"

    @pytest.mark.asyncio
    async def test_interleaved_text_between_function_calls_ids_self_consistent(
        self, mock_cls, text_engine, base_context, monkeypatch
    ):
        """DP-213: a stream that interleaves text parts between function-call
        parts synthesizes the same per-turn ids on the streaming path and the
        one-shot collect path, and the ids stay distinct — ids are per-turn
        correlation only, so self-consistency is the contract that matters."""
        monkeypatch.setenv("GOOGLE_GENERATIVEAI_API_KEY", "dummy")
        inst = mock_cls.return_value
        tools = [{"type": "function", "function": {"name": "get_a"}},
                 {"type": "function", "function": {"name": "get_b"}}]

        def chunks():
            return google_stream(
                _google_text_chunk("checking a "),
                _google_fc_chunk("get_a", {"x": 1}),
                _google_text_chunk("and then b "),
                _google_fc_chunk("get_b", {"y": 2}),
            )

        inst.models.generate_content_stream = AsyncMock(return_value=chunks())
        events = await _drain(text_engine._stream_google_response(
            {"model_name": "gemini-pro"}, base_context, tools
        ))
        stream_calls = next(e for e in events if e["type"] == "tool_calls")["calls"]

        inst.models.generate_content_stream = AsyncMock(return_value=chunks())
        result, _ = await text_engine.generate_response(
            {"model_name": "gemini-pro"}, base_context, tools=tools
        )

        assert result["type"] == "tool_calls"
        assert [c["name"] for c in stream_calls] == ["get_a", "get_b"]
        stream_ids = [c["id"] for c in stream_calls]
        assert [c["id"] for c in result["calls"]] == stream_ids
        assert len(set(stream_ids)) == len(stream_ids)
