# tests/test_engine.py

import os
import pytest
from unittest.mock import patch, AsyncMock, MagicMock
import base64
from openai import APIStatusError
import anthropic
import aiohttp
import json

from src.engine import TextEngine, LLMCommunicationError
from config.global_config import EMPTY_RESPONSE_RETRIES
from google.genai.types import Tool, GoogleSearch
from tests.helpers import engine_stream_events
from tests.provider_stream_mocks import (
    anthropic_stream,
    google_stream,
    openai_text_stream,
    openai_tool_call_stream,
)


def _one_shot_stream(result, payload=None):
    """An already-instantiated unified-event async generator for scripting
    `_stream_<provider>_response` mocks with one-shot (result, payload)."""
    async def _gen():
        for ev in engine_stream_events(result, payload):
            yield ev
    return _gen()


@pytest.fixture
def text_engine():
    """
    Provides a fresh, isolated TextEngine instance for each test function.
    This prevents state from bleeding between tests.
    """
    return TextEngine()


@pytest.fixture
def base_context():
    return {
        "persona_prompt": "You are a test bot.", "history": [],
        "current_message": {"text": "Hello"}
    }


@pytest.fixture
def openai_config():
    return {"model_name": "gpt-4"}


@pytest.fixture
def anthropic_config():
    return {"model_name": "claude-3-opus-20240229", "max_output_tokens": 100}


@pytest.fixture
def google_config():
    return {"model_name": "gemini-pro"}


@pytest.fixture
def local_config():
    return {"model_name": "local"}


class TestGenerateResponseLogic:
    @pytest.mark.asyncio
    @patch('src.engine.asyncio.sleep', new_callable=AsyncMock)
    @patch('src.engine.TextEngine._stream_openai_response')
    async def test_retry_on_empty_response_succeeds(self, mock_provider_call, mock_sleep, text_engine, openai_config, base_context):
        mock_provider_call.side_effect = [
            _one_shot_stream({}, {"payload": 1}),
            _one_shot_stream({"type": "text", "content": "Valid response"}, {"payload": 2}),
        ]
        response, _ = await text_engine.generate_response(openai_config, base_context)
        assert response == {"type": "text", "content": "Valid response"}
        assert mock_provider_call.call_count == 2

    @pytest.mark.asyncio
    @patch('src.engine.asyncio.sleep', new_callable=AsyncMock)
    @patch('src.engine.TextEngine._stream_openai_response')
    async def test_retry_on_empty_response_fails(self, mock_provider_call, mock_sleep, text_engine, openai_config, base_context):
        mock_provider_call.side_effect = lambda *a, **k: _one_shot_stream({}, {"payload": 1})
        with pytest.raises(LLMCommunicationError, match="LLM provider returned an empty or invalid response after all retries."):
            await text_engine.generate_response(openai_config, base_context)
        assert mock_provider_call.call_count == EMPTY_RESPONSE_RETRIES + 1

    @pytest.mark.asyncio
    @patch('src.engine.TextEngine._stream_openai_response')
    async def test_no_retry_on_rate_limit_error(self, mock_provider_call, text_engine, openai_config, base_context):
        """429 errors must abort immediately without consuming retry budget."""
        mock_provider_call.side_effect = LLMCommunicationError("Rate limited", rate_limited=True)
        with pytest.raises(LLMCommunicationError) as exc_info:
            await text_engine.generate_response(openai_config, base_context)
        assert exc_info.value.rate_limited is True
        assert mock_provider_call.call_count == 1

    @pytest.mark.asyncio
    @patch('src.engine.TextEngine._stream_google_response')
    async def test_rate_limit_falls_back_to_mapped_model(self, mock_provider_call, text_engine, base_context):
        """429 on a model with a _FALLBACK_MODELS entry reroutes to the
        fallback instead of aborting (DP-206b: policy lives in the driver)."""
        calls = []

        def _route(config, history_object, tools=None):
            calls.append(config["model_name"])
            if len(calls) == 1:
                raise LLMCommunicationError("429", rate_limited=True)
            return _one_shot_stream({"type": "text", "content": "fell back"}, {"p": 1})

        mock_provider_call.side_effect = _route
        response, _ = await text_engine.generate_response(
            {"model_name": "gemma-4-31b-it"}, base_context
        )
        assert response == {"type": "text", "content": "fell back"}
        assert calls == ["gemma-4-31b-it", "gemma-4-26b-a4b-it"]


@patch('src.engine.AsyncOpenAI')
class TestOpenAI:
    @pytest.mark.asyncio
    async def test_success_text_response(self, mock_openai_class, text_engine, openai_config, base_context, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "dummy_key_for_testing")
        mock_instance = mock_openai_class.return_value
        mock_instance.chat.completions.create = AsyncMock(
            return_value=openai_text_stream("Success")
        )
        response, _ = await text_engine.generate_response(openai_config, base_context)
        assert response == {"type": "text", "content": "Success"}

    @pytest.mark.asyncio
    async def test_success_tool_call_response(self, mock_openai_class, text_engine, openai_config, base_context, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "dummy_key_for_testing")
        mock_instance = mock_openai_class.return_value
        mock_instance.chat.completions.create = AsyncMock(
            return_value=openai_tool_call_stream(
                [("call_123", "get_weather", '{"location": "Boston"}')]
            )
        )
        # FIX: Pass a non-empty 'tools' list to trigger the tool-call logic path.
        response, _ = await text_engine.generate_response(openai_config, base_context, tools=[{"type": "function", "function": {"name": "get_weather"}}])
        assert response['type'] == 'tool_calls'
        assert response['calls'][0]['name'] == 'get_weather'

    @pytest.mark.asyncio
    async def test_api_error_raises_llm_error(self, mock_openai_class, text_engine, openai_config, base_context, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "dummy_key_for_testing")
        mock_instance = mock_openai_class.return_value
        error = APIStatusError("Server error", response=MagicMock(status_code=500), body=None)
        mock_instance.chat.completions.create.side_effect = error
        with pytest.raises(LLMCommunicationError, match="OpenAI API returned an error"):
            await text_engine.generate_response(openai_config, base_context)

    @pytest.mark.asyncio
    async def test_429_sets_rate_limited_flag(self, mock_openai_class, text_engine, openai_config, base_context, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "dummy_key_for_testing")
        mock_instance = mock_openai_class.return_value
        error = APIStatusError("Rate limit exceeded", response=MagicMock(status_code=429), body=None)
        mock_instance.chat.completions.create.side_effect = error
        with pytest.raises(LLMCommunicationError) as exc_info:
            await text_engine.generate_response(openai_config, base_context)
        assert exc_info.value.rate_limited is True
        assert mock_instance.chat.completions.create.call_count == 1

    @pytest.mark.asyncio
    async def test_tool_metadata_stripped_from_api_call(self, mock_openai_class, text_engine, openai_config,
                                                        base_context, monkeypatch):
        """Custom metadata fields (is_write, service_binding) must not leak into API calls."""
        monkeypatch.setenv("OPENAI_API_KEY", "dummy_key_for_testing")
        mock_instance = mock_openai_class.return_value
        mock_instance.chat.completions.create = AsyncMock(
            return_value=openai_text_stream("ok")
        )
        tools = [{
            "type": "function", "is_write": True, "service_binding": "zammad",
            "function": {"name": "create_ticket", "description": "Creates a ticket",
                         "parameters": {"type": "object", "properties": {}}}
        }]
        await text_engine.generate_response(openai_config, base_context, tools=tools)
        call_kwargs = mock_instance.chat.completions.create.call_args[1]
        for tool in call_kwargs["tools"]:
            assert "is_write" not in tool, "is_write leaked into OpenAI API call"
            assert "service_binding" not in tool, "service_binding leaked into OpenAI API call"
            assert set(tool.keys()) == {"type", "function"}


@patch('src.engine.anthropic.AsyncAnthropic')
class TestAnthropic:
    @pytest.mark.asyncio
    async def test_success_text_response(self, mock_anthropic_class, text_engine, anthropic_config, base_context, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "dummy_key_for_testing")
        mock_instance = mock_anthropic_class.return_value
        mock_instance.messages.stream.return_value = anthropic_stream(MagicMock(
            content=[MagicMock(text="Claude success")], stop_reason="end_turn"
        ), ["Claude success"])
        response, _ = await text_engine.generate_response(anthropic_config, base_context)
        assert response == {"type": "text", "content": "Claude success"}

    @pytest.mark.asyncio
    async def test_success_tool_call_response(self, mock_anthropic_class, text_engine, anthropic_config, base_context, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "dummy_key_for_testing")
        mock_instance = mock_anthropic_class.return_value
        mock_tool_use = MagicMock(type='tool_use', id='tool_123', input={'ticker': 'GOOG'})
        mock_tool_use.name = 'get_stock_price'
        mock_instance.messages.stream.return_value = anthropic_stream(
            MagicMock(content=[mock_tool_use], stop_reason="tool_use")
        )
        response, _ = await text_engine.generate_response(
            anthropic_config, base_context,
            tools=[{"type": "function", "function": {"name": "get_stock_price", "parameters": {}}}]
        )
        assert response['type'] == 'tool_calls'
        assert response['calls'][0]['name'] == 'get_stock_price'

    @pytest.mark.asyncio
    async def test_api_error_raises_llm_error(self, mock_anthropic_class, text_engine, anthropic_config, base_context, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "dummy_key_for_testing")
        mock_instance = mock_anthropic_class.return_value
        error = anthropic.APIStatusError("Server error", response=MagicMock(status_code=500), body=None)
        mock_instance.messages.stream.side_effect = error
        with pytest.raises(LLMCommunicationError, match="Anthropic API returned an error"):
            await text_engine.generate_response(anthropic_config, base_context)

    @pytest.mark.asyncio
    async def test_429_sets_rate_limited_flag(self, mock_anthropic_class, text_engine, anthropic_config, base_context, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "dummy_key_for_testing")
        mock_instance = mock_anthropic_class.return_value
        error = anthropic.APIStatusError("Rate limit exceeded", response=MagicMock(status_code=429), body=None)
        mock_instance.messages.stream.side_effect = error
        with pytest.raises(LLMCommunicationError) as exc_info:
            await text_engine.generate_response(anthropic_config, base_context)
        assert exc_info.value.rate_limited is True
        assert mock_instance.messages.stream.call_count == 1

    @pytest.mark.asyncio
    async def test_tool_metadata_stripped_from_api_call(self, mock_anthropic_class, text_engine, anthropic_config,
                                                        base_context, monkeypatch):
        """Custom metadata fields must be stripped and tools converted to Anthropic format."""
        monkeypatch.setenv("ANTHROPIC_API_KEY", "dummy_key_for_testing")
        mock_instance = mock_anthropic_class.return_value
        mock_instance.messages.stream.return_value = anthropic_stream(MagicMock(
            content=[MagicMock(text="ok")], stop_reason="end_turn"
        ), ["ok"])
        tools = [{
            "type": "function", "is_write": True, "service_binding": "zammad",
            "function": {"name": "create_ticket", "description": "Creates a ticket",
                         "parameters": {"type": "object", "properties": {}}}
        }]
        await text_engine.generate_response(anthropic_config, base_context, tools=tools)
        call_kwargs = mock_instance.messages.stream.call_args[1]
        for tool in call_kwargs["tools"]:
            assert "is_write" not in tool, "is_write leaked into Anthropic API call"
            assert "service_binding" not in tool, "service_binding leaked into Anthropic API call"
            assert "function" not in tool, "OpenAI-style nesting leaked into Anthropic API call"
            assert "name" in tool and "input_schema" in tool

    @pytest.mark.asyncio
    @patch('aiohttp.ClientSession.get')
    async def test_image_url_passed_to_anthropic(self, mock_get, mock_anthropic_class, text_engine, anthropic_config,
                                                 base_context, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "dummy_key_for_testing")

        # Mock the image download
        mock_response = AsyncMock()
        mock_response.read.return_value = b'imagedata'
        mock_response.content_type = 'image/png'
        mock_response.raise_for_status = MagicMock()
        mock_get.return_value.__aenter__.return_value = mock_response

        # Mock the Claude API response
        mock_instance = mock_anthropic_class.return_value
        mock_instance.messages.stream.return_value = anthropic_stream(MagicMock(
            content=[MagicMock(text="Image received")], stop_reason="end_turn"
        ), ["Image received"])

        base_context["current_message"]["image_url"] = "http://example.com/image.png"
        base_context["history"] = [{"role": "user", "content": "Check this out"}]

        await text_engine.generate_response(anthropic_config, base_context)

        # Verify that the image was included in the API call
        call_args = mock_instance.messages.stream.call_args[1]
        assert call_args['messages'][-1]['content'][-1]['type'] == 'image'
        assert call_args['messages'][-1]['content'][-1]['source']['data'] == base64.b64encode(b'imagedata').decode('utf-8')


@patch('src.engine.genai.client.AsyncClient')
class TestGoogle:
    @pytest.mark.asyncio
    async def test_success_text_response(self, mock_google_client_class, text_engine, google_config, base_context, monkeypatch):
        monkeypatch.setenv("GOOGLE_GENERATIVEAI_API_KEY", "dummy_key_for_testing")
        mock_instance = mock_google_client_class.return_value
        mock_part = MagicMock(text="Google success", function_call=None)
        mock_candidate = MagicMock(content=MagicMock(parts=[mock_part]), grounding_metadata=None)
        mock_instance.models.generate_content_stream = AsyncMock(
            return_value=google_stream(MagicMock(prompt_feedback=None, candidates=[mock_candidate]))
        )
        response, _ = await text_engine.generate_response(google_config, base_context)
        assert response == {"type": "text", "content": "Google success"}

    @pytest.mark.asyncio
    async def test_success_tool_call_response(self, mock_google_client_class, text_engine, google_config, base_context,
                                              monkeypatch):
        monkeypatch.setenv("GOOGLE_GENERATIVEAI_API_KEY", "dummy_key_for_testing")
        mock_instance = mock_google_client_class.return_value

        # Mock the specific Google API structure for a function call
        mock_function_call = MagicMock()
        mock_function_call.name = "search_web"
        # Note: Google's 'args' attribute is already a dict-like object, not a JSON string
        mock_function_call.args = {'query': 'python testing'}

        mock_part = MagicMock(text=None, function_call=mock_function_call)
        mock_candidate = MagicMock(content=MagicMock(parts=[mock_part]), grounding_metadata=None)
        mock_instance.models.generate_content_stream = AsyncMock(
            return_value=google_stream(MagicMock(prompt_feedback=None, candidates=[mock_candidate]))
        )

        # Pass a non-empty 'tools' list to trigger the tool-call logic path
        response, _ = await text_engine.generate_response(google_config, base_context, tools=[
            {"type": "function", "function": {"name": "search_web"}}])

        assert response['type'] == 'tool_calls'
        assert len(response['calls']) == 1
        assert response['calls'][0]['name'] == 'search_web'
        assert response['calls'][0]['arguments'] == {'query': 'python testing'}

    @pytest.mark.asyncio
    async def test_api_error_raises_llm_error(self, mock_google_client_class, text_engine, google_config, base_context, monkeypatch):
        monkeypatch.setenv("GOOGLE_GENERATIVEAI_API_KEY", "dummy_key_for_testing")
        mock_instance = mock_google_client_class.return_value
        mock_instance.models.generate_content_stream.side_effect = Exception("API failure")
        with pytest.raises(LLMCommunicationError, match="An error occurred with Google API"):
            await text_engine.generate_response(google_config, base_context)

    @pytest.mark.asyncio
    async def test_429_sets_rate_limited_flag(self, mock_google_client_class, text_engine, google_config, base_context, monkeypatch):
        monkeypatch.setenv("GOOGLE_GENERATIVEAI_API_KEY", "dummy_key_for_testing")
        mock_instance = mock_google_client_class.return_value
        mock_instance.models.generate_content_stream.side_effect = Exception("429 quota exceeded")
        with pytest.raises(LLMCommunicationError) as exc_info:
            await text_engine.generate_response(google_config, base_context)
        assert exc_info.value.rate_limited is True
        assert mock_instance.models.generate_content_stream.call_count == 1

    @pytest.mark.asyncio
    async def test_resource_exhausted_sets_rate_limited_flag(self, mock_google_client_class, text_engine, google_config, base_context, monkeypatch):
        monkeypatch.setenv("GOOGLE_GENERATIVEAI_API_KEY", "dummy_key_for_testing")
        mock_instance = mock_google_client_class.return_value
        mock_instance.models.generate_content_stream.side_effect = Exception("RESOURCE_EXHAUSTED: daily limit reached")
        with pytest.raises(LLMCommunicationError) as exc_info:
            await text_engine.generate_response(google_config, base_context)
        assert exc_info.value.rate_limited is True
        assert mock_instance.models.generate_content_stream.call_count == 1


    @pytest.mark.asyncio
    async def test_no_tools_passes_nothing_to_api(self, mock_google_client_class, text_engine, google_config, base_context, monkeypatch):
        """No tools enabled → no tools key sent to API (required for Gemma compatibility)."""
        monkeypatch.setenv("GOOGLE_GENERATIVEAI_API_KEY", "dummy_key_for_testing")
        mock_instance = mock_google_client_class.return_value
        mock_part = MagicMock(text="ok", function_call=None)
        mock_candidate = MagicMock(content=MagicMock(parts=[mock_part]), grounding_metadata=None)
        mock_instance.models.generate_content_stream = AsyncMock(
            return_value=google_stream(MagicMock(prompt_feedback=None, candidates=[mock_candidate]))
        )
        await text_engine.generate_response(google_config, base_context, tools=[])
        config = mock_instance.models.generate_content_stream.call_args.kwargs['config']
        assert not config.tools

    @pytest.mark.asyncio
    async def test_grounding_tool_injects_google_search(self, mock_google_client_class, text_engine, google_config, base_context, monkeypatch):
        """google_grounding_search in tools → GoogleSearch Tool injected into API config."""
        monkeypatch.setenv("GOOGLE_GENERATIVEAI_API_KEY", "dummy_key_for_testing")
        mock_instance = mock_google_client_class.return_value
        mock_part = MagicMock(text="ok", function_call=None)
        mock_candidate = MagicMock(content=MagicMock(parts=[mock_part]), grounding_metadata=None)
        mock_instance.models.generate_content_stream = AsyncMock(
            return_value=google_stream(MagicMock(prompt_feedback=None, candidates=[mock_candidate]))
        )
        grounding_tools = [{"type": "google_grounding", "function": {"name": "google_grounding_search"}}]
        await text_engine.generate_response(google_config, base_context, tools=grounding_tools)
        config = mock_instance.models.generate_content_stream.call_args.kwargs['config']
        assert config.tools
        assert any(hasattr(t, 'google_search') and t.google_search is not None for t in config.tools)

    @pytest.mark.asyncio
    async def test_function_tool_without_grounding(self, mock_google_client_class, text_engine, google_config, base_context, monkeypatch):
        """Function tool alone → function declarations present, no google_search injected."""
        monkeypatch.setenv("GOOGLE_GENERATIVEAI_API_KEY", "dummy_key_for_testing")
        mock_instance = mock_google_client_class.return_value
        mock_function_call = MagicMock(name="do_thing", args={})
        mock_part = MagicMock(text=None, function_call=mock_function_call)
        mock_candidate = MagicMock(content=MagicMock(parts=[mock_part]), grounding_metadata=None)
        mock_instance.models.generate_content_stream = AsyncMock(
            return_value=google_stream(MagicMock(prompt_feedback=None, candidates=[mock_candidate]))
        )
        function_tools = [{"type": "function", "function": {"name": "do_thing", "description": "does a thing", "parameters": {"type": "object", "properties": {}}}}]
        await text_engine.generate_response(google_config, base_context, tools=function_tools)
        config = mock_instance.models.generate_content_stream.call_args.kwargs['config']
        assert config.tools
        assert not any(hasattr(t, 'google_search') and t.google_search is not None for t in config.tools)
        assert any(hasattr(t, 'function_declarations') and t.function_declarations for t in config.tools)

    @pytest.mark.asyncio
    async def test_thought_signature_preserved_in_tool_calls(self, mock_google_client_class, text_engine, google_config,
                                                             base_context, monkeypatch):
        """Gemini 3.1 thinking models attach thought_signature to function call parts; it must be captured."""
        monkeypatch.setenv("GOOGLE_GENERATIVEAI_API_KEY", "dummy_key_for_testing")
        mock_instance = mock_google_client_class.return_value

        mock_function_call = MagicMock()
        mock_function_call.name = "web_search"
        mock_function_call.args = {'query': 'test'}

        mock_part = MagicMock(text=None, function_call=mock_function_call)
        mock_part.thought_signature = b'sig_abc123'
        mock_candidate = MagicMock(content=MagicMock(parts=[mock_part]), grounding_metadata=None)
        mock_instance.models.generate_content_stream = AsyncMock(
            return_value=google_stream(MagicMock(prompt_feedback=None, candidates=[mock_candidate]))
        )

        response, _ = await text_engine.generate_response(google_config, base_context, tools=[
            {"type": "function", "function": {"name": "web_search"}}])

        assert response['type'] == 'tool_calls'
        assert response['calls'][0]['thought_signature'] == base64.b64encode(b'sig_abc123').decode('utf-8')

    @pytest.mark.asyncio
    async def test_thought_signature_echoed_in_history(self, mock_google_client_class, text_engine, google_config,
                                                       base_context, monkeypatch):
        """When tool calls with thought_signature are in history, the signature must be echoed back to the API."""
        monkeypatch.setenv("GOOGLE_GENERATIVEAI_API_KEY", "dummy_key_for_testing")
        mock_instance = mock_google_client_class.return_value

        mock_part = MagicMock(text="Done", function_call=None)
        mock_candidate = MagicMock(content=MagicMock(parts=[mock_part]), grounding_metadata=None)
        mock_instance.models.generate_content_stream = AsyncMock(
            return_value=google_stream(MagicMock(prompt_feedback=None, candidates=[mock_candidate]))
        )

        # Simulate history with a tool call that has a thought_signature
        base_context["history"] = [
            {"role": "user", "content": "Search for test"},
            {"role": "assistant", "tool_calls": [
                {"id": "call_1", "name": "web_search", "arguments": {"query": "test"},
                 "thought_signature": base64.b64encode(b'sig_abc123').decode('utf-8')}
            ]},
            {"role": "tool", "tool_call_id": "call_1", "name": "web_search",
             "content": '{"result": "found"}'},
        ]

        await text_engine.generate_response(google_config, base_context)

        call_args = mock_instance.models.generate_content_stream.call_args[1]
        # The model turn (index 1: user, model, tool, ...) should have thought_signature
        model_turn = call_args['contents'][1]
        assert model_turn['role'] == 'model'
        assert model_turn['parts'][0].thought_signature == b'sig_abc123'

    @pytest.mark.asyncio
    async def test_no_thought_signature_when_absent(self, mock_google_client_class, text_engine, google_config,
                                                     base_context, monkeypatch):
        """Non-thinking models don't produce thought_signature; calls should omit it."""
        monkeypatch.setenv("GOOGLE_GENERATIVEAI_API_KEY", "dummy_key_for_testing")
        mock_instance = mock_google_client_class.return_value

        mock_function_call = MagicMock()
        mock_function_call.name = "web_search"
        mock_function_call.args = {'query': 'test'}

        mock_part = MagicMock(text=None, function_call=mock_function_call)
        mock_part.thought_signature = None
        mock_candidate = MagicMock(content=MagicMock(parts=[mock_part]), grounding_metadata=None)
        mock_instance.models.generate_content_stream = AsyncMock(
            return_value=google_stream(MagicMock(prompt_feedback=None, candidates=[mock_candidate]))
        )

        response, _ = await text_engine.generate_response(google_config, base_context, tools=[
            {"type": "function", "function": {"name": "web_search"}}])

        assert response['type'] == 'tool_calls'
        assert 'thought_signature' not in response['calls'][0]

    @pytest.mark.asyncio
    async def test_grounding_and_function_tools_combined(self, mock_google_client_class, text_engine, google_config, base_context, monkeypatch):
        """Both grounding and function tools → both present in API config."""
        monkeypatch.setenv("GOOGLE_GENERATIVEAI_API_KEY", "dummy_key_for_testing")
        mock_instance = mock_google_client_class.return_value
        mock_function_call = MagicMock(name="do_thing", args={})
        mock_part = MagicMock(text=None, function_call=mock_function_call)
        mock_candidate = MagicMock(content=MagicMock(parts=[mock_part]), grounding_metadata=None)
        mock_instance.models.generate_content_stream = AsyncMock(
            return_value=google_stream(MagicMock(prompt_feedback=None, candidates=[mock_candidate]))
        )
        mixed_tools = [
            {"type": "google_grounding", "function": {"name": "google_grounding_search"}},
            {"type": "function", "function": {"name": "do_thing", "description": "does a thing", "parameters": {"type": "object", "properties": {}}}},
        ]
        await text_engine.generate_response(google_config, base_context, tools=mixed_tools)
        config = mock_instance.models.generate_content_stream.call_args.kwargs['config']
        assert config.tools
        assert any(hasattr(t, 'google_search') and t.google_search is not None for t in config.tools)
        assert any(hasattr(t, 'function_declarations') and t.function_declarations for t in config.tools)

    @pytest.mark.asyncio
    @patch('aiohttp.ClientSession.get')
    async def test_image_url_passed_to_google(self, mock_get, mock_google_client_class, text_engine, google_config,
                                              base_context, monkeypatch):
        monkeypatch.setenv("GOOGLE_GENERATIVEAI_API_KEY", "dummy_key_for_testing")

        # Mock the image download
        mock_response = AsyncMock()
        mock_response.read.return_value = b'imagedata'
        mock_response.content_type = 'image/jpeg'
        mock_response.raise_for_status = MagicMock()
        mock_get.return_value.__aenter__.return_value = mock_response

        # Mock the Gemini API response
        mock_instance = mock_google_client_class.return_value
        mock_part = MagicMock()
        mock_part.function_call = None
        mock_part.text = "Image received"
        mock_candidate = MagicMock(content=MagicMock(parts=[mock_part]))
        mock_candidate.grounding_metadata = None
        mock_instance.models.generate_content_stream = AsyncMock(
            return_value=google_stream(MagicMock(prompt_feedback=None, candidates=[mock_candidate]))
        )

        base_context["current_message"]["image_url"] = "http://example.com/image.jpg"
        base_context["history"] = [{"role": "user", "content": "Check this out"}]

        await text_engine.generate_response(google_config, base_context)

        # Verify that the image was included in the API call
        call_args = mock_instance.models.generate_content_stream.call_args[1]
        assert len(call_args['contents'][-1]['parts']) == 2
        assert call_args['contents'][-1]['parts'][-1].inline_data.data == b'imagedata'


class TestLocalModel:
    """DP-206b: `local` one-shot rides the engine-owned kobold-native
    StreamEngine — generate_response = collect over `stream_local`, the same
    transport and `<tool_call>` protocol as the streaming portal path. The
    OpenAI-compat local transport is gone."""

    @staticmethod
    def _fake_local_engine(events):
        async def _gen(*a, **k):
            for ev in events:
                yield ev
        fake = MagicMock()
        fake.stream_local = MagicMock(side_effect=_gen)
        return fake

    @pytest.mark.asyncio
    async def test_success_text_response(self, local_config, base_context):
        fake = self._fake_local_engine([
            {"type": "api_payload", "payload": {"prompt": "<13 chars>", "genkey": "KCPP1234"}},
            {"type": "text_delta", "text": "Local success"},
            {"type": "done", "full_text": "Local success"},
        ])
        engine = TextEngine(stream_engine=fake)
        response, payload = await engine.generate_response(
            local_config, base_context, None, {"temperature": 0.5},
        )
        assert response == {"type": "text", "content": "Local success"}
        assert payload == {"prompt": "<13 chars>", "genkey": "KCPP1234"}
        fake.stream_local.assert_called_once()
        # The driver forwards (config, history_object, tools, local_inference_config).
        args = fake.stream_local.call_args[0]
        assert args[1] is base_context
        assert args[3] == {"temperature": 0.5}

    @pytest.mark.asyncio
    async def test_success_tool_call_response(self, local_config, base_context):
        """A `<tool_call>` block parsed out of the kobold token stream surfaces
        as a standard tool_calls result from the one-shot path."""
        calls = [{"id": "call_run_code_0", "name": "run_code",
                  "arguments": {"code": "print('hello from local')"}}]
        fake = self._fake_local_engine([
            {"type": "api_payload", "payload": {"prompt": "<10 chars>"}},
            {"type": "tool_calls", "calls": calls},
            {"type": "done", "full_text": ""},
        ])
        engine = TextEngine(stream_engine=fake)
        response, _ = await engine.generate_response(local_config, base_context, tools=[
            {"type": "function", "function": {"name": "run_code"}}])

        assert response['type'] == 'tool_calls'
        assert response['calls'] == calls

    @pytest.mark.asyncio
    async def test_transport_error_raises_llm_error(self, local_config, base_context):
        fake = MagicMock()
        fake.stream_local = MagicMock(side_effect=LLMCommunicationError(
            "Kobold native stream transport error: connection refused"
        ))
        engine = TextEngine(stream_engine=fake)
        with patch('src.engine.asyncio.sleep', new_callable=AsyncMock):
            with pytest.raises(LLMCommunicationError, match="Kobold native stream"):
                await engine.generate_response(local_config, base_context)
        # Transport errors are retried like any provider before surfacing.
        assert fake.stream_local.call_count == EMPTY_RESPONSE_RETRIES + 1

    def test_default_engine_owns_a_real_stream_engine(self):
        """Facade collapse: TextEngine() constructs its kobold-native local
        provider itself — no separate wiring at the composition root."""
        from src.stream_engine import StreamEngine
        engine = TextEngine()
        assert isinstance(engine.stream_engine, StreamEngine)


class TestProviderRouting:
    """Tests for _get_provider_route and model routing edge cases."""

    def test_unsupported_model_raises(self, text_engine):
        with pytest.raises(LLMCommunicationError, match="not supported"):
            text_engine._get_provider_route("unknown-model-v1")

    @pytest.mark.asyncio
    @patch('src.engine.TextEngine._stream_openai_response')
    async def test_image_unsupported_model_modifies_prompt(self, mock_provider, text_engine, base_context):
        """Models that don't support images get a system note appended and image_url cleared."""
        base_context["current_message"]["image_url"] = "http://example.com/photo.png"
        # gpt-3.5-turbo matches routing (starts with "gpt") but fails model_supports_images
        config = {"model_name": "gpt-3.5-turbo"}

        mock_provider.side_effect = lambda *a, **k: _one_shot_stream({"type": "text", "content": "ok"}, {})
        await text_engine.generate_response(config, base_context)

        assert base_context["current_message"]["image_url"] is None
        assert "cannot see" in base_context["persona_prompt"]


@patch('src.engine.AsyncOpenAI')
class TestOpenAIImage:
    @pytest.mark.asyncio
    async def test_image_url_passed_to_openai(self, mock_openai_class, text_engine,
                                              openai_config, base_context, monkeypatch):
        """OpenAI image attachment: URL is included as image_url content part."""
        monkeypatch.setenv("OPENAI_API_KEY", "dummy_key_for_testing")
        mock_instance = mock_openai_class.return_value
        mock_instance.chat.completions.create = AsyncMock(
            return_value=openai_text_stream("I see the image")
        )

        base_context["current_message"]["image_url"] = "http://example.com/photo.png"
        base_context["history"] = [{"role": "user", "content": "What's in this image?"}]

        response, _ = await text_engine.generate_response(openai_config, base_context)
        assert response == {"type": "text", "content": "I see the image"}

        call_args = mock_instance.chat.completions.create.call_args[1]
        last_msg = call_args['messages'][-1]
        assert isinstance(last_msg['content'], list)
        assert last_msg['content'][-1] == {"type": "image_url", "image_url": {"url": "http://example.com/photo.png"}}

    @pytest.mark.asyncio
    async def test_malformed_tool_call_json_skipped(self, mock_openai_class, text_engine,
                                                    openai_config, base_context, monkeypatch):
        """Tool calls with unparseable JSON arguments are skipped, not fatal."""
        monkeypatch.setenv("OPENAI_API_KEY", "dummy_key_for_testing")
        mock_instance = mock_openai_class.return_value
        mock_instance.chat.completions.create = AsyncMock(
            return_value=openai_tool_call_stream([
                ("call_1", "get_weather", '{"city": "NYC"}'),
                ("call_2", "broken_tool", '{not valid json'),
            ])
        )

        response, _ = await text_engine.generate_response(
            openai_config, base_context, tools=[{"type": "function", "function": {"name": "get_weather"}}]
        )
        assert response['type'] == 'tool_calls'
        assert len(response['calls']) == 1
        assert response['calls'][0]['name'] == 'get_weather'


@patch('src.engine.genai.client.AsyncClient')
class TestGoogleEdgeCases:
    @pytest.mark.asyncio
    async def test_blocked_response_raises(self, mock_google_client_class, text_engine,
                                           google_config, base_context, monkeypatch):
        """Google prompt blocking raises LLMCommunicationError."""
        monkeypatch.setenv("GOOGLE_GENERATIVEAI_API_KEY", "dummy_key_for_testing")
        mock_instance = mock_google_client_class.return_value

        mock_block_reason = MagicMock()
        mock_block_reason.name = "SAFETY"
        mock_prompt_feedback = MagicMock(block_reason=mock_block_reason)

        mock_instance.models.generate_content_stream = AsyncMock(
            return_value=google_stream(
                MagicMock(prompt_feedback=mock_prompt_feedback, candidates=[])
            )
        )

        with pytest.raises(LLMCommunicationError, match="blocked by Google.*SAFETY"):
            await text_engine.generate_response(google_config, base_context)

    @pytest.mark.asyncio
    async def test_empty_candidate_returns_empty_and_retries(self, mock_google_client_class,
                                                             text_engine, google_config,
                                                             base_context, monkeypatch):
        """Response with no candidates returns {} which triggers retry logic."""
        monkeypatch.setenv("GOOGLE_GENERATIVEAI_API_KEY", "dummy_key_for_testing")
        mock_instance = mock_google_client_class.return_value
        mock_instance.models.generate_content_stream = AsyncMock(
            return_value=google_stream(MagicMock(prompt_feedback=None, candidates=[]))
        )

        with pytest.raises(LLMCommunicationError, match="empty or invalid response after all retries"):
            await text_engine.generate_response(google_config, base_context)

    @pytest.mark.asyncio
    @patch('aiohttp.ClientSession.get')
    async def test_image_download_failure_gracefully_skipped(self, mock_get,
                                                             mock_google_client_class, text_engine,
                                                             google_config, base_context, monkeypatch):
        """Failed image download doesn't crash — response is still generated without the image."""
        monkeypatch.setenv("GOOGLE_GENERATIVEAI_API_KEY", "dummy_key_for_testing")

        mock_get.return_value.__aenter__.side_effect = aiohttp.ClientError("Connection refused")

        mock_instance = mock_google_client_class.return_value
        mock_part = MagicMock(text="Response without image", function_call=None)
        mock_candidate = MagicMock(content=MagicMock(parts=[mock_part]), grounding_metadata=None)
        mock_instance.models.generate_content_stream = AsyncMock(
            return_value=google_stream(MagicMock(prompt_feedback=None, candidates=[mock_candidate]))
        )

        base_context["current_message"]["image_url"] = "http://example.com/broken.png"
        base_context["history"] = [{"role": "user", "content": "Look at this"}]

        response, _ = await text_engine.generate_response(google_config, base_context)
        assert response == {"type": "text", "content": "Response without image"}

    @pytest.mark.asyncio
    @patch('aiohttp.ClientSession.get')
    async def test_unsupported_mime_type_skipped(self, mock_get,
                                                 mock_google_client_class, text_engine,
                                                 google_config, base_context, monkeypatch):
        """Image with unsupported MIME type (e.g. BMP) is skipped, not attached."""
        monkeypatch.setenv("GOOGLE_GENERATIVEAI_API_KEY", "dummy_key_for_testing")

        mock_response = AsyncMock()
        mock_response.read.return_value = b'bmpdata'
        mock_response.content_type = 'image/bmp'
        mock_response.raise_for_status = MagicMock()
        mock_get.return_value.__aenter__.return_value = mock_response

        mock_instance = mock_google_client_class.return_value
        mock_part = MagicMock(text="No image seen", function_call=None)
        mock_candidate = MagicMock(content=MagicMock(parts=[mock_part]), grounding_metadata=None)
        mock_instance.models.generate_content_stream = AsyncMock(
            return_value=google_stream(MagicMock(prompt_feedback=None, candidates=[mock_candidate]))
        )

        base_context["current_message"]["image_url"] = "http://example.com/image.bmp"
        base_context["history"] = [{"role": "user", "content": "Check this BMP"}]

        await text_engine.generate_response(google_config, base_context)

        call_args = mock_instance.models.generate_content_stream.call_args[1]
        user_turn = call_args['contents'][-1]
        assert len(user_turn['parts']) == 1  # text only, no image


@patch('src.engine.anthropic.AsyncAnthropic')
class TestAnthropicEdgeCases:
    @pytest.mark.asyncio
    @patch('aiohttp.ClientSession.get')
    async def test_image_download_failure_gracefully_skipped(self, mock_get,
                                                             mock_anthropic_class, text_engine,
                                                             anthropic_config, base_context,
                                                             monkeypatch):
        """Failed image download for Anthropic doesn't crash."""
        monkeypatch.setenv("ANTHROPIC_API_KEY", "dummy_key_for_testing")

        mock_get.return_value.__aenter__.side_effect = aiohttp.ClientError("Timeout")

        mock_instance = mock_anthropic_class.return_value
        mock_instance.messages.stream.return_value = anthropic_stream(MagicMock(
            content=[MagicMock(text="Response without image")], stop_reason="end_turn"
        ), ["Response without image"])

        base_context["current_message"]["image_url"] = "http://example.com/broken.png"
        base_context["history"] = [{"role": "user", "content": "Look at this"}]

        response, _ = await text_engine.generate_response(anthropic_config, base_context)
        assert response == {"type": "text", "content": "Response without image"}

    @pytest.mark.asyncio
    @patch('aiohttp.ClientSession.get')
    async def test_unsupported_mime_type_skipped(self, mock_get,
                                                 mock_anthropic_class, text_engine,
                                                 anthropic_config, base_context,
                                                 monkeypatch):
        """Image with unsupported MIME type for Anthropic is skipped."""
        monkeypatch.setenv("ANTHROPIC_API_KEY", "dummy_key_for_testing")

        mock_response = AsyncMock()
        mock_response.read.return_value = b'tiffdata'
        mock_response.content_type = 'image/tiff'
        mock_response.raise_for_status = MagicMock()
        mock_get.return_value.__aenter__.return_value = mock_response

        mock_instance = mock_anthropic_class.return_value
        mock_instance.messages.stream.return_value = anthropic_stream(MagicMock(
            content=[MagicMock(text="No image seen")], stop_reason="end_turn"
        ), ["No image seen"])

        base_context["current_message"]["image_url"] = "http://example.com/image.tiff"
        base_context["history"] = [{"role": "user", "content": "Check this TIFF"}]

        response, _ = await text_engine.generate_response(anthropic_config, base_context)
        assert response == {"type": "text", "content": "No image seen"}

        # Verify no image block was sent
        call_args = mock_instance.messages.stream.call_args[1]
        last_msg = call_args['messages'][-1]
        # content was converted to list (text part) but no image part added
        assert isinstance(last_msg['content'], list)
        assert all(block.get('type') != 'image' for block in last_msg['content'])


class TestExtractSystemPrompt:
    def test_merges_system_message_from_history(self, text_engine):
        context = {
            "persona_prompt": "Base prompt",
            "history": [
                {"role": "system", "content": "Extra system context"},
                {"role": "user", "content": "Hello"}
            ]
        }
        prompt, history = text_engine._extract_system_prompt(context)
        assert prompt == "Base prompt\n\nExtra system context"
        assert len(history) == 1
        assert history[0]["role"] == "user"

    def test_no_system_message_returns_persona_prompt(self, text_engine):
        context = {
            "persona_prompt": "Base prompt",
            "history": [{"role": "user", "content": "Hello"}]
        }
        prompt, history = text_engine._extract_system_prompt(context)
        assert prompt == "Base prompt"
        assert len(history) == 1

    def test_empty_history(self, text_engine):
        context = {"persona_prompt": "Base prompt", "history": []}
        prompt, history = text_engine._extract_system_prompt(context)
        assert prompt == "Base prompt"
        assert history == []


class TestWebSearch:
    @pytest.mark.asyncio
    @patch('ddgs.DDGS')
    async def test_web_search_returns_formatted_results(self, mock_ddgs_class):
        from src.tools.tool_manager import ToolManager, WebSearchHandler
        mock_ddgs_instance = MagicMock()
        mock_ddgs_class.return_value.__enter__ = MagicMock(return_value=mock_ddgs_instance)
        mock_ddgs_class.return_value.__exit__ = MagicMock(return_value=False)
        mock_ddgs_instance.text.return_value = [
            {"title": "Result One", "href": "http://example.com/1", "body": "Summary one."},
            {"title": "Result Two", "href": "http://example.com/2", "body": "Summary two."},
        ]
        manager = ToolManager()
        WebSearchHandler().register(manager)
        result = await manager.execute_tool("web_search", query="test query")
        assert "result" in result
        assert result["result"] == [
            {"title": "Result One", "url": "http://example.com/1", "summary": "Summary one."},
            {"title": "Result Two", "url": "http://example.com/2", "summary": "Summary two."},
        ]
        mock_ddgs_instance.text.assert_called_once_with("test query", max_results=5)

    @pytest.mark.asyncio
    @patch('ddgs.DDGS')
    async def test_web_search_respects_max_results(self, mock_ddgs_class):
        from src.tools.tool_manager import ToolManager, WebSearchHandler
        mock_ddgs_instance = MagicMock()
        mock_ddgs_class.return_value.__enter__ = MagicMock(return_value=mock_ddgs_instance)
        mock_ddgs_class.return_value.__exit__ = MagicMock(return_value=False)
        mock_ddgs_instance.text.return_value = []
        manager = ToolManager()
        WebSearchHandler().register(manager)
        await manager.execute_tool("web_search", query="test", max_results=3)
        mock_ddgs_instance.text.assert_called_once_with("test", max_results=3)


class TestAgyRenderAndConfig:
    """Sprint 1: SDK-free pieces of the agy route — prompt flattening, image
    policy, and limiter wiring. The handler/route themselves land in later
    sprints (they depend on the Antigravity SDK)."""

    def test_render_flattens_full_history_no_dup(self, text_engine):
        """Every prior turn AND the final user turn appear, exactly once each."""
        history = [
            {"role": "user", "content": "first question"},
            {"role": "assistant", "content": "first answer"},
            {"role": "user", "content": "current question"},
        ]
        rendered = text_engine._render_agy_prompt(history)

        assert "User: first question" in rendered
        assert "Assistant: first answer" in rendered
        assert "User: current question" in rendered
        # "nothing dropped": all three turns present
        assert rendered.count("User:") == 2
        assert rendered.count("Assistant:") == 1
        # "nothing duplicated": the final user turn appears exactly once
        assert rendered.count("current question") == 1
        # ordering preserved
        assert rendered.index("first question") < rendered.index("first answer") < rendered.index("current question")

    def test_render_handles_tool_turns(self, text_engine):
        """tool-role results and assistant tool_calls render with their tags —
        this is what lets the engine's multi-turn tool loop reach agy."""
        history = [
            {"role": "user", "content": "weather?"},
            {"role": "assistant", "content": "checking", "tool_calls": [
                {"id": "c1", "name": "get_weather", "arguments": {"city": "NYC"}},
            ]},
            {"role": "tool", "name": "get_weather", "content": '{"temp": 70}'},
            {"role": "user", "content": "thanks"},
        ]
        rendered = text_engine._render_agy_prompt(history)

        assert "Assistant: checking" in rendered
        assert 'Assistant (tool call get_weather): {"city": "NYC"}' in rendered
        assert 'Tool(get_weather): {"temp": 70}' in rendered
        assert "User: thanks" in rendered

    def test_render_tool_loop_followup_reaches_prompt(self, text_engine):
        """A history ending in a tool result (the engine's follow-up turn)
        renders that result so agy sees it on the next stateless call."""
        history = [
            {"role": "user", "content": "lookup"},
            {"role": "assistant", "tool_calls": [
                {"id": "c1", "name": "search", "arguments": {"q": "x"}},
            ]},
            {"role": "tool", "name": "search", "content": '{"hits": 3}'},
        ]
        rendered = text_engine._render_agy_prompt(history)
        assert 'Tool(search): {"hits": 3}' in rendered
        # assistant turn carrying only tool_calls still renders the call
        assert 'Assistant (tool call search): {"q": "x"}' in rendered

    def test_render_excludes_system_prompt(self, text_engine):
        """The persona is delivered via CustomSystemInstructions, never in the
        flattened transcript. _render_agy_prompt only receives post-extraction
        history, but guard against a stray system turn leaking through."""
        history = [{"role": "user", "content": "hi"}]
        rendered = text_engine._render_agy_prompt(history)
        assert "System" not in rendered

    def test_agy_excluded_from_image_support(self, text_engine):
        """agy is text-only in v1; excluding it means images get the existing
        'can't see image' note + strip rather than being silently dropped."""
        assert text_engine.model_supports_images("agy-flash") is False

    def test_agy_limiter_constructed(self, text_engine):
        """The agy rate limiter is wired at init, ready for the route."""
        from aiolimiter import AsyncLimiter
        assert isinstance(text_engine._agy_limiter, AsyncLimiter)


class TestAgyHandler:
    def test_tool_protocol_includes_tools(self, text_engine):
        tools = [{
            "function": {
                "name": "get_weather",
                "description": "Get the current weather",
                "parameters": {"type": "object", "properties": {"location": {"type": "string"}}}
            }
        }]
        protocol = text_engine._render_agy_tool_protocol(tools)
        assert "get_weather" in protocol
        assert "Get the current weather" in protocol
        assert "<tool_call>" in protocol

    def test_tool_protocol_empty_without_tools(self, text_engine):
        assert text_engine._render_agy_tool_protocol([]) == ""
        assert text_engine._render_agy_tool_protocol(None) == ""

    def test_parse_clean_tool_call(self, text_engine):
        text = '<tool_call>{"name": "get_weather", "arguments": {"location": "Tokyo"}}</tool_call>'
        parsed = text_engine._parse_agy_tool_call(text)
        assert parsed is not None
        assert len(parsed) == 1
        call = parsed[0]
        assert call["name"] == "get_weather"
        assert call["arguments"] == {"location": "Tokyo"}
        assert isinstance(call["id"], str)
        assert len(call["id"]) > 0

    def test_parse_no_block_returns_none(self, text_engine):
        assert text_engine._parse_agy_tool_call("This is plain text with no tool call.") is None

    def test_parse_strips_system_message(self, text_engine):
        text = '<SYSTEM_MESSAGE>system message noise</SYSTEM_MESSAGE>some prose\n<tool_call>{"name": "get_weather", "arguments": {"location": "Tokyo"}}</tool_call>\nmore prose'
        parsed = text_engine._parse_agy_tool_call(text)
        assert parsed is not None
        assert len(parsed) == 1
        call = parsed[0]
        assert call["name"] == "get_weather"
        assert call["arguments"] == {"location": "Tokyo"}

    def test_parse_malformed_json_returns_none(self, text_engine):
        text = '<tool_call>{"name": "get_weather", "arguments": </tool_call>'
        assert text_engine._parse_agy_tool_call(text) is None

    @pytest.mark.asyncio
    async def test_handler_text_path(self, text_engine, base_context, monkeypatch):
        mock_cli = AsyncMock(return_value="a plain answer")
        monkeypatch.setattr(text_engine, "_run_agy_cli", mock_cli)

        config = {"model_name": "agy-flash"}
        response, api_payload = await text_engine._generate_agy_response(config, base_context)

        assert response == {"type": "text", "content": "a plain answer"}
        assert isinstance(api_payload, dict)

    @pytest.mark.asyncio
    async def test_handler_tool_path(self, text_engine, base_context, monkeypatch):
        tool_output = '<tool_call>{"name": "get_weather", "arguments": {"location": "Tokyo"}}</tool_call>'
        mock_cli = AsyncMock(return_value=tool_output)
        monkeypatch.setattr(text_engine, "_run_agy_cli", mock_cli)

        config = {"model_name": "agy-flash"}
        tools = [{
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get the current weather",
                "parameters": {"type": "object", "properties": {"location": {"type": "string"}}}
            }
        }]

        response, api_payload = await text_engine._generate_agy_response(config, base_context, tools=tools)

        assert response["type"] == "tool_calls"
        assert len(response["calls"]) == 1
        call = response["calls"][0]
        assert call["name"] == "get_weather"
        assert call["arguments"] == {"location": "Tokyo"}
        assert isinstance(call["id"], str)
        assert len(call["id"]) > 0
        assert isinstance(api_payload, dict)

    @pytest.mark.asyncio
    async def test_handler_injects_system_and_tools_into_prompt(self, text_engine, base_context, monkeypatch):
        mock_cli = AsyncMock(return_value="mocked output")
        monkeypatch.setattr(text_engine, "_run_agy_cli", mock_cli)

        config = {"model_name": "agy-flash"}
        tools = [{
            "function": {
                "name": "get_weather",
                "description": "Get the current weather",
                "parameters": {"type": "object", "properties": {"location": {"type": "string"}}}
            }
        }]

        await text_engine._generate_agy_response(config, base_context, tools=tools)

        mock_cli.assert_called_once()
        prompt_arg = mock_cli.call_args[0][0]
        assert "You are a test bot." in prompt_arg
        assert "get_weather" in prompt_arg
        assert "<tool_call>" in prompt_arg

    @pytest.mark.asyncio
    async def test_handler_api_payload_has_no_secret(self, text_engine, base_context, monkeypatch):
        mock_cli = AsyncMock(return_value="a plain answer")
        monkeypatch.setattr(text_engine, "_run_agy_cli", mock_cli)

        config = {"model_name": "agy-flash"}
        _, api_payload = await text_engine._generate_agy_response(config, base_context)

        # Simple key-name and value scan for secrets
        payload_str = str(api_payload).lower()
        for forbidden in ["secret", "token", "oauth", "api_key"]:
            assert forbidden not in payload_str

    def test_route_resolves_to_agy_handler(self, text_engine, monkeypatch):
        # route resolution now calls the POSIX-only guard; no-op it so the
        # route-table assertion itself runs on any host
        monkeypatch.setattr(text_engine, "_ensure_agy_supported", lambda: None)
        handler, limiters = text_engine._get_provider_route("agy-flash")
        assert handler == text_engine._stream_agy_response
        assert limiters == [text_engine._agy_limiter]

    def test_route_refuses_agy_on_windows(self, text_engine, monkeypatch):
        """Selecting an agy model on native Windows fails at route resolution —
        before any temp dir or subprocess is created."""
        import src.engine as engine_mod

        monkeypatch.setattr(engine_mod.os, "name", "nt")
        with pytest.raises(LLMCommunicationError, match="native Windows"):
            text_engine._get_provider_route("agy-flash")

    @pytest.mark.asyncio
    async def test_generate_response_end_to_end_text(self, text_engine, base_context, monkeypatch):
        monkeypatch.setattr(text_engine, "_ensure_agy_supported", lambda: None)
        mock_cli = AsyncMock(return_value="end-to-end text answer")
        monkeypatch.setattr(text_engine, "_run_agy_cli", mock_cli)

        config = {"model_name": "agy-flash"}
        response, api_payload = await text_engine.generate_response(config, base_context)

        assert response == {"type": "text", "content": "end-to-end text answer"}
        assert isinstance(api_payload, dict)


class TestAgyCliInvocation:
    """Covers the real subprocess wiring of ``_run_agy_cli`` and the POSIX-only
    guard — the parts the mocked handler tests above skip.

    agy works on POSIX (macOS/Docker); on native Windows it is a TUI that only
    emits its response to a TTY, so DERPR's piped capture comes back empty.
    ``_ensure_agy_supported`` refuses the route on Windows rather than returning
    silent empty responses. We deliberately do NOT pass
    --dangerously-skip-permissions: agy must keep its own tools gated so it can
    never run them (DERPR drives every tool itself).
    """

    class _FakeProc:
        def __init__(self, *, stdout=b"", stderr=b"", returncode=0, pid=4321):
            self._stdout = stdout
            self._stderr = stderr
            self.returncode = returncode
            self.pid = pid

        async def communicate(self):
            return self._stdout, self._stderr

    @pytest.mark.asyncio
    async def test_run_agy_cli_args_match_working_posix_behavior(self, text_engine, monkeypatch, tmp_path):
        """Regression guard: the spawn args stay exactly the known-good POSIX set
        — no --dangerously-skip-permissions (would un-gate agy's own tools)."""
        import src.engine as engine_mod
        from config import global_config

        # keep the default persistent workspace out of the real data dir
        monkeypatch.setattr(global_config, "AGY_WORKSPACES_DIR", tmp_path / "workspaces")

        captured = {}

        async def fake_exec(*args, **kwargs):
            captured["args"] = args
            captured["kwargs"] = kwargs
            return self._FakeProc(stdout=b"hello from agy")

        monkeypatch.setattr(engine_mod.asyncio, "create_subprocess_exec", fake_exec)
        monkeypatch.setattr(engine_mod.shutil, "which", lambda name: "/usr/bin/agy")
        monkeypatch.delenv("ANTIGRAVITY_HARNESS_PATH", raising=False)
        # bypass the POSIX-only guard so the subprocess wiring runs on any host
        monkeypatch.setattr(text_engine, "_ensure_agy_supported", lambda: None)

        out = await text_engine._run_agy_cli("say hi", timeout=5)

        assert out == "hello from agy"
        assert captured["args"][0] == "/usr/bin/agy"
        assert "--dangerously-skip-permissions" not in captured["args"]
        assert "-p" in captured["args"] and "--print-timeout" in captured["args"]
        # OS-level sandbox on by default (defense-in-depth)
        assert "--sandbox" in captured["args"]
        # agy is isolated in its own POSIX session for cleanup
        assert captured["kwargs"].get("start_new_session") is True

    @pytest.mark.asyncio
    async def test_sandbox_flag_omitted_when_disabled(self, text_engine, monkeypatch, tmp_path):
        import src.engine as engine_mod
        from config import global_config

        monkeypatch.setattr(global_config, "AGY_WORKSPACES_DIR", tmp_path / "workspaces")
        monkeypatch.setattr(global_config, "AGY_SANDBOX", False)

        captured = {}

        async def fake_exec(*args, **kwargs):
            captured["args"] = args
            return self._FakeProc(stdout=b"ok")

        monkeypatch.setattr(engine_mod.asyncio, "create_subprocess_exec", fake_exec)
        monkeypatch.setattr(engine_mod.shutil, "which", lambda name: "/usr/bin/agy")
        monkeypatch.setattr(text_engine, "_ensure_agy_supported", lambda: None)

        await text_engine._run_agy_cli("hi", timeout=5)
        assert "--sandbox" not in captured["args"]

    def test_agy_unsupported_on_windows_raises_clear_error(self, text_engine, monkeypatch):
        import src.engine as engine_mod

        monkeypatch.setattr(engine_mod.os, "name", "nt")
        with pytest.raises(LLMCommunicationError, match="native Windows"):
            text_engine._ensure_agy_supported()

    def test_agy_supported_on_posix(self, text_engine, monkeypatch):
        import src.engine as engine_mod

        monkeypatch.setattr(engine_mod.os, "name", "posix")
        text_engine._ensure_agy_supported()  # no raise

    @pytest.mark.asyncio
    async def test_run_agy_cli_aborts_before_spawn_on_windows(self, text_engine, monkeypatch):
        """On Windows the guard must fire before any temp dir or subprocess."""
        import src.engine as engine_mod

        monkeypatch.setattr(engine_mod.os, "name", "nt")
        spawned = []
        made_dirs = []

        async def fake_exec(*a, **k):
            spawned.append(True)
            raise AssertionError("should not spawn agy on Windows")

        monkeypatch.setattr(engine_mod.asyncio, "create_subprocess_exec", fake_exec)
        monkeypatch.setattr(engine_mod.tempfile, "mkdtemp", lambda: made_dirs.append(True) or "x")

        with pytest.raises(LLMCommunicationError, match="native Windows"):
            await text_engine._run_agy_cli("hi", timeout=5)
        assert spawned == [] and made_dirs == []

    @pytest.mark.asyncio
    async def test_run_agy_cli_persistent_workspaces_persona(self, text_engine, monkeypatch, tmp_path):
        import src.engine as engine_mod
        from config import global_config
        
        captured_cwd = []
        async def fake_exec(*args, **kwargs):
            captured_cwd.append(kwargs.get("cwd"))
            return self._FakeProc(stdout=b"hello persistent persona")

        monkeypatch.setattr(engine_mod.asyncio, "create_subprocess_exec", fake_exec)
        monkeypatch.setattr(engine_mod.shutil, "which", lambda name: "/usr/bin/agy")
        monkeypatch.setattr(text_engine, "_ensure_agy_supported", lambda: None)
        
        monkeypatch.setattr(global_config, "AGY_PERSISTENT_WORKSPACES", True)
        monkeypatch.setattr(global_config, "AGY_WORKSPACE_MODE", "persona")
        monkeypatch.setattr(global_config, "AGY_WORKSPACES_DIR", tmp_path / "workspaces")
        
        # Test 1: Persona-specific workspace
        out = await text_engine._run_agy_cli("hi", persona_name="alice")
        assert out == "hello persistent persona"
        expected_dir = os.path.abspath(tmp_path / "workspaces" / "agy_alice")
        assert captured_cwd == [expected_dir]
        assert os.path.exists(expected_dir)

    @pytest.mark.asyncio
    async def test_run_agy_cli_persistent_workspaces_global(self, text_engine, monkeypatch, tmp_path):
        import src.engine as engine_mod
        from config import global_config
        
        captured_cwd = []
        async def fake_exec(*args, **kwargs):
            captured_cwd.append(kwargs.get("cwd"))
            return self._FakeProc(stdout=b"hello persistent global")

        monkeypatch.setattr(engine_mod.asyncio, "create_subprocess_exec", fake_exec)
        monkeypatch.setattr(engine_mod.shutil, "which", lambda name: "/usr/bin/agy")
        monkeypatch.setattr(text_engine, "_ensure_agy_supported", lambda: None)
        
        monkeypatch.setattr(global_config, "AGY_PERSISTENT_WORKSPACES", True)
        monkeypatch.setattr(global_config, "AGY_WORKSPACE_MODE", "global")
        monkeypatch.setattr(global_config, "AGY_WORKSPACES_DIR", tmp_path / "workspaces")
        
        # Test 2: Global mode (uses global workspace even if persona_name is passed)
        out = await text_engine._run_agy_cli("hi", persona_name="alice")
        assert out == "hello persistent global"
        expected_dir = os.path.abspath(tmp_path / "workspaces" / "agy_global")
        assert captured_cwd == [expected_dir]
        assert os.path.exists(expected_dir)

    @pytest.mark.asyncio
    async def test_run_agy_cli_stateless_fallback(self, text_engine, monkeypatch, tmp_path):
        import src.engine as engine_mod
        from config import global_config
        
        captured_cwd = []
        async def fake_exec(*args, **kwargs):
            captured_cwd.append(kwargs.get("cwd"))
            return self._FakeProc(stdout=b"hello stateless")

        monkeypatch.setattr(engine_mod.asyncio, "create_subprocess_exec", fake_exec)
        monkeypatch.setattr(engine_mod.shutil, "which", lambda name: "/usr/bin/agy")
        monkeypatch.setattr(text_engine, "_ensure_agy_supported", lambda: None)
        
        monkeypatch.setattr(global_config, "AGY_PERSISTENT_WORKSPACES", False)
        
        # Test 3: Stateless temp dir is created and removed
        out = await text_engine._run_agy_cli("hi")
        assert out == "hello stateless"
        assert len(captured_cwd) == 1
        # The temporary directory path should be deleted now
        assert not os.path.exists(captured_cwd[0])

    def _persistent_workspace_env(self, text_engine, monkeypatch, tmp_path, stdout=b"ok"):
        import src.engine as engine_mod
        from config import global_config

        captured_cwd = []

        async def fake_exec(*args, **kwargs):
            captured_cwd.append(kwargs.get("cwd"))
            return self._FakeProc(stdout=stdout)

        monkeypatch.setattr(engine_mod.asyncio, "create_subprocess_exec", fake_exec)
        monkeypatch.setattr(engine_mod.shutil, "which", lambda name: "/usr/bin/agy")
        monkeypatch.setattr(text_engine, "_ensure_agy_supported", lambda: None)
        monkeypatch.setattr(global_config, "AGY_PERSISTENT_WORKSPACES", True)
        monkeypatch.setattr(global_config, "AGY_WORKSPACE_MODE", "persona")
        monkeypatch.setattr(global_config, "AGY_WORKSPACES_DIR", tmp_path / "workspaces")
        return captured_cwd

    @pytest.mark.asyncio
    async def test_persona_name_is_sanitized_for_workspace_path(self, text_engine, monkeypatch, tmp_path):
        """Path separators and traversal in a persona name must not escape
        the workspaces dir."""
        captured_cwd = self._persistent_workspace_env(text_engine, monkeypatch, tmp_path)

        await text_engine._run_agy_cli("hi", persona_name="../evil/name")

        workspaces_root = os.path.abspath(tmp_path / "workspaces")
        assert len(captured_cwd) == 1
        assert captured_cwd[0] == os.path.join(workspaces_root, "agy_evil_name")
        assert os.path.dirname(captured_cwd[0]) == workspaces_root

    @pytest.mark.asyncio
    async def test_persona_mode_without_persona_name_falls_back_to_global(self, text_engine, monkeypatch, tmp_path):
        captured_cwd = self._persistent_workspace_env(text_engine, monkeypatch, tmp_path)

        await text_engine._run_agy_cli("hi")  # no persona_name

        expected = os.path.abspath(tmp_path / "workspaces" / "agy_global")
        assert captured_cwd == [expected]

    @pytest.mark.asyncio
    async def test_concurrent_calls_to_same_workspace_are_serialized(self, text_engine, monkeypatch, tmp_path):
        """Two in-flight calls sharing a persistent workspace must not overlap —
        the per-workspace lock serializes them."""
        import asyncio
        import src.engine as engine_mod
        from config import global_config

        in_flight = 0
        max_in_flight = 0

        class _SlowProc(self._FakeProc):
            async def communicate(inner):
                nonlocal in_flight, max_in_flight
                in_flight += 1
                max_in_flight = max(max_in_flight, in_flight)
                await asyncio.sleep(0.01)
                in_flight -= 1
                return inner._stdout, inner._stderr

        async def fake_exec(*args, **kwargs):
            return _SlowProc(stdout=b"ok")

        monkeypatch.setattr(engine_mod.asyncio, "create_subprocess_exec", fake_exec)
        monkeypatch.setattr(engine_mod.shutil, "which", lambda name: "/usr/bin/agy")
        monkeypatch.setattr(text_engine, "_ensure_agy_supported", lambda: None)
        monkeypatch.setattr(global_config, "AGY_PERSISTENT_WORKSPACES", True)
        monkeypatch.setattr(global_config, "AGY_WORKSPACE_MODE", "persona")
        monkeypatch.setattr(global_config, "AGY_WORKSPACES_DIR", tmp_path / "workspaces")

        await asyncio.gather(
            text_engine._run_agy_cli("a", persona_name="alice"),
            text_engine._run_agy_cli("b", persona_name="alice"),
        )
        assert max_in_flight == 1

    @pytest.mark.asyncio
    async def test_persistent_workspace_keeps_antigravitycli_state(self, text_engine, monkeypatch, tmp_path):
        """The symlink-target cleanup is temp-dir-only: persistent workspaces
        keep .antigravitycli state — that cache is the point of persistence."""
        captured_cwd = self._persistent_workspace_env(text_engine, monkeypatch, tmp_path)

        workspace = tmp_path / "workspaces" / "agy_alice"
        cli_dir = workspace / ".antigravitycli"
        cli_dir.mkdir(parents=True)
        cache_target = tmp_path / "cache_blob"
        cache_target.write_text("cached state")
        link = cli_dir / "cache_link"
        try:
            os.symlink(cache_target, link)
        except (OSError, NotImplementedError):
            pytest.skip("symlinks unavailable on this host")

        await text_engine._run_agy_cli("hi", persona_name="alice")

        assert captured_cwd == [os.path.abspath(workspace)]
        assert cache_target.exists()
        assert link.exists()


class TestClaudeCodeProvider:
    """DP-222: the `cc-*` Claude Code provider. Structural parity with agy
    (subprocess one-shot, POSIX-only-when-sandboxed, persistent per-persona
    workspace, dedicated limiter) but Claude Code runs its OWN sandboxed tools
    (`--dangerously-skip-permissions`), so the engine's `tools` arg is ignored
    and only the final text comes back."""

    class _FakeProc:
        def __init__(self, *, stdout=b"", stderr=b"", returncode=0, pid=4321):
            self._stdout = stdout
            self._stderr = stderr
            self.returncode = returncode
            self.pid = pid

        async def communicate(self):
            return self._stdout, self._stderr

    # --- model-name mapping -------------------------------------------------

    def test_cc_model_arg_strips_prefix(self, text_engine):
        assert text_engine._cc_model_arg("cc-sonnet") == "sonnet"
        assert text_engine._cc_model_arg("cc-opus") == "opus"
        assert text_engine._cc_model_arg("cc-haiku") == "haiku"

    def test_cc_model_arg_bare_prefix_defaults_sonnet(self, text_engine):
        assert text_engine._cc_model_arg("cc-") == "sonnet"

    def test_cc_prefix_not_classified_as_anthropic(self):
        """`cc-*` must not be captured by the substring `claude` check."""
        from src.utils.model_utils import get_model_prefix
        assert get_model_prefix("cc-sonnet") == "cc"
        assert get_model_prefix("claude-4-opus") == "claude"

    def test_cc_excluded_from_image_support(self, text_engine):
        assert text_engine.model_supports_images("cc-sonnet") is False

    def test_cc_limiter_constructed(self, text_engine):
        from aiolimiter import AsyncLimiter
        assert isinstance(text_engine._cc_limiter, AsyncLimiter)

    # --- routing ------------------------------------------------------------

    def test_route_resolves_to_cc_handler(self, text_engine, monkeypatch):
        monkeypatch.setattr(text_engine, "_ensure_cc_supported", lambda: None)
        handler, limiters = text_engine._get_provider_route("cc-sonnet")
        assert handler == text_engine._stream_cc_response
        assert limiters == [text_engine._cc_limiter]

    def test_route_cc_takes_precedence_over_anthropic(self, text_engine, monkeypatch):
        """A cc- model must route to Claude Code, never the Anthropic API."""
        monkeypatch.setattr(text_engine, "_ensure_cc_supported", lambda: None)
        handler, _ = text_engine._get_provider_route("cc-opus")
        assert handler == text_engine._stream_cc_response
        assert handler != text_engine._stream_anthropic_response

    def test_route_refuses_cc_on_windows_when_sandboxed(self, text_engine, monkeypatch):
        import src.engine as engine_mod
        from config import global_config
        monkeypatch.setattr(global_config, "CC_SANDBOX", True)
        monkeypatch.setattr(engine_mod.os, "name", "nt")
        with pytest.raises(LLMCommunicationError, match="native Windows"):
            text_engine._get_provider_route("cc-sonnet")

    def test_route_allows_cc_on_windows_when_unsandboxed(self, text_engine, monkeypatch):
        import src.engine as engine_mod
        from config import global_config
        monkeypatch.setattr(global_config, "CC_SANDBOX", False)
        monkeypatch.setattr(engine_mod.os, "name", "nt")
        handler, _ = text_engine._get_provider_route("cc-sonnet")
        assert handler == text_engine._stream_cc_response

    def test_ensure_cc_supported_ok_on_posix(self, text_engine, monkeypatch):
        import src.engine as engine_mod
        from config import global_config
        monkeypatch.setattr(global_config, "CC_SANDBOX", True)
        monkeypatch.setattr(engine_mod.os, "name", "posix")
        text_engine._ensure_cc_supported()  # no raise

    # --- argv construction --------------------------------------------------

    def test_build_cc_args_core_flags(self, text_engine, monkeypatch):
        from config import global_config
        monkeypatch.setattr(global_config, "CC_SANDBOX", True)
        monkeypatch.setattr(global_config, "CC_MAX_TURNS", 0)
        args = text_engine._build_cc_args("the prompt", "the persona", "sonnet")
        assert args[:2] == ["-p", "the prompt"]
        assert "--output-format" in args and "text" in args
        assert args[args.index("--model") + 1] == "sonnet"
        assert args[args.index("--system-prompt") + 1] == "the persona"
        assert "--dangerously-skip-permissions" in args
        # sandbox settings present and well-formed
        settings_raw = args[args.index("--settings") + 1]
        parsed = json.loads(settings_raw)
        assert parsed["sandbox"]["enabled"] is True
        assert parsed["sandbox"]["autoAllowBashIfSandboxed"] is True

    def test_build_cc_args_omits_system_prompt_when_empty(self, text_engine, monkeypatch):
        from config import global_config
        monkeypatch.setattr(global_config, "CC_SANDBOX", False)
        args = text_engine._build_cc_args("p", "", "sonnet")
        assert "--system-prompt" not in args

    def test_build_cc_args_no_settings_when_sandbox_off(self, text_engine, monkeypatch):
        from config import global_config
        monkeypatch.setattr(global_config, "CC_SANDBOX", False)
        args = text_engine._build_cc_args("p", "sys", "sonnet")
        assert "--settings" not in args

    def test_build_cc_args_no_yolo_when_sandbox_off(self, text_engine, monkeypatch):
        """Unsandboxed path must NEVER pass bare --dangerously-skip-permissions."""
        from config import global_config
        monkeypatch.setattr(global_config, "CC_SANDBOX", False)
        monkeypatch.setattr(global_config, "CC_ALLOWED_TOOLS", [])
        args = text_engine._build_cc_args("p", "sys", "sonnet")
        assert "--dangerously-skip-permissions" not in args
        assert "--allowedTools" not in args  # empty allowlist = default-deny

    def test_build_cc_args_allowlist_when_sandbox_off(self, text_engine, monkeypatch):
        """CC_ALLOWED_TOOLS feeds --allowedTools on the unsandboxed path (no yolo)."""
        from config import global_config
        monkeypatch.setattr(global_config, "CC_SANDBOX", False)
        monkeypatch.setattr(global_config, "CC_ALLOWED_TOOLS", ["Read", "Bash(npm run lint *)"])
        args = text_engine._build_cc_args("p", "sys", "sonnet")
        assert "--dangerously-skip-permissions" not in args
        idx = args.index("--allowedTools")
        assert args[idx + 1] == "Read"
        assert args[idx + 2] == "Bash(npm run lint *)"

    def test_build_cc_args_yolo_only_when_sandbox_on(self, text_engine, monkeypatch):
        """yolo is bounded by the OS sandbox; allowlist is ignored when sandboxed."""
        from config import global_config
        monkeypatch.setattr(global_config, "CC_SANDBOX", True)
        monkeypatch.setattr(global_config, "CC_ALLOWED_TOOLS", ["Read"])
        args = text_engine._build_cc_args("p", "sys", "sonnet")
        assert "--dangerously-skip-permissions" in args
        assert "--allowedTools" not in args

    def test_build_cc_args_max_turns(self, text_engine, monkeypatch):
        from config import global_config
        monkeypatch.setattr(global_config, "CC_SANDBOX", False)
        monkeypatch.setattr(global_config, "CC_MAX_TURNS", 4)
        args = text_engine._build_cc_args("p", "sys", "sonnet")
        assert args[args.index("--max-turns") + 1] == "4"

    def test_sandbox_settings_weaker_nested_and_domains(self, text_engine, monkeypatch):
        from config import global_config
        monkeypatch.setattr(global_config, "CC_SANDBOX", True)
        monkeypatch.setattr(global_config, "CC_SANDBOX_WEAKER_NESTED", True)
        monkeypatch.setattr(global_config, "CC_SANDBOX_ALLOWED_DOMAINS", ["github.com"])
        settings = text_engine._build_cc_sandbox_settings()
        assert settings["sandbox"]["enableWeakerNestedSandbox"] is True
        assert settings["sandbox"]["network"]["allowedDomains"] == ["github.com"]

    # --- handler ------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_handler_text_path(self, text_engine, base_context, monkeypatch):
        mock_cli = AsyncMock(return_value="  claude code answer  ")
        monkeypatch.setattr(text_engine, "_run_cc_cli", mock_cli)
        config = {"model_name": "cc-sonnet"}
        response, api_payload = await text_engine._generate_cc_response(config, base_context)
        assert response == {"type": "text", "content": "claude code answer"}
        assert api_payload["cc_model"] == "sonnet"

    @pytest.mark.asyncio
    async def test_handler_ignores_tools(self, text_engine, base_context, monkeypatch):
        """derpr tools are NOT advertised to Claude Code, and no <tool_call>
        protocol is injected into the prompt — CC uses its own tools."""
        mock_cli = AsyncMock(return_value="done")
        monkeypatch.setattr(text_engine, "_run_cc_cli", mock_cli)
        config = {"model_name": "cc-sonnet"}
        tools = [{"function": {"name": "get_weather", "description": "w",
                               "parameters": {"type": "object", "properties": {}}}}]
        response, api_payload = await text_engine._generate_cc_response(
            config, base_context, tools=tools
        )
        assert response["type"] == "text"
        assert api_payload["tools_ignored"] == ["get_weather"]
        # system prompt goes via the --system-prompt flag, not the -p prompt;
        # and the tool protocol is never rendered into the prompt.
        prompt_arg, system_arg = mock_cli.call_args[0][0], mock_cli.call_args[0][1]
        assert "<tool_call>" not in prompt_arg
        assert "get_weather" not in prompt_arg
        assert system_arg == "You are a test bot."

    @pytest.mark.asyncio
    async def test_handler_api_payload_has_no_secret(self, text_engine, base_context, monkeypatch):
        mock_cli = AsyncMock(return_value="answer")
        monkeypatch.setattr(text_engine, "_run_cc_cli", mock_cli)
        config = {"model_name": "cc-sonnet"}
        _, api_payload = await text_engine._generate_cc_response(config, base_context)
        payload_str = str(api_payload).lower()
        for forbidden in ["secret", "token", "oauth", "api_key", "password"]:
            assert forbidden not in payload_str

    @pytest.mark.asyncio
    async def test_generate_response_end_to_end_text(self, text_engine, base_context, monkeypatch):
        monkeypatch.setattr(text_engine, "_ensure_cc_supported", lambda: None)
        mock_cli = AsyncMock(return_value="end-to-end cc answer")
        monkeypatch.setattr(text_engine, "_run_cc_cli", mock_cli)
        config = {"model_name": "cc-sonnet"}
        response, _ = await text_engine.generate_response(config, base_context)
        assert response == {"type": "text", "content": "end-to-end cc answer"}

    # --- subprocess wiring + workspaces ------------------------------------

    @pytest.mark.asyncio
    async def test_run_cc_cli_spawns_claude_binary(self, text_engine, monkeypatch, tmp_path):
        import src.engine as engine_mod
        from config import global_config
        monkeypatch.setattr(global_config, "CC_SANDBOX", True)
        monkeypatch.setattr(global_config, "CC_WORKSPACE_DIR", None)
        monkeypatch.setattr(global_config, "CC_WORKSPACES_DIR", tmp_path / "workspaces")
        monkeypatch.setattr(global_config, "CC_WORKSPACE_MODE", "persona")

        captured = {}

        async def fake_exec(*args, **kwargs):
            captured["args"] = args
            captured["kwargs"] = kwargs
            return self._FakeProc(stdout=b"hi from claude")

        monkeypatch.setattr(engine_mod.asyncio, "create_subprocess_exec", fake_exec)
        monkeypatch.setattr(engine_mod.shutil, "which", lambda name: "/usr/bin/claude")
        monkeypatch.delenv("CLAUDE_CLI_PATH", raising=False)
        monkeypatch.setattr(text_engine, "_ensure_cc_supported", lambda: None)

        out = await text_engine._run_cc_cli("say hi", "be terse", "sonnet", persona_name="alice")

        assert out == "hi from claude"
        assert captured["args"][0] == "/usr/bin/claude"
        assert "--dangerously-skip-permissions" in captured["args"]
        assert captured["kwargs"].get("start_new_session") is True
        assert captured["kwargs"].get("cwd") == os.path.abspath(tmp_path / "workspaces" / "cc_alice")

    @pytest.mark.asyncio
    async def test_run_cc_cli_workspace_dir_override(self, text_engine, monkeypatch, tmp_path):
        import src.engine as engine_mod
        from config import global_config
        override = tmp_path / "derpr_checkout"
        override.mkdir()
        monkeypatch.setattr(global_config, "CC_WORKSPACE_DIR", str(override))

        captured_cwd = []

        async def fake_exec(*args, **kwargs):
            captured_cwd.append(kwargs.get("cwd"))
            return self._FakeProc(stdout=b"ok")

        monkeypatch.setattr(engine_mod.asyncio, "create_subprocess_exec", fake_exec)
        monkeypatch.setattr(engine_mod.shutil, "which", lambda name: "/usr/bin/claude")
        monkeypatch.setattr(text_engine, "_ensure_cc_supported", lambda: None)

        await text_engine._run_cc_cli("hi", "sys", "sonnet", persona_name="alice")
        # explicit override wins over per-persona dir
        assert captured_cwd == [os.path.abspath(override)]

    @pytest.mark.asyncio
    async def test_run_cc_cli_global_mode(self, text_engine, monkeypatch, tmp_path):
        import src.engine as engine_mod
        from config import global_config
        monkeypatch.setattr(global_config, "CC_WORKSPACE_DIR", None)
        monkeypatch.setattr(global_config, "CC_PERSISTENT_WORKSPACES", True)
        monkeypatch.setattr(global_config, "CC_WORKSPACE_MODE", "global")
        monkeypatch.setattr(global_config, "CC_WORKSPACES_DIR", tmp_path / "workspaces")

        captured_cwd = []

        async def fake_exec(*args, **kwargs):
            captured_cwd.append(kwargs.get("cwd"))
            return self._FakeProc(stdout=b"ok")

        monkeypatch.setattr(engine_mod.asyncio, "create_subprocess_exec", fake_exec)
        monkeypatch.setattr(engine_mod.shutil, "which", lambda name: "/usr/bin/claude")
        monkeypatch.setattr(text_engine, "_ensure_cc_supported", lambda: None)

        await text_engine._run_cc_cli("hi", "sys", "sonnet", persona_name="alice")
        assert captured_cwd == [os.path.abspath(tmp_path / "workspaces" / "cc_global")]

    @pytest.mark.asyncio
    async def test_run_cc_cli_stateless_fallback_cleans_temp(self, text_engine, monkeypatch, tmp_path):
        import src.engine as engine_mod
        from config import global_config
        monkeypatch.setattr(global_config, "CC_WORKSPACE_DIR", None)
        monkeypatch.setattr(global_config, "CC_PERSISTENT_WORKSPACES", False)

        captured_cwd = []

        async def fake_exec(*args, **kwargs):
            captured_cwd.append(kwargs.get("cwd"))
            return self._FakeProc(stdout=b"ok")

        monkeypatch.setattr(engine_mod.asyncio, "create_subprocess_exec", fake_exec)
        monkeypatch.setattr(engine_mod.shutil, "which", lambda name: "/usr/bin/claude")
        monkeypatch.setattr(text_engine, "_ensure_cc_supported", lambda: None)

        out = await text_engine._run_cc_cli("hi", "sys", "sonnet")
        assert out == "ok"
        assert len(captured_cwd) == 1
        assert not os.path.exists(captured_cwd[0])

    @pytest.mark.asyncio
    async def test_run_cc_cli_missing_binary_raises(self, text_engine, monkeypatch):
        import src.engine as engine_mod
        monkeypatch.setattr(text_engine, "_ensure_cc_supported", lambda: None)
        monkeypatch.setattr(engine_mod.shutil, "which", lambda name: None)
        monkeypatch.delenv("CLAUDE_CLI_PATH", raising=False)
        with pytest.raises(LLMCommunicationError, match="binary not found"):
            await text_engine._run_cc_cli("hi", "sys", "sonnet")

    # --- DP-227: per-call workspace override (fixr self-edit clone) ----------

    def test_resolve_cc_workspace_override_beats_workspace_dir(
        self, text_engine, monkeypatch, tmp_path
    ):
        """The per-call override (the fixr clone) wins over CC_WORKSPACE_DIR,
        which wins over the per-persona dir."""
        from config import global_config
        wsdir = tmp_path / "live_checkout"
        override = tmp_path / "fixr_clone"
        monkeypatch.setattr(global_config, "CC_WORKSPACE_DIR", str(wsdir))
        # override present -> override wins over CC_WORKSPACE_DIR
        assert text_engine._resolve_cc_workspace(
            "alice", str(override)
        ) == os.path.abspath(override)
        # no override -> CC_WORKSPACE_DIR wins (existing precedence preserved)
        assert text_engine._resolve_cc_workspace(
            "alice", None
        ) == os.path.abspath(wsdir)

    def test_resolve_cc_workspace_override_beats_persona_dir(
        self, text_engine, monkeypatch, tmp_path
    ):
        from config import global_config
        monkeypatch.setattr(global_config, "CC_WORKSPACE_DIR", None)
        monkeypatch.setattr(global_config, "CC_PERSISTENT_WORKSPACES", True)
        monkeypatch.setattr(global_config, "CC_WORKSPACE_MODE", "persona")
        monkeypatch.setattr(global_config, "CC_WORKSPACES_DIR", tmp_path / "workspaces")
        override = tmp_path / "fixr_clone"
        # override wins even over the per-persona dir
        assert text_engine._resolve_cc_workspace(
            "alice", str(override)
        ) == os.path.abspath(override)
        # without it, per-persona dir is used (unchanged behavior)
        assert text_engine._resolve_cc_workspace("alice", None) == os.path.abspath(
            tmp_path / "workspaces" / "cc_alice"
        )

    @pytest.mark.asyncio
    async def test_generate_cc_response_uses_config_override(
        self, text_engine, base_context, monkeypatch
    ):
        """`cc_workspace_override` in the engine config is threaded down to the
        CLI runner as the workspace."""
        monkeypatch.setattr(text_engine, "_ensure_cc_supported", lambda: None)
        captured = {}

        async def fake_run(prompt, system_prompt, model_arg, **kwargs):
            captured.update(kwargs)
            return "ok"

        monkeypatch.setattr(text_engine, "_run_cc_cli", fake_run)
        config = {
            "model_name": "cc-sonnet",
            "persona_name": "fixr",
            "cc_workspace_override": "/abs/fixr_clone",
        }
        result, _ = await text_engine._generate_cc_response(config, base_context)
        assert result == {"type": "text", "content": "ok"}
        assert captured.get("workspace_override") == "/abs/fixr_clone"
        assert captured.get("persona_name") == "fixr"
