# tests/test_engine.py

import logging
import os
import subprocess
import sys
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


@patch('src.engine.providers.openai.AsyncOpenAI')
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


@patch('src.engine.providers.openai.AsyncOpenAI')
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

    # ----------------------------------------------------------------
    # DP-338 -- batched blocks. hypr's prompt asks for independent reads
    # in one message; the parser kept block 1 and dropped the rest with
    # no execution, result or log line, so the model re-emitted the same
    # batch to chase answers it never got. The first block never changes,
    # so that is a fixed point: 15 identical `pve_status` calls, prod
    # 2026-08-21.
    # ----------------------------------------------------------------

    def test_parse_returns_every_block_not_just_the_first(self, text_engine):
        text = """Checking the node, the card and the unit list.
<tool_call>{"name": "pve_status", "arguments": {}}</tool_call>
<tool_call>{"name": "gpu_status", "arguments": {}}</tool_call>
<tool_call>{"name": "list_models", "arguments": {}}</tool_call>"""
        parsed = text_engine._parse_agy_tool_call(text)
        assert parsed is not None
        assert [c["name"] for c in parsed] == [
            "pve_status", "gpu_status", "list_models",
        ]
        # Distinct ids, or the loop pairs results to the wrong call.
        assert len({c["id"] for c in parsed}) == 3

    def test_parse_skips_a_malformed_block_among_good_ones(self, text_engine):
        """One bad block must not cost the calls the model got right --
        dropping the batch puts it straight back in the loop this ticket
        exists to remove. The 2nd block below is unparseable JSON and the
        3rd is missing `arguments`."""
        text = """<tool_call>{"name": "pve_status", "arguments": {}}</tool_call>
<tool_call>{"name": "gpu_status", "arguments": </tool_call>
<tool_call>{"name": "list_models"}</tool_call>
<tool_call>{"name": "hf_search", "arguments": {"q": "gguf"}}</tool_call>"""
        parsed = text_engine._parse_agy_tool_call(text)
        assert parsed is not None
        assert [c["name"] for c in parsed] == ["pve_status", "hf_search"]

    def test_parse_logs_every_block_it_drops(self, text_engine, caplog):
        """A skipped block must leave a trace. `strip_tool_call_blocks` removes
        the malformed block from the prose too, so without a log line the call
        vanishes from `calls`, from `content` and from the transcript at once
        -- the same total invisibility that let the prod spin run unnoticed."""
        text = """<tool_call>{"name": "pve_status", "arguments": {}}</tool_call>
<tool_call>{"name": "gpu_status", "arguments": </tool_call>
<tool_call>{"name": "list_models"}</tool_call>"""
        with caplog.at_level(logging.WARNING, logger="src.engine.providers.agy"):
            parsed = text_engine._parse_agy_tool_call(text)

        assert [c["name"] for c in parsed or []] == ["pve_status"]
        messages = [r.getMessage() for r in caplog.records]
        assert any("malformed <tool_call> block" in m for m in messages)
        assert any("missing 'name'/'arguments'" in m for m in messages)

    def test_parse_all_blocks_malformed_still_returns_none(self, text_engine):
        """The retry path keys off None; a response that made no USABLE call
        must stay indistinguishable from one that made no call at all."""
        text = """<tool_call>{"name": "a", "arguments": </tool_call>
<tool_call>not json at all</tool_call>"""
        assert text_engine._parse_agy_tool_call(text) is None

    def test_tool_protocol_no_longer_demands_exactly_one_block(
            self, text_engine):
        tools = [{"function": {"name": "pve_status", "description": "d",
                               "parameters": {}}}]
        protocol = text_engine._render_agy_tool_protocol(tools)
        assert "EXACTLY one block" not in protocol
        assert "one or more blocks" in protocol

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
    async def test_handler_returns_the_whole_batch(
            self, text_engine, base_context, monkeypatch):
        """One provider round trip, three calls -- what the loop then runs
        under `asyncio.gather`, and the only place batching's latency win
        can come from."""
        raw = """Checking the node, the card and the unit list.
<tool_call>{"name": "pve_status", "arguments": {}}</tool_call>
<tool_call>{"name": "gpu_status", "arguments": {}}</tool_call>
<tool_call>{"name": "list_models", "arguments": {}}</tool_call>"""
        monkeypatch.setattr(
            text_engine, "_run_agy_cli", AsyncMock(return_value=raw),
        )
        tools = [{"function": {"name": n, "description": "d",
                               "parameters": {}}}
                 for n in ("pve_status", "gpu_status", "list_models")]

        response, _ = await text_engine._generate_agy_response(
            {"model_name": "agy-flash"}, base_context, tools=tools,
        )

        assert response["type"] == "tool_calls"
        assert [c["name"] for c in response["calls"]] == [
            "pve_status", "gpu_status", "list_models",
        ]

    @pytest.mark.asyncio
    async def test_handler_carries_the_prose_beside_the_calls(
            self, text_engine, base_context, monkeypatch):
        """DP-338: the plan the model states before a batch rides back with
        it. Dropped, the next iteration re-read a transcript in which the
        calls had no stated reason -- so the model re-derived the plan off an
        identical history and re-emitted the same batch."""
        raw = """Checking the node and the card before proposing a swap.
<tool_call>{"name": "pve_status", "arguments": {}}</tool_call>
<tool_call>{"name": "gpu_status", "arguments": {}}</tool_call>"""
        monkeypatch.setattr(
            text_engine, "_run_agy_cli", AsyncMock(return_value=raw),
        )
        tools = [{"function": {"name": n, "description": "d",
                               "parameters": {}}}
                 for n in ("pve_status", "gpu_status")]

        response, _ = await text_engine._generate_agy_response(
            {"model_name": "agy-flash"}, base_context, tools=tools,
        )

        assert response["content"] == (
            "Checking the node and the card before proposing a swap."
        )
        # Blocks stripped: the markup is the calls, not the prose.
        assert "<tool_call>" not in response["content"]

    @pytest.mark.asyncio
    async def test_handler_call_only_response_omits_the_content_key(
            self, text_engine, base_context, monkeypatch):
        """No prose, no key. `collect_stream` omits it on a call-only
        response, so emitting an empty string here made the one-shot result
        something the event round trip could not reproduce."""
        raw = '<tool_call>{"name": "pve_status", "arguments": {}}</tool_call>'
        monkeypatch.setattr(
            text_engine, "_run_agy_cli", AsyncMock(return_value=raw),
        )
        tools = [{"function": {"name": "pve_status", "description": "d",
                               "parameters": {}}}]

        response, _ = await text_engine._generate_agy_response(
            {"model_name": "agy-flash"}, base_context, tools=tools,
        )

        assert response["type"] == "tool_calls"
        assert "content" not in response

    @pytest.mark.asyncio
    async def test_no_tools_means_no_tool_call_parsing(
            self, text_engine, base_context, monkeypatch):
        """A toolless call must come back as text, even if the reply contains
        a `<tool_call>` span (DP-335).

        The `if tools:` guards suppress *rendering* the protocol, not parsing
        it back, so this used to classify any reply containing the marker as
        `tool_calls` and discard the prose alongside it. The caller that hit
        this is the exhaustion wrap-up, whose prompt is a transcript of the
        turn's own tool calls plus a persona prompt naming tools by hand — so
        a `<tool_call>` reply is the likely output, and the whole DP-335
        feature silently degraded to its fallback on agy-flash, the provider
        whose measured turn motivated the ticket.
        """
        raw = (
            "Here is what I found.\n"
            '<tool_call>{"name": "get_weather", "arguments": {}}</tool_call>'
        )
        monkeypatch.setattr(
            text_engine, "_run_agy_cli", AsyncMock(return_value=raw),
        )

        response, _ = await text_engine._generate_agy_response(
            {"model_name": "agy-flash"}, base_context, tools=None,
        )

        assert response["type"] == "text"
        assert "Here is what I found." in response["content"]

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

    @pytest.mark.asyncio
    async def test_handler_clamps_oversized_prompt_to_argv_limit(
        self, text_engine, base_context, monkeypatch
    ):
        """DP-299: the prompt is one argv entry and execve caps a single argument
        at 128 KiB. An oversized history must be elided oldest-first, not handed
        to the spawn (which fails with OSError [Errno 7])."""
        from src.engine.providers import _subprocess

        mock_cli = AsyncMock(return_value="ok")
        monkeypatch.setattr(text_engine, "_run_agy_cli", mock_cli)

        ctx = dict(base_context)
        ctx["history"] = [
            {"role": "user", "content": "ancient question " + "a" * 80_000},
            {"role": "tool", "name": "read_file", "content": "b" * 80_000},
            {"role": "user", "content": "the latest question"},
        ]

        _, api_payload = await text_engine._generate_agy_response({"model_name": "agy-flash"}, ctx)

        prompt_arg = mock_cli.call_args[0][0]
        assert len(prompt_arg.encode("utf-8")) <= _subprocess.cli_prompt_budget()
        assert len(prompt_arg.encode("utf-8")) < _subprocess.MAX_ARG_STRLEN
        # System prompt and the newest turn survive; the oldest turn is elided.
        assert "You are a test bot." in prompt_arg
        assert "the latest question" in prompt_arg
        assert "ancient question" not in prompt_arg
        assert _subprocess.ELISION_NOTICE in prompt_arg
        assert api_payload["history_messages_elided"] >= 1

    @pytest.mark.asyncio
    async def test_handler_leaves_normal_prompt_untouched(
        self, text_engine, base_context, monkeypatch
    ):
        """The clamp must be inert for ordinary prompts (no elision marker), and
        must produce a prompt byte-identical to the pre-clamp construction."""
        from src.engine.providers import _subprocess

        mock_cli = AsyncMock(return_value="ok")
        monkeypatch.setattr(text_engine, "_run_agy_cli", mock_cli)

        ctx = dict(base_context)
        ctx["history"] = [{"role": "user", "content": "Hello"}]

        _, api_payload = await text_engine._generate_agy_response({"model_name": "agy-flash"}, ctx)

        prompt_arg = mock_cli.call_args[0][0]
        assert _subprocess.ELISION_NOTICE not in prompt_arg
        assert api_payload["history_messages_elided"] == 0
        assert prompt_arg == "You are a test bot.\n\nUser: Hello"

    def test_route_resolves_to_agy_handler(self, text_engine, monkeypatch):
        handler, limiters = text_engine._get_provider_route("agy-flash")
        assert handler == text_engine._stream_agy_response
        assert limiters == [text_engine._agy_limiter]

    def test_route_resolves_to_agy_handler_on_windows(self, text_engine, monkeypatch):
        """DP-324: agy is no longer refused on native Windows. agy >= 1.1.9
        writes its response to a pipe there, so route resolution must hand back
        the same handler it does on POSIX instead of raising."""
        import src.engine as engine_mod

        monkeypatch.setattr(engine_mod.os, "name", "nt")
        handler, limiters = text_engine._get_provider_route("agy-flash")
        assert handler == text_engine._stream_agy_response
        assert limiters == [text_engine._agy_limiter]

    @pytest.mark.asyncio
    async def test_generate_response_end_to_end_text(self, text_engine, base_context, monkeypatch):
        mock_cli = AsyncMock(return_value="end-to-end text answer")
        monkeypatch.setattr(text_engine, "_run_agy_cli", mock_cli)

        config = {"model_name": "agy-flash"}
        response, api_payload = await text_engine.generate_response(config, base_context)

        assert response == {"type": "text", "content": "end-to-end text answer"}
        assert isinstance(api_payload, dict)


class TestAgyCliInvocation:
    """Covers the real subprocess wiring of ``_run_agy_cli`` — the parts the
    mocked handler tests above skip.

    The route runs on every platform the `agy` CLI itself supports (DP-324
    dropped the POSIX-only guard: agy >= 1.1.9 writes its response to a pipe on
    native Windows too, which is what the guard existed for). Only the process
    isolation differs per platform. We deliberately do NOT pass
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
        # this test pins the POSIX spawn shape; the Windows shape has its own
        # test (test_run_agy_cli_spawns_on_windows), so force the branch rather
        # than letting it follow the host the suite happens to run on.
        monkeypatch.setattr(engine_mod.os, "name", "posix")

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

        await text_engine._run_agy_cli("hi", timeout=5)
        assert "--sandbox" not in captured["args"]

    @pytest.mark.asyncio
    async def test_run_agy_cli_wraps_e2big_with_payload_size(
        self, text_engine, monkeypatch, tmp_path
    ):
        """DP-299: a failed execve on an oversized argv/env must surface as an
        LLMCommunicationError, not a raw OSError that kills the turn — and must
        quote the payload size, which is what points at the byte budget."""
        import errno
        import src.engine as engine_mod
        from config import global_config

        monkeypatch.setattr(global_config, "AGY_WORKSPACES_DIR", tmp_path / "workspaces")

        async def fake_exec(*args, **kwargs):
            raise OSError(errno.E2BIG, "Argument list too long")

        monkeypatch.setattr(engine_mod.asyncio, "create_subprocess_exec", fake_exec)
        monkeypatch.setattr(engine_mod.shutil, "which", lambda name: "/usr/bin/agy")

        with pytest.raises(LLMCommunicationError, match="argv payload was"):
            await text_engine._run_agy_cli("hi", timeout=5)

    @pytest.mark.asyncio
    async def test_run_agy_cli_does_not_blame_size_for_other_spawn_errors(
        self, text_engine, monkeypatch, tmp_path
    ):
        """FileNotFoundError and PermissionError are OSError subclasses too. A
        blanket handler reports 'argv payload was N bytes' for a missing binary
        or an unreadable cwd, sending the investigation at the size limit — which
        is not the problem."""
        import src.engine as engine_mod
        from config import global_config

        monkeypatch.setattr(global_config, "AGY_WORKSPACES_DIR", tmp_path / "workspaces")

        async def fake_exec(*args, **kwargs):
            raise FileNotFoundError(2, "No such file or directory")

        monkeypatch.setattr(engine_mod.asyncio, "create_subprocess_exec", fake_exec)
        monkeypatch.setattr(engine_mod.shutil, "which", lambda name: "/usr/bin/agy")

        with pytest.raises(LLMCommunicationError) as exc:
            await text_engine._run_agy_cli("hi", timeout=5)

        assert "No such file or directory" in str(exc.value)
        assert "argv payload" not in str(exc.value)

    @pytest.mark.asyncio
    async def test_run_agy_cli_spawns_on_windows(self, text_engine, monkeypatch, tmp_path):
        """DP-324: the route must actually spawn on native Windows — the old
        guard aborted before the temp dir, so a regression that restores it
        would show up as no spawn at all rather than as a bad argv."""
        import src.engine as engine_mod
        from config import global_config

        monkeypatch.setattr(global_config, "AGY_WORKSPACES_DIR", tmp_path / "workspaces")
        monkeypatch.setattr(engine_mod.os, "name", "nt")
        captured = {}

        async def fake_exec(*args, **kwargs):
            captured["args"] = args
            captured["kwargs"] = kwargs
            return self._FakeProc(stdout=b"hello from agy on windows")

        monkeypatch.setattr(engine_mod.asyncio, "create_subprocess_exec", fake_exec)
        monkeypatch.setattr(engine_mod.shutil, "which", lambda name: r"C:\agy\agy.EXE")
        monkeypatch.delenv("ANTIGRAVITY_HARNESS_PATH", raising=False)

        out = await text_engine._run_agy_cli("say hi", timeout=5)

        assert out == "hello from agy on windows"
        assert captured["args"][0] == r"C:\agy\agy.EXE"
        # Windows has no setsid: isolation is a new process group instead, and
        # passing start_new_session there would be silently ignored.
        assert "start_new_session" not in captured["kwargs"]
        # getattr: the constant only exists in the Windows subprocess module, so
        # this assertion has to survive the suite running on the Linux CI host.
        assert captured["kwargs"]["creationflags"] == getattr(
            subprocess, "CREATE_NEW_PROCESS_GROUP", 0
        )

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


class TestFitCliPrompt:
    """DP-299: bounding the single argv entry the agy/cc CLIs take.

    The budget's unit is platform-dependent (DP-324): UTF-8 bytes on POSIX,
    escaped command-line characters on Windows. These cases assert the POSIX
    unit, so they pin `os.name` rather than inheriting whatever host the suite
    runs on; the Windows unit has its own cases in TestCliPlatformDifferences.
    """

    @pytest.fixture(autouse=True)
    def _posix_budget_unit(self, monkeypatch):
        from src.engine.providers import _subprocess
        monkeypatch.setattr(_subprocess.os, "name", "posix")

    @staticmethod
    def _blocks(*pairs):
        from src.engine.providers._subprocess import TranscriptBlock
        return [TranscriptBlock(role=r, text=t) for r, t in pairs]

    def test_under_budget_is_verbatim(self):
        from src.engine.providers._subprocess import fit_cli_prompt

        prompt, dropped = fit_cli_prompt(
            "SYS", self._blocks(("user", "User: hi"), ("assistant", "Assistant: yo")),
        )
        assert prompt == "SYS\n\nUser: hi\n\nAssistant: yo"
        assert dropped == 0

    def test_empty_history_matches_preamble_only(self):
        from src.engine.providers._subprocess import fit_cli_prompt

        prompt, dropped = fit_cli_prompt("SYS", [])
        assert prompt == "SYS"
        assert dropped == 0

    def test_drops_oldest_messages_first(self):
        from src.engine.providers._subprocess import ELISION_NOTICE, fit_cli_prompt

        blocks = self._blocks(*[("user", f"User: turn {i} " + "x" * 400) for i in range(10)])
        prompt, dropped = fit_cli_prompt("SYS", blocks, max_bytes=2000)
        assert len(prompt.encode("utf-8")) <= 2000
        assert dropped > 0
        assert ELISION_NOTICE in prompt
        assert "turn 9" in prompt and "turn 0" not in prompt

    def test_oversized_preamble_is_truncated(self):
        from src.engine.providers._subprocess import fit_cli_prompt

        prompt, _ = fit_cli_prompt("S" * 5000, self._blocks(("user", "User: hi")), max_bytes=1000)
        assert len(prompt.encode("utf-8")) <= 1000

    def test_multibyte_boundary_is_not_split(self):
        from src.engine.providers._subprocess import fit_cli_prompt

        prompt, _ = fit_cli_prompt("", self._blocks(("user", "User: " + "é" * 5000)), max_bytes=1000)
        assert len(prompt.encode("utf-8")) <= 1000
        prompt.encode("utf-8").decode("utf-8")  # round-trips: no split codepoint

    # --- the two policies the first cut got backwards ---

    def test_blank_lines_inside_a_message_do_not_split_it(self):
        """A message whose own content contains a blank line is ONE block. The
        earlier implementation recovered blocks by splitting the flat transcript
        on "\\n\\n", so a pretty-printed tool result became several 'blocks' and
        the cut landed mid-message — leaving an unlabeled fragment of raw tool
        output at the head of the prompt, read by the model as conversation."""
        from src.engine.providers._subprocess import fit_cli_prompt

        multi = "Tool(read_file): {\n\n  \"a\": 1,\n\n  \"b\": 2\n\n}"
        blocks = self._blocks(("user", "User: ancient " + "x" * 3000), ("tool", multi))
        prompt, dropped = fit_cli_prompt("", blocks, max_bytes=1200)

        assert dropped == 1                      # the old user turn, not a fragment
        assert multi in prompt                   # the tool result survives WHOLE
        assert "ancient" not in prompt

    def test_degenerate_case_keeps_the_question_not_the_tool_blob(self):
        """When nothing fits, prefer the newest USER block. In the tool loop the
        final block is the tool result; keeping it and eliding the question the
        model is supposed to answer leaves an unattributed blob and no task."""
        from src.engine.providers._subprocess import TRUNCATION_NOTICE, fit_cli_prompt

        blocks = self._blocks(
            ("user", "User: what is in the config file?"),
            ("assistant", "Assistant (tool call read_file): {}"),
            ("tool", "Tool(read_file): " + "z" * 5000),
        )
        prompt, _ = fit_cli_prompt("SYS", blocks, max_bytes=400)

        assert len(prompt.encode("utf-8")) <= 400
        assert "what is in the config file?" in prompt
        assert "zzz" not in prompt
        assert "SYS" in prompt

    def test_degenerate_case_keeps_role_label_when_truncating(self):
        """Truncation keeps the block's HEAD, so the role label survives. Cutting
        from the tail strips `Tool(name):` and presents raw tool output as if it
        were conversation."""
        from src.engine.providers._subprocess import TRUNCATION_NOTICE, fit_cli_prompt

        blocks = self._blocks(("tool", "Tool(read_file): " + "z" * 5000))
        prompt, _ = fit_cli_prompt("", blocks, max_bytes=300)

        assert len(prompt.encode("utf-8")) <= 300
        assert prompt.startswith("[...older conversation elided")
        assert "Tool(read_file):" in prompt      # label intact
        assert TRUNCATION_NOTICE in prompt

    def test_elided_count_is_messages_not_fragments(self):
        from src.engine.providers._subprocess import fit_cli_prompt

        blocks = self._blocks(
            ("user", "User: a\n\nsecond paragraph\n\nthird " + "x" * 2000),
            ("user", "User: keep me"),
        )
        _, dropped = fit_cli_prompt("", blocks, max_bytes=500)
        assert dropped == 1  # one message, though it renders as 3 blank-line groups


class TestRenderTranscriptBlocks:
    """The block renderer is the single source of truth; the flat transcript is
    its join. DP-299 — they must not be able to drift."""

    def test_flat_transcript_is_the_join_of_blocks(self):
        from src.engine.providers._subprocess import (
            render_transcript, render_transcript_blocks,
        )

        history = [
            {"role": "user", "content": "multi\n\nparagraph"},
            {"role": "assistant", "content": "text", "tool_calls": [
                {"id": "c1", "name": "t", "arguments": {"k": "v"}},
            ]},
            {"role": "tool", "name": "t", "content": "{}"},
        ]
        blocks = render_transcript_blocks(history)
        assert "\n\n".join(b.text for b in blocks) == render_transcript(history)

    def test_one_block_per_message_even_with_tool_calls(self):
        from src.engine.providers._subprocess import render_transcript_blocks

        history = [
            {"role": "assistant", "content": "text", "tool_calls": [
                {"id": "c1", "name": "t", "arguments": {}},
            ]},
        ]
        blocks = render_transcript_blocks(history)
        assert len(blocks) == 1                       # content + call = ONE message
        assert blocks[0].role == "assistant"
        assert "Assistant: text" in blocks[0].text
        assert "Assistant (tool call t)" in blocks[0].text

    def test_empty_assistant_turn_contributes_no_block(self):
        from src.engine.providers._subprocess import render_transcript_blocks

        assert render_transcript_blocks([{"role": "assistant", "content": ""}]) == []


class TestClampCliArg:
    """DP-299: `--system-prompt` is a second unbounded argv entry on the cc route."""

    @pytest.fixture(autouse=True)
    def _posix_budget_unit(self, monkeypatch):
        # See TestFitCliPrompt: `max_bytes` is only literally bytes on POSIX.
        from src.engine.providers import _subprocess
        monkeypatch.setattr(_subprocess.os, "name", "posix")

    def test_under_budget_is_verbatim(self):
        from src.engine.providers._subprocess import clamp_cli_arg

        assert clamp_cli_arg("short") == "short"

    def test_oversized_is_clamped(self):
        from src.engine.providers._subprocess import TRUNCATION_NOTICE, clamp_cli_arg

        out = clamp_cli_arg("S" * 5000, max_bytes=1000)
        assert len(out.encode("utf-8")) <= 1000
        assert TRUNCATION_NOTICE in out


class TestCliPlatformDifferences:
    """DP-324: the agy/cc subprocess runner is platform-agnostic, but the two
    platforms disagree about the command-line ceiling and about how you kill a
    process and its children. Each branch is pinned here so the suite asserts
    both regardless of the host it runs on."""

    # --- what an argv entry costs ------------------------------------------

    @staticmethod
    def _quote_dense(n):
        """n characters of the quote-dense text these routes actually carry: a
        rendered JSON tool result. `list2cmdline` escapes every `"`."""
        return ('{"ticket":"12345","state":"open"}, ' * (n // 34 + 1))[:n]

    def test_prompt_budget_fits_windows_command_line(self, monkeypatch):
        """Windows caps the WHOLE command line at 32767 chars — a shared pool.
        The prompt and the cc route's `--system-prompt` are the two large
        entries, so together they must still leave room for the flags.

        Measured on the ESCAPED command line, not on raw byte counts: Windows
        applies the cap to what `list2cmdline` produces, and asserting the raw
        sum passes happily while the real spawn dies with WinError 206.
        """
        from src.engine.providers import _subprocess

        monkeypatch.setattr(_subprocess.os, "name", "nt")
        prompt = _subprocess.fit_cli_prompt(
            "", [_subprocess.TranscriptBlock(
                role="tool", text=self._quote_dense(200_000))],
        )[0]
        system_prompt = _subprocess.clamp_cli_arg(self._quote_dense(60_000))
        argv = [r"C:\agy\agy.EXE", "--sandbox", "--print-timeout", "150s",
                "-p", prompt, "--system-prompt", system_prompt,
                "--model", "sonnet", "--output-format", "text"]
        assert len(subprocess.list2cmdline(argv)) < _subprocess.MAX_COMMAND_LINE_CHARS

    def test_cmdline_cost_is_bytes_on_posix_escaped_chars_on_windows(self, monkeypatch):
        """The budget unit differs because the OS contract does. POSIX passes an
        argv vector and caps a single entry in bytes; Windows passes ONE string
        built by `list2cmdline` and caps that, so embedded quotes cost double."""
        from src.engine.providers import _subprocess

        text = '{"a":"b"}'                      # 9 chars, 9 bytes, 4 quotes
        monkeypatch.setattr(_subprocess.os, "name", "posix")
        assert _subprocess._cmdline_cost(text) == 9
        monkeypatch.setattr(_subprocess.os, "name", "nt")
        assert _subprocess._cmdline_cost(text) == len(
            subprocess.list2cmdline([text])) == 13

    def test_windows_trim_accounts_for_escaping(self, monkeypatch):
        """Regression (DP-324 review): the budget was counted in raw UTF-8 bytes
        while CreateProcess measures the escaped command line. Quote-dense
        history passed the byte check and then failed the spawn — the exact
        failure fit_cli_prompt exists to prevent, and unfixable by trimming
        because the trimmer already believed it fit."""
        from src.engine.providers import _subprocess

        monkeypatch.setattr(_subprocess.os, "name", "nt")
        budget = _subprocess.cli_prompt_budget()
        blocks = [_subprocess.TranscriptBlock(
            role="tool", text="Tool(zammad): " + self._quote_dense(80_000))]

        prompt, dropped = _subprocess.fit_cli_prompt("SYS", blocks)

        assert len(subprocess.list2cmdline([prompt])) <= budget
        # ...and the raw text is meaningfully SHORTER than the budget, which is
        # what a byte-counting implementation would have handed back instead.
        assert len(prompt.encode("utf-8")) < budget

    def test_windows_clamp_cli_arg_accounts_for_escaping(self, monkeypatch):
        from src.engine.providers import _subprocess

        monkeypatch.setattr(_subprocess.os, "name", "nt")
        out = _subprocess.clamp_cli_arg(self._quote_dense(60_000))
        assert len(subprocess.list2cmdline([out])) <= _subprocess.cli_arg_budget()

    def test_truncate_block_never_exceeds_its_budget_on_either_platform(self, monkeypatch):
        """_truncate_block appends its marker AFTER cutting, so the split cost
        accounting has to stay conservative — on Windows the quoting is applied
        once to the joined string, not to each half."""
        from src.engine.providers import _subprocess

        for name in ("posix", "nt"):
            monkeypatch.setattr(_subprocess.os, "name", name)
            for text in (self._quote_dense(4000), '"' * 4000, "plain " * 700):
                for budget in (120, 300, 1000, 2500):
                    out = _subprocess._truncate_block(text, budget)
                    assert _subprocess._cmdline_cost(out) <= budget, (name, budget)

    def test_posix_budget_stays_under_per_arg_cap(self, monkeypatch):
        from src.engine.providers import _subprocess

        monkeypatch.setattr(_subprocess.os, "name", "posix")
        assert _subprocess.cli_prompt_budget() < _subprocess.MAX_ARG_STRLEN
        assert _subprocess.cli_arg_budget() < _subprocess.MAX_ARG_STRLEN

    def test_isolation_kwargs_posix(self, monkeypatch):
        from src.engine.providers import _subprocess

        monkeypatch.setattr(_subprocess.os, "name", "posix")
        assert _subprocess._spawn_isolation_kwargs() == {"start_new_session": True}

    def test_isolation_kwargs_windows(self, monkeypatch):
        """`start_new_session` is silently IGNORED by the Windows implementation,
        so passing it there would look like isolation while providing none."""
        from src.engine.providers import _subprocess

        monkeypatch.setattr(_subprocess.os, "name", "nt")
        kwargs = _subprocess._spawn_isolation_kwargs()
        assert "start_new_session" not in kwargs
        assert kwargs["creationflags"] == getattr(
            subprocess, "CREATE_NEW_PROCESS_GROUP", 0
        )

    def test_kill_process_tree_posix_kills_session(self, monkeypatch):
        from src.engine.providers import _subprocess

        import signal

        monkeypatch.setattr(_subprocess.os, "name", "posix")
        killed = []
        # SIGKILL/getpgid/killpg do not exist on a Windows dev host; supply them
        # so the POSIX branch is exercised wherever the suite runs.
        monkeypatch.setattr(signal, "SIGKILL", getattr(signal, "SIGKILL", 9), raising=False)
        monkeypatch.setattr(_subprocess.os, "getpgid", lambda pid: pid, raising=False)
        monkeypatch.setattr(
            _subprocess.os, "killpg", lambda pgid, sig: killed.append((pgid, sig)),
            raising=False,
        )

        proc = MagicMock(pid=4321, returncode=0)
        _subprocess._kill_process_tree(proc)

        assert killed and killed[0][0] == 4321

    def test_kill_process_tree_windows_falls_back_to_taskkill_without_a_job(
            self, monkeypatch):
        from src.engine.providers import _subprocess

        monkeypatch.setattr(_subprocess.os, "name", "nt")
        calls = []
        monkeypatch.setattr(
            _subprocess.subprocess, "Popen", lambda cmd, **kw: calls.append(cmd)
        )

        proc = MagicMock(pid=4321, returncode=None)  # still running
        _subprocess._kill_process_tree(proc, job=None)

        assert calls == [["taskkill", "/F", "/T", "/PID", "4321"]]

    def test_kill_process_tree_windows_does_not_block_the_event_loop(self, monkeypatch):
        """The fallback runs in the `finally` of an async call. `subprocess.run`
        would park the one event loop that also serves Discord, the portal and
        every other provider for up to its timeout; spawn and let go instead."""
        from src.engine.providers import _subprocess

        monkeypatch.setattr(_subprocess.os, "name", "nt")
        monkeypatch.setattr(_subprocess.subprocess, "Popen", lambda cmd, **kw: None)
        monkeypatch.setattr(
            _subprocess.subprocess, "run",
            lambda *a, **k: pytest.fail("teardown must not wait on taskkill"),
        )

        _subprocess._kill_process_tree(MagicMock(pid=4321, returncode=None), job=None)

    def test_kill_process_tree_windows_skips_exited_process_without_a_job(
            self, monkeypatch):
        """`taskkill /T` walks the tree from a LIVE parent pid; after exit the
        link is gone, so calling it would only spawn a doomed taskkill per turn.
        This is the degraded path — with a job the helpers still get killed."""
        from src.engine.providers import _subprocess

        monkeypatch.setattr(_subprocess.os, "name", "nt")
        calls = []
        monkeypatch.setattr(
            _subprocess.subprocess, "Popen", lambda cmd, **kw: calls.append(cmd)
        )

        proc = MagicMock(pid=4321, returncode=0)  # already exited
        _subprocess._kill_process_tree(proc, job=None)

        assert calls == []

    # --- the Windows analogue of setsid + killpg ---------------------------

    def test_adopt_process_tree_is_a_noop_on_posix(self, monkeypatch):
        """setsid at spawn already made the descendants reachable by killpg."""
        from src.engine.providers import _subprocess

        monkeypatch.setattr(_subprocess.os, "name", "posix")
        assert _subprocess._adopt_process_tree(MagicMock(pid=4321)) is None

    def test_kill_process_tree_windows_closes_the_job_and_skips_taskkill(
            self, monkeypatch):
        """Closing a KILL_ON_JOB_CLOSE job kills every process still inside it,
        which is what reaches the helpers a cleanly-exited CLI orphaned. taskkill
        cannot: verified on Windows, a grandchild outlives its parent and the
        walk then reports "process not found"."""
        from src.engine.providers import _subprocess

        monkeypatch.setattr(_subprocess.os, "name", "nt")
        closed, spawned = [], []
        monkeypatch.setattr(_subprocess, "_win_close_job", closed.append)
        monkeypatch.setattr(
            _subprocess.subprocess, "Popen", lambda cmd, **kw: spawned.append(cmd)
        )

        # returncode 0 = the clean-exit path the old code skipped entirely
        _subprocess._kill_process_tree(MagicMock(pid=4321, returncode=0), job=0xABCD)

        assert closed == [0xABCD]
        assert spawned == []

    @pytest.mark.asyncio
    async def test_exec_cli_adopts_then_releases_the_tree(self, monkeypatch):
        """The job has to be claimed right after the spawn and closed in the
        `finally`, or the whole mechanism is inert."""
        from src.engine.providers import _subprocess

        monkeypatch.setattr(_subprocess.os, "name", "nt")
        order = []

        class _Proc:
            pid, returncode = 4321, 0

            async def communicate(self):
                order.append("ran")
                return b"out", b""

        async def fake_exec(*a, **k):
            order.append("spawn")
            return _Proc()

        monkeypatch.setattr(_subprocess.asyncio, "create_subprocess_exec", fake_exec)
        monkeypatch.setattr(_subprocess, "_adopt_process_tree",
                            lambda proc: order.append("adopt") or 0xABCD)
        monkeypatch.setattr(_subprocess, "_win_close_job",
                            lambda job: order.append(f"close:{job:#x}"))

        assert await _subprocess.exec_cli("agy.EXE", ["-p", "hi"], ".", 5) == "out"
        assert order == ["spawn", "adopt", "ran", "close:0xabcd"]

    @pytest.mark.skipif(os.name != "nt", reason="job objects are a Windows API")
    def test_job_object_really_kills_an_orphaned_grandchild(self):
        """The load-bearing claim, against the real kernel: a helper whose parent
        already exited is still killed when the job handle closes."""
        import time
        from src.engine.providers import _subprocess

        parent = subprocess.Popen(
            [sys.executable, "-c",
             "import subprocess,sys;"
             "p=subprocess.Popen([sys.executable,'-c','import time;time.sleep(60)']);"
             "print(p.pid,flush=True)"],
            stdout=subprocess.PIPE, text=True,
        )
        job = _subprocess._adopt_process_tree(parent)
        assert job, "could not create the job object"
        grandchild = int(parent.stdout.readline().strip())
        parent.wait(timeout=15)          # the parent exits; the helper does not

        def alive():
            out = subprocess.run(["tasklist", "/FI", f"PID eq {grandchild}", "/NH"],
                                 capture_output=True, text=True).stdout
            return str(grandchild) in out

        try:
            assert alive(), "grandchild should outlive its parent (the whole point)"
            _subprocess._kill_process_tree(parent, job)
            deadline = time.monotonic() + 10
            while alive() and time.monotonic() < deadline:
                time.sleep(0.2)
            assert not alive(), "closing the job did not kill the orphaned helper"
        finally:
            subprocess.run(["taskkill", "/F", "/PID", str(grandchild)],
                           capture_output=True)

    @pytest.mark.asyncio
    async def test_windows_oversized_command_line_reports_payload_size(self, monkeypatch):
        """Windows signals a too-long command line as WinError 206 raised as a
        FileNotFoundError — matching on errno alone reads that as 'binary not
        found' and sends the investigation the wrong way."""
        from src.engine.providers import _subprocess

        # `winerror` is a read-only getset on real Windows OSErrors, so the
        # 206 is supplied by a subclass attribute — that keeps the test
        # constructible on the Linux CI host too.
        class _TooLongCommandLine(FileNotFoundError):
            winerror = 206

        err = _TooLongCommandLine(2, "The filename or extension is too long")

        async def fake_exec(*a, **k):
            raise err

        monkeypatch.setattr(_subprocess.asyncio, "create_subprocess_exec", fake_exec)

        with pytest.raises(LLMCommunicationError) as exc:
            await _subprocess.exec_cli("agy.EXE", ["-p", "hi"], ".", 5)

        assert "argv payload was" in str(exc.value)


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

    @pytest.mark.asyncio
    async def test_run_cc_cli_strips_api_key_from_env(self, text_engine, monkeypatch):
        """DP-232: cc-* must run on the Claude subscription, not the metered API —
        _run_cc_cli hands _exec_agy an env with ANTHROPIC_API_KEY removed."""
        from config import global_config
        monkeypatch.setattr(global_config, "CC_USE_SUBSCRIPTION", True)
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-secret")
        monkeypatch.setenv("CLAUDE_CODE_OAUTH_TOKEN", "oauth-keep")
        monkeypatch.setenv("CLAUDE_CLI_PATH", "claude")  # skip shutil.which
        monkeypatch.setattr(text_engine, "_ensure_cc_supported", lambda: None)
        monkeypatch.setattr(text_engine, "_resolve_cc_workspace", lambda *a, **k: None)

        captured = {}

        async def fake_exec(binary, args, workspace_dir, timeout, label="agy", env=None):
            captured["env"] = env
            return "ok"

        monkeypatch.setattr(text_engine, "_exec_agy", fake_exec)

        out = await text_engine._run_cc_cli("p", "s", "sonnet")
        assert out == "ok"
        env = captured["env"]
        assert env is not None and "ANTHROPIC_API_KEY" not in env
        assert env["CLAUDE_CODE_OAUTH_TOKEN"] == "oauth-keep"

    @pytest.mark.asyncio
    async def test_notes_are_prepared_before_argv_is_built(self, text_engine, monkeypatch, tmp_path):
        """DP-314 ordering guard. The sandbox `allowWrite` entry is derived from
        whether the notes clone EXISTS, so building argv before preparing notes
        ships an empty allowlist on the first call — the agent then gets a
        `memory/` it cannot write, failing with a bare EACCES. Order is load-
        bearing, so pin it."""
        import src.engine as engine_mod
        import src.engine.providers.cc as cc_mod
        from config import global_config
        monkeypatch.setattr(global_config, "CC_SANDBOX", True)
        monkeypatch.setattr(global_config, "CC_WORKSPACE_DIR", None)
        monkeypatch.setattr(global_config, "CC_WORKSPACES_DIR", tmp_path / "workspaces")
        monkeypatch.setattr(global_config, "CC_WORKSPACE_MODE", "persona")

        order = []
        monkeypatch.setattr(
            cc_mod, "prepare_workspace_notes",
            lambda ws, *a, **k: order.append("notes"),
        )
        real_build = text_engine._build_cc_args
        monkeypatch.setattr(
            text_engine, "_build_cc_args",
            lambda *a, **k: (order.append("argv"), real_build(*a, **k))[1],
        )

        async def fake_exec(*args, **kwargs):
            return self._FakeProc(stdout=b"ok")

        monkeypatch.setattr(engine_mod.asyncio, "create_subprocess_exec", fake_exec)
        monkeypatch.setattr(engine_mod.shutil, "which", lambda name: "/usr/bin/claude")
        monkeypatch.setattr(text_engine, "_ensure_cc_supported", lambda: None)

        await text_engine._run_cc_cli("hi", "sys", "sonnet", persona_name="alice")

        assert order == ["notes", "argv"]

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
        # cc sandboxed is POSIX-only; pin the branch so the shared runner's
        # Windows spawn shape can't make this assertion host-dependent.
        monkeypatch.setattr(engine_mod.os, "name", "posix")

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
