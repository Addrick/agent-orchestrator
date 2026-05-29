# tests/test_engine.py

import pytest
from unittest.mock import patch, AsyncMock, MagicMock
import base64
from openai import APIStatusError, APIConnectionError
import anthropic
import aiohttp
import json

from src.engine import TextEngine, LLMCommunicationError
from config.global_config import EMPTY_RESPONSE_RETRIES
from google.genai.types import Tool, GoogleSearch


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
    @patch('src.engine.TextEngine._generate_openai_response', new_callable=AsyncMock)
    async def test_retry_on_empty_response_succeeds(self, mock_provider_call, mock_sleep, text_engine, openai_config, base_context):
        mock_provider_call.side_effect = [
            ({}, {"payload": 1}),
            ({"type": "text", "content": "Valid response"}, {"payload": 2})
        ]
        response, _ = await text_engine.generate_response(openai_config, base_context)
        assert response == {"type": "text", "content": "Valid response"}
        assert mock_provider_call.call_count == 2

    @pytest.mark.asyncio
    @patch('src.engine.asyncio.sleep', new_callable=AsyncMock)
    @patch('src.engine.TextEngine._generate_openai_response', new_callable=AsyncMock)
    async def test_retry_on_empty_response_fails(self, mock_provider_call, mock_sleep, text_engine, openai_config, base_context):
        mock_provider_call.return_value = ({}, {"payload": 1})
        with pytest.raises(LLMCommunicationError, match="LLM provider returned an empty or invalid response after all retries."):
            await text_engine.generate_response(openai_config, base_context)
        assert mock_provider_call.call_count == EMPTY_RESPONSE_RETRIES + 1

    @pytest.mark.asyncio
    @patch('src.engine.TextEngine._generate_openai_response', new_callable=AsyncMock)
    async def test_no_retry_on_rate_limit_error(self, mock_provider_call, text_engine, openai_config, base_context):
        """429 errors must abort immediately without consuming retry budget."""
        mock_provider_call.side_effect = LLMCommunicationError("Rate limited", rate_limited=True)
        with pytest.raises(LLMCommunicationError) as exc_info:
            await text_engine.generate_response(openai_config, base_context)
        assert exc_info.value.rate_limited is True
        assert mock_provider_call.call_count == 1


@patch('src.engine.AsyncOpenAI')
class TestOpenAI:
    @pytest.mark.asyncio
    async def test_success_text_response(self, mock_openai_class, text_engine, openai_config, base_context, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "dummy_key_for_testing")
        mock_instance = mock_openai_class.return_value
        mock_instance.chat.completions.create = AsyncMock(
            return_value=MagicMock(choices=[MagicMock(message=MagicMock(content="Success", tool_calls=None))])
        )
        response, _ = await text_engine.generate_response(openai_config, base_context)
        assert response == {"type": "text", "content": "Success"}

    @pytest.mark.asyncio
    async def test_success_tool_call_response(self, mock_openai_class, text_engine, openai_config, base_context, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "dummy_key_for_testing")
        mock_instance = mock_openai_class.return_value
        mock_function = MagicMock()
        mock_function.name = "get_weather"
        mock_function.arguments = '{"location": "Boston"}'
        mock_tool_call = MagicMock(id="call_123", function=mock_function)
        mock_instance.chat.completions.create = AsyncMock(
            return_value=MagicMock(choices=[MagicMock(message=MagicMock(content=None, tool_calls=[mock_tool_call]))])
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
            return_value=MagicMock(choices=[MagicMock(message=MagicMock(content="ok", tool_calls=None))])
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


@patch('src.engine.anthropic.Anthropic')
class TestAnthropic:
    @pytest.mark.asyncio
    async def test_success_text_response(self, mock_anthropic_class, text_engine, anthropic_config, base_context, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "dummy_key_for_testing")
        mock_instance = mock_anthropic_class.return_value
        mock_instance.messages.create.return_value = MagicMock(
            content=[MagicMock(text="Claude success")], stop_reason="end_turn"
        )
        response, _ = await text_engine.generate_response(anthropic_config, base_context)
        assert response == {"type": "text", "content": "Claude success"}

    @pytest.mark.asyncio
    async def test_success_tool_call_response(self, mock_anthropic_class, text_engine, anthropic_config, base_context, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "dummy_key_for_testing")
        mock_instance = mock_anthropic_class.return_value
        mock_tool_use = MagicMock(type='tool_use', id='tool_123', input={'ticker': 'GOOG'})
        mock_tool_use.name = 'get_stock_price'
        mock_instance.messages.create.return_value = MagicMock(content=[mock_tool_use], stop_reason="tool_use")
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
        mock_instance.messages.create.side_effect = error
        with pytest.raises(LLMCommunicationError, match="Anthropic API returned an error"):
            await text_engine.generate_response(anthropic_config, base_context)

    @pytest.mark.asyncio
    async def test_429_sets_rate_limited_flag(self, mock_anthropic_class, text_engine, anthropic_config, base_context, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "dummy_key_for_testing")
        mock_instance = mock_anthropic_class.return_value
        error = anthropic.APIStatusError("Rate limit exceeded", response=MagicMock(status_code=429), body=None)
        mock_instance.messages.create.side_effect = error
        with pytest.raises(LLMCommunicationError) as exc_info:
            await text_engine.generate_response(anthropic_config, base_context)
        assert exc_info.value.rate_limited is True
        assert mock_instance.messages.create.call_count == 1

    @pytest.mark.asyncio
    async def test_tool_metadata_stripped_from_api_call(self, mock_anthropic_class, text_engine, anthropic_config,
                                                        base_context, monkeypatch):
        """Custom metadata fields must be stripped and tools converted to Anthropic format."""
        monkeypatch.setenv("ANTHROPIC_API_KEY", "dummy_key_for_testing")
        mock_instance = mock_anthropic_class.return_value
        mock_instance.messages.create.return_value = MagicMock(
            content=[MagicMock(text="ok")], stop_reason="end_turn"
        )
        tools = [{
            "type": "function", "is_write": True, "service_binding": "zammad",
            "function": {"name": "create_ticket", "description": "Creates a ticket",
                         "parameters": {"type": "object", "properties": {}}}
        }]
        await text_engine.generate_response(anthropic_config, base_context, tools=tools)
        call_kwargs = mock_instance.messages.create.call_args[1]
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
        mock_instance.messages.create.return_value = MagicMock(
            content=[MagicMock(text="Image received")], stop_reason="end_turn"
        )

        base_context["current_message"]["image_url"] = "http://example.com/image.png"
        base_context["history"] = [{"role": "user", "content": "Check this out"}]

        await text_engine.generate_response(anthropic_config, base_context)

        # Verify that the image was included in the API call
        call_args = mock_instance.messages.create.call_args[1]
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
        mock_instance.models.generate_content = AsyncMock(
            return_value=MagicMock(prompt_feedback=None, candidates=[mock_candidate])
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
        mock_instance.models.generate_content = AsyncMock(
            return_value=MagicMock(prompt_feedback=None, candidates=[mock_candidate])
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
        mock_instance.models.generate_content.side_effect = Exception("API failure")
        with pytest.raises(LLMCommunicationError, match="An error occurred with Google API"):
            await text_engine.generate_response(google_config, base_context)

    @pytest.mark.asyncio
    async def test_429_sets_rate_limited_flag(self, mock_google_client_class, text_engine, google_config, base_context, monkeypatch):
        monkeypatch.setenv("GOOGLE_GENERATIVEAI_API_KEY", "dummy_key_for_testing")
        mock_instance = mock_google_client_class.return_value
        mock_instance.models.generate_content.side_effect = Exception("429 quota exceeded")
        with pytest.raises(LLMCommunicationError) as exc_info:
            await text_engine.generate_response(google_config, base_context)
        assert exc_info.value.rate_limited is True
        assert mock_instance.models.generate_content.call_count == 1

    @pytest.mark.asyncio
    async def test_resource_exhausted_sets_rate_limited_flag(self, mock_google_client_class, text_engine, google_config, base_context, monkeypatch):
        monkeypatch.setenv("GOOGLE_GENERATIVEAI_API_KEY", "dummy_key_for_testing")
        mock_instance = mock_google_client_class.return_value
        mock_instance.models.generate_content.side_effect = Exception("RESOURCE_EXHAUSTED: daily limit reached")
        with pytest.raises(LLMCommunicationError) as exc_info:
            await text_engine.generate_response(google_config, base_context)
        assert exc_info.value.rate_limited is True
        assert mock_instance.models.generate_content.call_count == 1


    @pytest.mark.asyncio
    async def test_no_tools_passes_nothing_to_api(self, mock_google_client_class, text_engine, google_config, base_context, monkeypatch):
        """No tools enabled → no tools key sent to API (required for Gemma compatibility)."""
        monkeypatch.setenv("GOOGLE_GENERATIVEAI_API_KEY", "dummy_key_for_testing")
        mock_instance = mock_google_client_class.return_value
        mock_part = MagicMock(text="ok", function_call=None)
        mock_candidate = MagicMock(content=MagicMock(parts=[mock_part]), grounding_metadata=None)
        mock_instance.models.generate_content = AsyncMock(
            return_value=MagicMock(prompt_feedback=None, candidates=[mock_candidate])
        )
        await text_engine.generate_response(google_config, base_context, tools=[])
        config = mock_instance.models.generate_content.call_args.kwargs['config']
        assert not config.tools

    @pytest.mark.asyncio
    async def test_grounding_tool_injects_google_search(self, mock_google_client_class, text_engine, google_config, base_context, monkeypatch):
        """google_grounding_search in tools → GoogleSearch Tool injected into API config."""
        monkeypatch.setenv("GOOGLE_GENERATIVEAI_API_KEY", "dummy_key_for_testing")
        mock_instance = mock_google_client_class.return_value
        mock_part = MagicMock(text="ok", function_call=None)
        mock_candidate = MagicMock(content=MagicMock(parts=[mock_part]), grounding_metadata=None)
        mock_instance.models.generate_content = AsyncMock(
            return_value=MagicMock(prompt_feedback=None, candidates=[mock_candidate])
        )
        grounding_tools = [{"type": "google_grounding", "function": {"name": "google_grounding_search"}}]
        await text_engine.generate_response(google_config, base_context, tools=grounding_tools)
        config = mock_instance.models.generate_content.call_args.kwargs['config']
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
        mock_instance.models.generate_content = AsyncMock(
            return_value=MagicMock(prompt_feedback=None, candidates=[mock_candidate])
        )
        function_tools = [{"type": "function", "function": {"name": "do_thing", "description": "does a thing", "parameters": {"type": "object", "properties": {}}}}]
        await text_engine.generate_response(google_config, base_context, tools=function_tools)
        config = mock_instance.models.generate_content.call_args.kwargs['config']
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
        mock_instance.models.generate_content = AsyncMock(
            return_value=MagicMock(prompt_feedback=None, candidates=[mock_candidate])
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
        mock_instance.models.generate_content = AsyncMock(
            return_value=MagicMock(prompt_feedback=None, candidates=[mock_candidate])
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

        call_args = mock_instance.models.generate_content.call_args[1]
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
        mock_instance.models.generate_content = AsyncMock(
            return_value=MagicMock(prompt_feedback=None, candidates=[mock_candidate])
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
        mock_instance.models.generate_content = AsyncMock(
            return_value=MagicMock(prompt_feedback=None, candidates=[mock_candidate])
        )
        mixed_tools = [
            {"type": "google_grounding", "function": {"name": "google_grounding_search"}},
            {"type": "function", "function": {"name": "do_thing", "description": "does a thing", "parameters": {"type": "object", "properties": {}}}},
        ]
        await text_engine.generate_response(google_config, base_context, tools=mixed_tools)
        config = mock_instance.models.generate_content.call_args.kwargs['config']
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
        mock_instance.models.generate_content = AsyncMock(
            return_value=MagicMock(prompt_feedback=None, candidates=[mock_candidate])
        )

        base_context["current_message"]["image_url"] = "http://example.com/image.jpg"
        base_context["history"] = [{"role": "user", "content": "Check this out"}]

        await text_engine.generate_response(google_config, base_context)

        # Verify that the image was included in the API call
        call_args = mock_instance.models.generate_content.call_args[1]
        assert len(call_args['contents'][-1]['parts']) == 2
        assert call_args['contents'][-1]['parts'][-1].inline_data.data == b'imagedata'


class TestLocalModel:
    @pytest.mark.asyncio
    @patch('src.engine.AsyncOpenAI')
    async def test_success_text_response(self, mock_async_openai, text_engine, local_config, base_context):
        mock_client_instance = mock_async_openai.return_value
        mock_client_instance.chat.completions.create = AsyncMock(
            return_value=MagicMock(choices=[MagicMock(message=MagicMock(content="Local success", tool_calls=None))])
        )
        response, _ = await text_engine.generate_response(local_config, base_context)
        assert response == {"type": "text", "content": "Local success"}
        mock_client_instance.chat.completions.create.assert_awaited_once()

    @pytest.mark.asyncio
    @patch('src.engine.AsyncOpenAI')
    async def test_success_tool_call_response(self, mock_async_openai, text_engine, local_config, base_context):
        """
        Tests that a successful local model tool call is parsed correctly.
        """
        mock_client_instance = mock_async_openai.return_value

        # Mock the OpenAI-compatible response for a tool call
        mock_function = MagicMock()
        mock_function.name = "run_code"
        mock_function.arguments = '{"code": "print(\'hello from local\')"}'

        mock_tool_call = MagicMock(id="call_local_123", function=mock_function)
        mock_client_instance.chat.completions.create = AsyncMock(
            return_value=MagicMock(choices=[MagicMock(message=MagicMock(content=None, tool_calls=[mock_tool_call]))])
        )

        # Pass a non-empty 'tools' list to trigger the tool-call logic path
        response, _ = await text_engine.generate_response(local_config, base_context, tools=[
            {"type": "function", "function": {"name": "run_code"}}])

        assert response['type'] == 'tool_calls'
        assert len(response['calls']) == 1
        assert response['calls'][0]['name'] == 'run_code'
        assert response['calls'][0]['arguments'] == {'code': "print('hello from local')"}

    @pytest.mark.asyncio
    @patch('src.engine.AsyncOpenAI')
    async def test_connection_error_raises_llm_error(self, mock_async_openai, text_engine, local_config, base_context):
        mock_client_instance = mock_async_openai.return_value
        mock_client_instance.chat.completions.create.side_effect = APIConnectionError(request=MagicMock())
        with pytest.raises(LLMCommunicationError, match="Local API returned an error"):
            await text_engine.generate_response(local_config, base_context)


class TestProviderRouting:
    """Tests for _get_provider_route and model routing edge cases."""

    def test_unsupported_model_raises(self, text_engine):
        with pytest.raises(LLMCommunicationError, match="not supported"):
            text_engine._get_provider_route("unknown-model-v1")

    @pytest.mark.asyncio
    @patch('src.engine.TextEngine._generate_openai_response', new_callable=AsyncMock)
    async def test_image_unsupported_model_modifies_prompt(self, mock_provider, text_engine, base_context):
        """Models that don't support images get a system note appended and image_url cleared."""
        base_context["current_message"]["image_url"] = "http://example.com/photo.png"
        # gpt-3.5-turbo matches routing (starts with "gpt") but fails model_supports_images
        config = {"model_name": "gpt-3.5-turbo"}

        mock_provider.return_value = ({"type": "text", "content": "ok"}, {})
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
            return_value=MagicMock(choices=[MagicMock(message=MagicMock(content="I see the image", tool_calls=None))])
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

        good_fn = MagicMock()
        good_fn.name = "get_weather"
        good_fn.arguments = '{"city": "NYC"}'
        good_call = MagicMock(id="call_1", function=good_fn)

        bad_fn = MagicMock()
        bad_fn.name = "broken_tool"
        bad_fn.arguments = '{not valid json'
        bad_call = MagicMock(id="call_2", function=bad_fn)

        mock_instance.chat.completions.create = AsyncMock(
            return_value=MagicMock(choices=[MagicMock(message=MagicMock(content=None, tool_calls=[good_call, bad_call]))])
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

        mock_instance.models.generate_content = AsyncMock(
            return_value=MagicMock(prompt_feedback=mock_prompt_feedback, candidates=[])
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
        mock_instance.models.generate_content = AsyncMock(
            return_value=MagicMock(prompt_feedback=None, candidates=[])
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
        mock_instance.models.generate_content = AsyncMock(
            return_value=MagicMock(prompt_feedback=None, candidates=[mock_candidate])
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
        mock_instance.models.generate_content = AsyncMock(
            return_value=MagicMock(prompt_feedback=None, candidates=[mock_candidate])
        )

        base_context["current_message"]["image_url"] = "http://example.com/image.bmp"
        base_context["history"] = [{"role": "user", "content": "Check this BMP"}]

        await text_engine.generate_response(google_config, base_context)

        call_args = mock_instance.models.generate_content.call_args[1]
        user_turn = call_args['contents'][-1]
        assert len(user_turn['parts']) == 1  # text only, no image


@patch('src.engine.anthropic.Anthropic')
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
        mock_instance.messages.create.return_value = MagicMock(
            content=[MagicMock(text="Response without image")], stop_reason="end_turn"
        )

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
        mock_instance.messages.create.return_value = MagicMock(
            content=[MagicMock(text="No image seen")], stop_reason="end_turn"
        )

        base_context["current_message"]["image_url"] = "http://example.com/image.tiff"
        base_context["history"] = [{"role": "user", "content": "Check this TIFF"}]

        response, _ = await text_engine.generate_response(anthropic_config, base_context)
        assert response == {"type": "text", "content": "No image seen"}

        # Verify no image block was sent
        call_args = mock_instance.messages.create.call_args[1]
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
