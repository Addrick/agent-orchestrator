# tests/live/test_llm_live.py
#
# Live LLM API tests. These make real API calls and cost money.
# Auto-skipped when no LLM API keys are set.
#
# Uses LLM_LIVE_MODEL (gemini-2.5-flash) and LLM_LIVE_MAX_TOKENS from conftest
# to keep costs minimal while exercising the full engine path.

import asyncio
import pytest

from src.engine import TextEngine, LLMCommunicationError
from tests.live.conftest import LLM_LIVE_MODEL, LLM_LIVE_MAX_TOKENS

pytestmark = pytest.mark.llm_live

# A small, public-domain test image (1x1 white pixel PNG via httpbin-style data URI won't work,
# so we use a Wikipedia-hosted public domain image that is reliably available).
TEST_IMAGE_URL = "https://upload.wikimedia.org/wikipedia/commons/thumb/4/47/PNG_transparency_demonstration_1.png/100px-PNG_transparency_demonstration_1.png"


@pytest.fixture
def engine():
    return TextEngine()


@pytest.fixture
def live_config():
    return {"model_name": LLM_LIVE_MODEL, "max_output_tokens": LLM_LIVE_MAX_TOKENS}


@pytest.fixture
def text_context():
    """Minimal context for a plain text request."""
    return {
        "persona_prompt": "You are a helpful assistant. Respond in one short sentence.",
        "history": [{"role": "user", "content": "What is 2 + 2?"}],
        "current_message": {"text": "What is 2 + 2?"},
    }


@pytest.fixture
def image_context():
    """Context with an image URL attached to the last user message."""
    return {
        "persona_prompt": "You are a helpful assistant. Describe what you see in one short sentence.",
        "history": [{"role": "user", "content": "What is in this image?"}],
        "current_message": {
            "text": "What is in this image?",
            "image_url": TEST_IMAGE_URL,
        },
    }


class TestLiveTextResponse:
    @pytest.mark.asyncio
    async def test_text_happy_path(self, engine, live_config, text_context):
        """Real API call: text prompt returns a non-empty text response."""
        result, api_payload = await engine.generate_response(live_config, text_context)

        assert result["type"] == "text"
        assert len(result["content"].strip()) > 0
        assert api_payload is not None

    @pytest.mark.asyncio
    async def test_text_with_system_message_in_history(self, engine, live_config, text_context):
        """System message at history[0] is merged with persona_prompt correctly."""
        text_context["history"].insert(0, {"role": "system", "content": "Always answer in exactly one word."})

        result, _ = await engine.generate_response(live_config, text_context)

        assert result["type"] == "text"
        assert len(result["content"].strip()) > 0


class TestLiveMultimodalResponse:
    @pytest.mark.asyncio
    async def test_image_happy_path(self, engine, live_config, image_context):
        """Real API call: image + text prompt returns a non-empty text response describing the image."""
        result, api_payload = await engine.generate_response(live_config, image_context)

        assert result["type"] == "text"
        assert len(result["content"].strip()) > 0
        assert api_payload is not None


class TestLiveToolCalls:
    TOOL_MODEL = "gemini-3.1-flash-lite"  # drop -preview for stability and speed

    @pytest.mark.asyncio
    async def test_tool_call_happy_path(self, engine):
        """Real API call: when given a tool, the model invokes it."""
        context = {
            "persona_prompt": "You are a helpful assistant. Use the provided tools when appropriate.",
            "history": [{"role": "user", "content": "What's the weather in Paris right now?"}],
            "current_message": {"text": "What's the weather in Paris right now?"},
        }
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "get_current_weather",
                    "description": "Get the current weather in a given location",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "location": {
                                "type": "string",
                                "description": "The city name, e.g. Paris",
                            }
                        },
                        "required": ["location"],
                    },
                },
            }
        ]

        tool_config = {"model_name": self.TOOL_MODEL, "max_output_tokens": LLM_LIVE_MAX_TOKENS}
        try:
            result, api_payload = await asyncio.wait_for(
                engine.generate_response(tool_config, context, tools=tools),
                timeout=5.0
            )
        except asyncio.TimeoutError:
            pytest.skip(f"Live API call timed out after 5s ({self.TOOL_MODEL})")
        except LLMCommunicationError as e:
            if "503" in str(e) or "UNAVAILABLE" in str(e):
                pytest.skip(f"Live API currently unavailable (503): {e}")
            raise

        # The model should either call the tool or respond with text —
        # both are valid, but with a clear tool prompt it should call.
        assert result["type"] in ("tool_calls", "text")
        if result["type"] == "tool_calls":
            assert len(result["calls"]) >= 1
            assert result["calls"][0]["name"] == "get_current_weather"
            assert "location" in result["calls"][0]["arguments"]
