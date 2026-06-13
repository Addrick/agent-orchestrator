# tests/integration/test_image_support.py

from unittest.mock import MagicMock, AsyncMock, patch

import pytest

pytestmark = pytest.mark.integration
from tests.helpers import make_chat_system, route_stream_through_generate_response
from src.chat_system import ResponseType
from src.engine import TextEngine
from src.persona import Persona, MemoryMode


@pytest.fixture
def mock_memory_manager():
    return MagicMock()


@pytest.fixture
def mock_text_engine():
    """Real TextEngine with `generate_response` mocked. DP-206b: the test
    bridge routes the streaming pipeline back through the mock — same
    `(persona_config, history_object, ...)` shape, so contracts that read
    `call_args[0][1]` (history_object) are preserved."""
    engine = TextEngine()
    engine.generate_response = AsyncMock(  # type: ignore[method-assign]
        return_value=({"type": "text", "content": "Test response"}, {}),
    )
    route_stream_through_generate_response(engine)
    return engine


@pytest.fixture
def chat_system(mock_memory_manager, mock_text_engine):
    return make_chat_system(mock_memory_manager, mock_text_engine)


@pytest.mark.asyncio
async def test_image_url_passed_to_engine(chat_system, mock_text_engine):
    """
    Tests that the image URL is correctly passed to the TextEngine.
    """
    persona = Persona(
        persona_name="test_persona",
        model_name="gpt-4",
        prompt="You are a helpful assistant.",
        memory_mode=MemoryMode.PERSONAL
    )
    chat_system.personas["test_persona"] = persona

    with patch.object(mock_text_engine, 'model_supports_images', return_value=True):
        await chat_system.generate_response(
            persona_name="test_persona",
            user_identifier="user1",
            channel="test_channel",
            message="Check out this image!",
            image_url="http://example.com/image.png"
        )

        mock_text_engine.generate_response.assert_called_once()
        call_args = mock_text_engine.generate_response.call_args[0]
        context_object = call_args[1]
        assert context_object["current_message"]["image_url"] == "http://example.com/image.png"


@pytest.mark.asyncio
async def test_prompt_modified_for_unsupported_models(chat_system, mock_text_engine):
    """
    Tests that the persona prompt is modified when the model does not support images.
    """
    persona = Persona(
        persona_name="test_persona",
        model_name="gpt-3",
        prompt="You are a helpful assistant.",
        memory_mode=MemoryMode.PERSONAL
    )
    chat_system.personas["test_persona"] = persona

    # Since the logic is now in the engine, we need to use a real engine and
    # mock the canonical provider stream beneath the policy driver (DP-206b).
    from tests.helpers import engine_stream_events

    async def _fake_stream(*args, **kwargs):
        for ev in engine_stream_events({"type": "text", "content": "Test response"}, {}):
            yield ev

    real_engine = TextEngine()
    with patch.object(real_engine, '_stream_openai_response',
                      MagicMock(side_effect=_fake_stream)) as mock_openai_call, \
            patch.object(real_engine, 'model_supports_images', return_value=False):

        chat_system.text_engine = real_engine

        await chat_system.generate_response(
            persona_name="test_persona",
            user_identifier="user1",
            channel="test_channel",
            message="Check out this image!",
            image_url="http://example.com/image.png"
        )

    mock_openai_call.assert_called_once()
    call_args = mock_openai_call.call_args[0]
    context_object = call_args[1]
    assert "user has attached an image that you cannot see" in context_object["persona_prompt"]
    assert context_object["current_message"]["image_url"] is None
