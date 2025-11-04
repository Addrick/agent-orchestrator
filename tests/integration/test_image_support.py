# tests/integration/test_image_support.py

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import unittest
from unittest.mock import MagicMock, AsyncMock, patch

import pytest
from src.chat_system import ChatSystem, ResponseType
from src.engine import TextEngine
from src.persona import Persona, MemoryMode


@pytest.fixture
def mock_memory_manager():
    return MagicMock()


@pytest.fixture
def mock_zammad_client():
    return MagicMock()


@pytest.fixture
def mock_text_engine():
    engine = MagicMock(spec=TextEngine)
    engine.generate_response = AsyncMock(return_value=({"type": "text", "content": "Test response"}, {}))
    return engine


@pytest.fixture
def chat_system(mock_memory_manager, mock_text_engine, mock_zammad_client):
    return ChatSystem(mock_memory_manager, mock_text_engine, mock_zammad_client)


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

    # Since the logic is now in the engine, we need to use a real engine and mock its internal call
    real_engine = TextEngine()
    with patch.object(real_engine, '_generate_openai_response', new_callable=AsyncMock) as mock_openai_call, \
            patch.object(real_engine, 'model_supports_images', return_value=False):

        mock_openai_call.return_value = ({"type": "text", "content": "Test response"}, {})
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
