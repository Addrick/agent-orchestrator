# tests/test_google_history_grouping.py
import json
import pytest
from unittest.mock import MagicMock, AsyncMock, patch
from src.engine import TextEngine

# We mock Part since in the real SDK it's a genai.types.Part object
# We just need to check that it is instantiated correctly.
@pytest.fixture(autouse=True)
def mock_google_part(monkeypatch):
    mock_part_cls = MagicMock()
    # Mock Part to return a mock object that records kwargs passed to it
    def side_effect(**kwargs):
        instance = MagicMock()
        for k, v in kwargs.items():
            setattr(instance, k, v)
        # Store kwargs on the instance so we can inspect them in assertions
        instance._kwargs = kwargs
        return instance
    mock_part_cls.side_effect = side_effect
    monkeypatch.setattr('src.engine.Part', mock_part_cls)
    return mock_part_cls

@pytest.mark.asyncio
async def test_consecutive_tool_turns_grouped():
    """
    Ensures that consecutive tool response turns are merged into a single turn
    with multiple parts when building history for the Google API.
    """
    engine = TextEngine()

    history = [
        {
            "role": "tool",
            "name": "tool_1",
            "content": '{"result": "r1"}'
        },
        {
            "role": "tool",
            "name": "tool_2",
            "content": '{"result": "r2"}'
        }
    ]

    with patch('src.engine.TextEngine._download_image', new_callable=AsyncMock):
        history_for_api, serializable_history = await engine._build_google_history(
            "system prompt", history, None
        )

    # They should be merged into a single turn
    assert len(history_for_api) == 1
    tool_turn = history_for_api[0]
    assert tool_turn['role'] == 'tool'
    assert len(tool_turn['parts']) == 2
    assert tool_turn['parts'][0].function_response['name'] == 'tool_1'
    assert tool_turn['parts'][1].function_response['name'] == 'tool_2'

    # Check serializable history as well (index 0 is system prompt)
    assert len(serializable_history) == 2
    assert serializable_history[0]['role'] == 'system'
    assert serializable_history[1]['role'] == 'tool'
    assert len(serializable_history[1]['parts']) == 2
    assert serializable_history[1]['parts'][0]['function_response']['name'] == 'tool_1'
    assert serializable_history[1]['parts'][1]['function_response']['name'] == 'tool_2'

@pytest.mark.asyncio
async def test_consecutive_user_turns_grouped():
    """
    Ensures consecutive user turns are merged.
    """
    engine = TextEngine()

    history = [
        {
            "role": "user",
            "content": "Message 1"
        },
        {
            "role": "user",
            "content": "Message 2"
        }
    ]

    with patch('src.engine.TextEngine._download_image', new_callable=AsyncMock):
        history_for_api, serializable_history = await engine._build_google_history(
            "system prompt", history, None
        )

    assert len(history_for_api) == 1
    user_turn = history_for_api[0]
    assert user_turn['role'] == 'user'
    assert len(user_turn['parts']) == 2
    assert user_turn['parts'][0].text == "Message 1"
    assert user_turn['parts'][1].text == "Message 2"

    assert len(serializable_history) == 2
    assert serializable_history[1]['role'] == 'user'
    assert len(serializable_history[1]['parts']) == 2
    assert serializable_history[1]['parts'][0]['text'] == "Message 1"
    assert serializable_history[1]['parts'][1]['text'] == "Message 2"

@pytest.mark.asyncio
async def test_consecutive_model_turns_grouped():
    """
    Ensures consecutive model (assistant) turns (e.g. text then tool calls) are merged.
    """
    engine = TextEngine()

    history = [
        {
            "role": "assistant",
            "content": "Thinking about it..."
        },
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": "c1",
                    "name": "web_search",
                    "arguments": {"query": "test"}
                }
            ]
        }
    ]

    with patch('src.engine.TextEngine._download_image', new_callable=AsyncMock):
        history_for_api, serializable_history = await engine._build_google_history(
            "system prompt", history, None
        )

    assert len(history_for_api) == 1
    model_turn = history_for_api[0]
    assert model_turn['role'] == 'model'
    assert len(model_turn['parts']) == 2
    assert model_turn['parts'][0].text == "Thinking about it..."
    assert model_turn['parts'][1].function_call['name'] == 'web_search'

    assert len(serializable_history) == 2
    assert serializable_history[1]['role'] == 'assistant'
    assert len(serializable_history[1]['parts']) == 2
    assert serializable_history[1]['parts'][0]['text'] == "Thinking about it..."
    assert serializable_history[1]['parts'][1]['function_call']['name'] == 'web_search'

@pytest.mark.asyncio
async def test_alternating_turns_preserved():
    """
    Ensures that alternating user -> model -> tool -> model turns are preserved.
    """
    engine = TextEngine()

    history = [
        {
            "role": "user",
            "content": "Hello"
        },
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": "c1",
                    "name": "web_search",
                    "arguments": {"query": "test"}
                }
            ]
        },
        {
            "role": "tool",
            "name": "web_search",
            "content": '{"result": "ok"}'
        },
        {
            "role": "assistant",
            "content": "Here is the result."
        }
    ]

    with patch('src.engine.TextEngine._download_image', new_callable=AsyncMock):
        history_for_api, serializable_history = await engine._build_google_history(
            "system prompt", history, None
        )

    assert len(history_for_api) == 4
    assert history_for_api[0]['role'] == 'user'
    assert history_for_api[1]['role'] == 'model'
    assert history_for_api[2]['role'] == 'tool'
    assert history_for_api[3]['role'] == 'model'

    assert len(serializable_history) == 5  # system prompt + 4 history
