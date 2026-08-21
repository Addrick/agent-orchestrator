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
    monkeypatch.setattr('src.engine.providers.google.Part', mock_part_cls)
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


@pytest.mark.asyncio
async def test_prose_beside_tool_calls_reaches_the_wire():
    """DP-338: the plan the model wrote for its batch must survive the trip to
    Gemini. The builder read `tool_calls` and never looked at `content`, so the
    prose was stored in conversation_history and stripped again on the way
    out — the next iteration read the same reason-free transcript the ticket
    exists to remove."""
    engine = TextEngine()

    history = [
        {
            "role": "assistant",
            "content": "Checking the node and the card before proposing a swap.",
            "tool_calls": [
                {"id": "c1", "name": "pve_status", "arguments": {}},
                {"id": "c2", "name": "gpu_status", "arguments": {}},
            ],
        }
    ]

    with patch('src.engine.TextEngine._download_image', new_callable=AsyncMock):
        history_for_api, serializable_history = await engine._build_google_history(
            "system prompt", history, None
        )

    parts = history_for_api[0]['parts']
    assert len(parts) == 3
    assert parts[0].text == (
        "Checking the node and the card before proposing a swap."
    )
    assert parts[1].function_call['name'] == 'pve_status'
    assert parts[2].function_call['name'] == 'gpu_status'

    ser_parts = serializable_history[1]['parts']
    assert ser_parts[0] == {
        'text': "Checking the node and the card before proposing a swap."
    }
    assert len(ser_parts) == 3


@pytest.mark.asyncio
async def test_call_only_model_turn_gains_no_empty_text_part():
    """An empty/absent `content` must not add a blank Part — the Google API
    rejects empty text parts and the old shape has to stay byte-identical."""
    engine = TextEngine()

    history = [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {"id": "c1", "name": "pve_status", "arguments": {}},
            ],
        }
    ]

    with patch('src.engine.TextEngine._download_image', new_callable=AsyncMock):
        history_for_api, _ = await engine._build_google_history(
            "system prompt", history, None
        )

    parts = history_for_api[0]['parts']
    assert len(parts) == 1
    assert parts[0].function_call['name'] == 'pve_status'
