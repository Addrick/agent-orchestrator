# tests/test_thought_signature_serialization.py
import json
import base64
import pytest
from unittest.mock import MagicMock
from src.engine import TextEngine

@pytest.mark.asyncio
async def test_thought_signature_json_serialization():
    """
    Ensures that tool calls with thought_signature (from Gemini)
    are serializable to JSON after processing by TextEngine.
    """
    engine = TextEngine()
    
    # Mock a Google API response with a thought_signature
    # Note: in the real SDK, Part is a dataclass-like object
    mock_part = MagicMock()
    mock_part.function_call = MagicMock()
    mock_part.function_call.name = "test_tool"
    mock_part.function_call.args = {"arg": 1}
    mock_part.thought_signature = b"binary_signature_data"
    
    mock_candidate = MagicMock()
    mock_candidate.content.parts = [mock_part]
    mock_candidate.grounding_metadata = None
    
    response_obj = MagicMock()
    response_obj.candidates = [mock_candidate]
    response_obj.prompt_feedback = None
    
    # Process the response
    result, _ = engine._parse_google_response(response_obj, {})
    
    # Verify it is JSON serializable
    try:
        json_str = json.dumps(result)
        decoded = json.loads(json_str)
        
        # Verify the signature was encoded as base64
        expected_b64 = base64.b64encode(b"binary_signature_data").decode('utf-8')
        assert decoded['calls'][0]['thought_signature'] == expected_b64
    except TypeError as e:
        pytest.fail(f"Serialization failed: {e}")

@pytest.mark.asyncio
async def test_thought_signature_reconstruction():
    """
    Ensures that tool calls with encoded thought_signature are correctly
    reconstructed back to bytes when building history for the Google API.
    """
    engine = TextEngine()
    
    test_sig = b"binary_signature_data"
    encoded_sig = base64.b64encode(test_sig).decode('utf-8')
    
    history = [
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": "call_1",
                    "name": "test_tool",
                    "arguments": {"arg": 1},
                    "thought_signature": encoded_sig
                }
            ]
        }
    ]
    
    # Build history for API
    # We mock _download_image since it's not needed for this test
    with patch('src.engine.TextEngine._download_image', new_callable=AsyncMock):
        history_for_api, serializable_history = await engine._build_google_history(
            "system prompt", history, None
        )
    
    # The assistant turn is index 0
    model_turn = history_for_api[0]
    assert model_turn['role'] == 'model'
    # Check that it was decoded back to bytes
    assert model_turn['parts'][0].thought_signature == test_sig
    
    # Check that serializable version is safe
    assert serializable_history[1]['parts'][0]['thought_signature'] == '...present...'

from unittest.mock import AsyncMock, patch
