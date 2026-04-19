import pytest
import json
from src.stream_engine import StreamEngine

def test_render_prompt_default_chatml():
    engine = StreamEngine(None, None)
    messages = [
        {"role": "system", "content": "SysPrompt"},
        {"role": "user", "content": "UserMsg"}
    ]
    # Default uses ChatML base
    prompt, stop = engine._render_prompt(messages, "chatml")
    
    assert "<|im_start|>system\nSysPrompt<|im_end|>" in prompt
    assert "<|im_start|>user\nUserMsg<|im_end|>" in prompt
    assert prompt.endswith("<|im_start|>assistant\n")
    assert "<|im_end|>" in stop

def test_render_prompt_custom_markers():
    engine = StreamEngine(None, None)
    messages = [{"role": "user", "content": "Hello"}]
    # Simulation of detected Llama markers
    inference_config = {
        "user_marker": "<|turn|>user\n",
        "assistant_marker": "<|turn|>model\n",
        "stop_sequence": ["<|im_end|>"]
    }
    prompt, stop = engine._render_prompt(messages, "chatml", inference_config)
    
    # Should strictly use provided markers
    assert "<|turn|>user\nHello" in prompt
    assert prompt.endswith("<|turn|>model\n")
    # Should NOT have ChatML markers
    assert "<|im_start|>" not in prompt
    assert "<|im_end|>" in stop

def test_render_prompt_thinking_trigger_boundary():
    engine = StreamEngine(None, None)
    messages = [{"role": "user", "content": "Hello"}]
    inference_config = {
        "assistant_marker": "ASSISTANT: ",
        "thinking_trigger": "<|think|>"
    }
    prompt, _ = engine._render_prompt(messages, "chatml", inference_config)
    
    # Verify no accidental double space/newline but correct separation
    assert prompt == "ASSISTANT: HelloASSISTANT: \n<|think|>"

def test_render_prompt_tool_call_serialization():
    engine = StreamEngine(None, None)
    messages = [
        {"role": "assistant", "content": "Checking...", "tool_calls": [
            {"name": "get_weather", "arguments": {"city": "Berlin"}}
        ]}
    ]
    prompt, _ = engine._render_prompt(messages, "chatml")
    
    assert "Checking..." in prompt
    assert "<tool_call>{\"name\": \"get_weather\", \"arguments\": {\"city\": \"Berlin\"}}</tool_call>" in prompt

def test_render_prompt_stop_sequence_merging():
    engine = StreamEngine(None, None)
    messages = [{"role": "user", "content": "test"}]
    inference_config = {
        "stop_sequence": ["OVERRIDE_STOP"]
    }
    _, stop = engine._render_prompt(messages, "chatml", inference_config)
    
    assert "OVERRIDE_STOP" in stop
    assert "<|im_end|>" in stop # Should keep base stops too
    assert stop[0] == "OVERRIDE_STOP" # Priority 
