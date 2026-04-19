import pytest
import json
from src.stream_engine import StreamEngine, _render_prompt

def test_render_prompt_default_chatml():
    # Default uses ChatML base
    prompt, stop = _render_prompt([
        {"role": "system", "content": "SysPrompt"},
        {"role": "user", "content": "UserMsg"}
    ], "chatml")
    
    assert "<|im_start|>system\nSysPrompt<|im_end|>" in prompt
    assert "<|im_start|>user\nUserMsg<|im_end|>" in prompt
    assert prompt.endswith("<|im_start|>assistant\n")
    assert "<|im_end|>" in stop

def test_render_prompt_custom_markers():
    messages = [{"role": "user", "content": "Hello"}]
    # Simulation of detected Llama markers
    inference_config = {
        "user_marker": "<|turn|>user\n",
        "assistant_marker": "<|turn|>model\n",
        "stop_sequence": ["<|im_end|>"]
    }
    prompt, stop = _render_prompt(messages, "chatml", inference_config)
    
    # Should strictly use provided markers
    assert "<|turn|>user\nHello" in prompt
    assert prompt.endswith("<|turn|>model\n")
    # Should NOT have ChatML markers
    assert "<|im_start|>" not in prompt
    assert "<|im_end|>" in stop

def test_render_prompt_thinking_trigger_boundary():
    messages = [{"role": "user", "content": "Hello"}]
    inference_config = {
        "user_marker": "USER: ",
        "assistant_marker": "ASSISTANT: ",
        "thinking_trigger": "<|think|>"
    }
    prompt, _ = _render_prompt(messages, "chatml", inference_config)
    
    # Verify the thinking trigger is appended immediately after the assistant marker
    assert prompt.endswith("ASSISTANT: <|think|>")
    assert "USER: Hello" in prompt

def test_render_prompt_tool_call_serialization():
    messages = [
        {"role": "assistant", "content": "Checking...", "tool_calls": [
            {"name": "get_weather", "arguments": {"city": "Berlin"}}
        ]}
    ]
    prompt, _ = _render_prompt(messages, "chatml")
    
    assert "Checking..." in prompt
    assert "<tool_call>{\"name\": \"get_weather\", \"arguments\": {\"city\": \"Berlin\"}}</tool_call>" in prompt

def test_render_prompt_stop_sequence_merging():
    messages = [{"role": "user", "content": "test"}]
    inference_config = {
        "stop_sequence": ["OVERRIDE_STOP"]
    }
    _, stop = _render_prompt(messages, "chatml", inference_config)
    
    assert "OVERRIDE_STOP" in stop
    assert "<|im_start|>" in stop  # Should also contain base stops from ChatML
    assert stop[0] == "OVERRIDE_STOP" # Priority
