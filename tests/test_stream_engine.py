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

def test_render_prompt_template_selection():
    # Non-default template is picked up by name. Marker/thinking-trigger
    # overrides were intentionally dropped — the persona's chat_template owns
    # rendering and we pass through to kobold-lite otherwise.
    messages = [{"role": "user", "content": "Hello"}]
    prompt, _ = _render_prompt(messages, "gemma")
    assert "<start_of_turn>user\nHello<end_of_turn>" in prompt
    assert prompt.endswith("<start_of_turn>model\n")

def test_render_prompt_ignores_marker_overrides():
    # user_marker / assistant_marker / thinking_trigger in inference_config
    # are silently ignored; we never let runtime data reshape the template.
    messages = [{"role": "user", "content": "Hello"}]
    prompt, _ = _render_prompt(messages, "chatml", {
        "user_marker": "USER: ",
        "assistant_marker": "ASSISTANT: ",
        "thinking_trigger": "<|think|>",
    })
    assert "<|im_start|>user\nHello<|im_end|>" in prompt
    assert "USER:" not in prompt
    assert "<|think|>" not in prompt

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
