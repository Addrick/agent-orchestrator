"""Unit tests for the shared `<tool_call>` text-protocol primitives.

These cover the module directly (not just via the two call sites) so the
genuinely-shared extraction/decode/rendering core is pinned independently of
engine.py and stream_engine.py.
"""

from src.text_tool_protocol import (
    TOOL_CALL_OPEN,
    TOOL_CALL_CLOSE,
    TOOL_CALL_SYNTAX,
    decode_tool_call_payload,
    extract_first_tool_call_block,
    render_tool_descriptions,
)


def test_tag_constants():
    assert TOOL_CALL_OPEN == "<tool_call>"
    assert TOOL_CALL_CLOSE == "</tool_call>"
    assert TOOL_CALL_SYNTAX.startswith(TOOL_CALL_OPEN)
    assert TOOL_CALL_SYNTAX.endswith(TOOL_CALL_CLOSE)
    assert '"name"' in TOOL_CALL_SYNTAX and '"arguments"' in TOOL_CALL_SYNTAX


def test_extract_valid_block():
    text = '<tool_call>{"name": "get_weather", "arguments": {"city": "Tokyo"}}</tool_call>'
    inner = extract_first_tool_call_block(text)
    assert inner == '{"name": "get_weather", "arguments": {"city": "Tokyo"}}'


def test_extract_strips_surrounding_whitespace():
    text = '<tool_call>\n  {"name": "ping"}  \n</tool_call>'
    assert extract_first_tool_call_block(text) == '{"name": "ping"}'


def test_extract_no_block_returns_none():
    assert extract_first_tool_call_block("just plain text, no tools here") is None


def test_extract_empty_text_returns_none():
    assert extract_first_tool_call_block("") is None


def test_extract_returns_first_of_multiple():
    text = (
        '<tool_call>{"name": "a"}</tool_call>'
        'middle prose'
        '<tool_call>{"name": "b"}</tool_call>'
    )
    assert extract_first_tool_call_block(text) == '{"name": "a"}'


def test_extract_spans_newlines():
    text = '<tool_call>{"name": "a",\n "arguments": {}}</tool_call>'
    inner = extract_first_tool_call_block(text)
    assert inner is not None
    assert '"name": "a"' in inner


def test_decode_valid_object():
    parsed = decode_tool_call_payload('{"name": "x", "arguments": {"k": 1}}')
    assert parsed == {"name": "x", "arguments": {"k": 1}}


def test_decode_malformed_json_returns_none():
    assert decode_tool_call_payload('{"name": "x", "arguments": ') is None


def test_decode_non_object_returns_none():
    # Valid JSON, but not an object — callers expect a dict.
    assert decode_tool_call_payload("[1, 2, 3]") is None
    assert decode_tool_call_payload("42") is None
    assert decode_tool_call_payload('"a string"') is None


def test_decode_tolerates_surrounding_whitespace():
    assert decode_tool_call_payload('  {"name": "x"}  ') == {"name": "x"}


def test_render_tool_descriptions_one_line_per_tool():
    tools = [
        {"function": {"name": "a", "description": "first", "parameters": {"type": "object"}}},
        {"function": {"name": "b", "description": "second", "parameters": {}}},
    ]
    lines = render_tool_descriptions(tools)
    assert len(lines) == 2
    assert lines[0].startswith("name: a, description: first, parameters: ")
    assert lines[1].startswith("name: b, description: second, parameters: ")


def test_render_tool_descriptions_empty_list():
    assert render_tool_descriptions([]) == []


def test_render_extract_decode_round_trip():
    """A rendered tool name survives a synthesized call through extract+decode."""
    tools = [{"function": {"name": "lookup", "description": "d", "parameters": {}}}]
    line = render_tool_descriptions(tools)[0]
    assert "lookup" in line
    emitted = f'{TOOL_CALL_OPEN}{{"name": "lookup", "arguments": {{"q": "x"}}}}{TOOL_CALL_CLOSE}'
    inner = extract_first_tool_call_block(emitted)
    assert inner is not None
    parsed = decode_tool_call_payload(inner)
    assert parsed == {"name": "lookup", "arguments": {"q": "x"}}
