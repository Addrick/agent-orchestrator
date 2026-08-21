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
    extract_tool_call_blocks,
    render_tool_descriptions,
    strip_tool_call_blocks,
)


def test_tag_constants():
    assert TOOL_CALL_OPEN == "<tool_call>"
    assert TOOL_CALL_CLOSE == "</tool_call>"
    assert TOOL_CALL_SYNTAX.startswith(TOOL_CALL_OPEN)
    assert TOOL_CALL_SYNTAX.endswith(TOOL_CALL_CLOSE)
    assert '"name"' in TOOL_CALL_SYNTAX and '"arguments"' in TOOL_CALL_SYNTAX


def test_extract_valid_block():
    text = '<tool_call>{"name": "get_weather", "arguments": {"city": "Tokyo"}}</tool_call>'
    assert extract_tool_call_blocks(text) == [
        '{"name": "get_weather", "arguments": {"city": "Tokyo"}}'
    ]


def test_extract_strips_surrounding_whitespace():
    text = '<tool_call>\n  {"name": "ping"}  \n</tool_call>'
    assert extract_tool_call_blocks(text) == ['{"name": "ping"}']


def test_extract_no_block_returns_empty():
    assert extract_tool_call_blocks("just plain text, no tools here") == []


def test_extract_empty_text_returns_empty():
    assert extract_tool_call_blocks("") == []


def test_extract_keeps_blocks_separated_by_prose():
    text = (
        '<tool_call>{"name": "a"}</tool_call>'
        'middle prose'
        '<tool_call>{"name": "b"}</tool_call>'
    )
    assert extract_tool_call_blocks(text) == [
        '{"name": "a"}', '{"name": "b"}',
    ]


def test_extract_spans_newlines():
    text = '<tool_call>{"name": "a",\n "arguments": {}}</tool_call>'
    blocks = extract_tool_call_blocks(text)
    assert blocks and '"name": "a"' in blocks[0]


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
    blocks = extract_tool_call_blocks(emitted)
    assert len(blocks) == 1
    parsed = decode_tool_call_payload(blocks[0])
    assert parsed == {"name": "lookup", "arguments": {"q": "x"}}


# --------------------------------------------------------------------------
# DP-338 — every block, and the prose beside them. `extract_first_...` kept
# only block 1, which is what silently discarded a batched read on the agy
# route; these pin the all-blocks contract the streaming parser always had.
# --------------------------------------------------------------------------


def test_extract_blocks_returns_every_block_in_order():
    text = (
        "Checking three things at once.\n"
        '<tool_call>{"name": "a", "arguments": {}}</tool_call>\n'
        '<tool_call>{"name": "b", "arguments": {}}</tool_call>\n'
        '<tool_call>{"name": "c", "arguments": {}}</tool_call>'
    )
    assert extract_tool_call_blocks(text) == [
        '{"name": "a", "arguments": {}}',
        '{"name": "b", "arguments": {}}',
        '{"name": "c", "arguments": {}}',
    ]


def test_extract_blocks_no_block_returns_empty_list():
    assert extract_tool_call_blocks("plain prose, no tools") == []
    assert extract_tool_call_blocks("") == []


def test_extract_blocks_keeps_emission_order():
    text = (
        '<tool_call>{"name": "a"}</tool_call>'
        '<tool_call>{"name": "b"}</tool_call>'
    )
    assert extract_tool_call_blocks(text) == ['{"name": "a"}', '{"name": "b"}']


def test_strip_removes_every_block_and_keeps_the_prose():
    text = (
        "I need the node, the card and the units.\n"
        '<tool_call>{"name": "a", "arguments": {}}</tool_call>\n'
        '<tool_call>{"name": "b", "arguments": {}}</tool_call>'
    )
    assert strip_tool_call_blocks(text) == (
        "I need the node, the card and the units."
    )


def test_strip_collapses_the_blank_run_the_blocks_leave_behind():
    text = (
        "before\n\n"
        '<tool_call>{"name": "a"}</tool_call>\n\n'
        '<tool_call>{"name": "b"}</tool_call>\n\n'
        "after"
    )
    assert strip_tool_call_blocks(text) == "before\n\nafter"


def test_strip_of_a_call_only_response_is_empty():
    assert strip_tool_call_blocks(
        '<tool_call>{"name": "a", "arguments": {}}</tool_call>'
    ) == ""
    assert strip_tool_call_blocks("") == ""


def test_strip_drops_a_truncated_trailing_block():
    """The model hit its output cap mid-block. That fragment is not prose:
    this string is persisted to `tool_context` and replayed verbatim into the
    next request, so leaving it in self-poisons the model's own context with a
    half-written marker."""
    text = (
        "Checking the node and the card before proposing a swap.\n"
        '<tool_call>{"name": "pve_status", "arguments": {}}</tool_call>\n'
        '<tool_call>{"name": "list_mo'
    )
    assert strip_tool_call_blocks(text) == (
        "Checking the node and the card before proposing a swap."
    )


def test_strip_collapses_the_blank_run_on_crlf_too():
    """agy runs on Windows since DP-324 and its responses come back CRLF. An
    `\n`-only collapse never matched them, so the platform the route was
    opened for got the four-empty-lines render the function claims to
    prevent."""
    text = (
        "before\r\n\r\n"
        '<tool_call>{"name": "a"}</tool_call>\r\n\r\n'
        '<tool_call>{"name": "b"}</tool_call>\r\n\r\n'
        "after"
    )
    assert strip_tool_call_blocks(text) == "before\n\nafter"


def test_strip_of_an_inline_block_leaves_one_space_not_two():
    text = 'I will check <tool_call>{"name": "a", "arguments": {}}</tool_call> now.'
    assert strip_tool_call_blocks(text) == "I will check now."
