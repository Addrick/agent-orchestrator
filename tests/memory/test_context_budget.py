# tests/memory/test_context_budget.py

from memory.context_budget import estimate_tokens, truncate_messages_to_budget


# --- estimate_tokens boundaries ---

def test_estimate_empty():
    assert estimate_tokens("") == 0


def test_estimate_len_4():
    assert estimate_tokens("abcd") == 1


def test_estimate_len_5():
    assert estimate_tokens("abcde") == 2


def test_estimate_len_7():
    assert estimate_tokens("abcdefg") == 2


def test_estimate_len_8():
    assert estimate_tokens("abcdefgh") == 2


# --- truncate_messages_to_budget ---

def _msg(role: str, length: int) -> dict:
    return {"role": role, "content": "x" * length}


def test_truncate_under_budget_noop():
    msgs = [_msg("user", 4), _msg("assistant", 8)]
    out, dropped = truncate_messages_to_budget(msgs, 100)
    assert out == msgs
    assert dropped == 0


def test_truncate_none_budget_noop():
    msgs = [_msg("user", 1000)]
    out, dropped = truncate_messages_to_budget(msgs, None)
    assert out == msgs
    assert dropped == 0


def test_truncate_zero_or_negative_noop():
    msgs = [_msg("user", 100)]
    assert truncate_messages_to_budget(msgs, 0) == (msgs, 0)
    assert truncate_messages_to_budget(msgs, -1) == (msgs, 0)


def test_truncate_drops_oldest_non_system():
    # Each msg = 100 chars = 25 tokens. Budget = 60 tokens → keep 2 newest non-system + last user.
    msgs = [
        _msg("user", 100),       # 25 tok — oldest, droppable
        _msg("assistant", 100),  # 25 tok — droppable
        _msg("user", 100),       # 25 tok — droppable (not the latest user)
        _msg("assistant", 100),  # 25 tok — droppable
        _msg("user", 100),       # 25 tok — last user, preserved
    ]
    out, dropped = truncate_messages_to_budget(msgs, 60)
    assert dropped >= 1
    assert out[-1] == msgs[-1]  # last user preserved
    total = sum(len(m["content"]) // 4 + (1 if len(m["content"]) % 4 else 0) for m in out)
    assert total <= 60


def test_truncate_preserves_all_system():
    msgs = [
        {"role": "system", "content": "S" * 100},  # 25 tok system, preserved
        _msg("user", 200),                           # 50 tok, droppable
        _msg("assistant", 200),                      # 50 tok, droppable
        {"role": "system", "content": "S" * 100},  # 25 tok system, preserved
        _msg("user", 200),                           # 50 tok, last user, preserved
    ]
    out, dropped = truncate_messages_to_budget(msgs, 100)
    system_in = [m for m in out if m["role"] == "system"]
    assert len(system_in) == 2
    assert out[-1]["role"] == "user"
    assert dropped == 2  # both middle non-system messages


def test_truncate_last_user_preserved_even_when_alone_exceeds_budget():
    # Last user message alone is 100 tokens; budget is 50 — must still be returned.
    msgs = [
        _msg("user", 100),
        _msg("assistant", 100),
        _msg("user", 400),  # 100 tok last user
    ]
    out, dropped = truncate_messages_to_budget(msgs, 50)
    assert any(m is msgs[-1] for m in out)
    assert dropped == 2


def test_truncate_returns_dropped_count_correctly():
    msgs = [_msg("user", 40)] * 10  # 10 tok each = 100 total
    out, dropped = truncate_messages_to_budget(msgs, 30)
    assert dropped == len(msgs) - len(out)


# --- DP-296 review: tool-call / tool-result pairing ---

def _tool_span(call_id: str, result_chars: int):
    """An assistant tool_calls message plus the `tool` message answering it."""
    return [
        {"role": "assistant", "tool_calls": [{"id": call_id, "name": "get_ticket",
                                              "arguments": {}}]},
        {"role": "tool", "tool_call_id": call_id, "name": "get_ticket",
         "content": "R" * result_chars},
    ]


def test_tool_call_message_is_not_free():
    """A tool_calls message carries no `content` key. Scoring it 0 let the pruner
    drop it without reducing the running total while its results survived."""
    from memory.context_budget import _message_tokens
    assert _message_tokens(_tool_span("c1", 4)[0]) > 0


def test_truncate_never_splits_a_tool_span():
    """Cutting between an assistant tool_calls message and its results leaves an
    unpaired block at the head of the wire array; Anthropic and Gemini reject the
    whole request for it. DP-296 made such blocks routine."""
    msgs = [
        *_tool_span("c1", 400),   # oldest — this whole span should go
        _msg("assistant", 40),
        _msg("user", 40),
    ]
    out, dropped = truncate_messages_to_budget(msgs, 30)

    assert not any(m.get("role") == "tool" for m in out), \
        "orphaned tool_result survived its assistant tool_calls message"
    # Both halves of the span leave together; the span alone frees enough that
    # the two later messages stay.
    assert dropped == 2
    assert out == msgs[2:]


def test_truncate_keeps_span_whole_when_it_fits():
    msgs = [_msg("user", 40), *_tool_span("c1", 40), _msg("user", 40)]
    out, dropped = truncate_messages_to_budget(msgs, 10_000)
    assert dropped == 0
    assert out == msgs


def test_truncate_drops_whole_span_and_accounts_for_it():
    """The span's tokens must actually come off `running`, or the pruner keeps
    dropping past the budget (or stops short of it)."""
    msgs = [
        *_tool_span("c1", 400),   # 100 tok of results + the call itself
        _msg("user", 200),        # 50 tok
        _msg("user", 40),         # 10 tok, last user, preserved
    ]
    out, dropped = truncate_messages_to_budget(msgs, 60)
    assert out[-1] is msgs[-1]
    assert dropped == len(msgs) - len(out)
    assert not any(m.get("tool_calls") or m.get("role") == "tool" for m in out)
