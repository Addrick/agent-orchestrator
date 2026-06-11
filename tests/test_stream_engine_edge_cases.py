# tests/test_stream_engine_edge_cases.py
#
# DP-199 Batch 4 — StreamEngine SSE / parsing edge cases.

import json
from typing import AsyncIterator, List, Optional

import httpx
import pytest

from src.stream_engine import StreamEngine, _ToolCallStreamParser
from src.engine import LLMCommunicationError

# Reuse helpers from existing test by importing private symbols.
from tests.test_stream_engine import (
    _sse_token,
    _FakeResp,
    _FakeClient,
    _make_engine,
    _persona_config,
    _history,
    _drain,
)


# ------------------------------------------------------------------
# Tier 2 — SSE parsing edges
# ------------------------------------------------------------------


@pytest.mark.asyncio
async def test_kobold_sse_malformed_frame_skipped():
    """A frame missing the `event:` line must be silently skipped without
    aborting the stream — surrounding good frames still produce text."""
    malformed = "garbage line without event marker\n\n"
    chunks = [
        _sse_token("kept1 "),
        malformed,
        _sse_token("kept2"),
        _sse_token("", finish_reason="stop"),
    ]
    resp = _FakeResp(chunks=chunks)
    engine, _ = _make_engine(resp)

    events = await _drain(engine.stream_local(_persona_config(), _history()))
    visible = "".join(e["text"] for e in events if e["type"] == "text_delta")
    assert "kept1 " in visible
    assert "kept2" in visible


@pytest.mark.asyncio
async def test_sse_malformed_json_skipped():
    """SSE event with non-JSON data → frame is skipped, stream continues."""
    bad_frame = "event: message\ndata: {this is not json}\n\n"
    chunks = [
        bad_frame,
        _sse_token("after bad"),
        _sse_token("", finish_reason="stop"),
    ]
    resp = _FakeResp(chunks=chunks)
    engine, _ = _make_engine(resp)

    events = await _drain(engine.stream_local(_persona_config(), _history()))
    visible = "".join(e["text"] for e in events if e["type"] == "text_delta")
    assert visible == "after bad"


@pytest.mark.asyncio
async def test_tool_tag_partial_chunk_buffering():
    """A `<tool_call>` open tag split across SSE chunks must still be
    parsed correctly — and the partial '<tool' prefix must not leak as
    visible text."""
    body = '{"name": "ping", "arguments": {}}'
    chunks = [
        _sse_token("Hello <tool"),
        _sse_token("_call>" + body + "</tool_call> done"),
        _sse_token("", finish_reason="stop"),
    ]
    resp = _FakeResp(chunks=chunks)
    engine, _ = _make_engine(resp)

    events = await _drain(engine.stream_local(_persona_config(), _history()))
    visible = "".join(e["text"] for e in events if e["type"] == "text_delta")
    assert "<tool" not in visible
    assert "Hello " in visible
    assert "done" in visible
    tool_events = [e for e in events if e["type"] == "tool_calls"]
    assert len(tool_events) == 1
    assert tool_events[0]["calls"][0]["name"] == "ping"


@pytest.mark.asyncio
async def test_kobold_stream_timeout_raises():
    """httpx.ReadTimeout inside the stream context surfaces as
    LLMCommunicationError with api_payload preserved."""
    engine, _ = _make_engine(
        _FakeResp(), raise_on_iter=httpx.ReadTimeout("timed out")
    )
    with pytest.raises(LLMCommunicationError) as ei:
        async for _ in engine.stream_local(_persona_config(), _history()):
            pass
    assert "transport error" in str(ei.value).lower()
    assert ei.value.api_payload is not None


@pytest.mark.asyncio
async def test_kobold_stream_disconnect_raises():
    """httpx.RemoteProtocolError (upstream disconnect) is mapped to
    LLMCommunicationError, with abort POST issued because the stream
    did not finish cleanly."""
    engine, client = _make_engine(
        _FakeResp(),
        raise_on_iter=httpx.RemoteProtocolError("server disconnected"),
    )
    with pytest.raises(LLMCommunicationError):
        async for _ in engine.stream_local(_persona_config(), _history()):
            pass
    # finished_cleanly is False before the exception → abort POST is fired.
    abort_posts = [p for p in client.posts if p["url"].endswith("/api/extra/abort")]
    assert len(abort_posts) == 1


@pytest.mark.asyncio
async def test_kobold_empty_stream_yields_empty():
    """If kobold returns only the terminal stop frame with no tokens, we
    still emit the standard api_payload + done shape, and full_text is ''."""
    resp = _FakeResp(chunks=[_sse_token("", finish_reason="stop")])
    engine, _ = _make_engine(resp)

    events = await _drain(engine.stream_local(_persona_config(), _history()))
    types = [e["type"] for e in events]
    assert types[0] == "api_payload"
    assert types[-1] == "done"
    assert events[-1]["full_text"] == ""
    assert all(e["type"] != "text_delta" for e in events)


@pytest.mark.asyncio
async def test_non_local_stream_messages_no_upstream_abort():
    """When TextEngine.stream_messages routes a non-local model through the
    canonical provider streams, no kobold abort POST should ever be issued —
    there is no upstream kobold stream to abort. Asserts the StreamEngine is
    not involved at all."""
    from unittest.mock import patch, MagicMock

    from src.engine import TextEngine
    from src.generation_params import GenerationParams
    from tests.helpers import engine_stream_events

    fake_stream_engine = MagicMock()
    fake_stream_engine.stream_messages = MagicMock()
    engine = TextEngine(stream_engine=fake_stream_engine)

    async def _fake_openai_stream(*a, **k):
        for ev in engine_stream_events({"type": "text", "content": "ok"}, {"p": 1}):
            yield ev

    with patch.object(
        engine, "_stream_openai_response", new=_fake_openai_stream,
    ):
        events = []
        async for ev in engine.stream_messages(
            {"model_name": "gpt-4"},
            [{"role": "user", "content": "hi"}],
            GenerationParams(),
        ):
            events.append(ev)

    # StreamEngine.stream_messages must NOT have been called for a non-local model.
    fake_stream_engine.stream_messages.assert_not_called()
    # Standard wrap shape.
    assert events[0]["type"] == "api_payload"
    assert events[-1]["type"] == "done"


@pytest.mark.asyncio
async def test_stream_engine_aclose_idempotent():
    """Calling aclose twice on StreamEngine must not raise — idempotent
    contract used by app shutdown paths."""
    engine = StreamEngine()
    # First aclose with no client: should silently no-op.
    await engine.aclose()
    # Second aclose: still safe.
    await engine.aclose()
    assert engine._http_client is None


# ------------------------------------------------------------------
# Tier 2 — latent bug, skipped per DP-199
# ------------------------------------------------------------------


def test_tool_block_unterminated_visible():
    pytest.skip(
        "DP-199 deferred bug 5 — stream_engine.py:556-563 silently discards "
        "unterminated <tool_call> blocks at EOF. Latent bug; fall-through "
        "to visible text not yet implemented."
    )


# ------------------------------------------------------------------
# Bonus parser-level corroboration: documents *current* behavior of the
# parser for unterminated blocks (visible_text is restored on flush). The
# real bug lives at the SSE outer loop where unterminated blocks at stream
# EOF aren't always reaching flush() — kept as a parser unit assertion only,
# not a stream-EOF assertion.
# ------------------------------------------------------------------


def test_tool_call_parser_unterminated_block_restored_on_flush():
    """At the parser level: an unterminated `<tool_call>` block on flush
    is surfaced as visible text. This is the parser contract; the higher-
    level stream EOF discard bug (DP-199 deferred bug 5) is separate."""
    parser = _ToolCallStreamParser()
    parser.feed("hello <tool_call>")
    parser.feed('{"name": "x"')
    tail = parser.flush()
    assert "<tool_call>" in tail
    assert '"name": "x"' in tail
    assert parser.calls == []
