# tests/provider_stream_mocks.py
#
# DP-206 — mock transports for the canonical provider streaming drivers.
# The engine's one-shot provider paths drain real SDK streams now, so tests
# that used to mock `create(...)` returning a complete response instead mock
# the streaming call returning an (async) iterator of chunk objects.

from typing import Any, Iterable, List, Optional, Sequence, Tuple
from unittest.mock import AsyncMock, MagicMock


class AsyncIterList:
    """Wrap a list so it can stand in for an SDK async stream object."""

    def __init__(self, items: Iterable[Any]) -> None:
        self._items = list(items)

    def __aiter__(self) -> "AsyncIterList":
        self._it = iter(self._items)
        return self

    async def __anext__(self) -> Any:
        try:
            return next(self._it)
        except StopIteration:
            raise StopAsyncIteration


# ---------------------------------------------------------------------------
# OpenAI chat.completions streaming chunks
# ---------------------------------------------------------------------------

def openai_chunk(content: Optional[str] = None,
                 tool_call_deltas: Optional[List[Any]] = None,
                 finish_reason: Optional[str] = None) -> MagicMock:
    delta = MagicMock()
    delta.content = content
    delta.tool_calls = tool_call_deltas
    choice = MagicMock(delta=delta, finish_reason=finish_reason)
    return MagicMock(choices=[choice])


def openai_tool_call_delta(index: int = 0, id: Optional[str] = None,
                           name: Optional[str] = None,
                           arguments: Optional[str] = None) -> MagicMock:
    fn = MagicMock()
    fn.name = name
    fn.arguments = arguments
    return MagicMock(index=index, id=id, function=fn)


def openai_text_stream(text: str) -> AsyncIterList:
    """A minimal text completion stream: one content chunk + terminal chunk."""
    chunks = []
    if text is not None and text != "":
        chunks.append(openai_chunk(content=text))
    chunks.append(openai_chunk(finish_reason="stop"))
    return AsyncIterList(chunks)


# ---------------------------------------------------------------------------
# Anthropic messages.stream context manager
# ---------------------------------------------------------------------------

def anthropic_stream(final_message: Any,
                     text_chunks: Iterable[str] = ()) -> MagicMock:
    """Mock for `AsyncAnthropic().messages.stream(...)` (DP-211): an async
    context manager whose body exposes `text_stream` (async delta iteration)
    and awaitable `get_final_message()` (the SDK-accumulated complete message
    the engine parses)."""
    body = MagicMock()
    body.text_stream = AsyncIterList(text_chunks)
    body.get_final_message = AsyncMock(return_value=final_message)
    manager = MagicMock()
    manager.__aenter__ = AsyncMock(return_value=body)
    manager.__aexit__ = AsyncMock(return_value=False)
    return manager


# ---------------------------------------------------------------------------
# Google generate_content_stream chunks
# ---------------------------------------------------------------------------

def google_stream(*responses: Any) -> AsyncIterList:
    """Mock return value for `models.generate_content_stream(...)`: an async
    iterator of GenerateContentResponse-shaped chunks. A single complete
    response object is a valid one-chunk stream — tests reuse the same mock
    object they used to hand to `generate_content`."""
    return AsyncIterList(responses)


def openai_tool_call_stream(calls: Sequence[Tuple[Optional[str], str, str]]) -> AsyncIterList:
    """A stream emitting one complete tool-call delta per (id, name, args_json)."""
    deltas = [
        openai_tool_call_delta(index=i, id=cid, name=name, arguments=args)
        for i, (cid, name, args) in enumerate(calls)
    ]
    return AsyncIterList([
        openai_chunk(tool_call_deltas=deltas),
        openai_chunk(finish_reason="stop"),
    ])
