"""Cross-provider contract: a toolless completion returns the model's prose.

Every DP-335 test mocks `stream_messages` or `generate_response` — both of
which sit ABOVE every provider adapter — so the feature had twenty tests and
not one of them ran a line of provider code. The bug that shipped was in the
provider: `agy` parsed `<tool_call>` out of a reply it was never given tools
for, returned `type=tool_calls`, and the prose went in the bin.

The trap is that the wrap-up prompt is a transcript of the turn's OWN tool
calls plus a persona prompt naming tools by hand, so a `<tool_call>` in the
reply is the LIKELY output, not a corner case. And the failure is
provider-shaped: the kobold path strips the span and keeps the prose in
`visible_text`, so the dev box degrades gracefully while the deployed
one-shot provider loses the answer outright.

Each case here mocks at the **transport** — the CLI subprocess, the HTTP
stream — and asserts on what the caller receives, so no adapter is skipped.
"""

from __future__ import annotations

import json
from typing import Any, AsyncIterator, Dict, List, Optional
from unittest.mock import AsyncMock

import pytest

from src.engine import TextEngine
from src.stream_engine import StreamEngine


# The shape of the reply that broke it: real prose, then a tool-call span the
# model echoed back out of the transcript it was shown.
PROSE = "No gguf exists for that repo yet."
RAW_WITH_TOOL_SPAN = (
    f"{PROSE}\n"
    '<tool_call>{"name": "hf_search", "arguments": {"query": "qwen"}}</tool_call>'
)


def _history() -> Dict[str, Any]:
    return {
        "persona_prompt": "You are hypr. You can call hf_search and pve_status.",
        "history": [],
        "current_message": {"text": "which quant should I install?"},
    }


async def _drain(stream: AsyncIterator[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [ev async for ev in stream]


# -- per-provider drivers ----------------------------------------------------
#
# Each returns the unified event list for ONE toolless completion whose
# transport produced `RAW_WITH_TOOL_SPAN`.

async def _drive_agy(monkeypatch) -> List[Dict[str, Any]]:
    engine = TextEngine()
    monkeypatch.setattr(
        engine, "_run_agy_cli", AsyncMock(return_value=RAW_WITH_TOOL_SPAN),
    )
    return await _drain(
        engine._stream_agy_response({"model_name": "agy-flash"}, _history(),
                                    tools=None)
    )


async def _drive_cc(monkeypatch) -> List[Dict[str, Any]]:
    engine = TextEngine()
    monkeypatch.setattr(
        engine, "_run_cc_cli", AsyncMock(return_value=RAW_WITH_TOOL_SPAN),
    )
    monkeypatch.setattr(engine, "_ensure_cc_supported", lambda: None)
    return await _drain(
        engine._stream_cc_response({"model_name": "cc-sonnet"}, _history(),
                                   tools=None)
    )


async def _drive_local(monkeypatch) -> List[Dict[str, Any]]:
    # Import the kobold SSE fakes from the stream-engine suite rather than
    # re-deriving them: a second hand-written transport fake is a second
    # implementation of the contract, and it drifts toward whatever the test
    # that owns it needs.
    from tests.test_stream_engine import _FakeClient, _FakeResp, _sse_token

    # Keep the separators: kobold streams whitespace inside its tokens, and a
    # tokenizer that drops it makes the prose assertion pass or fail on the
    # test's own splitting rather than on the provider.
    chunks = [_sse_token(tok) for tok in RAW_WITH_TOOL_SPAN.splitlines(True)]
    chunks.append(_sse_token("", finish_reason="stop"))
    engine = StreamEngine()
    engine._http_client = _FakeClient(_FakeResp(chunks=chunks))
    return await _drain(engine.stream_local(
        {"model_name": "local", "max_output_tokens": 128,
         "temperature": 0.7, "top_p": 0.9, "top_k": 40,
         "chat_template": "chatml"},
        {"persona_prompt": "you are hypr",
         "message_history": [{"role": "user", "content": "which quant?"}]},
        None, None,
    ))


# `local` is the graceful-degradation control: it also parses the span, but
# keeps the prose in `visible_text`. Including it is the point — it is why the
# dev box could not see this bug, and it pins that the mitigation still holds.
DRIVERS = {"agy": _drive_agy, "cc": _drive_cc, "local": _drive_local}


@pytest.mark.asyncio
@pytest.mark.parametrize("provider", sorted(DRIVERS))
async def test_toolless_completion_returns_the_models_prose(provider, monkeypatch):
    """`tools=None` means the caller wants text, and text is what must arrive.

    `_answer_without_tools` reads the prose off `done.full_text` (falling back
    to the accumulated deltas). A provider that classifies the reply as
    `tool_calls` reports `full_text: ""` through
    `driver._events_from_one_shot`, and DP-335's exhaustion answer silently
    degrades to the canned list after paying for the round trip.
    """
    events = await DRIVERS[provider](monkeypatch)

    done = [e for e in events if e.get("type") == "done"]
    assert len(done) == 1, f"{provider}: expected exactly one done event"

    deltas = "".join(
        e.get("text") or "" for e in events if e.get("type") == "text_delta"
    )
    text = done[0].get("full_text") or deltas
    assert PROSE.split(".")[0] in text, (
        f"{provider} dropped the model's prose from a toolless completion"
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("provider", sorted(DRIVERS))
async def test_toolless_completion_never_reports_a_tool_call(provider, monkeypatch):
    """The stronger half: a provider must not invent a tool call the caller
    did not offer tools for.

    The caller has no way to run it — `_answer_without_tools` withholds tools
    precisely because the budget is spent — so a `tool_calls` event here is at
    best ignored and at worst (agy, cc) takes the prose down with it.
    """
    events = await DRIVERS[provider](monkeypatch)

    assert not [e for e in events if e.get("type") == "tool_calls"], (
        f"{provider} parsed a tool call out of a completion made with tools=None"
    )
