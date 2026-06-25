# tests/test_engine_payload_parity.py
#
# DP-206 — wire-payload byte-equivalence goldens.
#
# These goldens were captured from the pre-DP-206 one-shot provider handlers
# (engine.py at DP-205 `357bf6f`) with mocked SDK transports. They pin the
# EXACT request kwargs each provider sends, plus the api_payload dump and the
# parsed result tuple. The DP-206 cutover (one streaming generator per
# provider, one-shot = collect(stream)) must keep every golden byte-identical:
# when a provider's transport moves to its streaming SDK call, only the
# CAPTURE POINT in this file changes — the GOLDEN dicts must never change.
#
# If a golden assertion fails after an engine change, the wire payload drifted:
# fix the engine, do not update the golden (unless the wire change is an
# intentional, reviewed decision).

import base64
import copy
import pytest
from unittest.mock import patch, AsyncMock, MagicMock

from src.engine import TextEngine
from tests.provider_stream_mocks import (
    anthropic_stream,
    google_stream,
    openai_text_stream,
)


@pytest.fixture
def text_engine():
    return TextEngine()


def _ctx(history):
    return {
        "persona_prompt": "You are the parity bot.",
        "history": copy.deepcopy(history),
        "current_message": {"text": "current question"},
    }


PARITY_TOOLS = [
    {
        "type": "function",
        "is_write": False,
        "service_binding": "test",
        "function": {
            "name": "get_weather",
            "description": "Get the current weather",
            "parameters": {
                "type": "object",
                "properties": {"location": {"type": "string", "description": "City name"}},
                "required": ["location"],
            },
        },
    },
    {
        "type": "function",
        "is_write": True,
        "service_binding": "test",
        "function": {
            "name": "create_note",
            "description": "Create a note",
            "parameters": {
                "type": "object",
                "properties": {"text": {"type": "string"}},
                "required": ["text"],
            },
        },
    },
]


# ---------------------------------------------------------------------------
# OpenAI
# ---------------------------------------------------------------------------

OPENAI_CONFIG = {
    "model_name": "gpt-4",
    "temperature": 0.7,
    "top_p": 0.9,
    "max_output_tokens": 256,
}
OPENAI_HISTORY = [
    {"role": "user", "content": "first question"},
    {"role": "assistant", "content": "first answer"},
    {"role": "user", "content": "current question"},
]

OPENAI_GOLDEN_WIRE = {
    "max_tokens": 256,
    "messages": [
        {"content": "You are the parity bot.", "role": "system"},
        {"content": "first question", "role": "user"},
        {"content": "first answer", "role": "assistant"},
        {"content": "current question", "role": "user"},
    ],
    "model": "gpt-4",
    "temperature": 0.7,
    "tool_choice": "auto",
    "tools": [
        {
            "function": {
                "description": "Get the current weather",
                "name": "get_weather",
                "parameters": {
                    "properties": {"location": {"description": "City name", "type": "string"}},
                    "required": ["location"],
                    "type": "object",
                },
            },
            "type": "function",
        },
        {
            "function": {
                "description": "Create a note",
                "name": "create_note",
                "parameters": {
                    "properties": {"text": {"type": "string"}},
                    "required": ["text"],
                    "type": "object",
                },
            },
            "type": "function",
        },
    ],
    "top_p": 0.9,
}
OPENAI_GOLDEN_PAYLOAD = {
    **{k: v for k, v in OPENAI_GOLDEN_WIRE.items() if k != "tools"},
    "tools": ["get_weather", "create_note"],
}


@pytest.mark.asyncio
async def test_openai_wire_payload_matches_golden(text_engine, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "dummy")
    with patch("src.engine.providers.openai.AsyncOpenAI") as cls:
        inst = cls.return_value
        inst.chat.completions.create = AsyncMock(
            return_value=openai_text_stream("parity ok")
        )
        result, payload = await text_engine.generate_response(
            dict(OPENAI_CONFIG), _ctx(OPENAI_HISTORY), tools=copy.deepcopy(PARITY_TOOLS)
        )
        captured = dict(inst.chat.completions.create.call_args.kwargs)

    # DP-206 cutover: the one-shot path drains the canonical stream, so the
    # transport gains `stream=True` — everything else must stay byte-equal.
    assert captured.pop("stream") is True

    # The payload dump rewrites `tools` to a name list on the SAME dict object
    # that was sent over the wire, so reconstruct the wire view from the dump
    # contract: everything except tools is the wire payload verbatim.
    assert {k: v for k, v in captured.items() if k != "tools"} == \
        {k: v for k, v in OPENAI_GOLDEN_WIRE.items() if k != "tools"}
    assert payload == OPENAI_GOLDEN_PAYLOAD
    assert result == {"type": "text", "content": "parity ok"}


# ---------------------------------------------------------------------------
# Anthropic
# ---------------------------------------------------------------------------

ANTHROPIC_CONFIG = {
    "model_name": "claude-3-opus-20240229",
    "temperature": 0.5,
    "top_p": 0.8,
    "top_k": 40,
    "max_output_tokens": 256,
}
ANTHROPIC_HISTORY = [
    {"role": "system", "content": "Extra system context"},
    {"role": "user", "content": "first question"},
    {"role": "assistant", "content": "first answer"},
    {"role": "user", "content": "current question"},
]

ANTHROPIC_GOLDEN_WIRE = {
    "max_tokens": 256,
    "messages": [
        {"content": "first question", "role": "user"},
        {"content": "first answer", "role": "assistant"},
        {"content": "current question", "role": "user"},
    ],
    "model": "claude-3-opus-20240229",
    "system": "You are the parity bot.\n\nExtra system context",
    "temperature": 0.5,
    "tools": [
        {
            "description": "Get the current weather",
            "input_schema": {
                "properties": {"location": {"description": "City name", "type": "string"}},
                "required": ["location"],
                "type": "object",
            },
            "name": "get_weather",
        },
        {
            "description": "Create a note",
            "input_schema": {
                "properties": {"text": {"type": "string"}},
                "required": ["text"],
                "type": "object",
            },
            "name": "create_note",
        },
    ],
    "top_k": 40,
    "top_p": 0.8,
}
ANTHROPIC_GOLDEN_PAYLOAD = {
    **{k: v for k, v in ANTHROPIC_GOLDEN_WIRE.items() if k != "tools"},
    "tools": ["get_weather", "create_note"],
}


@pytest.mark.asyncio
async def test_anthropic_wire_payload_matches_golden(text_engine, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "dummy")
    with patch("src.engine.anthropic.AsyncAnthropic") as cls:
        inst = cls.return_value
        # DP-206 cutover: the transport moved from `messages.create` to
        # `messages.stream` (same kwargs — the SDK adds the wire-level
        # `stream: true` itself). Capture point change only; goldens frozen.
        inst.messages.stream.return_value = anthropic_stream(MagicMock(
            content=[MagicMock(text="parity ok")], stop_reason="end_turn"
        ), ["parity ok"])
        result, payload = await text_engine.generate_response(
            dict(ANTHROPIC_CONFIG), _ctx(ANTHROPIC_HISTORY), tools=copy.deepcopy(PARITY_TOOLS)
        )
        captured = dict(inst.messages.stream.call_args.kwargs)

    assert {k: v for k, v in captured.items() if k != "tools"} == \
        {k: v for k, v in ANTHROPIC_GOLDEN_WIRE.items() if k != "tools"}
    assert payload == ANTHROPIC_GOLDEN_PAYLOAD
    assert result == {"type": "text", "content": "parity ok"}


# ---------------------------------------------------------------------------
# Google
# ---------------------------------------------------------------------------

GOOGLE_CONFIG = {
    "model_name": "gemini-pro",
    "temperature": 0.6,
    "top_p": 0.95,
    "top_k": 32,
    "max_output_tokens": 512,
    "thinking_level": "low",
}
THOUGHT_SIG_B64 = base64.b64encode(b"sig_parity").decode("utf-8")
GOOGLE_HISTORY = [
    {"role": "user", "content": "search for x"},
    {"role": "assistant", "tool_calls": [
        {"id": "call_1", "name": "get_weather", "arguments": {"location": "NYC"},
         "thought_signature": THOUGHT_SIG_B64},
    ]},
    {"role": "tool", "tool_call_id": "call_1", "name": "get_weather",
     "content": '{"temp": 70}'},
    {"role": "user", "content": "current question"},
]
GOOGLE_TOOLS = PARITY_TOOLS + [
    {"type": "google_grounding", "function": {"name": "google_grounding_search"}},
]

GOOGLE_GOLDEN_WIRE = {
    "config": {
        "max_output_tokens": 512,
        "safety_settings": [
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
        ],
        "system_instruction": "You are the parity bot.",
        "temperature": 0.6,
        "thinking_config": {"thinking_level": "LOW"},
        "tool_config": {"function_calling_config": {"mode": "AUTO"}},
        "tools": [
            {"google_search": {}},
            {"function_declarations": [
                {"description": "Get the current weather",
                 "name": "get_weather",
                 "parameters": {
                     "properties": {"location": {"description": "City name", "type": "STRING"}},
                     "required": ["location"],
                     "type": "OBJECT",
                 }},
            ]},
            {"function_declarations": [
                {"description": "Create a note",
                 "name": "create_note",
                 "parameters": {
                     "properties": {"text": {"type": "STRING"}},
                     "required": ["text"],
                     "type": "OBJECT",
                 }},
            ]},
        ],
        "top_k": 32.0,
        "top_p": 0.95,
    },
    "contents": [
        {"parts": [{"text": "search for x"}], "role": "user"},
        {"parts": [{"function_call": {"args": {"location": "NYC"}, "name": "get_weather"},
                    "thought_signature": "c2lnX3Bhcml0eQ=="}],
         "role": "model"},
        {"parts": [{"function_response": {"name": "get_weather", "response": {"temp": 70}}}],
         "role": "tool"},
        {"parts": [{"text": "current question"}], "role": "user"},
    ],
    "model": "gemini-pro",
}


def _normalize_google_call(kwargs):
    """Project the (model, contents, config) kwargs onto plain JSON-able
    structures so they can be compared against a literal golden."""
    contents = [
        {"role": c["role"],
         "parts": [p.model_dump(mode="json", exclude_none=True) for p in c["parts"]]}
        for c in kwargs["contents"]
    ]
    config = kwargs["config"].model_dump(mode="json", exclude_none=True)
    return {"model": kwargs["model"], "contents": contents, "config": config}


@pytest.mark.asyncio
async def test_google_wire_payload_matches_golden(text_engine, monkeypatch):
    monkeypatch.setenv("GOOGLE_GENERATIVEAI_API_KEY", "dummy")
    with patch("src.engine.genai.client.AsyncClient") as cls:
        inst = cls.return_value
        part = MagicMock(text="parity ok", function_call=None)
        part.thought_signature = None
        cand = MagicMock(content=MagicMock(parts=[part]), grounding_metadata=None)
        inst.models.generate_content_stream = AsyncMock(
            return_value=google_stream(MagicMock(prompt_feedback=None, candidates=[cand]))
        )
        result, payload = await text_engine.generate_response(
            dict(GOOGLE_CONFIG), _ctx(GOOGLE_HISTORY), tools=copy.deepcopy(GOOGLE_TOOLS)
        )
        captured = _normalize_google_call(inst.models.generate_content_stream.call_args.kwargs)

    assert captured == GOOGLE_GOLDEN_WIRE
    # The api_payload dump lists tools by name and carries the serializable
    # history (with the system row and the original message fields preserved).
    assert payload["model"] == "gemini-pro"
    assert payload["config"]["tools"] == ["google_search", "get_weather", "create_note"]
    assert payload["contents"][0] == {
        "role": "system", "parts": [{"text": "You are the parity bot."}]
    }
    assert payload["contents"][2]["parts"][0]["thought_signature"] == "...present..."
    assert result == {"type": "text", "content": "parity ok"}
