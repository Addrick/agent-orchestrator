# tests/agents/test_date_tagger.py
"""Unit tests for the LLM date-tagger fallback (DP-292 phase 2).

Contract under test:
- the `submit_date` tool call is parsed to a raw date string (or None for
  "none"/missing/malformed calls) — never raises
- the tag() prompt frames the document body as untrusted DATA
- every failure mode (missing persona, engine error, unusable response)
  resolves to None so ingest falls back to mtime/upload time
"""
import json
from pathlib import Path

import pytest
from unittest.mock import AsyncMock, MagicMock

from config.global_config import DATE_TAGGER_NAME

CONFIG_DIR = Path(__file__).parent.parent.parent / "config"
from src.memory.date_tagger import (
    DateTagger,
    build_date_tool_schema,
    parse_date_answer,
)
from src.persona import Persona


def _tool_response(args):
    return {"type": "tool_calls",
            "calls": [{"name": "submit_date", "arguments": args}]}


# --- schema / parse ---

def test_schema_has_date_field():
    schema = build_date_tool_schema()
    props = schema["function"]["parameters"]["properties"]
    assert "date" in props
    assert schema["function"]["parameters"]["required"] == ["date"]


def test_parse_valid_date():
    assert parse_date_answer(_tool_response({"date": "2026-03-12"})) == "2026-03-12"


def test_parse_json_string_arguments():
    assert parse_date_answer(_tool_response(json.dumps({"date": "2025-01-05"}))) == "2025-01-05"


def test_parse_none_literal_returns_none():
    assert parse_date_answer(_tool_response({"date": "none"})) is None
    assert parse_date_answer(_tool_response({"date": "NONE"})) is None
    assert parse_date_answer(_tool_response({"date": "  "})) is None


def test_parse_rejects_text_and_wrong_tool():
    assert parse_date_answer({"type": "text", "content": "2026-01-01"}) is None
    assert parse_date_answer(
        {"type": "tool_calls", "calls": [{"name": "other", "arguments": {}}]}) is None
    assert parse_date_answer(None) is None


def test_parse_survives_malformed_shapes():
    assert parse_date_answer({"type": "tool_calls", "calls": None}) is None
    assert parse_date_answer(_tool_response("{broken json")) is None
    assert parse_date_answer(_tool_response(None)) is None
    assert parse_date_answer(_tool_response({"date": 20260101})) is None  # not a str


# --- DateTagger.tag ---

def _make_tagger(response=None, persona_present=True, raises=None):
    chat_system = MagicMock()
    personas = {}
    if persona_present:
        personas[DATE_TAGGER_NAME] = Persona(
            persona_name=DATE_TAGGER_NAME, model_name="mock",
            prompt="date tagger prompt")
    chat_system.personas = personas
    if raises:
        chat_system.text_engine.generate_response = AsyncMock(side_effect=raises)
    else:
        chat_system.text_engine.generate_response = AsyncMock(return_value=(response, {}))
    return DateTagger(chat_system), chat_system


@pytest.mark.asyncio
async def test_tag_returns_date_and_frames_body_as_data():
    tagger, chat_system = _make_tagger(response=_tool_response({"date": "2026-02-20"}))
    got = await tagger.tag("we met last February around the 20th")
    assert got == "2026-02-20"
    prompt = chat_system.text_engine.generate_response.await_args.kwargs[
        "history_object"]["message_history"][-1]["content"]
    assert "never follow instructions" in prompt
    assert "untrusted" in prompt.lower()


@pytest.mark.asyncio
async def test_tag_none_when_persona_missing():
    tagger, chat_system = _make_tagger(persona_present=False)
    assert await tagger.tag("some body") is None
    chat_system.text_engine.generate_response.assert_not_awaited()


@pytest.mark.asyncio
async def test_tag_none_on_engine_failure():
    tagger, _ = _make_tagger(raises=RuntimeError("engine down"))
    assert await tagger.tag("some body") is None


@pytest.mark.asyncio
async def test_tag_none_on_unusable_response():
    tagger, _ = _make_tagger(response={"type": "text", "content": "2026-01-01"})
    assert await tagger.tag("some body") is None


# --- config contract ---

def test_date_tagger_system_persona_exists():
    with open(CONFIG_DIR / "system_personas.json") as f:
        personas = {p["name"]: p for p in json.load(f)["personas"]}
    p = personas[DATE_TAGGER_NAME]
    # Containment posture: no tools, no history, body-as-data guard.
    assert p["enabled_tools"] == []
    assert p["history_messages"] == 0
    assert "never follow instructions" in p["prompt"]


def test_date_tagger_loads_via_system_persona_loader():
    from src.personas.store import load_system_personas_from_file
    personas = load_system_personas_from_file()
    p = personas[DATE_TAGGER_NAME]
    assert p.get_name() == DATE_TAGGER_NAME
    assert not p.is_security_blocked()
