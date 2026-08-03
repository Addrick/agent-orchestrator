# tests/live/test_repropose_harness.py
#
# Verifies the `test_repropose_live.py` probe harness itself, with a scripted
# engine and no API call. Deliberately NOT marked `llm_live`, so it runs in CI.
#
# A probe that silently measures nothing is worse than no probe: every
# assertion in the live file would pass, and the conclusion drawn from it would
# be "the model behaves", when the truth is "the harness sees nothing". The
# `_suppressed_since` DB scan is the fragile part — it reads tool_context blobs
# out of SQLite looking for duplicate-guard hits, and it is asserted below
# against a case that really produces one.

import os
import random
from typing import Any, Dict, List

import pytest

from src.engine import TextEngine
from src.memory.memory_manager import MemoryManager
from src.persona import ExecutionMode, MemoryMode, Persona
from tests.helpers import make_chat_system
from tests.live.test_repropose_live import _Probe, PERSONA


def _script(streams):
    it = iter(streams)

    async def stream_messages(*a, **k):
        for ev in next(it):
            yield ev
    return stream_messages


@pytest.fixture
def probe_harness():
    db_path = f"smoke_{random.randint(1000, 9999)}.db"
    mm = MemoryManager(db_path=db_path)
    mm.create_schema()
    persona = Persona(
        persona_name=PERSONA, model_name="local", prompt="p",
        enabled_tools=["create_ticket"],
        memory_mode=MemoryMode.CHANNEL_ISOLATED, history_messages=20,
    )
    persona.set_execution_mode(ExecutionMode.CONFIRM)
    cs = make_chat_system(memory_manager=mm, text_engine=TextEngine(),
                          personas={PERSONA: persona})
    executed: List[Dict[str, Any]] = []

    async def stub(**kwargs):
        executed.append({"name": "create_ticket", "arguments": kwargs})
        return {"ticket_id": 1}
    cs.tool_manager.register("create_ticket", stub)
    yield _Probe(cs, mm, executed), cs
    mm.close()
    if os.path.exists(db_path):
        try:
            os.remove(db_path)
        except PermissionError:
            pass


@pytest.mark.asyncio
async def test_probe_detects_a_park(probe_harness):
    probe, cs = probe_harness
    cs.text_engine.stream_messages = _script([
        [{"type": "tool_calls", "calls": [
            {"id": "w1", "name": "create_ticket",
             "arguments": {"title": "t", "body": "b"}}]},
         {"type": "done", "full_text": ""}],
        [{"type": "text_delta", "text": "Proposed."},
         {"type": "done", "full_text": "Proposed."}],
    ])
    rep = await probe.say("laptop broken", label="t1")
    assert rep.proposed
    assert len(rep.new_parks) == 1
    assert not rep.reproposed_but_suppressed


@pytest.mark.asyncio
async def test_probe_detects_suppressed_duplicate(probe_harness):
    """The _suppressed_since DB scan must actually see a duplicate-guard hit."""
    probe, cs = probe_harness
    cs.text_engine.stream_messages = _script([
        [{"type": "tool_calls", "calls": [
            {"id": "w1", "name": "create_ticket",
             "arguments": {"title": "t", "body": "b"}}]},
         {"type": "done", "full_text": ""}],
        [{"type": "text_delta", "text": "Proposed."},
         {"type": "done", "full_text": "Proposed."}],
        # turn 2 re-proposes the identical write while it is still pending
        [{"type": "tool_calls", "calls": [
            {"id": "w2", "name": "create_ticket",
             "arguments": {"title": "t", "body": "b"}}]},
         {"type": "done", "full_text": ""}],
        [{"type": "text_delta", "text": "Still waiting."},
         {"type": "done", "full_text": "Still waiting."}],
    ])
    first = await probe.say("laptop broken", label="t1")
    assert first.proposed
    second = await probe.say("again please", label="t2")
    assert not second.proposed, "guard should stop the second park"
    assert second.reproposed_but_suppressed, "probe failed to SEE the intent"


@pytest.mark.asyncio
async def test_probe_decide_reports_execution(probe_harness):
    probe, cs = probe_harness
    cs.text_engine.stream_messages = _script([
        [{"type": "tool_calls", "calls": [
            {"id": "w1", "name": "create_ticket",
             "arguments": {"title": "t", "body": "b"}}]},
         {"type": "done", "full_text": ""}],
        [{"type": "text_delta", "text": "Proposed."},
         {"type": "done", "full_text": "Proposed."}],
        [{"type": "text_delta", "text": "Opened."},
         {"type": "done", "full_text": "Opened."}],
    ])
    token = await probe.park_one("laptop broken", label="t1")
    rep = await probe.decide(token, approved=True, label="approve")
    assert rep.executed, "decide() must report the stub execution"
    assert not rep.proposed
