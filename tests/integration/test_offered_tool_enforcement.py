# tests/integration/test_offered_tool_enforcement.py
"""DP-404 end to end: the persona's policy and service bindings decide what a
call may execute, not just what the model is shown.

Real ChatSystem, real ToolManager, real `filter_tools_for_persona`. The model
names a tool that IS registered in the process but that this persona was not
given — the shape of an injected `<tool_call>`.
"""

import pytest

from src.persona import ExecutionMode
from tests.helpers import offer_tools

pytestmark = pytest.mark.integration


def _script(chat_system, *results):
    it = iter(results)

    async def fake_generate_response(*_a, **_k):
        return next(it)
    chat_system.text_engine.generate_response.side_effect = fake_generate_response


def _call(name, **arguments):
    return ({"type": "tool_calls",
             "calls": [{"id": "c1", "name": name, "arguments": arguments}]}, {})


def _text(content):
    return ({"type": "text", "content": content}, {})


def _recording(chat_system, name):
    ran = []

    async def handler(**kwargs):
        ran.append(kwargs)
        return {"status": "ran"}
    chat_system.tool_manager.register(name, handler)
    return ran


async def _turn(chat_system):
    return [ev async for ev in chat_system.stream_response(
        "test_persona", "user", "chan", "check the agent please",
    )]


@pytest.mark.asyncio
async def test_registered_tool_outside_the_persona_bindings_does_not_run(mocked_chat_system):
    chat_system, _ = mocked_chat_system
    persona = chat_system.personas["test_persona"]
    persona.set_execution_mode(ExecutionMode.AUTONOMOUS)
    persona.set_enabled_tools(["*"])
    # get_agent_status is `agents`-bound; the persona has no bindings, so the
    # wildcard does not reach it.
    ran = _recording(chat_system, "get_agent_status")

    _script(chat_system, _call("get_agent_status", agent_id="a"), _text("ok"))
    await _turn(chat_system)
    assert ran == []

    # Positive control: grant the binding and the identical call runs.
    offer_tools(chat_system, "test_persona", "get_agent_status")
    _script(chat_system, _call("get_agent_status", agent_id="a"), _text("ok"))
    await _turn(chat_system)
    assert ran == [{"agent_id": "a"}]


@pytest.mark.asyncio
async def test_registered_tool_outside_the_persona_allowlist_does_not_run(mocked_chat_system):
    chat_system, _ = mocked_chat_system
    persona = chat_system.personas["test_persona"]
    persona.set_execution_mode(ExecutionMode.AUTONOMOUS)
    persona.set_enabled_tools(["get_agent_history"])
    offer_tools(chat_system, "test_persona", "get_agent_history", "get_agent_status")
    ran = _recording(chat_system, "get_agent_status")

    _script(chat_system, _call("get_agent_status", agent_id="a"), _text("ok"))
    await _turn(chat_system)
    assert ran == []
