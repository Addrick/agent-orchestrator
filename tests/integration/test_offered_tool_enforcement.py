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

    # Positive control: widen the allowlist and the identical call runs, so
    # the refusal above was the allowlist and nothing else.
    persona.set_enabled_tools(["get_agent_history", "get_agent_status"])
    _script(chat_system, _call("get_agent_status", agent_id="a"), _text("ok"))
    await _turn(chat_system)
    assert ran == [{"agent_id": "a"}]


@pytest.mark.asyncio
async def test_write_naming_another_bank_is_refused_before_it_parks(
        mocked_chat_system, monkeypatch):
    """DP-475: a bound tool's own scope is fixed, so a write that names
    another bank is answered with a refusal and never parked — an approval
    card reading bank_id='managr' must not exist for a call that could only
    run in the caller's own bank."""
    from src.tools import definitions
    from src.tools.definitions import ALL_TOOL_DEFINITIONS, ToolDefinitionRegistry

    monkeypatch.setattr(
        definitions, "_REGISTRY", ToolDefinitionRegistry(ALL_TOOL_DEFINITIONS))
    name = "mcp__hindsight__delete_mental_model"
    definitions.register_tool_definition({
        "type": "function", "dynamic": True, "is_write": True,
        "service_binding": "mcp:hindsight",
        "capabilities": {"produces_untrusted": True, "irreversible": True,
                         "locality": "network", "sensitivity": "pii"},
        "bind_args": {"bank_id": "{PERSONA}"},
        "function": {"name": name, "description": "d", "parameters": {
            "type": "object", "properties": {"mental_model_id": {"type": "string"}},
        }},
    })
    chat_system, _ = mocked_chat_system
    persona = chat_system.personas["test_persona"]
    persona.set_enabled_tools([name])
    ran = _recording(chat_system, name)
    offer_tools(chat_system, "test_persona", name)

    _script(chat_system,
            _call(name, mental_model_id="x", bank_id="managr"), _text("ok"))
    events = await _turn(chat_system)
    assert ran == []
    assert chat_system.confirmations.pending == {}
    assert any("fixed to 'test_persona'" in (getattr(ev, "error", None) or "")
               for ev in events)
