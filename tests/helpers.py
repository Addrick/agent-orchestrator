# tests/helpers.py
"""Shared construction helpers for tests (DP-201).

`make_chat_system` builds a ChatSystem directly from explicit dependencies —
no filesystem reads and no `patch('src.bootstrap.load_personas_from_file')`
ritual. Tests state exactly the personas/tools they need; anything omitted
gets a hermetic default.

Production code and entrypoints keep using `src.bootstrap.create_chat_system`;
tests whose *subject* is the bootstrap wiring (tests/integration/
test_startup_wiring.py) should keep exercising bootstrap too.
"""

from typing import Any, Dict, Optional, Set
from unittest.mock import MagicMock

from src.bootstrap import build_tool_manager
from src.chat_system import ChatSystem
from src.engine import TextEngine
from src.memory.memory_manager import MemoryManager
from src.message_handler import BotLogic
from src.origin import Origin
from src.persona import Persona

# DP-277: canonical origins for exercising the control-plane gate. Tests that
# drive dev commands as "the operator" pass OPERATOR_ORIGIN; gate tests pass
# ANON_ORIGIN (or build a specific Origin) to assert refusal.
OPERATOR_ORIGIN = Origin(transport="test", operator=True)
ANON_ORIGIN = Origin(transport="test", operator=False)


def make_chat_system(
    memory_manager: Optional[Any] = None,
    text_engine: Optional[Any] = None,
    personas: Optional[Dict[str, Persona]] = None,
    tool_manager: Optional[Any] = None,
    embedding_service: Optional[Any] = None,
    *,
    system_persona_names: Optional[Set[str]] = None,
    models_available: Optional[Dict[str, Any]] = None,
) -> ChatSystem:
    """Build a ChatSystem with explicit deps and hermetic defaults.

    Defaults: a spec'd MagicMock MemoryManager (with a stub `.backend`,
    which spec= would otherwise hide — it's an instance attribute), a spec'd
    MagicMock TextEngine, an empty persona map, an empty model catalog, and a
    real ToolManager with the core handlers registered (same registration as
    bootstrap, so tool dispatch behaves like production).
    """
    if memory_manager is None:
        memory_manager = MagicMock(spec=MemoryManager)
        memory_manager.backend = MagicMock()
    if text_engine is None:
        text_engine = MagicMock(spec=TextEngine)
    if personas is None:
        personas = {}
    if tool_manager is None:
        tool_manager = build_tool_manager(memory_manager, personas)
    return ChatSystem(
        memory_manager=memory_manager,
        text_engine=text_engine,
        embedding_service=embedding_service,
        personas=personas,
        system_persona_names=system_persona_names or set(),
        tool_manager=tool_manager,
        models_available=models_available,
    )


def engine_stream_events(result: Dict[str, Any],
                         payload: Optional[Dict[str, Any]] = None) -> list:
    """Unified-event list equivalent to a one-shot (result, api_payload) pair
    — the same synthesis `TextEngine._events_from_one_shot` performs."""
    events: list = [{"type": "api_payload", "payload": {} if payload is None else payload}]
    if result.get("type") == "tool_calls":
        events.append({"type": "tool_calls", "calls": list(result.get("calls", []))})
        events.append({"type": "done", "full_text": ""})
    else:
        text = result.get("content", "") or ""
        if text:
            events.append({"type": "text_delta", "text": text})
        events.append({"type": "done", "full_text": text})
    return events


def route_stream_through_generate_response(text_engine: Any) -> None:
    """DP-206b test bridge: make the engine's streaming pipeline consume a
    *mocked* ``generate_response``.

    Pre-cutover, ``stream_messages`` wrapped ``generate_response``, so dozens
    of tests scripted the LLM by replacing ``text_engine.generate_response``
    with an AsyncMock. The cutover routed the pipeline through the
    ``_stream_response`` policy driver instead. This bridge re-points the
    driver at ``generate_response`` (same ``(persona_config, history_object,
    tools, local_inference_config)`` argument shape), so those tests keep
    scripting — and asserting on — ``text_engine.generate_response``.

    Only install this on engines whose ``generate_response`` is replaced by a
    mock: the real ``generate_response`` drains ``_stream_response``, so
    bridging an unmocked engine would recurse.
    """
    async def _bridged_stream(persona_config, history_object, tools=None,
                              local_inference_config=None):
        result, payload = await text_engine.generate_response(
            persona_config, history_object, tools, local_inference_config,
        )
        for ev in engine_stream_events(result, payload):
            yield ev

    text_engine._stream_response = _bridged_stream


def make_bot_logic(state: Any) -> BotLogic:
    """Build a BotLogic over a mutable state bucket (DP-202 explicit deps).

    `state` is any object carrying the attributes BotLogic's deps read —
    `personas` (dict), `models_available` (dict), `last_api_requests` /
    `last_api_iterations` (dump caches; `state` itself stands in for
    TurnPersistence), `text_engine`, `tool_manager`, `memory_manager`,
    optionally `system_persona_names` (set, for visible_personas filtering).
    A plain MagicMock works: tests mutate the attributes and the closures
    dereference the live values, mirroring how ChatSystem wires production.
    """
    return BotLogic(
        personas=lambda: state.personas,
        visible_personas=lambda: {
            name: persona
            for name, persona in state.personas.items()
            if name not in (getattr(state, "system_persona_names", None) or set())
        },
        text_engine=lambda: state.text_engine,
        tool_manager=lambda: state.tool_manager,
        turn_persistence=state,
        memory_manager=state.memory_manager,
        get_models_available=lambda: state.models_available,
        set_models_available=lambda models: setattr(state, "models_available", models),
    )
