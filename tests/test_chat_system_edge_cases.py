# tests/test_chat_system_edge_cases.py
"""DP-199 Batch 1 — ChatSystem orchestrator edge cases.

CONFIRM-mode resume edges, concurrent-turn ContextVar isolation, and
timeout-boundary checks. Uses the shared `chat_system_with_mocks` fixture
from tests/test_chat_system.py.

No production-code changes — bugs noted in DP-199-edge-cases.md
"Latent-bug fix list" are skipped, not patched.
"""

import asyncio
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.chat_system import ResponseType
from src.confirmations import ParkedWrite
from src.persona import Persona, ExecutionMode
from src.tools.turn_context import get_turn_context
from memory.memory_manager import MemoryManager
from src.engine import TextEngine

# Reuse the shared fixture
from tests.helpers import only_pending_token, pending_tokens
from tests.test_chat_system import chat_system_with_mocks  # noqa: F401


# --- CONFIRM resume / deny edges ------------------------------------------

@pytest.mark.asyncio
async def test_confirm_deny_then_retry_creates_new_pending(chat_system_with_mocks):
    """After denying a gated write, re-issuing the same request parks a NEW
    proposal with its own token (no stale state from the prior deny)."""
    system, _, text_engine_mock, persona, tool_manager_mock = chat_system_with_mocks
    persona.set_execution_mode(ExecutionMode.CONFIRM)
    persona.set_enabled_tools(['*'])

    # First turn: LLM returns a write call → gate it
    text_engine_mock.generate_response.side_effect = [
        ({'type': 'tool_calls',
          'calls': [{'id': 'call_1', 'name': 'update_ticket',
                     'arguments': {'ticket_id': 1, 'state': 'closed'}}]}, {}),
        ({'type': 'text', 'content': 'Proposed.'}, {}),
    ]
    await system.generate_response('test_persona', 'user', 'channel', 'close it')
    first_token = only_pending_token(system, 'user', 'test_persona')

    # Deny it
    text_engine_mock.generate_response.side_effect = None
    text_engine_mock.generate_response.return_value = (
        {'type': 'text', 'content': 'OK, not closing.'}, {})
    await system.resolve_park('user', 'test_persona', first_token, approved=False)
    assert pending_tokens(system, 'user', 'test_persona') == []

    # Now ask again → fresh tool call → fresh proposal, distinct token
    text_engine_mock.generate_response.side_effect = [
        ({'type': 'tool_calls',
          'calls': [{'id': 'call_2', 'name': 'update_ticket',
                     'arguments': {'ticket_id': 1, 'state': 'closed'}}]}, {}),
        ({'type': 'text', 'content': 'Proposed again.'}, {}),
    ]
    await system.generate_response(
        'test_persona', 'user', 'channel', 'really close it')

    parks = system.confirmations.list_for('user', 'test_persona')
    assert len(parks) == 1
    assert parks[0].write_call['id'] == 'call_2'
    assert parks[0].token != first_token


@pytest.mark.asyncio
async def test_confirm_deny_continuation_parks_nothing_new(chat_system_with_mocks):
    """A denial's continuation returns clean text and leaves nothing pending,
    provided the model doesn't propose again."""
    system, _, text_engine_mock, persona, tool_manager_mock = chat_system_with_mocks
    persona.set_execution_mode(ExecutionMode.CONFIRM)
    persona.set_enabled_tools(['*'])

    text_engine_mock.generate_response.side_effect = [
        ({'type': 'tool_calls',
          'calls': [{'id': 'call_1', 'name': 'update_ticket',
                     'arguments': {'ticket_id': 1, 'state': 'closed'}}]}, {}),
        ({'type': 'text', 'content': 'Proposed.'}, {}),
    ]
    await system.generate_response('test_persona', 'user', 'channel', 'close it')
    token = only_pending_token(system, 'user', 'test_persona')

    text_engine_mock.generate_response.side_effect = None
    text_engine_mock.generate_response.return_value = (
        {'type': 'text', 'content': 'Acknowledged denial.'}, {})
    _, response_type, _, _ = await system.resolve_park(
        'user', 'test_persona', token, approved=False)
    assert response_type == ResponseType.LLM_GENERATION
    tool_manager_mock.execute_tool.assert_not_called()
    assert pending_tokens(system, 'user', 'test_persona') == []


@pytest.mark.asyncio
async def test_continuation_may_propose_again(chat_system_with_mocks):
    """A continuation runs the full tool loop, so the model may gate ANOTHER
    write off the back of a resolved one. That new proposal must be a live,
    independently resolvable park — this chaining is what the old blocking
    resume could not express (the 2026-07-26 dangling-proposal bug)."""
    system, _, text_engine_mock, persona, tool_manager_mock = chat_system_with_mocks
    persona.set_execution_mode(ExecutionMode.CONFIRM)
    persona.set_enabled_tools(['*'])

    text_engine_mock.generate_response.side_effect = [
        ({'type': 'tool_calls',
          'calls': [{'id': 'call_1', 'name': 'update_ticket',
                     'arguments': {'ticket_id': 1, 'state': 'closed'}}]}, {}),
        ({'type': 'text', 'content': 'Proposed.'}, {}),
    ]
    await system.generate_response('test_persona', 'user', 'channel', 'close it')
    first = only_pending_token(system, 'user', 'test_persona')

    tool_manager_mock.execute_tool.return_value = {'result': 'closed'}
    text_engine_mock.generate_response.side_effect = [
        ({'type': 'tool_calls',
          'calls': [{'id': 'call_2', 'name': 'create_ticket',
                     'arguments': {'title': 'follow-up'}}]}, {}),
        ({'type': 'text', 'content': 'Closed it; proposed a follow-up.'}, {}),
    ]
    _, response_type, _, _ = await system.resolve_park(
        'user', 'test_persona', first, approved=True)

    assert response_type == ResponseType.LLM_GENERATION
    parks = system.confirmations.list_for('user', 'test_persona')
    assert len(parks) == 1
    assert parks[0].write_call['name'] == 'create_ticket'
    assert parks[0].token != first


@pytest.mark.asyncio
async def test_resolve_a_park_registered_directly(chat_system_with_mocks):
    """A park registered without any preceding turn still resolves.

    Guards the store's independence from the turn that created it: since
    DP-297 a ParkedWrite carries no conversation snapshot, so nothing about
    resolution may depend on one existing.
    """
    system, _, text_engine_mock, persona, tool_manager_mock = chat_system_with_mocks
    persona.set_execution_mode(ExecutionMode.CONFIRM)
    persona.set_enabled_tools(['*'])

    park = ParkedWrite(
        token='w1-token',
        write_call={'id': 'w1', 'name': 'update_ticket',
                    'arguments': {'ticket_id': 7, 'state': 'closed'}},
        audit_info={'actions': [], 'tainted': False, 'taint_sources': [],
                    'model_reasoning': None, 'execution_mode': 'CONFIRM'},
        confirmation_text='Close #7?',
        user_identifier='user',
        persona_name='test_persona',
        channel='channel',
    )
    system.confirmations.park(park)

    tool_manager_mock.execute_tool.return_value = {'result': 'closed'}
    text_engine_mock.generate_response.return_value = (
        {'type': 'text', 'content': 'Done.'}, {},
    )

    response, response_type, _, _ = await system.resolve_park(
        'user', 'test_persona', 'w1-token', approved=True)
    assert response_type == ResponseType.LLM_GENERATION
    assert response == 'Done.'
    tool_manager_mock.execute_tool.assert_called_once_with(
        'update_ticket', ticket_id=7, state='closed')


@pytest.mark.asyncio
async def test_expired_park_is_not_executed_on_approval(chat_system_with_mocks):
    """A park past its TTL must refuse even an explicit approval.

    Race the guard exists for: the deadline crossed between the operator
    seeing the affordance and clicking it.
    """
    system, _, text_engine_mock, persona, tool_manager_mock = chat_system_with_mocks
    persona.set_execution_mode(ExecutionMode.CONFIRM)
    persona.set_enabled_tools(['*'])

    text_engine_mock.generate_response.side_effect = [
        ({'type': 'tool_calls',
          'calls': [{'id': 'call_1', 'name': 'update_ticket',
                     'arguments': {'ticket_id': 1, 'state': 'closed'}}]}, {}),
        ({'type': 'text', 'content': 'Proposed.'}, {}),
    ]
    await system.generate_response('test_persona', 'user', 'channel', 'close it')
    token = only_pending_token(system, 'user', 'test_persona')

    # Backdate past any reasonable PENDING_ACTION_TTL.
    system.confirmations.pending[token].created_at = (
        time.time() - (60 * 60 * 24 * 365)  # 1 year ago
    )

    response, response_type, _, _ = await system.resolve_park(
        'user', 'test_persona', token, approved=True)
    assert response_type == ResponseType.DEV_COMMAND
    assert 'expired' in response.lower()
    # Critically, the write was NOT executed despite approval
    tool_manager_mock.execute_tool.assert_not_called()


# --- Concurrency / ContextVar isolation ------------------------------------

@pytest.mark.asyncio
async def test_concurrent_turns_context_isolation(chat_system_with_mocks):
    """Two _orchestrate calls running concurrently with different personas
    must each see their own TurnContext — no bleed via the ContextVar.

    Captures the TurnContext as observed mid-turn (during preprocess_message)
    from each task, then asserts the two captures are scope-isolated.
    """
    system, _, text_engine_mock, persona_a, _ = chat_system_with_mocks
    # Add a second persona so both turns can run with distinct scope
    persona_b = Persona('persona_b', 'mock_model', 'prompt_b')
    system.personas['persona_b'] = persona_b

    captured = {}

    # Hook _prepare_request (called AFTER set_turn_context in _orchestrate)
    # so we can observe each task's TurnContext from inside the active turn.
    real_prepare = system.request_builder.prepare_request

    async def capture_in_prepare(ctx, is_retry=False):
        captured[ctx.persona_name] = get_turn_context()
        # Yield so the two gathered tasks actually interleave on the event loop
        await asyncio.sleep(0)
        # Re-check after the await — confirms the ContextVar survived the
        # context-switch (ContextVar copy-on-task means each task's value
        # is preserved across awaits).
        captured[ctx.persona_name + "_after"] = get_turn_context()
        return await real_prepare(ctx, is_retry=is_retry)

    # Force the path past preprocess_message and persona lookup so we hit
    # _prepare_request, then raise to short-circuit before the LLM call.
    system.bot_logic.preprocess_message = AsyncMock(return_value=None)

    with patch.object(system.request_builder, 'prepare_request', side_effect=capture_in_prepare):
        # _prepare_request returns None; orchestrate continues into the
        # ToolLoop. Stub the ToolLoop's engine call to short-circuit cheaply.
        with patch('src.chat_system.ToolLoop') as MockLoop:
            async def empty_run(**kwargs):
                from src.tools.tool_loop import _LoopFinishedEvent
                yield _LoopFinishedEvent(
                    final_text="", response_type=ResponseType.LLM_GENERATION,
                )
            MockLoop.return_value.run = lambda **kw: empty_run(**kw)

            results = await asyncio.gather(
                system.generate_response('test_persona', 'user_a', 'ch_a', 'hi'),
                system.generate_response('persona_b', 'user_b', 'ch_b', 'hi'),
            )

    # Each turn's preprocess saw its own scope — no cross-task leakage
    assert captured['test_persona'] is not None
    assert captured['persona_b'] is not None
    assert captured['test_persona'].persona_name == 'test_persona'
    assert captured['test_persona'].user_identifier == 'user_a'
    assert captured['test_persona'].channel == 'ch_a'
    assert captured['persona_b'].persona_name == 'persona_b'
    assert captured['persona_b'].user_identifier == 'user_b'
    assert captured['persona_b'].channel == 'ch_b'

    # ContextVar survived the await inside each task — i.e. it did NOT
    # leak across to the other concurrent task's value.
    assert captured['test_persona_after'].persona_name == 'test_persona'
    assert captured['persona_b_after'].persona_name == 'persona_b'

    # And after both turns finish, the ContextVar must be back to None
    # (each turn called reset_turn_context in its finally path)
    assert get_turn_context() is None
