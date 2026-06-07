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

from src.chat_system import (
    ChatSystem, ResponseType, PendingConfirmation,
)
from src.persona import Persona, ExecutionMode
from src.tools.turn_context import get_turn_context
from memory.memory_manager import MemoryManager
from src.engine import TextEngine

# Reuse the shared fixture
from tests.test_chat_system import chat_system_with_mocks  # noqa: F401


# --- CONFIRM resume / deny edges ------------------------------------------

@pytest.mark.asyncio
async def test_confirm_deny_then_retry_creates_new_pending(chat_system_with_mocks):
    """After denying a pending write, re-issuing the same request should
    park a NEW PendingConfirmation (no stale state from the prior deny)."""
    system, _, text_engine_mock, persona, tool_manager_mock = chat_system_with_mocks
    persona.set_execution_mode(ExecutionMode.CONFIRM)
    persona.set_enabled_tools(['*'])

    # First turn: LLM returns a write call → park pending
    tool_call = {'type': 'tool_calls',
                 'calls': [{'id': 'call_1', 'name': 'update_ticket',
                            'arguments': {'ticket_id': 1, 'state': 'closed'}}]}
    text_engine_mock.generate_response.return_value = (tool_call, {})
    await system.generate_response('test_persona', 'user', 'channel', 'close it')
    assert ('user', 'test_persona') in system._pending_confirmations

    # Deny the pending
    final_response = {'type': 'text', 'content': 'OK, not closing.'}
    text_engine_mock.generate_response.return_value = (final_response, {})
    await system.resume_pending_confirmation('user', 'test_persona', approved=False)
    # Resume must have cleared the slot
    assert ('user', 'test_persona') not in system._pending_confirmations

    # Now ask again → fresh tool call → fresh pending
    tool_call2 = {'type': 'tool_calls',
                  'calls': [{'id': 'call_2', 'name': 'update_ticket',
                             'arguments': {'ticket_id': 1, 'state': 'closed'}}]}
    text_engine_mock.generate_response.return_value = (tool_call2, {})
    _, response_type, _, _ = await system.generate_response(
        'test_persona', 'user', 'channel', 'really close it')
    assert response_type == ResponseType.PENDING_CONFIRMATION
    new_pending = system._pending_confirmations.get(('user', 'test_persona'))
    assert new_pending is not None
    assert new_pending.write_calls[0]['id'] == 'call_2'


@pytest.mark.asyncio
async def test_confirm_deny_resume_max_iterations(chat_system_with_mocks):
    """After denial, the post-resume LLM call still respects max-iterations
    semantics: if the model immediately produces another write, that lands
    via the *next* generate_response call, not via resume itself.
    Resume should return a clean LLM_GENERATION with the denial-aware text."""
    system, _, text_engine_mock, persona, tool_manager_mock = chat_system_with_mocks
    persona.set_execution_mode(ExecutionMode.CONFIRM)
    persona.set_enabled_tools(['*'])

    # Park
    tool_call = {'type': 'tool_calls',
                 'calls': [{'id': 'call_1', 'name': 'update_ticket',
                            'arguments': {'ticket_id': 1, 'state': 'closed'}}]}
    text_engine_mock.generate_response.return_value = (tool_call, {})
    await system.generate_response('test_persona', 'user', 'channel', 'close it')

    # Resume denied — even if LLM tries to call another write tool, resume
    # returns text (resume path uses generate_response which yields the text
    # content of whatever the LLM emits; it does not re-enter the tool loop).
    persistent = {'type': 'text', 'content': 'Acknowledged denial.'}
    text_engine_mock.generate_response.return_value = (persistent, {})
    _, response_type, _, _ = await system.resume_pending_confirmation(
        'user', 'test_persona', approved=False)
    assert response_type == ResponseType.LLM_GENERATION
    tool_manager_mock.execute_tool.assert_not_called()
    # No new pending was parked by resume
    assert ('user', 'test_persona') not in system._pending_confirmations


@pytest.mark.asyncio
async def test_resume_minimal_conversation_history(chat_system_with_mocks):
    """Resume must work even when the parked conversation_history is
    minimal (only the assistant tool_calls turn, no prior user/system).
    Approval should still execute the write and return LLM text."""
    system, _, text_engine_mock, persona, tool_manager_mock = chat_system_with_mocks
    persona.set_execution_mode(ExecutionMode.CONFIRM)
    persona.set_enabled_tools(['*'])

    # Hand-construct a minimal PendingConfirmation and drop it in.
    pending = PendingConfirmation(
        write_calls=[{'id': 'w1', 'name': 'update_ticket',
                      'arguments': {'ticket_id': 7, 'state': 'closed'}}],
        conversation_history=[
            {'role': 'assistant', 'tool_calls': [
                {'id': 'w1', 'name': 'update_ticket',
                 'arguments': {'ticket_id': 7, 'state': 'closed'}}
            ]},
        ],
        persona_name='test_persona',
        tools_for_llm=[],
        image_url=None,
        channel='channel',
        server_id=None,
        turn_tainted=False,
        audit_info={'actions': [], 'tainted': False, 'taint_sources': [],
                    'model_reasoning': None, 'execution_mode': 'CONFIRM'},
    )
    system._pending_confirmations[('user', 'test_persona')] = pending

    tool_manager_mock.execute_tool.return_value = {'result': 'closed'}
    text_engine_mock.generate_response.return_value = (
        {'type': 'text', 'content': 'Done.'}, {},
    )

    response, response_type, _, _ = await system.resume_pending_confirmation(
        'user', 'test_persona', approved=True)
    assert response_type == ResponseType.LLM_GENERATION
    assert response == 'Done.'
    tool_manager_mock.execute_tool.assert_called_once_with(
        'update_ticket', ticket_id=7, state='closed')


@pytest.mark.asyncio
async def test_confirmation_timeout_boundary(chat_system_with_mocks):
    """If `time.time() - pending.created_at > PENDING_CONFIRMATION_TIMEOUT`
    at resume entry, resume must reject with the expired text — even if
    the user clicked approve. (Race: timer crossed boundary before resume.)"""
    system, _, text_engine_mock, persona, tool_manager_mock = chat_system_with_mocks
    persona.set_execution_mode(ExecutionMode.CONFIRM)
    persona.set_enabled_tools(['*'])

    # Park
    tool_call = {'type': 'tool_calls',
                 'calls': [{'id': 'call_1', 'name': 'update_ticket',
                            'arguments': {'ticket_id': 1, 'state': 'closed'}}]}
    text_engine_mock.generate_response.return_value = (tool_call, {})
    await system.generate_response('test_persona', 'user', 'channel', 'close it')

    pending = system._pending_confirmations[('user', 'test_persona')]
    # Backdate the parked timestamp far enough that any reasonable
    # PENDING_CONFIRMATION_TIMEOUT is exceeded.
    pending.created_at = time.time() - (60 * 60 * 24 * 365)  # 1 year ago

    response, response_type, _, _ = await system.resume_pending_confirmation(
        'user', 'test_persona', approved=True)
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
    real_prepare = system._prepare_request

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

    with patch.object(system, '_prepare_request', side_effect=capture_in_prepare):
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
