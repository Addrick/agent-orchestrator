# tests/test_chat_system_review_fixes.py
"""Regression tests for the stale code-review findings on ChatSystem.

Each test targets one genuine bug confirmed still present in master and is
written to FAIL against the unpatched code, then pass once fixed:

- _store_api_request eviction is FIFO, not LRU (finding #4)
- empty/whitespace user message reaches the LLM prompt (finding #6)
- retry update never persists the new tool_context (finding #8)
- a second parked write silently overwrites an unresolved one (finding #9)
- the client-fallback log reports the fallback size as the DB row count (#14)
- _conversation_taints grows unbounded with no eviction (finding #15)

Uses the shared `chat_system_with_mocks` fixture from tests/test_chat_system.py.
"""

import logging

import pytest

from config.global_config import MAX_CACHED_API_REQUESTS
from src.chat_system import ResponseType
from src.request_builder import RequestContext
from src.persona import Persona, ExecutionMode


# Reuse the shared fixture
from tests.test_chat_system import chat_system_with_mocks  # noqa: F401


# --- #4: _store_api_request eviction should be LRU, not FIFO ----------------

def test_store_api_request_eviction_is_lru(chat_system_with_mocks):
    """Re-storing a user's payload marks them most-recently-used, so a later
    eviction drops a stale user rather than the just-touched one."""
    system, *_ = chat_system_with_mocks
    cap = MAX_CACHED_API_REQUESTS

    for i in range(cap):
        system.turn_persistence.store_api_request(f"u{i}", "p", {"payload": i})
    assert len(system.last_api_requests) == cap

    # Touch the earliest-inserted user — under LRU this must move it to MRU.
    system.turn_persistence.store_api_request("u0", "p", {"payload": "touched"})

    # One more distinct user tips us over capacity, forcing one eviction.
    system.turn_persistence.store_api_request(f"u{cap}", "p", {"payload": "new"})

    assert len(system.last_api_requests) == cap
    assert "u0" in system.last_api_requests, "touched user must survive (LRU)"
    assert "u1" not in system.last_api_requests, "least-recently-used must be evicted"


def test_store_api_request_eviction_does_not_orphan_iterations(chat_system_with_mocks):
    """Eviction must drop the evicted user from both caches in lockstep."""
    system, *_ = chat_system_with_mocks
    cap = MAX_CACHED_API_REQUESTS
    for i in range(cap + 1):
        system.turn_persistence.store_api_request(f"v{i}", "p", {"payload": i}, is_first_iteration=True)
    # Whatever set of users remain, the two caches must agree on membership.
    assert set(system.last_api_requests) == set(system.last_api_iterations)


# --- #6: empty/whitespace user message must not reach the LLM prompt --------

@pytest.mark.asyncio
async def test_prepare_request_skips_empty_user_message(chat_system_with_mocks):
    """A blank message (kobold-lite continue/prefetch) must not append a
    `{'role':'user','content':''}` turn to the LLM prompt — mirroring the
    DB-side guard in _log_user_turn."""
    system, mm, _, persona, _ = chat_system_with_mocks
    mm.get_channel_history.return_value = []

    ctx = RequestContext(
        persona=persona, persona_name="test_persona",
        user_identifier="u", channel="c", message="   ",
    )
    await system.request_builder.prepare_request(ctx, is_retry=False)

    empties = [m for m in ctx.conversation_history
               if m.get("role") == "user" and not (m.get("content") or "").strip()]
    assert empties == [], f"empty user turn leaked into prompt: {empties}"


@pytest.mark.asyncio
async def test_prepare_request_keeps_real_user_message(chat_system_with_mocks):
    """Guard must not over-fire: a real message is still appended."""
    system, mm, _, persona, _ = chat_system_with_mocks
    mm.get_channel_history.return_value = []

    ctx = RequestContext(
        persona=persona, persona_name="test_persona",
        user_identifier="u", channel="c", message="hello there",
    )
    await system.request_builder.prepare_request(ctx, is_retry=False)
    assert ctx.conversation_history[-1] == {"role": "user", "content": "hello there"}


# --- #8: retry must persist the regenerated turn's tool_context -------------

def test_retry_update_persists_tool_context(chat_system_with_mocks):
    """On retry, _commit_or_update_assistant must forward the new
    tool_context so the stored row's tool_context matches its new content."""
    system, mm, _, _, _ = chat_system_with_mocks
    mm.update_interaction_content.return_value = True

    rid = system.turn_persistence.commit_or_update_assistant(
        persona_name="test_persona", user_identifier="u", channel="c",
        server_id=None, final_text="regenerated answer",
        response_type=ResponseType.LLM_GENERATION,
        user_interaction_id=None, retry_assistant_id=42,
        tool_context_json='[{"role": "tool", "name": "get_ticket"}]',
    )
    assert rid == 42
    mm.update_interaction_content.assert_called_once()
    kwargs = mm.update_interaction_content.call_args.kwargs
    assert kwargs.get("tool_context") == '[{"role": "tool", "name": "get_ticket"}]'


# --- #9: overwriting an unresolved parked write must be audited -------------

@pytest.mark.asyncio
async def test_overwriting_pending_confirmation_is_audited(chat_system_with_mocks):
    """Parking write B while write A is still pending for the same
    (user, persona) evicts A — that eviction must emit an audit event."""
    system, mm, text_engine_mock, persona, _ = chat_system_with_mocks
    persona.set_execution_mode(ExecutionMode.CONFIRM)
    persona.set_enabled_tools(["*"])

    call_a = {"type": "tool_calls",
              "calls": [{"id": "A", "name": "update_ticket",
                         "arguments": {"ticket_id": 1, "state": "closed"}}]}
    text_engine_mock.generate_response.return_value = (call_a, {})
    await system.generate_response("test_persona", "user", "channel", "close 1")
    assert ("user", "test_persona") in system.confirmations.pending

    mm.log_audit_event.reset_mock()

    # Park a second write WITHOUT resolving the first.
    call_b = {"type": "tool_calls",
              "calls": [{"id": "B", "name": "update_ticket",
                         "arguments": {"ticket_id": 2, "state": "closed"}}]}
    text_engine_mock.generate_response.return_value = (call_b, {})
    await system.generate_response("test_persona", "user", "channel", "close 2")

    event_types = [c.kwargs.get("event_type") for c in mm.log_audit_event.call_args_list]
    assert "audit_parked_evicted" in event_types, (
        f"silent overwrite of pending A not audited; events={event_types}")


# --- #14: client-fallback log must report the real DB row count -------------

@pytest.mark.asyncio
async def test_client_fallback_log_reports_db_row_count(chat_system_with_mocks, caplog):
    """The fallback log must distinguish the discarded DB row count from the
    client-message count — they must not both render the fallback size."""
    system, mm, _, persona, _ = chat_system_with_mocks
    # DB returns 3 rows; client supplies a single (non-matching) message.
    mm.get_channel_history.return_value = [
        {"author_role": "user", "author_name": None, "content": f"db {i}",
         "interaction_id": i}
        for i in range(3)
    ]
    ctx = RequestContext(
        persona=persona, persona_name="test_persona",
        user_identifier="u", channel="c", message="brand new",
        client_messages=[{"role": "user", "content": "client only"}],
    )
    with caplog.at_level(logging.INFO):
        await system.request_builder.prepare_request(ctx, is_retry=False)

    msgs = [r.getMessage() for r in caplog.records]
    assert any("DB result (3 rows) discarded" in m for m in msgs), (
        f"DB row count mis-reported in fallback log; messages={msgs}")


# --- #15: _conversation_taints must be bounded ------------------------------

def test_conversation_taints_is_bounded(chat_system_with_mocks):
    """The sticky-taint map must not grow without bound across distinct
    (user, persona, channel, server) tuples."""
    from src.request_builder import MAX_CONVERSATION_TAINTS
    system, *_ = chat_system_with_mocks
    for i in range(MAX_CONVERSATION_TAINTS + 50):
        system.request_builder.set_conversation_taint((f"u{i}", "p", "c", None), True)
    assert len(system.request_builder.conversation_taints) <= MAX_CONVERSATION_TAINTS
