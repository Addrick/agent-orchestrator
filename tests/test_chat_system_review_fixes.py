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
from tests.helpers import only_pending_token
from tests.test_chat_system import chat_system_with_mocks  # noqa: F401


# --- #4: _store_api_request eviction should be LRU, not FIFO ----------------

def test_store_api_request_eviction_is_lru(chat_system_with_mocks):
    """Re-storing a user's payload marks them most-recently-used, so a later
    eviction drops a stale user rather than the just-touched one."""
    system, *_ = chat_system_with_mocks
    cap = MAX_CACHED_API_REQUESTS

    for i in range(cap):
        system.turn_persistence.store_api_request(f"u{i}", "p", {"payload": i})
    assert len(system.turn_persistence.last_api_requests) == cap

    # Touch the earliest-inserted user — under LRU this must move it to MRU.
    system.turn_persistence.store_api_request("u0", "p", {"payload": "touched"})

    # One more distinct user tips us over capacity, forcing one eviction.
    system.turn_persistence.store_api_request(f"u{cap}", "p", {"payload": "new"})

    assert len(system.turn_persistence.last_api_requests) == cap
    assert "u0" in system.turn_persistence.last_api_requests, "touched user must survive (LRU)"
    assert "u1" not in system.turn_persistence.last_api_requests, "least-recently-used must be evicted"


def test_store_api_request_eviction_does_not_orphan_iterations(chat_system_with_mocks):
    """Eviction must drop the evicted user from both caches in lockstep."""
    system, *_ = chat_system_with_mocks
    cap = MAX_CACHED_API_REQUESTS
    for i in range(cap + 1):
        system.turn_persistence.store_api_request(f"v{i}", "p", {"payload": i}, is_first_iteration=True)
    # Whatever set of users remain, the two caches must agree on membership.
    assert set(system.turn_persistence.last_api_requests) == set(system.turn_persistence.last_api_iterations)


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
async def test_second_park_does_not_evict_the_first(chat_system_with_mocks):
    """Parking write B while A is still pending keeps BOTH.

    Inverts the pre-DP-297 contract. The store was keyed `(user, persona)` and
    held one park, so B silently superseded A; the best that could be done was
    to audit the eviction (`audit_parked_evicted`). Since parks are token-keyed
    they are siblings, so nothing is evicted and that event no longer exists.
    """
    system, mm, text_engine_mock, persona, _ = chat_system_with_mocks
    persona.set_execution_mode(ExecutionMode.CONFIRM)
    persona.set_enabled_tools(["*"])

    text_engine_mock.generate_response.side_effect = [
        ({"type": "tool_calls",
          "calls": [{"id": "A", "name": "update_ticket",
                     "arguments": {"ticket_id": 1, "state": "closed"}}]}, {}),
        ({"type": "text", "content": "Proposed A."}, {}),
    ]
    await system.generate_response("test_persona", "user", "channel", "close 1")
    first = [p.token for p in system.confirmations.list_for("user", "test_persona")]
    assert len(first) == 1

    mm.log_audit_event.reset_mock()

    # Park a second write WITHOUT resolving the first.
    text_engine_mock.generate_response.side_effect = [
        ({"type": "tool_calls",
          "calls": [{"id": "B", "name": "update_ticket",
                     "arguments": {"ticket_id": 2, "state": "closed"}}]}, {}),
        ({"type": "text", "content": "Proposed B."}, {}),
    ]
    await system.generate_response("test_persona", "user", "channel", "close 2")

    parks = system.confirmations.list_for("user", "test_persona")
    assert [p.write_call["id"] for p in parks] == ["A", "B"], \
        "the first pending write was lost when a second one parked"
    assert parks[0].token == first[0], "A's token must be stable across B's park"

    event_types = [c.kwargs.get("event_type") for c in mm.log_audit_event.call_args_list]
    assert "audit_parked_evicted" not in event_types, (
        f"nothing should be evicted any more; events={event_types}")


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


# --- DP-200 review: non-string content in client_messages must not crash ----

@pytest.mark.asyncio
async def test_prepare_request_tolerates_non_string_trailing_client_content(
        chat_system_with_mocks):
    """The trailing-user dedupe compares client content to ctx.message; a
    client-supplied turn with content=None or an OAI multimodal list must be
    kept as-is, not crash on .strip()."""
    system, mm, _, persona, _ = chat_system_with_mocks
    mm.get_channel_history.return_value = []

    for weird_content in (None, [{"type": "text", "text": "hi"}]):
        ctx = RequestContext(
            persona=persona, persona_name="test_persona",
            user_identifier="u", channel="c", message="hi",
            client_messages=[
                {"role": "user", "content": "earlier turn"},
                {"role": "assistant", "content": "earlier reply"},
                {"role": "user", "content": weird_content},
            ],
        )
        await system.request_builder.prepare_request(ctx, is_retry=False)
        # The non-string turn can't match ctx.message, so it stays and the
        # fresh user message is appended after it.
        assert ctx.conversation_history[-1] == {"role": "user", "content": "hi"}


# --- DP-200 review: retry + park must not overwrite the archived row --------

def test_retry_park_does_not_overwrite_archived_row(chat_system_with_mocks):
    """A retried turn that ends PENDING_CONFIRMATION must not UPDATE the
    archived assistant row with the ephemeral confirmation text — the park
    renders unpersisted (DP-130) and the resumed continuation commits the
    real text."""
    system, mm, _, _, _ = chat_system_with_mocks

    rid = system.turn_persistence.commit_or_update_assistant(
        persona_name="test_persona", user_identifier="u", channel="c",
        server_id=None, final_text="I'd like to perform the following actions:",
        response_type=ResponseType.PENDING_CONFIRMATION,
        user_interaction_id=None, retry_assistant_id=42,
        tool_context_json=None,
    )
    assert rid is None
    mm.update_interaction_content.assert_not_called()


# --- DP-200 review: retry linkage must survive the park/resume cycle --------

@pytest.mark.asyncio
async def test_retried_turn_that_gates_a_write_updates_the_archived_row(
        chat_system_with_mocks):
    """A retried turn that gates a write still lands on the archived row.

    The retry linkage no longer has to survive the park, which is the point:
    the gating turn now ends with its own text and commits immediately, so it
    consumes `retry_assistant_id` itself. The continuation that follows an
    approval is a *separate* turn and correctly writes its own row — where the
    pre-DP-297 park deferred everything to the resume, which then had to carry
    the linkage across.
    """
    system, mm, text_engine_mock, persona, tool_manager_mock = chat_system_with_mocks
    persona.set_execution_mode(ExecutionMode.CONFIRM)
    persona.set_enabled_tools(["*"])
    mm.handle_portal_retry.return_value = 42
    mm.get_channel_history.return_value = []

    # Retry turn proposes a write, then finishes with text.
    text_engine_mock.generate_response.side_effect = [
        ({"type": "tool_calls",
          "calls": [{"id": "c1", "name": "update_ticket",
                     "arguments": {"state": "closed"}}]}, {}),
        ({"type": "text", "content": "I've proposed closing it."}, {}),
    ]
    async for _ in system.stream_response(
            "test_persona", "user", "channel", "regenerate", is_retry=True):
        pass

    token = only_pending_token(system, "user", "test_persona")
    # The retried turn UPDATEd the archived row rather than inserting beside it.
    mm.update_interaction_content.assert_called_once()
    assert mm.update_interaction_content.call_args.args[0] == 42
    assistant_inserts = [c for c in mm.log_message.call_args_list
                         if c.kwargs.get("author_role") == "assistant"]
    assert assistant_inserts == []
    # And that row is the one the proposal will patch when it resolves.
    assert system.confirmations.pending[token].parked_assistant_id == 42

    # Approving runs a fresh continuation turn, which gets its own row.
    mm.update_interaction_content.reset_mock()
    tool_manager_mock.execute_tool.return_value = {"ok": True}
    text_engine_mock.generate_response.side_effect = None
    text_engine_mock.generate_response.return_value = (
        {"type": "text", "content": "Done, ticket closed."}, {})
    _, rtype, _, _ = await system.resolve_park(
        "user", "test_persona", token, approved=True)

    assert rtype == ResponseType.LLM_GENERATION
    mm.update_interaction_content.assert_not_called()


# --- DP-200 review: RequestBuilder must see a post-init tool_manager swap ---

def test_request_builder_sees_post_init_tool_manager_swap(chat_system_with_mocks):
    """Tool filtering resolves the tool manager per call (lookup closure, like
    ConfirmationManager): after a post-init rebind of chat_system.tool_manager,
    the request must offer the model the new manager's tools."""
    from unittest.mock import MagicMock

    system, _, _, persona, original_tm = chat_system_with_mocks
    persona.set_enabled_tools(["*"])

    swapped_tm = MagicMock()
    swapped_tm.get_tool_definitions.return_value = []
    system.tool_manager = swapped_tm

    system.request_builder.filter_tools_for_persona(persona)

    swapped_tm.get_tool_definitions.assert_called_once()
    original_tm.get_tool_definitions.assert_not_called()


# --- #15: _conversation_taints must be bounded ------------------------------

def test_conversation_taints_is_bounded(chat_system_with_mocks):
    """The sticky-taint map must not grow without bound across distinct
    (user, persona, channel, server) tuples."""
    from src.request_builder import MAX_CONVERSATION_TAINTS
    system, *_ = chat_system_with_mocks
    for i in range(MAX_CONVERSATION_TAINTS + 50):
        system.request_builder.set_conversation_taint((f"u{i}", "p", "c", None), True)
    assert len(system.request_builder.conversation_taints) <= MAX_CONVERSATION_TAINTS


# --- DP-296 review: retry paths must land tool context without blanking text --

def test_retry_with_empty_text_does_not_blank_the_canonical_row(chat_system_with_mocks):
    """The error path commits whatever prose accumulated, which is "" whenever
    the model emitted tool calls and died before writing any. On a portal retry
    that took the UPDATE branch and wrote content="" + reasoning=None, which
    fails `_is_renderable` — dropping the message *and* its version chevron from
    the transcript and stranding the archived original."""
    system, mm, _, _, _ = chat_system_with_mocks

    rid = system.turn_persistence.commit_or_update_assistant(
        persona_name="test_persona", user_identifier="u", channel="c",
        server_id=None, final_text="",
        response_type=ResponseType.LLM_GENERATION,
        user_interaction_id=None, retry_assistant_id=42,
        tool_context_json='[{"role": "tool", "name": "get_ticket"}]',
    )

    assert rid == 42
    mm.update_interaction_content.assert_not_called()
    mm.set_tool_context.assert_called_once_with(
        42, '[{"role": "tool", "name": "get_ticket"}]')


def test_retry_park_still_records_its_tool_context(chat_system_with_mocks):
    """DP-296's park row is INSERTed on the non-retry path only, so "retry a turn
    → model proposes a write → operator never answers" used to leave zero trace
    and a None parked_assistant_id. Attach the sealed context to the archived
    row instead — without touching the prior attempt's text, which that row
    still renders."""
    system, mm, _, _, _ = chat_system_with_mocks
    sealed = ('[{"role": "assistant", "tool_calls": [{"id": "c1"}]}, '
              '{"role": "tool", "tool_call_id": "c1", "content": "{}"}]')

    rid = system.turn_persistence.commit_or_update_assistant(
        persona_name="test_persona", user_identifier="u", channel="c",
        server_id=None, final_text="I'd like to perform the following actions:",
        response_type=ResponseType.PENDING_CONFIRMATION,
        user_interaction_id=None, retry_assistant_id=42,
        tool_context_json=sealed,
    )

    assert rid == 42, "the park must report a row the resume can later clear"
    mm.update_interaction_content.assert_not_called()
    mm.set_tool_context.assert_called_once_with(42, sealed)


# --- DP-297 review #1: a parked turn that then exhausts MAX_TOOL_CALLS -------

def test_non_llm_turn_with_tool_context_is_persisted(chat_system_with_mocks):
    """The DP-296 rescue keyed off PENDING_CONFIRMATION, which DP-297 stopped
    producing anywhere. The max-iterations exit emits DEV_COMMAND plus a sealed
    span holding every `awaiting_human_approval` entry, so it fell through to
    `return None` — dropping the only row those parks can ever be patched
    through."""
    system, mm, _, _, _ = chat_system_with_mocks
    mm.log_message.return_value = 999
    sealed = ('[{"role": "assistant", "tool_calls": [{"id": "c1"}]}, '
              '{"role": "tool", "tool_call_id": "c1", '
              '"content": "{\\"status\\": \\"awaiting_human_approval\\"}"}]')

    rid = system.turn_persistence.commit_or_update_assistant(
        persona_name="test_persona", user_identifier="u", channel="c",
        server_id=None,
        final_text="I seem to be stuck in a loop. Could you please clarify?",
        response_type=ResponseType.DEV_COMMAND,
        user_interaction_id=None, retry_assistant_id=None,
        tool_context_json=sealed,
    )

    assert rid == 999
    mm.log_message.assert_called_once()
    kwargs = mm.log_message.call_args.kwargs
    assert kwargs["tool_context"] == sealed
    # Only PENDING_CONFIRMATION blanks its text (DP-130). This turn's prose is
    # what the user actually saw and has to survive.
    assert kwargs["content"].startswith("I seem to be stuck")


def test_non_llm_turn_without_tool_context_still_skipped(chat_system_with_mocks):
    """The widened rescue must stay keyed on there being a context to preserve —
    an ordinary dev command has no proposals and must not start landing rows."""
    system, mm, _, _, _ = chat_system_with_mocks

    rid = system.turn_persistence.commit_or_update_assistant(
        persona_name="test_persona", user_identifier="u", channel="c",
        server_id=None, final_text="Current model: mock_model",
        response_type=ResponseType.DEV_COMMAND,
        user_interaction_id=None, retry_assistant_id=None,
        tool_context_json=None,
    )

    assert rid is None
    mm.log_message.assert_not_called()


def test_register_parks_drops_parks_when_the_row_is_missing(
        chat_system_with_mocks, caplog):
    """`log_message` can still fail. Registering anyway binds the park to
    `parked_assistant_id=None`, so approving it executes a real write while
    `patch_parked_entry` silently no-ops — the action happens and history never
    mentions it. Failing closed costs a refused click instead."""
    from src.confirmations import ParkedWrite, new_token
    system, *_ = chat_system_with_mocks
    parked = ParkedWrite(
        token=new_token(),
        write_call={"id": "c1", "name": "create_ticket", "arguments": {}},
        audit_info={"actions": []},
        confirmation_text="create a ticket?",
        user_identifier="u",
        persona_name="test_persona",
    )

    with caplog.at_level(logging.ERROR):
        system._register_parks([parked], None)

    assert system.confirmations.pending == {}
    assert parked.token not in system.confirmations.pending
    assert "dropping" in caplog.text


# --- DP-297 review #4/#13: the pre-apply guards in stream_resolve_park -------

@pytest.mark.asyncio
async def test_missing_persona_restores_the_park(chat_system_with_mocks):
    """`take()` has already claimed the park by the time this guard runs. The
    wrong-conversation branch above it restores; this one dropped the object on
    the floor, destroying both the proposal and the operator's decision — the
    write never ran, and its history entry read `awaiting_human_approval`
    forever with no park left for the duplicate guard to match."""
    from src.confirmations import ParkedWrite, new_token
    system, *_ = chat_system_with_mocks
    parked = ParkedWrite(
        token=new_token(),
        write_call={"id": "c1", "name": "update_ticket", "arguments": {}},
        audit_info={"actions": []},
        confirmation_text="update?",
        user_identifier="u",
        persona_name="test_persona",
    )
    system.confirmations.park(parked)
    # The persona is renamed / removed while the write sits parked.
    system.personas = {}

    text, _, _, _ = await system.resolve_park(
        "u", "test_persona", parked.token, approved=True)

    assert "Persona not found" in text
    assert system.confirmations.pending.get(parked.token) is parked, (
        "the park must survive so it is still resolvable once the persona is"
    )


@pytest.mark.asyncio
async def test_expiry_at_click_audits_like_the_sweep(chat_system_with_mocks):
    """Clicking approve on a park that aged out past its TTL but was not yet
    swept is the one park-terminating path that logged nothing — so the fact
    that a human tried to approve an expired irreversible action was recorded
    nowhere. It also hardcoded the literal "expired" instead of the constant
    every other consumer keys off."""
    import time
    from config.global_config import PENDING_ACTION_TTL
    from src.confirmations import ParkedWrite, new_token
    from src.tools.tool_loop import PARK_STATUS_EXPIRED
    system, mm, _, _, _ = chat_system_with_mocks
    parked = ParkedWrite(
        token=new_token(),
        write_call={"id": "c1", "name": "update_ticket", "arguments": {}},
        audit_info={"actions": []},
        confirmation_text="update?",
        user_identifier="u",
        persona_name="test_persona",
        created_at=time.time() - PENDING_ACTION_TTL - 10,
    )
    system.confirmations.pending[parked.token] = parked
    system.confirmations._by_key.setdefault(
        ("u", "test_persona"), []).append(parked.token)

    text, _, _, _ = await system.resolve_park(
        "u", "test_persona", parked.token, approved=True)

    assert "expired" in text
    kinds = [c.kwargs.get("event_type")
             for c in mm.log_audit_event.call_args_list]
    assert "audit_park_expired" in kinds
    states = [c.kwargs.get("new_state")
              for c in mm.log_audit_event.call_args_list]
    assert PARK_STATUS_EXPIRED in states


# --- DP-297 review #6: a failed history patch must not be silent ------------

@pytest.mark.asyncio
async def test_unpatchable_decision_is_stated_in_the_nudge(
        chat_system_with_mocks):
    """`patch_parked_entry` returns False on four reachable conditions, and
    `apply()` discarded that. The write has already run, so the continuation
    rebuilds history, reads its own proposal as still pending, and summarizes
    the wrong outcome. The nudge is the only channel left once the durable one
    failed."""
    from src.confirmations import Decision, ParkedWrite, new_token
    from src.chat_system import _render_resolution_nudge
    system, *_ = chat_system_with_mocks
    parked = ParkedWrite(
        token=new_token(),
        write_call={"id": "c1", "name": "update_ticket", "arguments": {}},
        audit_info={"actions": []},
        confirmation_text="update?",
        user_identifier="u",
        persona_name="test_persona",
        parked_assistant_id=None,   # nothing to patch → patch returns False
    )
    system.confirmations.park(parked)
    decision = Decision(park=parked, approved=True)

    await system.confirmations.apply(decision)

    assert decision.patched is False
    assert "history entry could not be updated" in \
        _render_resolution_nudge([decision])


def test_nudge_stays_clean_when_the_patch_succeeded():
    """The warning must not appear on the ordinary path — it would train the
    model to distrust a tool context that is in fact correct."""
    from src.confirmations import Decision, ParkedWrite, new_token
    from src.chat_system import _render_resolution_nudge
    parked = ParkedWrite(
        token=new_token(), write_call={"id": "c1", "name": "update_ticket"},
        audit_info={}, confirmation_text="", user_identifier="u",
        persona_name="p",
    )
    nudge = _render_resolution_nudge(
        [Decision(park=parked, approved=True, ok=True, patched=True)])
    assert "could not be updated" not in nudge
    assert "approved and executed" in nudge
