# tests/agents/test_managr_agent.py
"""Unit tests for ManagrAgent (DP-280 Phase 0 — read-only manager's report).

Everything mocked: no network, no DB. The Phase 0 contract under test:
- a cycle = board snapshot -> analyst briefs -> planner report -> digest
- the agent has NO write path: the only outbound side effect is
  notification_router.send
- degraded modes (empty board, missing personas, failed notification) resolve
  to explicit outcomes instead of raising
"""

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from config.global_config import (
    MANAGR_PLANNER_NAME,
    MANAGR_STALE_ANALYST_NAME,
    MANAGR_PATTERN_ANALYST_NAME,
)
from src.agents.managr_agent import ManagrAgent
from src.persona import Persona

CONFIG_DIR = Path(__file__).parent.parent.parent / "config"

TICKETS = [
    {"id": 1, "number": "10001", "title": "Printer offline at front desk",
     "created_at": "2026-07-01T10:00:00Z", "updated_at": "2026-07-02T10:00:00Z"},
    {"id": 2, "number": "10002", "title": "VPN drops every hour",
     "created_at": "2026-06-20T10:00:00Z", "updated_at": "2026-06-21T10:00:00Z"},
]


def _text(content):
    return ({"type": "text", "content": content}, {})


def _make_personas():
    return {
        name: Persona(persona_name=name, model_name="mock", prompt=f"{name} prompt")
        for name in (MANAGR_PLANNER_NAME, MANAGR_STALE_ANALYST_NAME,
                     MANAGR_PATTERN_ANALYST_NAME)
    }


def _make_agent(tickets=TICKETS, personas=None, send_result=True, agent_config=None):
    chat_system = MagicMock()
    chat_system.text_engine = MagicMock()
    chat_system.memory_manager = MagicMock()
    chat_system.memory_manager.log_agent_action = MagicMock(return_value=1)
    chat_system.memory_manager.get_relevant_agent_actions = MagicMock(return_value=[])
    chat_system.memory_manager.list_standing_orders = MagicMock(return_value=[])

    zammad = MagicMock()
    zammad.search_tickets = MagicMock(return_value=tickets)
    zammad.get_tags = MagicMock(return_value=[])

    router = MagicMock()
    router.send = AsyncMock(return_value=send_result)

    if agent_config is None:
        agent_config = {
            "notification_targets": [{"channel": "discord_dm", "recipient": "adrich"}],
            "_recipients": {"adrich": {"discord_user_id": "321"}},
        }

    with patch("src.agents.base.load_system_personas_from_file", return_value={}):
        agent = ManagrAgent(chat_system, zammad, router, agent_config=agent_config)

    chat_system.personas = personas if personas is not None else _make_personas()
    # Retain bridging is exercised in the integration test; here it would
    # trip over MagicMock rows.
    agent._retain_action_series = AsyncMock()
    return agent, chat_system, zammad, router


def _final_outcome(agent):
    """(outcome, payload) of the last update_agent_action_outcome on the root."""
    calls = agent.memory_manager.update_agent_action_outcome.call_args_list
    assert calls, "no outcome was finalized"
    args = calls[-1].args
    payload = json.loads(args[2]) if args[2] else {}
    return args[1], payload


@pytest.mark.asyncio
async def test_deploy_happy_path_sends_plan_digest():
    agent, chat_system, zammad, router = _make_agent()
    chat_system.text_engine.generate_response = AsyncMock(side_effect=[
        _text("stale brief"), _text("patterns brief"), _text("THE PLAN"),
    ])
    agent.text_engine = chat_system.text_engine

    await agent.deploy()

    # Two analyst calls + one planner call
    assert chat_system.text_engine.generate_response.await_count == 3
    # Digest went out with the plan as the body, to the resolved discord id
    router.send.assert_awaited_once()
    kwargs = router.send.await_args.kwargs
    assert kwargs["body"] == "THE PLAN"
    assert kwargs["channel"] == "discord_dm"
    assert kwargs["recipient"] == "321"
    assert "Manager's Report" in kwargs["subject"]

    outcome, payload = _final_outcome(agent)
    assert outcome == "success"
    assert payload["plan_excerpt"] == "THE PLAN"
    assert payload["briefs"] == ["patterns", "stale"]
    agent._retain_action_series.assert_awaited_once()


@pytest.mark.asyncio
async def test_planner_prompt_includes_board_and_briefs():
    agent, chat_system, zammad, router = _make_agent()
    chat_system.text_engine.generate_response = AsyncMock(side_effect=[
        _text("stale brief"), _text("patterns brief"), _text("THE PLAN"),
    ])
    agent.text_engine = chat_system.text_engine

    await agent.deploy()

    planner_call = chat_system.text_engine.generate_response.await_args_list[-1]
    prompt = planner_call.kwargs["history_object"]["message_history"][-1]["content"]
    assert "#10001 Printer offline at front desk" in prompt
    assert "ANALYST BRIEF (stale):\nstale brief" in prompt
    assert "ANALYST BRIEF (patterns):\npatterns brief" in prompt
    # Read-only agent: no tools offered to any call
    for call in chat_system.text_engine.generate_response.await_args_list:
        assert call.kwargs["tools"] is None


@pytest.mark.asyncio
async def test_no_open_tickets_skips_cycle():
    agent, chat_system, zammad, router = _make_agent(tickets=[])
    chat_system.text_engine.generate_response = AsyncMock()
    agent.text_engine = chat_system.text_engine

    await agent.deploy()

    chat_system.text_engine.generate_response.assert_not_awaited()
    router.send.assert_not_awaited()
    outcome, payload = _final_outcome(agent)
    assert outcome == "skipped"


@pytest.mark.asyncio
async def test_missing_planner_persona_fails_without_digest():
    personas = _make_personas()
    del personas[MANAGR_PLANNER_NAME]
    agent, chat_system, zammad, router = _make_agent(personas=personas)
    chat_system.text_engine.generate_response = AsyncMock(side_effect=[
        _text("stale brief"), _text("patterns brief"),
    ])
    agent.text_engine = chat_system.text_engine

    await agent.deploy()

    router.send.assert_not_awaited()
    outcome, _ = _final_outcome(agent)
    assert outcome == "failed"


@pytest.mark.asyncio
async def test_missing_analyst_still_produces_plan():
    personas = _make_personas()
    del personas[MANAGR_STALE_ANALYST_NAME]
    agent, chat_system, zammad, router = _make_agent(personas=personas)
    chat_system.text_engine.generate_response = AsyncMock(side_effect=[
        _text("patterns brief"), _text("THE PLAN"),
    ])
    agent.text_engine = chat_system.text_engine

    await agent.deploy()

    router.send.assert_awaited_once()
    outcome, payload = _final_outcome(agent)
    assert outcome == "success"
    assert payload["briefs"] == ["patterns"]


@pytest.mark.asyncio
async def test_notification_failure_is_surfaced():
    agent, chat_system, zammad, router = _make_agent(send_result=False)
    chat_system.text_engine.generate_response = AsyncMock(side_effect=[
        _text("stale brief"), _text("patterns brief"), _text("THE PLAN"),
    ])
    agent.text_engine = chat_system.text_engine

    await agent.deploy()

    outcome, _ = _final_outcome(agent)
    assert outcome == "notification_failed"


@pytest.mark.asyncio
async def test_no_notification_targets_config_key():
    """Old config files without notification_targets must not crash the cycle."""
    agent, chat_system, zammad, router = _make_agent(agent_config={})
    chat_system.text_engine.generate_response = AsyncMock(side_effect=[
        _text("stale brief"), _text("patterns brief"), _text("THE PLAN"),
    ])
    agent.text_engine = chat_system.text_engine

    await agent.deploy()

    router.send.assert_not_awaited()
    outcome, _ = _final_outcome(agent)
    assert outcome == "notification_failed"


def test_format_ticket_line_is_defensive():
    agent, *_ = _make_agent()
    now = datetime(2026, 7, 4, 12, 0, tzinfo=timezone.utc)
    # Full search payloads carry expanded state/priority; bare ones don't.
    full = agent._format_ticket_line(
        {"number": "1", "title": "T", "state": "open", "priority": "3 high",
         "created_at": "2026-07-01T12:00:00Z", "updated_at": "2026-07-04T09:00:00Z"},
        now, tags=[],
    )
    assert full == "- #1 T | state=open | priority=3 high | age=3d | last_update=3h"
    bare = agent._format_ticket_line({"id": 7}, now, tags=[])
    assert bare == "- #7 No Title"
    # tags omitted/None = tag state unknown = title withheld (fail-closed)
    unknown = agent._format_ticket_line({"id": 7, "title": "bait"}, now)
    assert "bait" not in unknown
    assert "title withheld" in unknown


def test_resolve_recipient_mappings():
    agent, *_ = _make_agent()
    assert agent._resolve_recipient("discord_dm", "adrich") == "321"
    assert agent._resolve_recipient("discord_dm", "12345") == "12345"
    assert agent._resolve_recipient("discord_dm", "unknown") == "unknown"


# --- Config contract guards ---

def test_agents_json_has_managr_entry():
    config = json.loads((CONFIG_DIR / "agents.json").read_text())
    managr = config["agents"]["managr"]
    assert managr["persona"] == MANAGR_PLANNER_NAME
    # Live smoketest posture (Adam, 2026-07-06): managr is propose-only, so
    # it auto-starts — one cycle at container startup, then daily. Writes
    # still require human approval via the proposal queue.
    assert managr["auto_start"] is True, "managr should auto-start for the daily report"
    assert "daily_at" in managr["schedule"]
    assert managr["notification_targets"], "digest needs at least one target"


def test_system_personas_define_managr_fleet():
    config = json.loads((CONFIG_DIR / "system_personas.json").read_text())
    by_name = {p["name"]: p for p in config["personas"]}
    for name in (MANAGR_PLANNER_NAME, MANAGR_STALE_ANALYST_NAME,
                 MANAGR_PATTERN_ANALYST_NAME):
        assert name in by_name, f"missing system persona {name}"
        # Neutered: analysis personas must never carry tools
        assert by_name[name]["enabled_tools"] == []


def test_managr_registered_at_startup():
    """CLAUDE.md startup-registration rule: the agent must actually be wired."""
    from src.main import _register_agents

    manager = MagicMock()
    _register_agents(manager, zammad_client=MagicMock())
    registered = {c.args[0]: c.args[1] for c in manager.register.call_args_list}
    assert registered.get("managr") is ManagrAgent

    manager_no_zammad = MagicMock()
    _register_agents(manager_no_zammad, zammad_client=None)
    names = {c.args[0] for c in manager_no_zammad.register.call_args_list}
    assert "managr" not in names


# --- Phase 1: proposal extraction (DP-282) ---

def _tool_calls(proposals):
    return ({"type": "tool_calls",
             "calls": [{"name": "submit_proposals",
                        "arguments": {"proposals": proposals}}]}, {})


def _proposals_config():
    return {
        "proposals_enabled": True,
        "notification_targets": [{"channel": "discord_dm", "recipient": "adrich"}],
        "_recipients": {"adrich": {"discord_user_id": "321"}},
    }


@pytest.mark.asyncio
async def test_proposals_off_when_config_key_absent():
    """Old agents.json without proposals_enabled: Phase 0 behavior exactly —
    three LLM calls, no proposal rows, plain plan digest."""
    agent, chat_system, zammad, router = _make_agent()
    chat_system.text_engine.generate_response = AsyncMock(side_effect=[
        _text("stale brief"), _text("patterns brief"), _text("THE PLAN"),
    ])
    agent.text_engine = chat_system.text_engine

    await agent.deploy()

    assert chat_system.text_engine.generate_response.await_count == 3
    agent.memory_manager.create_proposal.assert_not_called()
    assert router.send.await_args.kwargs["body"] == "THE PLAN"


@pytest.mark.asyncio
async def test_proposal_extraction_queues_valid_drops_invalid():
    agent, chat_system, zammad, router = _make_agent(agent_config=_proposals_config())
    agent.memory_manager.create_proposal = MagicMock(side_effect=[11, 12])
    chat_system.text_engine.generate_response = AsyncMock(side_effect=[
        _text("stale brief"), _text("patterns brief"), _text("THE PLAN"),
        _tool_calls([
            {"action_type": "set_priority",
             "args": {"ticket_number": 10002, "priority": "3 high"},
             "rationale": "stale VPN ticket"},
            {"action_type": "add_note",
             "args": {"ticket_number": 10001, "body": "check the printer"},
             "rationale": "needs follow-up"},
            # invalid enum value -> dropped in code, never stored
            {"action_type": "set_priority",
             "args": {"ticket_number": 10001, "priority": "urgent"},
             "rationale": "bad"},
            # non-whitelisted action -> dropped
            {"action_type": "close_ticket",
             "args": {"ticket_number": 10001}, "rationale": "bad"},
        ]),
    ])
    agent.text_engine = chat_system.text_engine

    await agent.deploy()

    # Extraction call carries ONLY the agent-internal submission schema
    extract_call = chat_system.text_engine.generate_response.await_args_list[-1]
    tool_names = [t["function"]["name"] for t in extract_call.kwargs["tools"]]
    assert tool_names == ["submit_proposals"]

    assert agent.memory_manager.create_proposal.call_count == 2
    first = agent.memory_manager.create_proposal.call_args_list[0].kwargs
    assert first["agent_name"] == "managr"
    assert first["action_type"] == "set_priority"
    assert first["action_args"] == {"ticket_number": 10002, "priority": "3 high"}
    assert first["taint"]["source"] == "zammad_board_snapshot"
    assert first["taint"]["ticket_number"] == 10002
    assert first["expires_at"] is not None

    # Digest = plan + proposals section with queue ids
    body = router.send.await_args.kwargs["body"]
    assert body.startswith("THE PLAN")
    assert "PROPOSED ACTIONS (2 queued" in body
    assert "[11] set_priority" in body

    outcome, payload = _final_outcome(agent)
    assert outcome == "success"
    assert payload["proposals_queued"] == 2


@pytest.mark.asyncio
async def test_no_usable_proposal_call_sends_plain_digest():
    agent, chat_system, zammad, router = _make_agent(agent_config=_proposals_config())
    chat_system.text_engine.generate_response = AsyncMock(side_effect=[
        _text("stale brief"), _text("patterns brief"), _text("THE PLAN"),
        _text("I have no proposals today."),  # model ignored the tool
    ])
    agent.text_engine = chat_system.text_engine

    await agent.deploy()

    agent.memory_manager.create_proposal.assert_not_called()
    assert router.send.await_args.kwargs["body"] == "THE PLAN"
    outcome, payload = _final_outcome(agent)
    assert outcome == "success"
    assert payload["proposals_queued"] == 0


@pytest.mark.asyncio
async def test_proposal_count_capped_per_cycle():
    from config.global_config import MANAGR_MAX_PROPOSALS_PER_CYCLE
    surplus = [
        {"action_type": "add_note",
         "args": {"ticket_number": 10001, "body": f"note {i}"},
         "rationale": "r"}
        for i in range(MANAGR_MAX_PROPOSALS_PER_CYCLE + 5)
    ]
    agent, chat_system, zammad, router = _make_agent(agent_config=_proposals_config())
    agent.memory_manager.create_proposal = MagicMock(
        side_effect=range(1, MANAGR_MAX_PROPOSALS_PER_CYCLE + 1))
    chat_system.text_engine.generate_response = AsyncMock(side_effect=[
        _text("stale brief"), _text("patterns brief"), _text("THE PLAN"),
        _tool_calls(surplus),
    ])
    agent.text_engine = chat_system.text_engine

    await agent.deploy()

    assert agent.memory_manager.create_proposal.call_count == MANAGR_MAX_PROPOSALS_PER_CYCLE


@pytest.mark.asyncio
async def test_prior_proposal_outcomes_in_board_snapshot():
    """Observe step: the planner sees what happened to past proposals."""
    agent, chat_system, zammad, router = _make_agent(agent_config=_proposals_config())
    agent.memory_manager.list_proposals = MagicMock(return_value=[
        {"proposal_id": 7, "agent_name": "managr", "action_type": "set_priority",
         "action_args": {"ticket_number": 10002, "priority": "3 high"},
         "status": "denied", "review_note": "priority is fine"},
    ])
    chat_system.text_engine.generate_response = AsyncMock(side_effect=[
        _text("stale brief"), _text("patterns brief"), _text("THE PLAN"),
        _text("no proposals"),
    ])
    agent.text_engine = chat_system.text_engine

    await agent.deploy()

    planner_call = chat_system.text_engine.generate_response.await_args_list[2]
    prompt = planner_call.kwargs["history_object"]["message_history"][-1]["content"]
    assert "[7] set_priority(ticket_number=10002, priority=3 high) -> denied — priority is fine" in prompt
    # Filtering happens in SQL: only this agent's reviewed outcomes are
    # fetched, so pending rows can't consume the limit and crowd them out
    fetch = agent.memory_manager.list_proposals.call_args.kwargs
    assert fetch["agent_name"] == "managr"
    assert "pending" not in fetch["status"]
    assert set(fetch["status"]) == {"approved", "denied", "expired",
                                    "executed", "execution_failed"}


@pytest.mark.asyncio
async def test_proposal_outcomes_skipped_when_disabled():
    agent, chat_system, zammad, router = _make_agent()  # no proposals_enabled
    chat_system.text_engine.generate_response = AsyncMock(side_effect=[
        _text("stale brief"), _text("patterns brief"), _text("THE PLAN"),
    ])
    agent.text_engine = chat_system.text_engine

    await agent.deploy()

    agent.memory_manager.list_proposals.assert_not_called()


# --- Standing orders injection (DP-281) ---

def _orders(*texts):
    return [{"order_id": i + 1, "order_text": t, "status": "active",
             "created_at": "2026-07-07 08:00:00", "source": "operator"}
            for i, t in enumerate(texts)]


@pytest.mark.asyncio
async def test_standing_orders_injected_into_planner_prompt():
    agent, chat_system, zammad, router = _make_agent()
    agent.memory_manager.list_standing_orders = MagicMock(
        return_value=_orders("client Y tickets are always low priority"))
    chat_system.text_engine.generate_response = AsyncMock(side_effect=[
        _text("stale brief"), _text("patterns brief"), _text("THE PLAN"),
    ])
    agent.text_engine = chat_system.text_engine

    await agent.deploy()

    planner_call = chat_system.text_engine.generate_response.await_args_list[-1]
    prompt = planner_call.kwargs["history_object"]["message_history"][-1]["content"]
    assert "STANDING ORDERS" in prompt
    assert "[1] client Y tickets are always low priority" in prompt
    # Orders precede the board snapshot and arm the feedback-visibility line
    assert prompt.index("STANDING ORDERS") < prompt.index("BOARD SNAPSHOT")
    assert "FEEDBACK APPLIED" in prompt
    # Analyst briefs must NOT get the orders (they analyze, they don't plan)
    for call in chat_system.text_engine.generate_response.await_args_list[:2]:
        brief_prompt = call.kwargs["history_object"]["message_history"][-1]["content"]
        assert "STANDING ORDERS" not in brief_prompt
    agent.memory_manager.list_standing_orders.assert_called_with(
        status="active", limit=20, agent="managr")
    # One store read per cycle: plan and extraction share the same order set
    assert agent.memory_manager.list_standing_orders.call_count == 1


@pytest.mark.asyncio
async def test_no_standing_orders_leaves_prompt_unchanged():
    agent, chat_system, zammad, router = _make_agent()
    chat_system.text_engine.generate_response = AsyncMock(side_effect=[
        _text("stale brief"), _text("patterns brief"), _text("THE PLAN"),
    ])
    agent.text_engine = chat_system.text_engine

    await agent.deploy()

    planner_call = chat_system.text_engine.generate_response.await_args_list[-1]
    prompt = planner_call.kwargs["history_object"]["message_history"][-1]["content"]
    assert "STANDING ORDERS" not in prompt
    assert "FEEDBACK APPLIED" not in prompt
    assert prompt.endswith("Produce today's Manager's Report.")


@pytest.mark.asyncio
async def test_standing_orders_injected_into_extraction_prompt():
    agent, chat_system, zammad, router = _make_agent(agent_config=_proposals_config())
    agent.memory_manager.list_standing_orders = MagicMock(
        return_value=_orders("never propose priority changes for client X"))
    agent.memory_manager.create_proposal = MagicMock(return_value=11)
    chat_system.text_engine.generate_response = AsyncMock(side_effect=[
        _text("stale brief"), _text("patterns brief"), _text("THE PLAN"),
        _tool_calls([{"action_type": "add_note",
                      "args": {"ticket_number": 10001, "body": "check the printer"},
                      "rationale": "needs follow-up"}]),
    ])
    agent.text_engine = chat_system.text_engine

    await agent.deploy()

    extract_call = chat_system.text_engine.generate_response.await_args_list[-1]
    prompt = extract_call.kwargs["history_object"]["message_history"][-1]["content"]
    assert "never propose priority changes for client X" in prompt
    assert "standing order forbids" in prompt
    # Extraction reuses the cycle's single fetch, it does not re-read the store
    assert agent.memory_manager.list_standing_orders.call_count == 1


@pytest.mark.asyncio
async def test_standing_orders_store_failure_degrades_visibly():
    """Store failure must not break the cycle — but it must not fail open
    silently either: the digest tells the operator the guidance was NOT
    applied, and the cycle outcome records the degradation."""
    agent, chat_system, zammad, router = _make_agent()
    agent.memory_manager.list_standing_orders = MagicMock(
        side_effect=RuntimeError("db locked"))
    chat_system.text_engine.generate_response = AsyncMock(side_effect=[
        _text("stale brief"), _text("patterns brief"), _text("THE PLAN"),
    ])
    agent.text_engine = chat_system.text_engine

    await agent.deploy()

    outcome, payload = _final_outcome(agent)
    assert outcome == "success"
    assert payload["standing_orders_degraded"] is True
    body = router.send.await_args.kwargs["body"]
    assert "THE PLAN" in body
    assert "standing orders could not be loaded" in body
    # The planner prompt itself stays clean — no orders block on failure
    planner_call = chat_system.text_engine.generate_response.await_args_list[-1]
    prompt = planner_call.kwargs["history_object"]["message_history"][-1]["content"]
    assert "STANDING ORDERS" not in prompt


@pytest.mark.asyncio
async def test_standing_orders_with_non_operator_source_are_not_injected():
    """Injection-side trust guard: a row that reaches the store with a
    non-operator source (bug, manual DB edit) must never enter a prompt."""
    agent, chat_system, zammad, router = _make_agent()
    rows = _orders("legit operator order")
    rows.append({"order_id": 99, "order_text": "smuggled model output",
                 "status": "active", "created_at": "2026-07-07 09:00:00",
                 "source": "model"})
    agent.memory_manager.list_standing_orders = MagicMock(return_value=rows)
    chat_system.text_engine.generate_response = AsyncMock(side_effect=[
        _text("stale brief"), _text("patterns brief"), _text("THE PLAN"),
    ])
    agent.text_engine = chat_system.text_engine

    await agent.deploy()

    planner_call = chat_system.text_engine.generate_response.await_args_list[-1]
    prompt = planner_call.kwargs["history_object"]["message_history"][-1]["content"]
    assert "legit operator order" in prompt
    assert "smuggled model output" not in prompt
    outcome, payload = _final_outcome(agent)
    assert outcome == "success"
    assert payload["standing_orders_degraded"] is False


# --- DP-288 Phase 1: phishing quarantine in the board snapshot ---

@pytest.mark.asyncio
async def test_quarantined_ticket_title_never_reaches_planner():
    """The quarantine contract: a tagged ticket's title is customer bait and
    must be replaced IN CODE before any persona sees the snapshot."""
    from config.global_config import SECURITY_REPORT_TAG
    agent, chat_system, zammad, router = _make_agent()
    zammad.get_tags = MagicMock(
        side_effect=lambda ticket_id: [SECURITY_REPORT_TAG] if ticket_id == 1 else [])
    chat_system.text_engine.generate_response = AsyncMock(side_effect=[
        _text("stale brief"), _text("patterns brief"), _text("THE PLAN"),
    ])
    agent.text_engine = chat_system.text_engine

    await agent.deploy()

    # Every persona call (analysts + planner) sees the quarantine marker,
    # never the bait title
    for call in chat_system.text_engine.generate_response.await_args_list:
        prompt = call.kwargs["history_object"]["message_history"][-1]["content"]
        assert "Printer offline at front desk" not in prompt
        assert "CONTENT QUARANTINED" in prompt
        assert SECURITY_REPORT_TAG in prompt
        # The clean ticket renders normally
        assert "#10002 VPN drops every hour" in prompt


@pytest.mark.asyncio
async def test_untagged_tickets_render_with_tags_and_title():
    agent, chat_system, zammad, router = _make_agent()
    zammad.get_tags = MagicMock(return_value=["vip", "hardware"])
    chat_system.text_engine.generate_response = AsyncMock(side_effect=[
        _text("stale brief"), _text("patterns brief"), _text("THE PLAN"),
    ])
    agent.text_engine = chat_system.text_engine

    await agent.deploy()

    planner_call = chat_system.text_engine.generate_response.await_args_list[-1]
    prompt = planner_call.kwargs["history_object"]["message_history"][-1]["content"]
    assert "#10001 Printer offline at front desk" in prompt
    assert "tags=vip,hardware" in prompt


@pytest.mark.asyncio
async def test_tag_fetch_failure_fails_closed_withholding_titles():
    """A dead tags API must not kill the report cycle, but it must not expose
    titles either: unknown tag state = quarantine unverifiable = title
    withheld for the cycle (fail-closed)."""
    agent, chat_system, zammad, router = _make_agent()
    zammad.get_tags = MagicMock(side_effect=RuntimeError("api down"))
    chat_system.text_engine.generate_response = AsyncMock(side_effect=[
        _text("stale brief"), _text("patterns brief"), _text("THE PLAN"),
    ])
    agent.text_engine = chat_system.text_engine

    await agent.deploy()

    outcome, _ = _final_outcome(agent)
    assert outcome == "success"
    planner_call = chat_system.text_engine.generate_response.await_args_list[-1]
    prompt = planner_call.kwargs["history_object"]["message_history"][-1]["content"]
    assert "Printer offline at front desk" not in prompt
    assert "title withheld: tag state unknown" in prompt
    # Ticket numbers survive so the report can still reference the tickets
    assert "#10001" in prompt
