# tests/integration/test_managr_action_trajectory.py
"""End-to-end integration test for the ManagrAgent report cycle (DP-280).

Real MemoryManager + real Persona objects; Zammad, the LLM, and the
notification router are mocked. Verifies one full cycle produces:
- a root manager_report row with plan_excerpt in outcome_payload
- child steps for the snapshot, each analyst brief, the planner call,
  and the notification send
- a Hindsight bridge retain with the managr bank + stable document_id
"""

import json

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from config.global_config import (
    MANAGR_PLANNER_NAME,
    MANAGR_STALE_ANALYST_NAME,
    MANAGR_PATTERN_ANALYST_NAME,
)
from src.agents.managr_agent import ManagrAgent
from src.clients.notification import NotificationRouter
from src.persona import Persona


@pytest.mark.integration
@pytest.mark.asyncio
async def test_managr_end_to_end_trajectory(mocked_chat_system):
    chat_system, memory_manager = mocked_chat_system

    zammad = MagicMock()
    zammad.search_tickets = MagicMock(return_value=[
        {"id": 1, "number": "10001", "title": "Printer offline at front desk",
         "created_at": "2026-07-01T10:00:00Z", "updated_at": "2026-07-02T10:00:00Z"},
        {"id": 2, "number": "10002", "title": "VPN drops every hour",
         "created_at": "2026-06-20T10:00:00Z", "updated_at": "2026-06-21T10:00:00Z"},
    ])

    router = NotificationRouter()
    router.send = AsyncMock(return_value=True)

    with patch('src.agents.base.load_system_personas_from_file', return_value={}):
        agent = ManagrAgent(
            chat_system, zammad, router,
            agent_config={
                "notification_targets": [
                    {"channel": "discord_dm", "recipient": "adrich"},
                ],
                "_recipients": {"adrich": {"discord_user_id": "321"}},
            },
        )

    for name in (MANAGR_PLANNER_NAME, MANAGR_STALE_ANALYST_NAME,
                 MANAGR_PATTERN_ANALYST_NAME):
        chat_system.personas[name] = Persona(
            persona_name=name, model_name="mock_model", prompt=f"{name} prompt",
        )

    chat_system.text_engine.generate_response = AsyncMock(side_effect=[
        ({"type": "text", "content": "stale: #10002 no touch in 13d"}, {}),
        ({"type": "text", "content": "patterns: none significant"}, {}),
        ({"type": "text", "content": "BOARD HEALTH: fine\nTOP PRIORITIES: #10002"}, {}),
    ])
    agent.text_engine = chat_system.text_engine

    retain_mock = AsyncMock(return_value="")
    memory_manager.retain_experience = retain_mock

    await agent.deploy()

    actions = memory_manager.get_agent_actions("managr", limit=50)
    roots = [a for a in actions
             if a.get("parent_id") is None and a["action_type"] == "manager_report"]
    assert len(roots) == 1
    root = roots[0]
    root_id = root["id"]

    assert root["outcome"] == "success"
    op = json.loads(root["outcome_payload"])
    assert op["sent_targets"] == 1
    assert op["briefs"] == ["patterns", "stale"]
    assert "TOP PRIORITIES: #10002" in op["plan_excerpt"]

    steps = memory_manager.get_action_steps(root_id)
    step_types = {s["action_type"] for s in steps}
    assert "tool:zammad.search_tickets" in step_types
    assert "brief:stale" in step_types
    assert "brief:patterns" in step_types
    assert "llm_step" in step_types
    assert "tool:notification.send" in step_types
    assert all(s["parent_id"] == root_id for s in steps)

    # Analyst briefs completed, planner step carries the plan excerpt
    by_type = {s["action_type"]: s for s in steps}
    assert by_type["brief:stale"]["outcome"] == "success"
    assert by_type["llm_step"]["outcome"] == "success"

    # The only outbound side effect is the digest send
    router.send.assert_awaited_once()
    send_kwargs = router.send.await_args.kwargs
    assert send_kwargs["recipient"] == "321"
    assert "BOARD HEALTH" in send_kwargs["body"]

    # Hindsight bridge fired once for the series
    retain_mock.assert_awaited_once()
    kwargs = retain_mock.await_args.kwargs
    assert kwargs["document_id"] == f"agent_action:{root_id}"
    assert kwargs["bank_id"] == "managr"
    assert kwargs["action_type"] == "manager_report"
    prose = kwargs["content_override"]
    assert f"action_id={root_id}" in prose
    assert "outcome=success" in prose
    assert "brief:stale" in prose
