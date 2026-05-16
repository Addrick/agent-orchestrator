# tests/integration/test_dispatch_action_trajectory.py
"""End-to-end integration test for DP-116a enriched dispatch trajectory.

Verifies that a successful dispatch produces:
- a root Agent_Actions row with populated action_payload + outcome_payload
- at least one child row (parent_id = root id) capturing a tool / LLM step
- at least two Agent_Action_Contexts rows attached to the root
"""

import json

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from src.agents.dispatch_agent import DispatchAgent
from src.clients.notification import NotificationRouter


@pytest.mark.integration
@pytest.mark.asyncio
async def test_dispatch_end_to_end_trajectory(mocked_chat_system):
    chat_system, memory_manager = mocked_chat_system

    zammad = MagicMock()
    zammad.get_ticket = MagicMock(return_value={
        "id": 1234, "number": "10042", "title": "Billing portal returns 500",
    })
    zammad.get_ticket_articles = MagicMock(return_value=[
        {"body": "Customer cannot pay. AI TRIAGE CONTEXT DUMP: invoice service down.",
         "internal": True},
    ])
    zammad.add_tag = MagicMock()

    router = NotificationRouter()
    router.send = AsyncMock(return_value=True)

    with patch('src.agents.base.load_system_personas_from_file', return_value={}):
        agent = DispatchAgent(
            chat_system, zammad, router,
            agent_config={"notification_defaults": {"channel": "zammad"}},
        )

    decision = {"priority": "high", "summary": "Invoice service outage",
                "reasoning": "Billing portal 500s; payment blocked."}
    agent._get_dispatch_decision = AsyncMock(return_value=decision)

    await agent._dispatch_ticket(1234)

    actions = memory_manager.get_agent_actions("dispatch", limit=50)
    # All rows ordered newest-first; find the root
    roots = [a for a in actions if a.get("parent_id") is None and a["action_type"] == "dispatch"]
    assert len(roots) == 1, f"expected exactly one root dispatch row, got {len(roots)}"
    root = roots[0]
    root_id = root["id"]

    # Root has structured payloads
    assert root["outcome"] == "success"
    op = json.loads(root["outcome_payload"])
    assert op["priority"] == "high"
    assert op["sent"] is True
    assert op["decision"]["summary"] == "Invoice service outage"
    ap = json.loads(root["action_payload"])
    assert ap["ticket_id"] == 1234

    # At least one child step under the root
    steps = memory_manager.get_action_steps(root_id)
    assert len(steps) >= 1
    step_types = {s["action_type"] for s in steps}
    assert "llm_step" in step_types
    assert any(t.startswith("tool:") for t in step_types)
    # All steps share the same parent_id
    assert all(s["parent_id"] == root_id for s in steps)

    # Contexts: at least two rows on the root
    conn = memory_manager._get_connection()
    rows = conn.execute(
        "SELECT context_type, context_value FROM Agent_Action_Contexts WHERE action_id = ?",
        (root_id,),
    ).fetchall()
    contexts = {(r["context_type"], r["context_value"]) for r in rows}
    assert len(contexts) >= 2
    ctx_types = {t for t, _ in contexts}
    assert "ticket_id" in ctx_types
    assert "priority" in ctx_types
    assert "channel" in ctx_types
