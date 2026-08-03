# tests/test_audit_log.py

import pytest
import json
import asyncio
from datetime import datetime
from unittest.mock import MagicMock, AsyncMock, patch

from src.memory.memory_manager import MemoryManager
from src.chat_system import ResponseType
from src.confirmations import ParkedWrite
from tests.helpers import make_chat_system
from src.persona import Persona, ExecutionMode
from src.engine import TextEngine
from src.tools.tool_manager import ToolManager

@pytest.fixture
def mem_manager():
    manager = MemoryManager(db_path=":memory:")
    manager.create_schema()
    yield manager
    manager.close()

@pytest.fixture
def chat_system(mem_manager):
    text_engine = MagicMock(spec=TextEngine)
    tool_manager = MagicMock(spec=ToolManager)
    tool_manager.get_tool_definitions.return_value = []
    
    return make_chat_system(
        memory_manager=mem_manager, text_engine=text_engine,
        tool_manager=tool_manager,
    )

def test_memory_manager_log_audit_event(mem_manager):
    metadata = {"key": "value"}
    mem_manager.log_audit_event(
        event_type="test_event",
        target_id=123,
        operator_id="user1",
        prior_state="old",
        new_state="new",
        reason="testing",
        metadata=metadata
    )
    
    conn = mem_manager._get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM Audit_Log WHERE event_type = 'test_event'")
    row = cursor.fetchone()
    
    assert row is not None
    assert row['target_id'] == 123
    assert row['operator_id'] == "user1"
    assert row['prior_state'] == "old"
    assert row['new_state'] == "new"
    assert row['reason'] == "testing"
    assert json.loads(row['metadata']) == metadata

@pytest.mark.asyncio
async def test_chat_system_audit_parked(chat_system, mem_manager):
    # Mock Persona
    persona = Persona("test_p", "model", "prompt")
    persona.set_execution_mode(ExecutionMode.CONFIRM)
    persona.set_enabled_tools(["*"])
    
    # Mock ToolLoop events: a gated write mid-turn, then a normal finish.
    from src.tools.tool_loop import WriteParkedEvent, _LoopFinishedEvent
    audit_info = {"actions": [{"tool": "write_tool", "args": {}}]}
    park_ev = WriteParkedEvent(
        token="tok-audit-1",
        write_call={"id": "c1", "name": "write_tool", "arguments": {}},
        audit_info=audit_info,
        confirmation_text="Parking",
        turn_tainted=True,
    )
    finish_ev = _LoopFinishedEvent(
        final_text="Proposed a write.",
        response_type=ResponseType.LLM_GENERATION,
        turn_tainted=True,
    )

    # Mock ToolLoop.run
    with patch('src.chat_system.ToolLoop') as mock_loop_cls:
        mock_loop = mock_loop_cls.return_value
        async def mock_run(*args, **kwargs):
            yield park_ev
            yield finish_ev
        mock_loop.run = mock_run
        
        # Setup ChatSystem for orchestrate
        chat_system.personas["test_p"] = persona
        chat_system.bot_logic.preprocess_message = AsyncMock(return_value=None)
        
        # Drive the kernel through its public streaming entry
        events = []
        async for ev in chat_system.stream_response("test_p", "user_id", "chan", "msg"):
            events.append(ev)
            
        # Verify audit log has 'audit_parked'
        conn = mem_manager._get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM Audit_Log WHERE event_type = 'audit_parked'")
        row = cursor.fetchone()
        
        assert row is not None
        assert row['operator_id'] == "user_id"
        assert row['new_state'] == "pending"
        assert json.loads(row['metadata']) == audit_info

def _seed_park(chat_system, audit_info, *, token="tok-1", turn_tainted=False):
    """Register a gated write directly, as a completed turn would have."""
    park = ParkedWrite(
        token=token,
        write_call={"id": "c1", "name": "write_tool", "arguments": {}},
        audit_info=audit_info,
        confirmation_text="Approve?",
        user_identifier="user_id",
        persona_name="test_p",
        channel="chan",
        turn_tainted=turn_tainted,
    )
    chat_system.confirmations.park(park)
    return park.token


@pytest.mark.asyncio
async def test_chat_system_audit_decision_approved(chat_system, mem_manager):
    audit_info = {"actions": [{"tool": "write_tool", "args": {}}]}
    token = _seed_park(chat_system, audit_info)
    chat_system.personas["test_p"] = Persona("test_p", "model", "prompt")

    chat_system.tool_manager.execute_tool = AsyncMock(return_value={"ok": True})
    chat_system.text_engine.generate_response = AsyncMock(return_value=({"content": "Done"}, {}))

    await chat_system.resolve_park("user_id", "test_p", token, approved=True)

    # Verify audit log
    conn = mem_manager._get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM Audit_Log WHERE event_type = 'audit_decision' AND new_state = 'approved'")
    row = cursor.fetchone()

    assert row is not None
    assert row['operator_id'] == "user_id"
    assert row['prior_state'] == "pending"
    assert "Human approved" in row['reason']
    meta = json.loads(row['metadata'])
    assert meta['audit_info'] == audit_info
    # The token is on the row, so an audit trail can tie a decision to the
    # exact proposal — necessary now that several can be open at once.
    assert meta['token'] == token

@pytest.mark.asyncio
async def test_chat_system_audit_decision_denied(chat_system, mem_manager):
    audit_info = {"actions": [{"tool": "write_tool", "args": {}}]}
    token = _seed_park(chat_system, audit_info, turn_tainted=True)
    chat_system.personas["test_p"] = Persona("test_p", "model", "prompt")

    chat_system.text_engine.generate_response = AsyncMock(return_value=({"content": "Denied"}, {}))

    await chat_system.resolve_park("user_id", "test_p", token, approved=False)

    # Verify audit log
    conn = mem_manager._get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM Audit_Log WHERE event_type = 'audit_decision' AND new_state = 'denied'")
    row = cursor.fetchone()
    
    assert row is not None
    assert row['operator_id'] == "user_id"
    assert row['prior_state'] == "pending"
    assert "Human denied" in row['reason']
    meta = json.loads(row['metadata'])
    assert meta['turn_tainted'] is True


@pytest.mark.asyncio
async def test_audit_decision_metadata_carries_no_raw_write_call(
        chat_system, mem_manager):
    """The decision row describes the write without re-serializing the raw call.

    `write_calls` used to duplicate the tool name and arguments that
    `audit_info["actions"]` already carries, differing only in being
    unredacted. The sink scrubs now, so this is defence in depth — but the
    reviewable content must survive the field's removal, which is what the
    audit_info and call_id assertions below pin.
    """
    audit_info = {"actions": [{"tool": "write_tool",
                               "arguments": {"title": "t"},
                               "irreversible": False}]}
    token = _seed_park(chat_system, audit_info)
    chat_system.personas["test_p"] = Persona("test_p", "model", "prompt")

    chat_system.tool_manager.execute_tool = AsyncMock(return_value={"ok": True})
    chat_system.text_engine.generate_response = AsyncMock(
        return_value=({"content": "Done"}, {}))

    await chat_system.resolve_park("user_id", "test_p", token, approved=True)

    cursor = mem_manager._get_connection().cursor()
    cursor.execute(
        "SELECT * FROM Audit_Log WHERE event_type = 'audit_decision'"
    )
    row = cursor.fetchone()
    assert row is not None
    meta = json.loads(row['metadata'])

    assert "write_calls" not in meta
    # Everything the removed field contributed is still reachable.
    assert meta['audit_info']['actions'][0]['tool'] == "write_tool"
    assert meta['audit_info']['actions'][0]['arguments'] == {"title": "t"}
    # `call_id` replaces the raw call's only unique field, and is what ties the
    # row to the patched tool_context entry.
    assert meta['call_id'] == "c1"
