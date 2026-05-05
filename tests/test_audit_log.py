# tests/test_audit_log.py

import pytest
import json
import asyncio
from datetime import datetime
from unittest.mock import MagicMock, AsyncMock, patch

from src.memory.memory_manager import MemoryManager
from src.chat_system import ChatSystem, ResponseType, PendingConfirmation
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
    
    with patch('src.chat_system.load_personas_from_file', return_value={}):
        system = ChatSystem(memory_manager=mem_manager, text_engine=text_engine)
        system.tool_manager = tool_manager
        return system

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
    
    # Mock ToolLoop event
    from src.tools.tool_loop import _LoopFinishedEvent
    audit_info = {"actions": [{"tool": "write_tool", "args": {}}]}
    finish_ev = _LoopFinishedEvent(
        final_text="Parking",
        response_type=ResponseType.PENDING_CONFIRMATION,
        pending_writes=[{"name": "write_tool", "arguments": {}}],
        audit_info=audit_info,
        turn_tainted=True
    )
    
    # Mock ToolLoop.run
    with patch('src.chat_system.ToolLoop') as mock_loop_cls:
        mock_loop = mock_loop_cls.return_value
        async def mock_run(*args, **kwargs):
            yield finish_ev
        mock_loop.run = mock_run
        
        # Setup ChatSystem for orchestrate
        chat_system.personas["test_p"] = persona
        chat_system.bot_logic.preprocess_message = AsyncMock(return_value=None)
        
        # Drive orchestrate
        events = []
        async for ev in chat_system._orchestrate("test_p", "user_id", "chan", "msg"):
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

@pytest.mark.asyncio
async def test_chat_system_audit_decision_approved(chat_system, mem_manager):
    # Setup pending confirmation
    audit_info = {"actions": [{"tool": "write_tool", "args": {}}]}
    pending = PendingConfirmation(
        write_calls=[{"name": "write_tool", "arguments": {}}],
        conversation_history=[],
        persona_name="test_p",
        tools_for_llm=[],
        image_url=None,
        channel="chan",
        server_id=None,
        turn_tainted=False,
        audit_info=audit_info
    )
    chat_system._pending_confirmations[("user_id", "test_p")] = pending
    chat_system.personas["test_p"] = Persona("test_p", "model", "prompt")
    
    # Mock dependencies for resume
    chat_system._execute_write_calls = AsyncMock()
    chat_system.text_engine.generate_response = AsyncMock(return_value=({"content": "Done"}, {}))
    
    # Resume with approval
    await chat_system.resume_pending_confirmation("user_id", "test_p", approved=True)
    
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

@pytest.mark.asyncio
async def test_chat_system_audit_decision_denied(chat_system, mem_manager):
    # Setup pending confirmation
    audit_info = {"actions": [{"tool": "write_tool", "args": {}}]}
    pending = PendingConfirmation(
        write_calls=[{"name": "write_tool", "arguments": {}}],
        conversation_history=[],
        persona_name="test_p",
        tools_for_llm=[],
        image_url=None,
        channel="chan",
        server_id=None,
        turn_tainted=True,
        audit_info=audit_info
    )
    chat_system._pending_confirmations[("user_id", "test_p")] = pending
    chat_system.personas["test_p"] = Persona("test_p", "model", "prompt")
    
    # Mock dependencies for resume
    chat_system._append_denied_tool_results = MagicMock()
    chat_system.text_engine.generate_response = AsyncMock(return_value=({"content": "Denied"}, {}))
    
    # Resume with denial
    await chat_system.resume_pending_confirmation("user_id", "test_p", approved=False)
    
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
