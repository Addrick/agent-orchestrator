# tests/test_tool_security_phase_6.py

import pytest
from datetime import datetime
from src.memory.memory_manager import MemoryManager
from src.message_handler import BotLogic
from unittest.mock import MagicMock
from config.global_config import EMBEDDING_MODEL
import json

EMBEDDING_BLOB = b'\x00' * (3072 * 4)

@pytest.fixture
def mem_manager():
    manager = MemoryManager(db_path=":memory:")
    manager.create_schema()
    yield manager
    manager.close()

def test_audit_log_schema(mem_manager):
    conn = mem_manager._get_connection()
    cursor = conn.cursor()
    cursor.execute("PRAGMA table_info(Audit_Log)")
    columns = {row['name'] for row in cursor.fetchall()}
    expected = {'audit_id', 'event_type', 'target_id', 'operator_id', 'timestamp', 'prior_state', 'new_state', 'reason', 'metadata'}
    assert expected.issubset(columns)

def test_mark_trusted_untrusted(mem_manager):
    # Seed a summary
    seg_id = mem_manager.store_segment("chan", None, "p1", 1, 5, 5, datetime.now())
    # Initially untrusted=1 (simulating a tainted memory)
    sum_id = mem_manager.store_summary(seg_id, "test summary", EMBEDDING_BLOB, EMBEDDING_MODEL, datetime.now(), untrusted=1)
    
    # Verify initial state
    summaries = mem_manager.get_summaries_for_channel("chan", "p1")
    assert summaries[0]['untrusted'] == 1
    
    # Mark trusted
    success = mem_manager.mark_trusted(sum_id, "operator1", "Manual verification")
    assert success is True
    
    # Verify state change
    summaries = mem_manager.get_summaries_for_channel("chan", "p1")
    assert summaries[0]['untrusted'] == 0
    
    # Verify audit log
    conn = mem_manager._get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM Audit_Log WHERE target_id = ?", (sum_id,))
    audit = cursor.fetchone()
    assert audit['event_type'] == "operator_override"
    assert audit['operator_id'] == "operator1"
    assert audit['prior_state'] == "untrusted"
    assert audit['new_state'] == "trusted"
    assert audit['reason'] == "Manual verification"

    # Mark untrusted again
    success = mem_manager.mark_untrusted(sum_id, "operator2", "Re-tainting")
    assert success is True
    summaries = mem_manager.get_summaries_for_channel("chan", "p1")
    assert summaries[0]['untrusted'] == 1
    
    cursor.execute("SELECT * FROM Audit_Log WHERE target_id = ? ORDER BY audit_id DESC", (sum_id,))
    audit = cursor.fetchone()
    assert audit['operator_id'] == "operator2"
    assert audit['prior_state'] == "trusted"
    assert audit['new_state'] == "untrusted"

def test_bot_logic_trust_commands(mem_manager):
    chat_system = MagicMock()
    chat_system.memory_manager = mem_manager
    bot_logic = BotLogic(chat_system)
    
    persona = MagicMock()
    persona.get_name.return_value = "p1"
    
    # Seed a summary
    seg_id = mem_manager.store_segment("chan", None, "p1", 1, 5, 5, datetime.now())
    sum_id = mem_manager.store_summary(seg_id, "test summary", EMBEDDING_BLOB, EMBEDDING_MODEL, datetime.now(), untrusted=1)
    
    # Test 'trust' command
    response, mutated = bot_logic._handle_trust([str(sum_id), "Verified", "it's", "safe"], persona, "user123")
    assert "TRUSTED" in response
    assert mutated is True
    
    summaries = mem_manager.get_summaries_for_channel("chan", "p1")
    assert summaries[0]['untrusted'] == 0
    
    # Test 'untrust' command
    response, mutated = bot_logic._handle_untrust([str(sum_id), "Suspect"], persona, "user123")
    assert "UNTRUSTED" in response
    assert mutated is True
    
    summaries = mem_manager.get_summaries_for_channel("chan", "p1")
    assert summaries[0]['untrusted'] == 1
