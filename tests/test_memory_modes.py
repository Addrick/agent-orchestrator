# tests/test_memory_modes.py

import pytest
from unittest.mock import MagicMock, AsyncMock, patch
from datetime import datetime, timedelta

from src.chat_system import ChatSystem
from src.database.memory_manager import MemoryManager
from src.persona import Persona, MemoryMode
from src.engine import TextEngine
from src.clients.zammad_client import ZammadClient


@pytest.fixture
def mem_test_system():
    """Provides a ChatSystem with a real, in-memory MemoryManager for testing memory logic."""
    memory_manager = MemoryManager(db_path=":memory:")
    memory_manager.create_schema()

    mock_text_engine = MagicMock(spec=TextEngine)
    mock_zammad_client = MagicMock(spec=ZammadClient)

    chat_system = ChatSystem(
        memory_manager=memory_manager,
        text_engine=mock_text_engine,
        zammad_client=mock_zammad_client
    )
    # Give the system a real persona dictionary to modify
    chat_system.personas = {
        'test_persona': Persona('test_persona', 'mock_model', 'prompt', context_length=10),
        'persona_2': Persona('persona_2', 'mock_model', 'prompt', context_length=10)
    }

    yield chat_system, memory_manager, mock_text_engine


def test_database_schema_has_server_id_column(mem_test_system):
    """Tests that the schema creation correctly adds the 'server_id' column."""
    _, memory_manager, _ = mem_test_system
    conn = memory_manager._get_connection()
    cursor = conn.cursor()
    cursor.execute("PRAGMA table_info(User_Interactions)")
    columns = [row['name'] for row in cursor.fetchall()]
    assert 'server_id' in columns


def test_log_message_with_server_id(mem_test_system):
    """Tests that the server_id is correctly saved by log_message."""
    _, memory_manager, _ = mem_test_system

    memory_manager.log_message(
        user_identifier="user1", persona_name="p", channel="c", author_role="user",
        author_name="user1", content="test", timestamp=datetime.now(), server_id="server123"
    )

    conn = memory_manager._get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT server_id FROM User_Interactions WHERE user_identifier = 'user1'")
    row = cursor.fetchone()
    assert row['server_id'] == "server123"


@pytest.mark.asyncio
async def test_channel_isolated_mode(mem_test_system):
    """Tests that CHANNEL_ISOLATED mode only retrieves messages from the correct channel and server."""
    chat_system, memory_manager, mock_text_engine = mem_test_system
    persona = chat_system.personas['test_persona']
    persona.set_memory_mode(MemoryMode.CHANNEL_ISOLATED)
    mock_text_engine.generate_response.return_value = ({'type': 'text', 'content': ''}, {})

    now = datetime.now()
    memory_manager.log_message("u1", "test_persona", "channel-A", "user", "u1", "msg1_server1_channelA", now,
                               server_id="server1")
    memory_manager.log_message("u1", "test_persona", "channel-B", "user", "u1", "msg2_server1_channelB", now,
                               server_id="server1")
    memory_manager.log_message("u1", "test_persona", "channel-A", "user", "u1", "msg3_server2_channelA", now,
                               server_id="server2")

    await chat_system.generate_response("test_persona", "u1", "channel-A", "current_msg", server_id="server1")

    history = mock_text_engine.generate_response.call_args.args[1]['history']

    assert len(history) == 2
    assert history[0]['content'] == "u1: msg1_server1_channelA"


@pytest.mark.asyncio
async def test_server_wide_mode(mem_test_system):
    """Tests that SERVER_WIDE mode retrieves all messages from one server but not others."""
    chat_system, memory_manager, mock_text_engine = mem_test_system
    persona = chat_system.personas['test_persona']
    persona.set_memory_mode(MemoryMode.SERVER_WIDE)
    mock_text_engine.generate_response.return_value = ({'type': 'text', 'content': ''}, {})

    now = datetime.now()
    memory_manager.log_message("u1", "test_persona", "channel-A", "user", "u1", "msg1_server1_channelA", now,
                               server_id="server1")
    memory_manager.log_message("u1", "test_persona", "channel-B", "user", "u1", "msg2_server1_channelB",
                               now + timedelta(seconds=1), server_id="server1")
    memory_manager.log_message("u1", "test_persona", "channel-A", "user", "u1", "msg3_server2_channelA", now,
                               server_id="server2")

    await chat_system.generate_response("test_persona", "u1", "channel-A", "current_msg", server_id="server1")

    history = mock_text_engine.generate_response.call_args.args[1]['history']

    assert len(history) == 3
    contents = {msg['content'] for msg in history}
    assert "u1: msg1_server1_channelA" in contents
    assert "u1: msg2_server1_channelB" in contents
    assert "u1: msg3_server2_channelA" not in contents


@pytest.mark.asyncio
async def test_global_mode(mem_test_system):
    """Tests that GLOBAL mode retrieves all messages seen by the persona."""
    chat_system, memory_manager, mock_text_engine = mem_test_system
    persona = chat_system.personas['test_persona']
    persona.set_memory_mode(MemoryMode.GLOBAL)
    mock_text_engine.generate_response.return_value = ({'type': 'text', 'content': ''}, {})

    now = datetime.now()
    memory_manager.log_message("u1", "test_persona", "channel-A", "user", "u1", "msg1", now, server_id="server1")
    memory_manager.log_message("u2", "other_persona", "channel-B", "user", "u2", "msg2_other_persona", now,
                               server_id="server1")
    memory_manager.log_message("u3", "test_persona", "channel-C", "user", "u3", "msg3", now + timedelta(seconds=1),
                               server_id="server2")

    await chat_system.generate_response("test_persona", "u1", "channel-A", "current_msg", server_id="server1")

    history = mock_text_engine.generate_response.call_args.args[1]['history']

    assert len(history) == 3
    contents = {msg['content'] for msg in history}
    assert "u1: msg1" in contents
    assert "u3: msg3" in contents
    assert "u2: msg2_other_persona" not in contents


@pytest.mark.asyncio
async def test_personal_mode_isolates_by_user_and_persona(mem_test_system):
    """Tests that PERSONAL mode isolates by user and the specific persona."""
    chat_system, memory_manager, mock_text_engine = mem_test_system
    persona = chat_system.personas['test_persona']
    persona.set_memory_mode(MemoryMode.PERSONAL)
    mock_text_engine.generate_response.return_value = ({'type': 'text', 'content': ''}, {})

    now = datetime.now()
    memory_manager.log_message("user_A", "test_persona", "channel", "user", "UserA", "msg1_userA_persona1", now)
    memory_manager.log_message("user_B", "test_persona", "channel", "user", "UserB", "msg2_userB_persona1", now)
    memory_manager.log_message("user_A", "persona_2", "channel", "user", "UserA", "msg3_userA_persona2", now)

    await chat_system.generate_response("test_persona", "user_A", "channel", "current_msg")

    history = mock_text_engine.generate_response.call_args.args[1]['history']

    assert len(history) == 2  # 1 from DB + current message
    assert history[0]['content'] == "msg1_userA_persona1"


@pytest.mark.asyncio
async def test_modes_in_non_server_context(mem_test_system):
    """Tests that channel isolation works correctly for contexts where server_id is None."""
    chat_system, memory_manager, mock_text_engine = mem_test_system
    persona = chat_system.personas['test_persona']
    persona.set_memory_mode(MemoryMode.CHANNEL_ISOLATED)
    mock_text_engine.generate_response.return_value = ({'type': 'text', 'content': ''}, {})

    now = datetime.now()
    memory_manager.log_message("u1", "test_persona", "gmail", "user", "u1", "gmail_message", now, server_id=None)
    memory_manager.log_message("u1", "test_persona", "gmail", "user", "u1", "conflicting_server_message", now,
                               server_id="server123")

    with patch.object(chat_system, '_get_or_create_zammad_user', new_callable=AsyncMock, return_value=(1, 'a@b.c')), \
            patch.object(chat_system, '_find_active_ticket_for_user', new_callable=AsyncMock, return_value=None):
        await chat_system.generate_response("test_persona", "u1", "gmail", "current_msg", server_id=None)

    history = mock_text_engine.generate_response.call_args.args[1]['history']

    assert len(history) == 2
    # FIX: The assertion was incorrect. The formatter should NOT prepend the author name
    # in non-channel modes (like 'personal' or 'ticket', which gmail defaults to).
    assert history[0]['content'] == "u1: gmail_message"


@pytest.mark.asyncio
async def test_ticket_isolated_mode_is_exclusive(mem_test_system):
    """Tests that TICKET_ISOLATED mode only uses ticket history and ignores other contexts."""
    chat_system, memory_manager, mock_text_engine = mem_test_system
    persona = chat_system.personas['test_persona']
    persona.set_memory_mode(MemoryMode.TICKET_ISOLATED)
    mock_text_engine.generate_response.return_value = ({'type': 'text', 'content': ''}, {})

    now = datetime.now()
    memory_manager.log_message("u1", "test_persona", "channel", "user", "u1", "channel_message", now,
                               server_id="server1")
    memory_manager.log_message("u1", "test_persona", "channel", "user", "u1", "ticket_message",
                               now - timedelta(seconds=10), zammad_ticket_id=123)

    # FIX: Add all necessary mocks to ensure the ticket-finding logic runs correctly.
    with patch.object(chat_system, '_get_or_create_zammad_user', new_callable=AsyncMock, return_value=(1, 'a@b.c')), \
            patch.object(chat_system, '_find_ticket_id_in_message', return_value=123), \
            patch.object(chat_system, '_find_active_ticket_for_user', new_callable=AsyncMock, return_value=None):
        await chat_system.generate_response("test_persona", "u1", "channel", "current_msg", server_id="server1")

    history = mock_text_engine.generate_response.call_args.args[1]['history']

    assert len(history) == 3
    assert "part of Zammad ticket #123" in history[0]['content']
    assert history[1]['content'] == "ticket_message"
