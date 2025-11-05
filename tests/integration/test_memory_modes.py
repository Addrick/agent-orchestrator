# tests/test_memory_modes.py

import pytest
from unittest.mock import MagicMock, AsyncMock, patch
from datetime import datetime, timedelta

from src.chat_system import ChatSystem
from src.database.memory_manager import MemoryManager
from src.persona import Persona, MemoryMode
from src.engine import TextEngine
from src.clients.zammad_client import ZammadClient

# Mark all tests in this file as 'integration'.
pytestmark = pytest.mark.integration
@pytest.fixture
def mem_test_system():
    """Provides a ChatSystem with a real, in-memory MemoryManager and mocked dependencies."""
    memory_manager = MemoryManager(db_path=":memory:")
    memory_manager.create_schema()

    mock_text_engine = MagicMock(spec=TextEngine)
    mock_zammad_client = MagicMock(spec=ZammadClient)

    chat_system = ChatSystem(
        memory_manager=memory_manager,
        text_engine=mock_text_engine,
        zammad_client=mock_zammad_client
    )
    chat_system.personas = {
        'test_persona': Persona('test_persona', 'mock_model', 'prompt', context_length=10),
        'persona_2': Persona('persona_2', 'mock_model', 'prompt', context_length=10)
    }
    yield chat_system, memory_manager, mock_text_engine, mock_zammad_client


@pytest.fixture
def real_test_system():
    """Provides a ChatSystem with REAL, integrated components using an in-memory DB."""
    memory_manager = MemoryManager(db_path=":memory:")
    memory_manager.create_schema()
    text_engine = TextEngine()
    zammad_client = ZammadClient()
    chat_system = ChatSystem(
        memory_manager=memory_manager,
        text_engine=text_engine,
        zammad_client=zammad_client
    )
    chat_system.personas = {
        'test_persona': Persona('test_persona', 'mock_model', 'prompt', context_length=10),
    }
    yield chat_system, memory_manager


def test_database_schema_has_server_id_column(mem_test_system):
    """Tests that the schema creation correctly adds the 'server_id' column."""
    _, memory_manager, _, _ = mem_test_system
    conn = memory_manager._get_connection()
    cursor = conn.cursor()
    cursor.execute("PRAGMA table_info(User_Interactions)")
    columns = [row['name'] for row in cursor.fetchall()]
    assert 'server_id' in columns


def test_log_message_with_server_id(mem_test_system):
    """Tests that the server_id is correctly saved by log_message."""
    _, memory_manager, _, _ = mem_test_system
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
    chat_system, memory_manager, mock_text_engine, _ = mem_test_system
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
    chat_system, memory_manager, mock_text_engine, _ = mem_test_system
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
    chat_system, memory_manager, mock_text_engine, _ = mem_test_system
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
    chat_system, memory_manager, mock_text_engine, _ = mem_test_system
    persona = chat_system.personas['test_persona']
    persona.set_memory_mode(MemoryMode.PERSONAL)
    mock_text_engine.generate_response.return_value = ({'type': 'text', 'content': ''}, {})

    now = datetime.now()
    memory_manager.log_message("user_A", "test_persona", "channel", "user", "UserA", "msg1_userA_persona1", now)
    memory_manager.log_message("user_B", "test_persona", "channel", "user", "UserB", "msg2_userB_persona1", now)
    memory_manager.log_message("user_A", "persona_2", "channel", "user", "UserA", "msg3_userA_persona2", now)

    await chat_system.generate_response("test_persona", "user_A", "channel", "current_msg")

    history = mock_text_engine.generate_response.call_args.args[1]['history']

    assert len(history) == 2
    assert history[0]['content'] == "msg1_userA_persona1"



@pytest.mark.asyncio
@patch('src.clients.zammad_client.requests.request')
@patch('src.engine.AsyncOpenAI')
async def test_channel_mode_in_non_server_context_integration(
        mock_async_openai, mock_requests_request, real_test_system
):
    """
    An integration test to verify that CHANNEL_ISOLATED mode works correctly in a
    non-server context (e.g., DMs, Gmail) by using real system components and
    patching only the outgoing network calls.
    """
    # 1. SETUP: Use real components from the fixture
    chat_system, memory_manager = real_test_system
    persona = chat_system.personas['test_persona']
    persona.set_memory_mode(MemoryMode.CHANNEL_ISOLATED)
    persona.set_model_name('local')

    # 2. CONFIGURE PATCHES for external network calls
    # Mock the Zammad client's HTTP requests
    def zammad_side_effect(*args, **kwargs):
        url = args[1]
        mock_response = MagicMock()
        mock_response.raise_for_status.return_value = None
        if 'users/search' in url: mock_response.json.return_value = []
        elif 'users' in url and args[0] == 'post': mock_response.json.return_value = {'id': 99, 'email': 'a@b.c'}
        elif 'tickets/search' in url: mock_response.json.return_value = []
        else: mock_response.json.return_value = {}
        return mock_response
    mock_requests_request.side_effect = zammad_side_effect

    # Configure the mock AsyncOpenAI class
    mock_client_instance = mock_async_openai.return_value
    mock_client_instance.chat.completions.create = AsyncMock(
        return_value=MagicMock(choices=[MagicMock(message=MagicMock(content="mocked llm response", tool_calls=None))])
    )

    # 3. SEED DATABASE
    now = datetime.now()
    memory_manager.log_message("u1", "test_persona", "gmail", "user", "u1", "gmail_message", now, server_id=None)
    memory_manager.log_message("u1", "test_persona", "gmail", "user", "u1", "conflicting_server_message", now, server_id="server123")

    # 4. ACTION
    await chat_system.generate_response(
        persona_name="test_persona", user_identifier="u1", channel="gmail",
        message="current_msg", server_id=None
    )

    # 5. ASSERTION
    assert 'u1' in chat_system.last_api_requests
    assert 'test_persona' in chat_system.last_api_requests['u1']
    final_payload = chat_system.last_api_requests['u1']['test_persona']
    messages = final_payload.get('messages', [])
    history_string = " ".join([m.get('content', '') for m in messages])

    assert "gmail_message" in history_string
    assert "conflicting_server_message" not in history_string
    assert "current_msg" in history_string


@pytest.mark.asyncio
@patch('src.chat_system.ChatSystem._should_create_ticket', return_value=True)
@patch('src.clients.zammad_client.requests.request')
@patch('src.engine.AsyncOpenAI')
async def test_ticket_isolated_mode_is_exclusive(
        mock_async_openai, mock_requests_request, mock_should_create, real_test_system
):
    """
    Tests that TICKET_ISOLATED mode only uses ticket history and ignores other contexts
    using a full integration path.
    """
    # 1. SETUP
    chat_system, memory_manager = real_test_system
    persona = chat_system.personas['test_persona']
    persona.set_memory_mode(MemoryMode.TICKET_ISOLATED)
    persona.set_model_name('local')

    # 2. CONFIGURE PATCHES
    def zammad_side_effect(*args, **kwargs):
        url = args[1]
        mock_response = MagicMock()
        mock_response.raise_for_status.return_value = None
        if 'users/search' in url: mock_response.json.return_value = [{'id': 1, 'email': 'a@b.c'}]
        elif 'tickets/search' in url and 'number:54321' in kwargs.get('params', {}).get('query', ''):
            mock_response.json.return_value = [{'id': 123}]
        else: mock_response.json.return_value = []
        return mock_response
    mock_requests_request.side_effect = zammad_side_effect

    # Configure the mock AsyncOpenAI class
    mock_client_instance = mock_async_openai.return_value
    mock_client_instance.chat.completions.create = AsyncMock(
        return_value=MagicMock(choices=[MagicMock(message=MagicMock(content="mocked llm response", tool_calls=None))])
    )

    # 3. SEED DATABASE
    now = datetime.now()
    memory_manager.log_message("u1", "test_persona", "support", "user", "u1", "channel_message", now, server_id="server1")
    memory_manager.log_message("u1", "test_persona", "support", "user", "u1", "ticket_message", now - timedelta(seconds=10), zammad_ticket_id=123)
    memory_manager.log_message("u1", "test_persona", "support", "user", "u1", "other_ticket_msg", now - timedelta(seconds=5), zammad_ticket_id=456)

    # 4. ACTION
    await chat_system.generate_response("test_persona", "u1", "support", "current_msg for [Ticket#54321]", server_id="server1")

    # 5. ASSERTION
    assert 'u1' in chat_system.last_api_requests
    final_payload = chat_system.last_api_requests['u1']['test_persona']
    messages = final_payload.get('messages', [])
    history_string = " ".join([m.get('content', '') for m in messages])

    assert "part of Zammad ticket #54321" in history_string
    assert "ticket_message" in history_string
    assert "current_msg for [Ticket#54321]" in history_string
    assert "channel_message" not in history_string
    assert "other_ticket_msg" not in history_string