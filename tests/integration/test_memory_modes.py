# tests/test_memory_modes.py

import math
import struct
import pytest
from unittest.mock import MagicMock, AsyncMock, patch
from datetime import datetime, timedelta

from src.chat_system import ChatSystem
from src.database.memory_manager import MemoryManager
from src.persona import Persona, MemoryMode
from src.engine import TextEngine
from src.clients.zammad_client import ZammadClient
from src.clients.zammad_service import ZammadIntegration

# Mark all tests in this file as 'integration'.
pytestmark = pytest.mark.integration
@pytest.fixture
def mem_test_system():
    """Provides a ChatSystem with a real, in-memory MemoryManager and mocked dependencies."""
    memory_manager = MemoryManager(db_path=":memory:")
    memory_manager.create_schema()

    mock_text_engine = MagicMock(spec=TextEngine)

    chat_system = ChatSystem(
        memory_manager=memory_manager,
        text_engine=mock_text_engine,
    )
    chat_system.personas = {
        'test_persona': Persona('test_persona', 'mock_model', 'prompt', context_length=10),
        'persona_2': Persona('persona_2', 'mock_model', 'prompt', context_length=10)
    }
    yield chat_system, memory_manager, mock_text_engine


@pytest.fixture
def real_test_system(monkeypatch):
    """Provides a ChatSystem with REAL, integrated components using an in-memory DB.
    Sets dummy Zammad env vars since the two tests using this fixture patch requests.request."""
    monkeypatch.setenv("ZAMMAD_URL", "http://zammad.test")
    monkeypatch.setenv("ZAMMAD_API_KEY", "fake-token-for-tests")
    memory_manager = MemoryManager(db_path=":memory:")
    memory_manager.create_schema()
    text_engine = TextEngine()
    zammad_client = ZammadClient()
    chat_system = ChatSystem(
        memory_manager=memory_manager,
        text_engine=text_engine,
    )
    chat_system.register_service(ZammadIntegration(zammad_client))
    chat_system.personas = {
        'test_persona': Persona('test_persona', 'mock_model', 'prompt', context_length=10),
    }
    yield chat_system, memory_manager


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


# --- Helpers for Long-Term Memory Integration Tests ---

def _unit_blob(*components):
    """Create a normalized float32 BLOB from components."""
    vec = list(components)
    norm = math.sqrt(sum(v * v for v in vec))
    if norm > 0:
        vec = [v / norm for v in vec]
    return struct.pack(f'{len(vec)}f', *vec)


# --- End-to-End Long-Term Memory Tests ---

@pytest.fixture
def memory_e2e_system():
    """ChatSystem with real in-memory DB and mocked text engine for memory tests."""
    memory_manager = MemoryManager(db_path=":memory:")
    memory_manager.create_schema()
    mock_text_engine = MagicMock(spec=TextEngine)
    mock_text_engine.generate_response = AsyncMock(
        return_value=({'type': 'text', 'content': ''}, {})
    )
    chat_system = ChatSystem(
        memory_manager=memory_manager,
        text_engine=mock_text_engine,
    )
    chat_system.personas = {
        'test_persona': Persona('test_persona', 'mock_model', 'prompt', context_length=5),
    }
    yield chat_system, memory_manager, mock_text_engine


@pytest.mark.asyncio
@patch('src.chat_system.MEMORY_RETRIEVAL_ENABLED', True)
async def test_e2e_memory_injection_in_prepare_request(memory_e2e_system):
    """End-to-end: store messages -> create segments/summaries -> verify memory
    block appears in _prepare_request conversation history."""
    system, mm, mock_engine = memory_e2e_system

    # 1. Seed older messages (outside the sliding window of context_length=5)
    now = datetime.now()
    for i in range(1, 8):
        mm.log_message("u1", "test_persona", "general", "user", "Alice",
                       f"old message {i}", now + timedelta(seconds=i), server_id="s1")

    # 2. Create a segment + summary for the old messages (simulating batch agent output)
    seg_id = mm.store_segment("general", "s1", "test_persona",
                              start_id=1, end_id=3, message_count=3, created_at=now)
    summary_emb = _unit_blob(1.0, 0.0)
    mm.store_summary(seg_id, "- Alice discussed Python 3.13 JIT compiler improvements",
                     summary_emb, "test-model", now)

    # 3. Set up a mock embedding service on ChatSystem
    mock_emb_service = MagicMock()
    mock_emb_service.model_name = "test-model"
    mock_emb_service.encode = AsyncMock(return_value=[_unit_blob(1.0, 0.0)])
    system._embedding_service = mock_emb_service

    # 4. Trigger generate_response which calls _prepare_request internally
    persona = system.personas['test_persona']
    persona.set_memory_mode(MemoryMode.CHANNEL_ISOLATED)

    await system.generate_response("test_persona", "u1", "general",
                                   "Tell me about the JIT", server_id="s1")

    # 5. Verify the memory block was injected
    call_args = mock_engine.generate_response.call_args
    context = call_args.args[1]
    history = context['history']

    # First message should be the memory block
    assert any("<memory>" in msg.get('content', '') for msg in history), \
        "Memory block not found in conversation history"
    memory_msg = next(m for m in history if "<memory>" in m.get('content', ''))
    assert "Python 3.13 JIT compiler" in memory_msg['content']
    assert "#general" in memory_msg['content']


@pytest.mark.asyncio
@patch('src.chat_system.MEMORY_RETRIEVAL_ENABLED', True)
async def test_e2e_recency_filter_no_information_gap(memory_e2e_system):
    """Recency filter integration: a segment straddling the sliding window boundary
    is included (not filtered), preventing information gaps for messages that are
    too old for the window but inside a segment that extends into it."""
    system, mm, mock_engine = memory_e2e_system

    now = datetime.now()

    # Seed 10 messages — IDs 1-10
    for i in range(1, 11):
        mm.log_message("u1", "test_persona", "general", "user", "Alice",
                       f"message {i}", now + timedelta(seconds=i), server_id="s1")

    # Segment A: IDs 1-4 (fully outside a sliding window starting at ID 6)
    seg_a = mm.store_segment("general", "s1", "test_persona",
                             start_id=1, end_id=4, message_count=4, created_at=now)
    mm.store_summary(seg_a, "- Old topic: infrastructure migration",
                     _unit_blob(1.0, 0.0), "test-model", now)

    # Segment B: IDs 5-8 (straddles — start=5 < window start, end=8 inside window)
    seg_b = mm.store_segment("general", "s1", "test_persona",
                             start_id=5, end_id=8, message_count=4, created_at=now)
    mm.store_summary(seg_b, "- Straddling topic: database schema review",
                     _unit_blob(0.9, 0.1), "test-model", now)

    # Segment C: IDs 9-10 (fully inside window)
    seg_c = mm.store_segment("general", "s1", "test_persona",
                             start_id=9, end_id=10, message_count=2, created_at=now)
    mm.store_summary(seg_c, "- Recent topic: deployment checklist",
                     _unit_blob(0.8, 0.2), "test-model", now)

    # Mock embedding service
    mock_emb_service = MagicMock()
    mock_emb_service.model_name = "test-model"
    mock_emb_service.encode = AsyncMock(return_value=[_unit_blob(1.0, 0.0)])
    system._embedding_service = mock_emb_service

    persona = system.personas['test_persona']
    persona.set_memory_mode(MemoryMode.CHANNEL_ISOLATED)

    await system.generate_response("test_persona", "u1", "general",
                                   "What about the migration?", server_id="s1")

    call_args = mock_engine.generate_response.call_args
    context = call_args.args[1]
    history = context['history']

    memory_msg = next((m for m in history if "<memory>" in m.get('content', '')), None)
    assert memory_msg is not None, "Memory block should be present"

    content = memory_msg['content']
    # Segment A (fully outside window) — included
    assert "infrastructure migration" in content
    # Segment B (straddling) — included, not filtered
    assert "database schema review" in content
    # Segment C (fully inside window, start_id=9 >= oldest_interaction_id) — filtered
    assert "deployment checklist" not in content


