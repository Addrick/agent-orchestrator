# tests/test_memory_retrieval.py

import math
import struct
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.database.memory_manager import MemoryManager
from src.chat_system import ChatSystem, _relative_time
from src.embedding_service import EmbeddingService
from src.persona import MemoryMode


# --- Helpers ---

def _unit_blob(*components):
    """Create a normalized float32 BLOB of dimension 3072."""
    # Use the requested components for the start, pad with zeros to 3072
    vec = [0.0] * 3072
    for i, val in enumerate(components):
        if i < 3072:
            vec[i] = val
    
    norm = math.sqrt(sum(v * v for v in vec))
    if norm > 0:
        vec = [v / norm for v in vec]
    return struct.pack('3072f', *vec)


# --- MemoryManager.retrieve_relevant_summaries Tests ---

@pytest.fixture
def mem_manager():
    manager = MemoryManager(db_path=":memory:")
    manager.create_schema()
    yield manager
    manager.close()


def _seed_data(mm, channel, persona, server_id=None, seg_start=1, n_ambient=False):
    """Insert a segment + summary for testing retrieval."""
    ts = datetime.now()
    for i in range(seg_start, seg_start + 3):
        mm.log_message("u1", persona, channel, "user", "Alice",
                       f"msg {i}", ts, server_id=server_id)

    seg_id = mm.store_segment(channel, server_id, persona, seg_start, seg_start + 2, 3, ts)
    emb = _unit_blob(1.0, 0.0, 0.0, 0.0)
    mm.store_summary(seg_id, "- Test fact", emb, "test-model", ts)

    if n_ambient:
        seg_id2 = mm.store_segment(channel, server_id, "ambient", seg_start + 10, seg_start + 12, 3, ts)
        mm.store_summary(seg_id2, "- Ambient fact", emb, "test-model", ts)


def test_retrieve_channel_isolated(mem_manager):
    """CHANNEL_ISOLATED returns only summaries for the specified channel+persona."""
    _seed_data(mem_manager, "chan-a", "p1")
    _seed_data(mem_manager, "chan-b", "p1", seg_start=10)

    results = mem_manager.retrieve_relevant_summaries(
        "p1", "chan-a", memory_mode="channel", include_ambient=False
    )
    assert len(results) == 1
    assert results[0]['channel'] == 'chan-a'


def test_retrieve_server_wide(mem_manager):
    """SERVER_WIDE returns summaries across all channels in a server."""
    _seed_data(mem_manager, "chan-a", "p1", server_id="srv1")
    _seed_data(mem_manager, "chan-b", "p1", server_id="srv1", seg_start=10)
    _seed_data(mem_manager, "chan-c", "p1", server_id="srv2", seg_start=20)

    results = mem_manager.retrieve_relevant_summaries(
        "p1", "chan-a", server_id="srv1", memory_mode="server", include_ambient=False
    )
    assert len(results) == 2
    channels = {r['channel'] for r in results}
    assert channels == {'chan-a', 'chan-b'}


def test_retrieve_global(mem_manager):
    """GLOBAL returns all summaries for a persona."""
    _seed_data(mem_manager, "chan-a", "p1")
    _seed_data(mem_manager, "chan-b", "p1", seg_start=10)
    _seed_data(mem_manager, "chan-c", "p2", seg_start=20)  # different persona

    results = mem_manager.retrieve_relevant_summaries(
        "p1", "chan-a", memory_mode="global", include_ambient=False
    )
    assert len(results) == 2


def test_retrieve_ticket_returns_empty(mem_manager):
    """TICKET_ISOLATED always returns empty."""
    _seed_data(mem_manager, "chan-a", "p1")

    results = mem_manager.retrieve_relevant_summaries(
        "p1", "chan-a", memory_mode="ticket"
    )
    assert results == []


def test_retrieve_includes_ambient(mem_manager):
    """include_ambient=True adds ambient persona summaries."""
    _seed_data(mem_manager, "chan-a", "p1", n_ambient=True)

    results = mem_manager.retrieve_relevant_summaries(
        "p1", "chan-a", memory_mode="channel", include_ambient=True
    )
    assert len(results) == 2
    personas = {r['persona_name'] for r in results}
    assert personas == {'p1', 'ambient'}


def test_retrieve_excludes_ambient(mem_manager):
    """include_ambient=False excludes ambient persona summaries."""
    _seed_data(mem_manager, "chan-a", "p1", n_ambient=True)

    results = mem_manager.retrieve_relevant_summaries(
        "p1", "chan-a", memory_mode="channel", include_ambient=False
    )
    assert len(results) == 1
    assert results[0]['persona_name'] == 'p1'


def test_retrieve_recency_filter(mem_manager):
    """Recency filter excludes segments starting inside the window."""
    ts = datetime.now()
    # Segment A: IDs 1-3 (outside window)
    seg_a = mem_manager.store_segment("chan", None, "p1", 1, 3, 3, ts)
    mem_manager.store_summary(seg_a, "Old facts", _unit_blob(1.0, 0.0), "m", ts)
    # Segment B: IDs 6-8 (straddles window starting at 7)
    seg_b = mem_manager.store_segment("chan", None, "p1", 6, 8, 3, ts)
    mem_manager.store_summary(seg_b, "Straddle facts", _unit_blob(1.0, 0.0), "m", ts)
    # Segment C: IDs 10-12 (fully inside window)
    seg_c = mem_manager.store_segment("chan", None, "p1", 10, 12, 3, ts)
    mem_manager.store_summary(seg_c, "Recent facts", _unit_blob(1.0, 0.0), "m", ts)

    results = mem_manager.retrieve_relevant_summaries(
        "p1", "chan", memory_mode="channel", include_ambient=False,
        exclude_after_interaction_id=7,
    )
    assert len(results) == 2
    contents = {r['content'] for r in results}
    assert contents == {"Old facts", "Straddle facts"}


def test_retrieve_model_name_filter(mem_manager):
    """model_name filter excludes summaries from different embedding models."""
    ts = datetime.now()
    seg = mem_manager.store_segment("chan", None, "p1", 1, 3, 3, ts)
    mem_manager.store_summary(seg, "Facts", _unit_blob(1.0, 0.0), "old-model", ts)

    # Matching model name should return the summary
    assert len(mem_manager.retrieve_relevant_summaries(
        "p1", "chan", memory_mode="channel", include_ambient=False, model_name="old-model"
    )) == 1
    # Different model name should exclude it (prevents cross-model similarity noise)
    assert len(mem_manager.retrieve_relevant_summaries(
        "p1", "chan", memory_mode="channel", include_ambient=False, model_name="new-model"
    )) == 0


# --- ChatSystem._retrieve_memory_block Tests ---

@pytest.fixture
def mock_persona():
    p = MagicMock()
    p.get_name.return_value = "test_persona"
    p.get_memory_mode.return_value = MemoryMode.CHANNEL_ISOLATED
    return p


@pytest.fixture
def chat_system_with_memory():
    """ChatSystem with mocked dependencies for memory retrieval tests."""
    mm = MagicMock()
    te = MagicMock()
    with patch('src.chat_system.load_personas_from_file', return_value={}), \
         patch('src.chat_system.get_model_list', return_value={}):
        system = ChatSystem(memory_manager=mm, text_engine=te)

    # Set up embedding service
    emb_service = MagicMock(spec=EmbeddingService)
    emb_service.model_name = "test-model"
    system._embedding_service = emb_service
    return system, mm, emb_service


@pytest.mark.asyncio
@patch('src.chat_system.MEMORY_RETRIEVAL_ENABLED', True)
async def test_retrieve_memory_block_returns_formatted_block(chat_system_with_memory, mock_persona):
    """Memory block is returned with proper formatting."""
    system, mm, emb_service = chat_system_with_memory

    mm.retrieve_relevant_summaries.return_value = [
        {
            'summary_id': 1, 'segment_id': 1, 'content': '- Security alert found',
            'embedding': _unit_blob(1.0, 0.0), 'model_name': 'test-model',
            'created_at': datetime.now() - timedelta(days=2),
            'channel': 'security', 'persona_name': 'sage',
            'start_interaction_id': 1, 'end_interaction_id': 5,
        }
    ]
    emb_service.encode = AsyncMock(return_value=[_unit_blob(1.0, 0.0)])

    result = await system._retrieve_memory_block(
        mock_persona, "user1", "chan", None,
        [{"role": "user", "content": "What about the security issue?"}],
    )

    assert result is not None
    assert "<memory>" in result
    assert "</memory>" in result
    assert "Security alert found" in result
    assert "#security" in result


@pytest.mark.asyncio
@patch('src.chat_system.MEMORY_RETRIEVAL_ENABLED', False)
async def test_retrieve_memory_block_disabled(chat_system_with_memory, mock_persona):
    """Returns None when feature is disabled."""
    system, _, _ = chat_system_with_memory

    result = await system._retrieve_memory_block(
        mock_persona, "user1", "chan", None,
        [{"role": "user", "content": "Hello"}],
    )
    assert result is None


@pytest.mark.asyncio
@patch('src.chat_system.MEMORY_RETRIEVAL_ENABLED', True)
async def test_retrieve_memory_block_no_embedding_service(mock_persona):
    """Returns None when embedding service is not initialized."""
    with patch('src.chat_system.load_personas_from_file', return_value={}), \
         patch('src.chat_system.get_model_list', return_value={}):
        system = ChatSystem(memory_manager=MagicMock(), text_engine=MagicMock())

    result = await system._retrieve_memory_block(
        mock_persona, "user1", "chan", None,
        [{"role": "user", "content": "Hello"}],
    )
    assert result is None


@pytest.mark.asyncio
@patch('src.chat_system.MEMORY_RETRIEVAL_ENABLED', True)
async def test_retrieve_memory_block_no_summaries(chat_system_with_memory, mock_persona):
    """Returns None when no summaries are found."""
    system, mm, _ = chat_system_with_memory
    mm.retrieve_relevant_summaries.return_value = []

    result = await system._retrieve_memory_block(
        mock_persona, "user1", "chan", None,
        [{"role": "user", "content": "Hello"}],
    )
    assert result is None


@pytest.mark.asyncio
@patch('src.chat_system.MEMORY_RETRIEVAL_ENABLED', True)
async def test_retrieve_memory_block_empty_history(chat_system_with_memory, mock_persona):
    """Returns None when conversation history has no text content."""
    system, mm, _ = chat_system_with_memory
    mm.retrieve_relevant_summaries.return_value = [
        {'summary_id': 1, 'content': 'facts', 'embedding': _unit_blob(1.0, 0.0),
         'channel': 'chan', 'persona_name': 'p1', 'created_at': datetime.now(),
         'segment_id': 1, 'model_name': 'm', 'start_interaction_id': 1, 'end_interaction_id': 3}
    ]

    result = await system._retrieve_memory_block(
        mock_persona, "user1", "chan", None,
        [],  # empty history
    )
    assert result is None


@pytest.mark.asyncio
@patch('src.chat_system.MEMORY_RETRIEVAL_ENABLED', True)
@patch('src.chat_system.MEMORY_MAX_SUMMARIES_IN_CONTEXT', 2)
async def test_retrieve_memory_block_top_k_limit(chat_system_with_memory, mock_persona):
    """Only top-K summaries are fetched from the database."""
    system, mm, emb_service = chat_system_with_memory

    mm.retrieve_relevant_summaries.return_value = [
        {'summary_id': i, 'segment_id': i, 'content': f'Fact {i}',
         'embedding': _unit_blob(1.0, 0.0), 'model_name': 'test-model',
         'created_at': datetime.now(), 'channel': 'chan', 'persona_name': 'p1',
         'start_interaction_id': i, 'end_interaction_id': i + 2}
        for i in range(2)
    ]
    emb_service.encode = AsyncMock(return_value=[_unit_blob(1.0, 0.0)])

    result = await system._retrieve_memory_block(
        mock_persona, "user1", "chan", None,
        [{"role": "user", "content": "Hello"}],
    )

    assert result is not None
    # Verify the database limit argument was correctly passed
    mm.retrieve_relevant_summaries.assert_called_once()
    assert mm.retrieve_relevant_summaries.call_args.kwargs['limit'] == 2


@pytest.mark.asyncio
@patch('src.chat_system.MEMORY_RETRIEVAL_ENABLED', True)
async def test_retrieve_memory_block_ambient_tag(chat_system_with_memory, mock_persona):
    """Ambient summaries are tagged in the formatted block."""
    system, mm, emb_service = chat_system_with_memory

    mm.retrieve_relevant_summaries.return_value = [
        {
            'summary_id': 1, 'segment_id': 1, 'content': '- Ambient observation',
            'embedding': _unit_blob(1.0, 0.0), 'model_name': 'test-model',
            'created_at': datetime.now(), 'channel': 'general',
            'persona_name': 'ambient',
            'start_interaction_id': 1, 'end_interaction_id': 3,
        }
    ]
    emb_service.encode = AsyncMock(return_value=[_unit_blob(1.0, 0.0)])

    result = await system._retrieve_memory_block(
        mock_persona, "user1", "chan", None,
        [{"role": "user", "content": "What happened?"}],
    )

    assert result is not None
    assert "ambient" in result
    assert "#general" in result


# --- _relative_time Tests ---

def test_relative_time_days():
    assert "2 days ago" == _relative_time(datetime.now() - timedelta(days=2))


def test_relative_time_weeks():
    assert "1 week ago" == _relative_time(datetime.now() - timedelta(weeks=1))


def test_relative_time_hours():
    assert "3 hours ago" == _relative_time(datetime.now() - timedelta(hours=3))


# --- Memory Block Formatting ---

def test_format_memory_block_score_ordered():
    """Entries are in the order passed (already score-sorted by caller)."""
    summaries = [
        (0.9, {'channel': 'chan-a', 'persona_name': 'p1',
                'created_at': datetime.now(), 'content': '- High relevance'}),
        (0.3, {'channel': 'chan-b', 'persona_name': 'p1',
                'created_at': datetime.now(), 'content': '- Low relevance'}),
    ]
    result = ChatSystem._format_memory_block(summaries)
    assert result is not None
    high_idx = result.index("High relevance")
    low_idx = result.index("Low relevance")
    assert high_idx < low_idx


def test_format_memory_block_empty():
    """Returns None for empty input."""
    assert ChatSystem._format_memory_block([]) is None
