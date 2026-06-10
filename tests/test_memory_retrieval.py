# tests/test_memory_retrieval.py

import math
import struct
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from memory.memory_manager import MemoryManager
from src.request_builder import RequestBuilder, _relative_time
from src.embedding_service import EmbeddingService
from src.memory.backend.base import MemoryHit
from src.persona import MemoryMode
from tests.helpers import make_chat_system


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


# --- RequestBuilder.retrieve_memory_block Tests ---

@pytest.fixture
def mock_persona():
    p = MagicMock()
    p.get_name.return_value = "test_persona"
    p.get_memory_mode.return_value = MemoryMode.CHANNEL_ISOLATED
    return p


@pytest.fixture
def chat_system_with_memory():
    """ChatSystem with mocked dependencies for memory retrieval tests.

    DP-113: ChatSystem now calls `memory_backend.recall(...)` rather than
    `memory_manager.retrieve_relevant_summaries(...)`. Tests stub the
    backend's recall directly via `system.memory_backend.recall = AsyncMock(...)`.
    """
    mm = MagicMock()
    mm.backend = MagicMock()  # spec= would block this; use plain MagicMock
    te = MagicMock()
    system = make_chat_system(memory_manager=mm, text_engine=te)

    emb_service = MagicMock(spec=EmbeddingService)
    emb_service.model_name = "test-model"
    system._embedding_service = emb_service
    return system, mm, emb_service


@pytest.mark.asyncio
@patch('src.request_builder.MEMORY_RETRIEVAL_ENABLED', True)
async def test_retrieve_memory_block_returns_formatted_block(chat_system_with_memory, mock_persona):
    """Memory block is returned with proper formatting."""
    system, mm, emb_service = chat_system_with_memory

    system.memory_backend.recall = AsyncMock(return_value=[
        MemoryHit(
            id="1", content="- Security alert found", score=0.9, untrusted=False,
            tags=["channel:security", "persona:sage"],
            timestamp=datetime.now() - timedelta(days=2),
            metadata={"segment_id": 1},
        ),
    ])

    result, is_untrusted = await system.request_builder.retrieve_memory_block(
        mock_persona, "user1", "chan", None,
        [{"role": "user", "content": "What about the security issue?"}],
    )

    assert result is not None
    assert "<memory>" in result
    assert "</memory>" in result
    assert "Security alert found" in result
    assert "#security" in result


@pytest.mark.asyncio
@patch('src.request_builder.MEMORY_RETRIEVAL_ENABLED', False)
async def test_retrieve_memory_block_disabled(chat_system_with_memory, mock_persona):
    """Returns None when feature is disabled."""
    system, _, _ = chat_system_with_memory

    result, is_untrusted = await system.request_builder.retrieve_memory_block(
        mock_persona, "user1", "chan", None,
        [{"role": "user", "content": "Hello"}],
    )
    assert result is None
    assert is_untrusted is False


@pytest.mark.asyncio
@patch('src.request_builder.MEMORY_RETRIEVAL_ENABLED', True)
async def test_retrieve_memory_block_no_embedding_service(mock_persona):
    """Returns None when embedding service is not initialized."""
    system = make_chat_system(memory_manager=MagicMock(), text_engine=MagicMock())

    result, is_untrusted = await system.request_builder.retrieve_memory_block(
        mock_persona, "user1", "chan", None,
        [{"role": "user", "content": "Hello"}],
    )
    assert result is None
    assert is_untrusted is False


@pytest.mark.asyncio
@patch('src.request_builder.MEMORY_RETRIEVAL_ENABLED', True)
async def test_retrieve_memory_block_no_summaries(chat_system_with_memory, mock_persona):
    """Returns None when no summaries are found."""
    system, mm, _ = chat_system_with_memory
    system.memory_backend.recall = AsyncMock(return_value=[])

    result, is_untrusted = await system.request_builder.retrieve_memory_block(
        mock_persona, "user1", "chan", None,
        [{"role": "user", "content": "Hello"}],
    )
    assert result is None
    assert is_untrusted is False


@pytest.mark.asyncio
@patch('src.request_builder.MEMORY_RETRIEVAL_ENABLED', True)
async def test_retrieve_memory_block_empty_history(chat_system_with_memory, mock_persona):
    """Returns None when conversation history has no text content."""
    system, mm, _ = chat_system_with_memory
    system.memory_backend.recall = AsyncMock(return_value=[
        MemoryHit(id="1", content="facts", score=0.5, tags=["channel:chan", "persona:p1"]),
    ])

    result, is_untrusted = await system.request_builder.retrieve_memory_block(
        mock_persona, "user1", "chan", None,
        [],  # empty history
    )
    assert result is None
    assert is_untrusted is False


@pytest.mark.asyncio
@patch('src.request_builder.MEMORY_RETRIEVAL_ENABLED', True)
@patch('src.request_builder.MEMORY_MAX_SUMMARIES_IN_CONTEXT', 2)
async def test_retrieve_memory_block_top_k_limit(chat_system_with_memory, mock_persona):
    """Only top-K summaries are fetched from the database."""
    system, mm, emb_service = chat_system_with_memory

    system.memory_backend.recall = AsyncMock(return_value=[
        MemoryHit(id=str(i), content=f"Fact {i}", score=0.5,
                  tags=["channel:chan", "persona:p1"])
        for i in range(2)
    ])

    result, is_untrusted = await system.request_builder.retrieve_memory_block(
        mock_persona, "user1", "chan", None,
        [{"role": "user", "content": "Hello"}],
    )

    assert result is not None
    system.memory_backend.recall.assert_called_once()
    assert system.memory_backend.recall.call_args.kwargs['k'] == 2


@pytest.mark.asyncio
@patch('src.request_builder.MEMORY_RETRIEVAL_ENABLED', True)
async def test_retrieve_memory_block_ambient_tag(chat_system_with_memory, mock_persona):
    """Ambient summaries are tagged in the formatted block."""
    system, mm, emb_service = chat_system_with_memory

    system.memory_backend.recall = AsyncMock(return_value=[
        MemoryHit(id="1", content="- Ambient observation", score=0.7,
                  tags=["channel:general", "persona:ambient"],
                  timestamp=datetime.now()),
    ])

    result, is_untrusted = await system.request_builder.retrieve_memory_block(
        mock_persona, "user1", "chan", None,
        [{"role": "user", "content": "What happened?"}],
    )

    assert result is not None
    assert "ambient" in result
    assert "#general" in result


@pytest.mark.asyncio
@patch('src.request_builder.MEMORY_RETRIEVAL_ENABLED', True)
async def test_retrieve_memory_block_untrusted_propagation(chat_system_with_memory, mock_persona):
    """Untrusted flag is propagated when an untrusted summary is retrieved."""
    system, mm, emb_service = chat_system_with_memory

    system.memory_backend.recall = AsyncMock(return_value=[
        MemoryHit(id="1", content="- Untrusted content", score=0.6,
                  untrusted=True, tags=["channel:web", "persona:p1"]),
    ])

    result, is_untrusted = await system.request_builder.retrieve_memory_block(
        mock_persona, "user1", "chan", None,
        [{"role": "user", "content": "Hello"}],
    )

    assert result is not None
    assert is_untrusted is True
    assert "Untrusted content" in result


# --- _relative_time Tests ---

def test_relative_time_days():
    assert "2 days ago" == _relative_time(datetime.now() - timedelta(days=2))


def test_relative_time_weeks():
    assert "1 week ago" == _relative_time(datetime.now() - timedelta(weeks=1))


def test_relative_time_hours():
    assert "3 hours ago" == _relative_time(datetime.now() - timedelta(hours=3))


# --- Memory Block Formatting ---

def test_format_memory_block_score_ordered():
    """Entries appear in the order passed (caller is responsible for sorting)."""
    hits = [
        MemoryHit(id="1", content="- High relevance", score=0.9,
                  tags=["channel:chan-a", "persona:p1"], timestamp=datetime.now()),
        MemoryHit(id="2", content="- Low relevance", score=0.3,
                  tags=["channel:chan-b", "persona:p1"], timestamp=datetime.now()),
    ]
    result = RequestBuilder.format_memory_block(hits)
    assert result is not None
    high_idx = result.index("High relevance")
    low_idx = result.index("Low relevance")
    assert high_idx < low_idx


def test_format_memory_block_empty():
    """Returns None for empty input."""
    assert RequestBuilder.format_memory_block([]) is None
