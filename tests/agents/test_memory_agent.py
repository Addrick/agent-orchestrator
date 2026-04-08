# tests/agents/test_memory_agent.py

import math
import struct
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

from src.agents.memory_agent import MemoryAgent
from config.global_config import GEMINI_EMBEDDING_001_TPM


# --- Helpers ---

def _unit_blob(*components: float) -> bytes:
    """Create a normalized float32 BLOB from components."""
    vec = list(components)
    norm = math.sqrt(sum(v * v for v in vec))
    if norm > 0:
        vec = [v / norm for v in vec]
    return struct.pack(f'{len(vec)}f', *vec)


def _blob_dim(dim: Any = 4,axis: Any = 0) -> Any:
    """Create a unit vector BLOB along a specific axis."""
    vec = [0.0] * dim
    vec[axis] = 1.0
    return struct.pack(f'{dim}f', *vec)


# --- Fixtures ---

@pytest.fixture
def mock_chat_system() -> Any:
    cs = MagicMock()
    cs.text_engine = MagicMock()
    cs.memory_manager = MagicMock()
    cs.personas = {}
    return cs


@pytest.fixture
@patch('src.agents.base.load_system_personas_from_file', return_value={})
def memory_agent(mock_load: Any,mock_chat_system: Any) -> Any:
    agent = MemoryAgent(mock_chat_system, agent_config={
        "similarity_threshold": 0.3,
        "min_segment_size": 3,
        "batch_size": 200,
        "persona": "memory_summarizer",
    })
    return agent


# --- Init Tests ---

class TestMemoryAgentInit:
    @patch('src.agents.base.load_system_personas_from_file', return_value={})
    def test_init_stores_config(self,mock_load: Any,mock_chat_system: Any) -> None:
        agent = MemoryAgent(mock_chat_system, agent_config={
            "similarity_threshold": 0.5,
            "min_segment_size": 5,
            "batch_size": 100,
        })
        assert agent._similarity_threshold == 0.5
        assert agent._min_segment_size == 5
        assert agent._batch_size == 100

    @patch('src.agents.base.load_system_personas_from_file', return_value={})
    def test_init_defaults(self,mock_load: Any,mock_chat_system: Any) -> None:
        agent = MemoryAgent(mock_chat_system)
        assert agent._similarity_threshold == 0.3
        assert agent._min_segment_size == 3
        assert agent._batch_size == 100

    @patch('src.agents.base.load_system_personas_from_file', return_value={})
    def test_lazy_embedding_service(self,mock_load: Any,mock_chat_system: Any) -> None:
        agent = MemoryAgent(mock_chat_system)
        assert agent._embedding_service is None


# --- Segmentation Tests ---

class TestSegmentation:
    def _no_seed(self,memory_agent: Any) -> Any:
        """Configure mocks so centroid seeding returns None (no previous segment)."""
        memory_agent.memory_manager.get_last_segment_tail_embeddings.return_value = None

    def test_all_similar_messages_one_segment(self,memory_agent: Any) -> None:
        """All messages with identical embeddings produce one segment."""
        self._no_seed(memory_agent)
        blob = _blob_dim(4, axis=0)
        messages = [
            {'interaction_id': i, 'content': f'msg {i}', 'author_role': 'user', 'author_name': 'Alice'}
            for i in range(5)
        ]
        embeddings = [blob] * 5

        segments = memory_agent._segment_by_similarity(
            messages, embeddings, "chan", "p1", None
        )
        assert len(segments) == 1
        assert segments[0]['count'] == 5
        assert segments[0]['start_id'] == 0
        assert segments[0]['end_id'] == 4

    def test_orthogonal_messages_multiple_segments(self,memory_agent: Any) -> None:
        """Messages with orthogonal embeddings produce multiple segments (after min_size)."""
        self._no_seed(memory_agent)
        messages = [
            {'interaction_id': i, 'content': f'msg {i}', 'author_role': 'user', 'author_name': 'Alice'}
            for i in range(8)
        ]
        # First 4 messages along axis 0, then 4 along axis 1 (orthogonal)
        embeddings = [_blob_dim(4, axis=0)] * 4 + [_blob_dim(4, axis=1)] * 4

        # Threshold 0.3, min_size 3 -> should cut after first group
        segments = memory_agent._segment_by_similarity(
            messages, embeddings, "chan", "p1", None
        )
        assert len(segments) >= 2

    def test_min_segment_size_enforced(self,memory_agent: Any) -> None:
        """No cut happens when current segment is smaller than min_segment_size."""
        self._no_seed(memory_agent)
        messages = [
            {'interaction_id': i, 'content': f'msg {i}', 'author_role': 'user', 'author_name': 'Alice'}
            for i in range(4)
        ]
        # Alternate axes every message — would cut if min_size weren't enforced
        embeddings = [
            _blob_dim(4, axis=0),
            _blob_dim(4, axis=0),  # 2 similar
            _blob_dim(4, axis=1),  # switch — but only 2 in current, min is 3
            _blob_dim(4, axis=1),
        ]

        segments = memory_agent._segment_by_similarity(
            messages, embeddings, "chan", "p1", None
        )
        # No cut possible — 2 < min_size=3 at the switch point
        assert len(segments) == 1

    def test_too_few_messages(self,memory_agent: Any) -> None:
        """Empty input produces no segments."""
        self._no_seed(memory_agent)
        segments = memory_agent._segment_by_similarity([], [], "chan", "p1", None)
        assert len(segments) == 0

    def test_centroid_renormalization(self,memory_agent: Any) -> None:
        """After incremental updates, centroid magnitude stays approximately 1.0."""
        self._no_seed(memory_agent)
        blob = _blob_dim(4, axis=0)
        messages = [
            {'interaction_id': i, 'content': f'msg {i}', 'author_role': 'user', 'author_name': 'Alice'}
            for i in range(20)
        ]
        embeddings = [blob] * 20

        # Run segmentation — internally the centroid is updated 19 times
        segments = memory_agent._segment_by_similarity(
            messages, embeddings, "chan", "p1", None
        )
        # All same direction -> one segment
        assert len(segments) == 1


class TestCentroidSeeding:
    def test_seed_from_previous_segment(self,memory_agent: Any) -> None:
        """Centroid seeding loads tail embeddings from DB and averages them."""
        blob_a = _blob_dim(4, axis=0)
        blob_b = _blob_dim(4, axis=1)
        memory_agent.memory_manager.get_last_segment_tail_embeddings.return_value = [blob_a, blob_b]
        memory_agent._embedding_service = MagicMock()
        memory_agent._embedding_service.model_name = "test-model"

        centroid = memory_agent._seed_centroid_from_previous("chan", "p1", None)
        assert centroid is not None
        # Centroid should be normalized
        assert abs(np.linalg.norm(centroid) - 1.0) < 1e-5

    def test_seed_returns_none_when_no_previous(self,memory_agent: Any) -> None:
        """Returns None when no previous segment exists."""
        memory_agent.memory_manager.get_last_segment_tail_embeddings.return_value = None
        memory_agent._embedding_service = MagicMock()
        memory_agent._embedding_service.model_name = "test-model"

        centroid = memory_agent._seed_centroid_from_previous("chan", "p1", None)
        assert centroid is None

    def test_topic_continuation_across_batches(self,memory_agent: Any) -> None:
        """When seeded with previous tail, same-topic messages extend rather than split."""
        blob = _blob_dim(4, axis=0)
        memory_agent.memory_manager.get_last_segment_tail_embeddings.return_value = [blob, blob, blob]
        memory_agent._embedding_service = MagicMock()
        memory_agent._embedding_service.model_name = "test-model"

        # New messages are the same topic as the seed
        messages = [
            {'interaction_id': i, 'content': f'msg {i}', 'author_role': 'user', 'author_name': 'Alice'}
            for i in range(5)
        ]
        embeddings = [blob] * 5

        segments = memory_agent._segment_by_similarity(
            messages, embeddings, "chan", "p1", None
        )
        # Should be one segment — no artificial split at batch boundary
        assert len(segments) == 1


# --- Summarization Tests ---

class TestSummarization:
    @pytest.mark.asyncio
    async def test_summarize_calls_llm(self, memory_agent: Any) -> None:
        """Summarization calls text_engine with correct persona and transcript."""
        mock_persona = MagicMock()
        mock_persona.get_prompt.return_value = "prompt"
        mock_persona.get_config_for_engine.return_value = {"model_name": "test"}
        memory_agent.chat_system.personas = {"memory_summarizer": mock_persona}

        memory_agent.text_engine.generate_response = AsyncMock(return_value=(
            {"type": "text", "content": "- Fact 1\n- Fact 2"}, {}
        ))

        mock_emb_service = MagicMock()
        mock_emb_service.encode_single = AsyncMock(return_value=b'\x00' * 16)

        segment = {
            'messages': [
                {'author_role': 'user', 'author_name': 'Alice', 'content': 'Hello'},
                {'author_role': 'assistant', 'author_name': 'Bot', 'content': 'Hi there'},
            ],
        }

        result = await memory_agent._summarize_segment(segment, mock_emb_service)
        assert result is not None
        facts_text, summary_emb = result
        assert "Fact 1" in facts_text
        assert isinstance(summary_emb, bytes)

        # Verify LLM was called
        memory_agent.text_engine.generate_response.assert_called_once()

    @pytest.mark.asyncio
    async def test_summarize_handles_missing_persona(self, memory_agent: Any) -> None:
        """Returns None when persona is not found."""
        memory_agent.chat_system.personas = {}

        mock_emb_service = MagicMock()
        segment: dict[str, Any] = {'messages': []}

        result = await memory_agent._summarize_segment(segment, mock_emb_service)
        assert result is None

    @pytest.mark.asyncio
    async def test_summarize_handles_llm_error(self, memory_agent: Any) -> None:
        """Returns None when LLM call fails."""
        mock_persona = MagicMock()
        mock_persona.get_config_for_engine.return_value = {"model_name": "test"}
        memory_agent.chat_system.personas = {"memory_summarizer": mock_persona}
        memory_agent.text_engine.generate_response = AsyncMock(side_effect=Exception("API error"))

        mock_emb_service = MagicMock()
        segment = {
            'messages': [{'author_role': 'user', 'author_name': 'Alice', 'content': 'Hello'}],
        }

        result = await memory_agent._summarize_segment(segment, mock_emb_service)
        assert result is None


# --- Deploy Tests ---

class TestDeploy:
    @pytest.mark.asyncio
    @patch('src.agents.memory_agent.GeminiEmbeddingProvider')
    async def test_deploy_processes_channels(self, mock_provider_cls: Any, memory_agent: Any) -> None:
        """Deploy processes each active channel."""
        mock_provider = MagicMock()
        mock_provider.model_name = "test-model"
        mock_provider.dimensions = 4
        mock_provider.max_input_tokens = None
        mock_provider.encode = AsyncMock(return_value=[[1.0, 0.0, 0.0, 0.0]])
        mock_provider_cls.return_value = mock_provider

        memory_agent.memory_manager.get_active_channels.return_value = [
            ("chan-a", "p1", None),
        ]
        memory_agent._process_channel = AsyncMock()

        await memory_agent.deploy()
        memory_agent._process_channel.assert_called_once()

    @pytest.mark.asyncio
    @patch('src.agents.memory_agent.GeminiEmbeddingProvider')
    async def test_deploy_no_channels(self, mock_provider_cls: Any, memory_agent: Any) -> None:
        """Deploy does nothing when no channels need processing."""
        mock_provider = MagicMock()
        mock_provider.model_name = "test-model"
        mock_provider.dimensions = 4
        mock_provider.max_input_tokens = None
        mock_provider_cls.return_value = mock_provider

        memory_agent.memory_manager.get_active_channels.return_value = []
        memory_agent._process_channel = AsyncMock()

        await memory_agent.deploy()
        memory_agent._process_channel.assert_not_called()

    @pytest.mark.asyncio
    @patch('src.agents.memory_agent.GeminiEmbeddingProvider')
    async def test_deploy_skips_below_min_segment_size(self, mock_provider_cls: Any, memory_agent: Any) -> None:
        """Channels with fewer messages than min_segment_size are skipped."""
        mock_provider = MagicMock()
        mock_provider.model_name = "test-model"
        mock_provider.dimensions = 4
        mock_provider.max_input_tokens = None
        mock_provider.encode = AsyncMock(return_value=[[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]])
        mock_provider_cls.return_value = mock_provider

        memory_agent.memory_manager.get_active_channels.return_value = [("chan", "p1", None)]
        # Phase 1: None to embed
        memory_agent.memory_manager.get_unembedded_messages.return_value = []
        # Phase 2: Only 2 messages, min_segment_size is 3
        memory_agent.memory_manager.get_unsegmented_embedded_messages.return_value = [
            {'interaction_id': 1, 'content': 'hi', 'author_role': 'user', 'author_name': 'Alice', 'embedding': b'00', 'timestamp': None},
            {'interaction_id': 2, 'content': 'hello', 'author_role': 'user', 'author_name': 'Bob', 'embedding': b'00', 'timestamp': None},
        ]
        memory_agent.memory_manager.get_last_segment_tail_embeddings.return_value = None

        await memory_agent.deploy()
        # No segments/summaries should be stored
        memory_agent.memory_manager.transaction.assert_not_called()

    @pytest.mark.asyncio
    @patch('src.agents.memory_agent.GeminiEmbeddingProvider')
    async def test_deploy_respects_shutdown(self, mock_provider_cls: Any, memory_agent: Any) -> None:
        """Deploy stops processing channels when shutdown is signalled."""
        mock_provider = MagicMock()
        mock_provider.model_name = "test-model"
        mock_provider.dimensions = 4
        mock_provider.max_input_tokens = None
        mock_provider_cls.return_value = mock_provider

        memory_agent.memory_manager.get_active_channels.return_value = [
            ("chan-a", "p1", None),
            ("chan-b", "p1", None),
        ]
        memory_agent._process_channel = AsyncMock()
        memory_agent._shutdown_event.set()

        await memory_agent.deploy()
        memory_agent._process_channel.assert_not_called()

    @pytest.mark.asyncio
    @patch('src.agents.memory_agent.GeminiEmbeddingProvider')
    async def test_deploy_continues_on_channel_error(self, mock_provider_cls: Any, memory_agent: Any) -> None:
        """Deploy continues to next channel if one fails."""
        mock_provider = MagicMock()
        mock_provider.model_name = "test-model"
        mock_provider.dimensions = 4
        mock_provider.max_input_tokens = None
        mock_provider_cls.return_value = mock_provider

        memory_agent.memory_manager.get_active_channels.return_value = [
            ("chan-a", "p1", None),
            ("chan-b", "p1", None),
        ]
        call_count = 0

        async def side_effect(*args: Any, **kwargs: Any) -> None:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise Exception("API error")

        memory_agent._process_channel = AsyncMock(side_effect=side_effect)

        await memory_agent.deploy()
        assert memory_agent._process_channel.call_count == 2


# --- Transaction Model Tests ---

class TestTransactionModel:
    @pytest.mark.asyncio
    @patch('asyncio.sleep')
    async def test_embedding_failure_does_not_prevent_segmentation(self, mock_sleep: Any, memory_agent: Any) -> None:
        """If embedding fails, Phase 2 still runs on any prior embeddings."""
        memory_agent._embedding_service = MagicMock()
        memory_agent._embedding_service.model_name = "test-model"
        memory_agent._embedding_service.encode = AsyncMock(side_effect=Exception("API error"))

        memory_agent.memory_manager.get_unembedded_messages.return_value = [
            {'interaction_id': i, 'content': f'msg {i}', 'author_role': 'user', 'author_name': 'Alice'}
            for i in range(5)
        ]
        # Phase 2 query returns nothing — no prior embeddings to segment
        memory_agent.memory_manager.get_unsegmented_embedded_messages.return_value = []

        # Should not raise — Phase 1 failure is caught, Phase 2 runs
        await memory_agent._process_channel("chan", "p1", None, memory_agent._embedding_service)

        # Phase 2 was attempted despite Phase 1 failure
        memory_agent.memory_manager.get_unsegmented_embedded_messages.assert_called_once()


# --- Config Injection Tests ---

class TestConfigInjection:
    @patch('src.agents.base.load_system_personas_from_file', return_value={})
    def test_batch_size_from_config(self,mock_load: Any,mock_chat_system: Any) -> None:
        agent = MemoryAgent(mock_chat_system, agent_config={"batch_size": 50})
        assert agent._batch_size == 50

    @patch('src.agents.base.load_system_personas_from_file', return_value={})
    def test_max_tokens_per_chunk_default(self,mock_load: Any,mock_chat_system: Any) -> None:
        agent = MemoryAgent(mock_chat_system)
        assert agent._max_tokens_per_chunk == GEMINI_EMBEDDING_001_TPM

    @patch('src.agents.base.load_system_personas_from_file', return_value={})
    def test_max_tokens_per_chunk_from_config(self,mock_load: Any,mock_chat_system: Any) -> None:
        agent = MemoryAgent(mock_chat_system, agent_config={"max_tokens_per_chunk": 5000})
        assert agent._max_tokens_per_chunk == 5000

    @patch('src.agents.base.load_system_personas_from_file', return_value={})
    def test_agent_name(self,mock_load: Any,mock_chat_system: Any) -> None:
        agent = MemoryAgent(mock_chat_system)
        assert agent.agent_name == "memory"

    @patch('src.agents.base.load_system_personas_from_file', return_value={})
    def test_allowed_channels_default_none(self,mock_load: Any,mock_chat_system: Any) -> None:
        agent = MemoryAgent(mock_chat_system)
        assert agent._allowed_channels is None

    @patch('src.agents.base.load_system_personas_from_file', return_value={})
    def test_allowed_channels_from_config(self,mock_load: Any,mock_chat_system: Any) -> None:
        agent = MemoryAgent(mock_chat_system, agent_config={
            "allowed_channels": [
                {"channel": "chan-a", "server_id": "s1"},
                {"channel": "chan-b", "server_id": "s2"},
            ]
        })
        assert agent._allowed_channels == [
            {"channel": "chan-a", "server_id": "s1"},
            {"channel": "chan-b", "server_id": "s2"},
        ]


class TestChunkMessages:
    """Token-aware chunking respects both item count and token budget."""

    def test_single_chunk_under_limits(self) -> None:
        msgs = [{'content': 'short'} for _ in range(5)]
        chunks = MemoryAgent._chunk_messages(msgs, max_items=100, max_tokens=10000)
        assert len(chunks) == 1
        assert len(chunks[0]) == 5

    def test_splits_on_item_count(self) -> None:
        msgs = [{'content': 'x'} for _ in range(250)]
        chunks = MemoryAgent._chunk_messages(msgs, max_items=100, max_tokens=10**9)
        assert [len(c) for c in chunks] == [100, 100, 50]

    def test_splits_on_token_budget(self) -> None:
        # 4 chars/token estimate; 4000 chars = 1000 tokens per message
        msgs = [{'content': 'a' * 4000} for _ in range(10)]
        chunks = MemoryAgent._chunk_messages(msgs, max_items=100, max_tokens=2500)
        # Each chunk fits at most 2 messages (2000 tokens) before the 3rd would exceed 2500
        assert all(len(c) <= 2 for c in chunks)
        assert sum(len(c) for c in chunks) == 10

    def test_oversized_message_gets_own_chunk(self) -> None:
        # A single message larger than the budget must still be emitted
        # (EmbeddingService truncates it before sending).
        msgs = [
            {'content': 'a' * 200000},  # ~50k tokens, way over budget
            {'content': 'short'},
        ]
        chunks = MemoryAgent._chunk_messages(msgs, max_items=100, max_tokens=25000)
        assert len(chunks) == 2
        assert chunks[0] == [msgs[0]]
        assert chunks[1] == [msgs[1]]

    def test_empty_input(self) -> None:
        assert MemoryAgent._chunk_messages([], max_items=100, max_tokens=25000) == []


class TestAllowedChannels:
    @pytest.mark.asyncio
    @patch('src.agents.memory_agent.GeminiEmbeddingProvider')
    async def test_deploy_filters_to_allowed_channels(self, mock_provider_cls: Any, memory_agent: Any) -> None:
        """Deploy only processes channels in allowed_channels when configured."""
        mock_provider = MagicMock()
        mock_provider.model_name = "test-model"
        mock_provider.dimensions = 4
        mock_provider.max_input_tokens = None
        mock_provider_cls.return_value = mock_provider

        memory_agent._allowed_channels = [{"channel": "chan-b", "server_id": "s1"}]
        memory_agent.memory_manager.get_active_channels.return_value = [
            ("chan-a", "p1", None),
            ("chan-b", "p1", None),
            ("chan-b", "p1", "s1"),
            ("chan-c", "p1", "s1"),
        ]
        memory_agent._process_channel = AsyncMock()

        await memory_agent.deploy()
        memory_agent._process_channel.assert_called_once()
        call_args = memory_agent._process_channel.call_args
        assert call_args[0][0] == "chan-b"
        assert call_args[0][2] == "s1"

    @pytest.mark.asyncio
    @patch('src.agents.memory_agent.GeminiEmbeddingProvider')
    async def test_deploy_no_filter_when_allowed_channels_none(self, mock_provider_cls: Any, memory_agent: Any) -> None:
        """Deploy processes all channels when allowed_channels is not set."""
        mock_provider = MagicMock()
        mock_provider.model_name = "test-model"
        mock_provider.dimensions = 4
        mock_provider.max_input_tokens = None
        mock_provider_cls.return_value = mock_provider

        memory_agent._allowed_channels = None
        memory_agent.memory_manager.get_active_channels.return_value = [
            ("chan-a", "p1", None),
            ("chan-b", "p1", None),
        ]
        memory_agent._process_channel = AsyncMock()

        await memory_agent.deploy()
        assert memory_agent._process_channel.call_count == 2
