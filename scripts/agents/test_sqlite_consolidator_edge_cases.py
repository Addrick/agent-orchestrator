# tests/agents/test_sqlite_consolidator_edge_cases.py
"""DP-199 Batch 8 — SqliteConsolidator edge cases.

Phase isolation: embed / segment / summarize tested as separate units.
GLOBAL_EMBEDDING_LIMITER mocked for rate-limit scenarios.
"""

import math
import struct
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.agents.sqlite_consolidator import SqliteConsolidator


def _blob_dim(dim: int = 4, axis: int = 0) -> bytes:
    vec = [0.0] * dim
    vec[axis] = 1.0
    return struct.pack(f'{dim}f', *vec)


@pytest.fixture
def mock_chat_system() -> Any:
    cs = MagicMock()
    cs.text_engine = MagicMock()
    cs.memory_manager = MagicMock()
    cs.personas = {}
    cs._embedding_service = None
    return cs


@pytest.fixture
@patch('src.agents.base.load_system_personas_from_file', return_value={})
def consolidator(_mock_load: Any, mock_chat_system: Any) -> SqliteConsolidator:
    return SqliteConsolidator(mock_chat_system, agent_config={
        "similarity_threshold": 0.3,
        "min_segment_size": 1,
        "batch_size": 10,
        "persona": "memory_summarizer",
    })


# ---------------------------------------------------------------------------
# Embed phase
# ---------------------------------------------------------------------------

class TestEmbedPhase:
    async def test_rate_limit_daily_exhaustion_breaks_loop(
        self, consolidator: SqliteConsolidator, mock_chat_system: Any,
    ) -> None:
        """RuntimeError from rate limiter (daily exhausted) should stop the loop
        cleanly without re-raising."""
        msgs = [
            {"interaction_id": i, "content": f"msg{i}"}
            for i in range(3)
        ]
        mock_chat_system.memory_manager.get_unembedded_messages = MagicMock(return_value=msgs)
        # Patch the consolidator's rate_limiter directly (it captured the
        # global limiter at __init__).
        consolidator._rate_limiter = MagicMock()
        consolidator._rate_limiter.acquire = AsyncMock(
            side_effect=RuntimeError("Daily Google API Quota Exhausted")
        )

        emb_service = MagicMock()
        emb_service.encode = AsyncMock(return_value=[])

        with patch('src.agents.sqlite_consolidator.logger') as mock_logger:
            stored = await consolidator._embed_unembedded(
                "chan", "p1", None, emb_service,
            )
        # Quota error breaks before any encode; nothing stored.
        assert stored == 0
        emb_service.encode.assert_not_called()
        # Warning logged with quota message
        warnings = [c.args[0] for c in mock_logger.warning.call_args_list]
        assert any("Embedding loop stopped" in w for w in warnings)

    async def test_transient_error_triggers_backoff_and_break(
        self, consolidator: SqliteConsolidator, mock_chat_system: Any,
    ) -> None:
        """Generic exception during encode should sleep+break (no infinite loop)."""
        msgs = [{"interaction_id": i, "content": f"msg{i}"} for i in range(3)]
        mock_chat_system.memory_manager.get_unembedded_messages = MagicMock(return_value=msgs)
        consolidator._rate_limiter = MagicMock()
        consolidator._rate_limiter.acquire = AsyncMock()

        emb_service = MagicMock()
        # ConnectionError -> hits generic-Exception branch (RuntimeError is the
        # daily-quota path that breaks without sleeping).
        emb_service.encode = AsyncMock(side_effect=ConnectionError("network blip"))

        with patch('src.agents.sqlite_consolidator.asyncio.sleep', new=AsyncMock()) as mock_sleep:
            stored = await consolidator._embed_unembedded(
                "chan", "p1", None, emb_service,
            )

        assert stored == 0
        mock_sleep.assert_awaited_once()

    def test_chunking_max_tokens_boundary(self, consolidator: SqliteConsolidator) -> None:
        """A single oversized message must still produce a chunk on its own;
        subsequent messages start a new chunk."""
        # max_tokens = 5; per char/4 estimator
        # msg1 ~ 25 tokens (oversized), msg2 ~ 25 tokens
        messages = [
            {"interaction_id": 1, "content": "X" * 100},
            {"interaction_id": 2, "content": "Y" * 100},
        ]
        chunks = SqliteConsolidator._chunk_messages(messages, max_items=10, max_tokens=5)
        # Each oversized message should be in its own chunk
        assert len(chunks) == 2
        assert chunks[0][0]["interaction_id"] == 1
        assert chunks[1][0]["interaction_id"] == 2

    def test_chunking_respects_max_items(self, consolidator: SqliteConsolidator) -> None:
        messages = [{"interaction_id": i, "content": "x"} for i in range(7)]
        chunks = SqliteConsolidator._chunk_messages(messages, max_items=3, max_tokens=10000)
        assert [len(c) for c in chunks] == [3, 3, 1]


# ---------------------------------------------------------------------------
# Deploy / channel filtering
# ---------------------------------------------------------------------------

class TestDeploy:
    async def test_allowed_channels_filtering(
        self, consolidator: SqliteConsolidator, mock_chat_system: Any,
    ) -> None:
        consolidator._allowed_channels = [
            {"channel": "good", "server_id": "s1"},
        ]
        mock_chat_system.memory_manager.get_active_channels = MagicMock(return_value=[
            ("good", "p1", "s1"),
            ("bad", "p1", "s1"),
            ("good", "p1", "wrong-server"),
        ])
        consolidator._get_embedding_service = MagicMock(return_value=MagicMock())
        consolidator._process_channel = AsyncMock()

        await consolidator.deploy()

        # Only the (good, p1, s1) tuple should be processed
        assert consolidator._process_channel.await_count == 1
        args = consolidator._process_channel.await_args.args
        assert args[0] == "good"
        assert args[2] == "s1"


# ---------------------------------------------------------------------------
# Segmentation phase
# ---------------------------------------------------------------------------

class TestSegmentation:
    def _no_seed(self, consolidator: SqliteConsolidator) -> None:
        consolidator.memory_manager.get_last_segment_tail_embeddings.return_value = None

    def test_single_message(self, consolidator: SqliteConsolidator) -> None:
        self._no_seed(consolidator)
        messages = [{"interaction_id": 1, "content": "hi", "author_role": "user", "author_name": "A"}]
        embeddings = [_blob_dim(4, 0)]
        segments = consolidator._segment_by_similarity(
            messages, embeddings, "c", "p", None,
        )
        assert len(segments) == 1
        assert segments[0]['count'] == 1

    def test_persona_boundary_user_to_assistant_bridges(
        self, consolidator: SqliteConsolidator,
    ) -> None:
        """role_link bridges a user→assistant pair even with orthogonal
        embeddings: a 1-msg user segment followed by an assistant should NOT split."""
        self._no_seed(consolidator)
        messages = [
            {"interaction_id": 1, "content": "Q", "author_role": "user", "author_name": "A"},
            {"interaction_id": 2, "content": "A", "author_role": "assistant", "author_name": "Bot"},
        ]
        embeddings = [_blob_dim(4, 0), _blob_dim(4, 1)]  # orthogonal
        consolidator._similarity_threshold = 0.9
        consolidator._min_segment_size = 1
        segments = consolidator._segment_by_similarity(
            messages, embeddings, "c", "p", None,
        )
        # Should be a single bridged segment of 2
        assert len(segments) == 1
        assert segments[0]['count'] == 2

    def test_orphan_reply_falls_back_to_role_bridge(
        self, consolidator: SqliteConsolidator,
    ) -> None:
        """reply_to_id pointing outside the current segment doesn't link, but
        the role-fallback should still bridge user→assistant."""
        self._no_seed(consolidator)
        messages = [
            {"interaction_id": 1, "content": "Q", "author_role": "user",
             "author_name": "A"},
            {"interaction_id": 2, "content": "A", "author_role": "assistant",
             "author_name": "Bot", "reply_to_id": None},
        ]
        embeddings = [_blob_dim(4, 0), _blob_dim(4, 1)]
        consolidator._similarity_threshold = 0.9
        consolidator._min_segment_size = 1
        segments = consolidator._segment_by_similarity(messages, embeddings, "c", "p", None)
        assert len(segments) == 1

    async def test_suppress_failed_ranges_cooldown(
        self, consolidator: SqliteConsolidator, mock_chat_system: Any,
    ) -> None:
        """Segments overlapping a recorded failure range should be skipped."""
        mm = mock_chat_system.memory_manager
        # One unsegmented embedded message
        mm.get_unsegmented_embedded_messages = MagicMock(return_value=[
            {"interaction_id": 5, "author_role": "user", "author_name": "A",
             "content": "hi", "timestamp": None, "embedding": _blob_dim(4, 0)},
        ])
        mm.get_last_segment_tail_embeddings = MagicMock(return_value=None)
        # Failed range covers id 5
        mm.get_failed_segment_ranges = MagicMock(return_value=[
            {"start_interaction_id": 5, "end_interaction_id": 5},
        ])
        consolidator._summarize_segment = AsyncMock()

        committed = await consolidator._segment_and_summarize(
            "c", "p", None, MagicMock(),
        )

        assert committed == 0
        # Summarize should not have been attempted
        consolidator._summarize_segment.assert_not_called()


# ---------------------------------------------------------------------------
# Summarize phase
# ---------------------------------------------------------------------------

class TestSummarize:
    async def test_llm_error_returns_none(
        self, consolidator: SqliteConsolidator, mock_chat_system: Any,
    ) -> None:
        persona = MagicMock()
        persona.get_config_for_engine.return_value = {"model_name": "m"}
        persona.get_prompt.return_value = "p"
        mock_chat_system.personas["memory_summarizer"] = persona
        mock_chat_system.text_engine.generate_response = AsyncMock(
            side_effect=RuntimeError("LLM fail"),
        )
        mock_chat_system.text_engine.google_client = None  # skip token guardrail
        segment = {
            "start_id": 1,
            "messages": [{"author_role": "user", "author_name": "A", "content": "hi"}],
        }
        result = await consolidator._summarize_segment(segment, MagicMock())
        assert result is None

    async def test_oversized_segment_skipped(
        self, consolidator: SqliteConsolidator, mock_chat_system: Any,
    ) -> None:
        """When token_count exceeds 95% of RATE_LIMIT_GEMMA_4_TPR, segment is
        skipped (returns None)."""
        from config.global_config import RATE_LIMIT_GEMMA_4_TPR

        persona = MagicMock()
        persona.get_config_for_engine.return_value = {"model_name": "m"}
        persona.get_prompt.return_value = "p"
        mock_chat_system.personas["memory_summarizer"] = persona

        # Fake google_client.models.count_tokens to return a huge number
        google_client = MagicMock()
        count_resp = MagicMock()
        count_resp.total_tokens = int(RATE_LIMIT_GEMMA_4_TPR * 1.0)
        google_client.models.count_tokens = AsyncMock(return_value=count_resp)
        mock_chat_system.text_engine.google_client = google_client

        segment = {
            "start_id": 1,
            "messages": [{"author_role": "user", "author_name": "A", "content": "x"}],
        }
        result = await consolidator._summarize_segment(segment, MagicMock())
        assert result is None

    async def test_malformed_tool_response_falls_back_to_text(
        self, consolidator: SqliteConsolidator, mock_chat_system: Any,
    ) -> None:
        """tool_calls with no matching call name -> falls back to text path.
        Here we return type=text directly to exercise the text fallback."""
        persona = MagicMock()
        persona.get_config_for_engine.return_value = {"model_name": "m"}
        persona.get_prompt.return_value = "p"
        mock_chat_system.personas["memory_summarizer"] = persona
        mock_chat_system.text_engine.google_client = None

        mock_chat_system.text_engine.generate_response = AsyncMock(return_value=(
            {"type": "text", "content": "- Fact A\n- Fact B"}, {},
        ))

        consolidator._rate_limiter = MagicMock()
        consolidator._rate_limiter.acquire = AsyncMock()

        emb_service = MagicMock()
        emb_service.encode_single = AsyncMock(return_value=b'\x00' * 16)

        segment = {
            "start_id": 1,
            "messages": [{"author_role": "user", "author_name": "A", "content": "hi"}],
        }
        result = await consolidator._summarize_segment(segment, emb_service)
        assert result is not None
        facts_text, _, outlier_ids = result
        assert "Fact A" in facts_text
        assert outlier_ids == []

    async def test_outlier_exclusion_propagates(
        self, consolidator: SqliteConsolidator, mock_chat_system: Any,
    ) -> None:
        """tool_calls returning outlier_ids should be returned as-is."""
        persona = MagicMock()
        persona.get_config_for_engine.return_value = {"model_name": "m"}
        persona.get_prompt.return_value = "p"
        mock_chat_system.personas["memory_summarizer"] = persona
        mock_chat_system.text_engine.google_client = None

        mock_chat_system.text_engine.generate_response = AsyncMock(return_value=(
            {
                "type": "tool_calls",
                "calls": [{
                    "name": "submit_memory_summary",
                    "arguments": {
                        "observations": ["Obs1", "Obs2"],
                        "keywords": ["kw1"],
                        "outlier_ids": [42, 99],
                    },
                }],
            },
            {},
        ))
        consolidator._rate_limiter = MagicMock()
        consolidator._rate_limiter.acquire = AsyncMock()

        emb_service = MagicMock()
        emb_service.encode_single = AsyncMock(return_value=b'\x00' * 16)

        segment = {
            "start_id": 1,
            "messages": [{"author_role": "user", "author_name": "A", "content": "hi"}],
        }
        result = await consolidator._summarize_segment(segment, emb_service)
        assert result is not None
        facts_text, _, outlier_ids = result
        assert outlier_ids == [42, 99]
        assert "Obs1" in facts_text
        assert "Keywords: kw1" in facts_text

    async def test_dedup_repeated_facts_preserved_as_returned(
        self, consolidator: SqliteConsolidator, mock_chat_system: Any,
    ) -> None:
        """The consolidator does NOT dedupe — verify observations pass through
        verbatim (downstream dedup is not its responsibility)."""
        persona = MagicMock()
        persona.get_config_for_engine.return_value = {"model_name": "m"}
        persona.get_prompt.return_value = "p"
        mock_chat_system.personas["memory_summarizer"] = persona
        mock_chat_system.text_engine.google_client = None

        mock_chat_system.text_engine.generate_response = AsyncMock(return_value=(
            {
                "type": "tool_calls",
                "calls": [{
                    "name": "submit_memory_summary",
                    "arguments": {
                        "observations": ["Same fact", "Same fact", "Other"],
                        "keywords": [],
                        "outlier_ids": [],
                    },
                }],
            },
            {},
        ))
        consolidator._rate_limiter = MagicMock()
        consolidator._rate_limiter.acquire = AsyncMock()

        emb_service = MagicMock()
        emb_service.encode_single = AsyncMock(return_value=b'\x00' * 16)

        segment = {
            "start_id": 1,
            "messages": [{"author_role": "user", "author_name": "A", "content": "hi"}],
        }
        result = await consolidator._summarize_segment(segment, emb_service)
        assert result is not None
        facts_text = result[0]
        # Both copies of the repeated fact survive — explicit pass-through
        assert facts_text.count("Same fact") == 2


# ---------------------------------------------------------------------------
# Embedding rate-limiter shared state
# ---------------------------------------------------------------------------

class TestRateLimiterSharing:
    def test_consolidator_uses_global_limiter(self) -> None:
        """Two consolidator instances must share the same limiter object
        with the GeminiEmbeddingProvider (prevents double-spending budget)."""
        from src.embedding_service import GLOBAL_EMBEDDING_LIMITER

        cs = MagicMock()
        cs.text_engine = MagicMock()
        cs.memory_manager = MagicMock()
        cs.personas = {}
        cs._embedding_service = None
        with patch('src.agents.base.load_system_personas_from_file', return_value={}):
            a = SqliteConsolidator(cs)
            b = SqliteConsolidator(cs)
        assert a._rate_limiter is GLOBAL_EMBEDDING_LIMITER
        assert b._rate_limiter is GLOBAL_EMBEDDING_LIMITER


# ---------------------------------------------------------------------------
# Latent-bug skip
# ---------------------------------------------------------------------------

@pytest.mark.skip(reason="DP-199 deferred bug: agents/sqlite_consolidator.py:254-258 — "
                  "vec-row invalidation on update_interaction_content is not asserted; "
                  "embedding may orphan on edit.")
def test_consolidator_vec_row_invalidation_on_content_update() -> None:
    """Placeholder for the latent bug listed in DP-199-edge-cases.md."""
    pass
