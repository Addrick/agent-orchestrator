# src/agents/memory_agent.py

import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from src.agents.base import Agent
from src.chat_system import ChatSystem
from src.embedding_service import EmbeddingService, GeminiEmbeddingProvider

logger = logging.getLogger(__name__)


class MemoryAgent(Agent):
    """
    Batch agent that segments conversations by topic, extracts facts via LLM,
    and stores embedded summaries for retrieval-augmented conversation context.

    Pipeline per channel:
      1. Fetch unembedded messages
      2. Embed messages (batch API call)
      3. Segment by topic similarity (sliding-window centroid)
      4. Summarize each segment via LLM (fact extraction)
      5. Embed summaries
      6. Write all results to DB in one transaction
    """

    agent_name: str = "memory"

    def __init__(
        self,
        chat_system: ChatSystem,
        agent_config: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(chat_system)
        self.agent_config = agent_config or {}
        self._embedding_service: Optional[EmbeddingService] = None

        # Config with defaults
        self._similarity_threshold: float = float(
            self.agent_config.get("similarity_threshold", 0.3)
        )
        self._min_segment_size: int = int(
            self.agent_config.get("min_segment_size", 3)
        )
        self._batch_size: int = int(
            self.agent_config.get("batch_size", 200)
        )
        self._persona_name: str = self.agent_config.get("persona", "memory_summarizer")
        self._allowed_channels = self._parse_allowed_channels(
            self.agent_config.get("allowed_channels")
        )

    @staticmethod
    def _parse_allowed_channels(
        raw: Optional[List[Any]],
    ) -> Optional[List[Dict[str, str]]]:
        """Parse allowed_channels config into channel+server pairs.

        Each entry must be {"channel": "name", "server_id": "id"}.
        """
        if raw is None:
            return None
        result = []
        for entry in raw:
            if not isinstance(entry, dict) or "channel" not in entry or "server_id" not in entry:
                raise ValueError(
                    f"allowed_channels entries must have 'channel' and 'server_id': {entry}"
                )
            result.append({
                "channel": entry["channel"],
                "server_id": entry["server_id"],
            })
        return result

    def _get_embedding_service(self) -> EmbeddingService:
        """Lazily initialize the embedding service on first use."""
        if self._embedding_service is None:
            provider_name = self.agent_config.get("embedding_provider", "gemini")
            if provider_name == "gemini":
                provider = GeminiEmbeddingProvider()
            else:
                raise ValueError(f"Unknown embedding provider: {provider_name}")

            self._embedding_service = EmbeddingService(provider)
            # Share with ChatSystem for query-time retrieval
            self.chat_system._embedding_service = self._embedding_service
            logger.info(
                f"MemoryAgent initialized EmbeddingService with provider "
                f"'{provider_name}' (model: {self._embedding_service.model_name})"
            )

        return self._embedding_service

    async def deploy(self) -> None:
        """Discover channels with unprocessed messages and process each."""
        embedding_service = self._get_embedding_service()
        model_name = embedding_service.model_name

        all_channels = self.memory_manager.get_active_channels(model_name=model_name)

        # Filter to allowed channels if configured (for staged rollout / testing)
        if self._allowed_channels is not None:
            channels = [
                (ch, pn, sid) for ch, pn, sid in all_channels
                if any(
                    a["channel"] == ch and a["server_id"] == sid
                    for a in self._allowed_channels
                )
            ]
        else:
            channels = all_channels

        if not channels:
            logger.debug("MemoryAgent: no channels with unprocessed messages.")
            return

        logger.info(f"MemoryAgent: found {len(channels)} channel(s) to process.")

        for channel, persona_name, server_id in channels:
            if self._shutdown_event.is_set():
                break

            try:
                await self._process_channel(
                    channel, persona_name, server_id, embedding_service
                )
            except Exception as e:
                is_rate_limit = "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e)
                if is_rate_limit:
                    logger.warning(
                        f"MemoryAgent: {channel}/{persona_name} rate-limited, will retry next cycle."
                    )
                else:
                    logger.error(
                        f"MemoryAgent: error processing {channel}/{persona_name}: {e}",
                        exc_info=True,
                    )

    async def _process_channel(
        self,
        channel: str,
        persona_name: str,
        server_id: Optional[str],
        embedding_service: EmbeddingService,
    ) -> None:
        """Process a single channel: embed -> segment -> summarize -> store."""
        model_name = embedding_service.model_name

        # 1. Fetch unembedded messages
        messages = self.memory_manager.get_unembedded_messages(
            persona_name=persona_name,
            channel=channel,
            server_id=server_id,
            limit=self._batch_size,
            model_name=model_name,
        )

        if len(messages) < self._min_segment_size:
            logger.debug(
                f"MemoryAgent: {channel}/{persona_name} has {len(messages)} messages "
                f"(< min_segment_size={self._min_segment_size}), skipping."
            )
            return

        # 2. Embed all messages (single batch API call)
        texts = [msg['content'] for msg in messages]
        embeddings = await embedding_service.encode(texts)

        # 3. Persist embeddings immediately so they survive later failures
        now = datetime.now(timezone.utc)
        with self.memory_manager.transaction() as conn:
            for msg, emb in zip(messages, embeddings):
                conn.execute(
                    """INSERT OR REPLACE INTO Message_Embeddings
                       (interaction_id, embedding, model_name, created_at)
                       VALUES (?, ?, ?, ?)""",
                    (msg['interaction_id'], emb, model_name, now)
                )
        logger.info(
            f"MemoryAgent: {channel}/{persona_name} — stored {len(embeddings)} embeddings."
        )

        # 4. Segment by topic similarity
        segments = self._segment_by_similarity(
            messages, embeddings, channel, persona_name, server_id
        )

        if not segments:
            logger.debug(f"MemoryAgent: no segments produced for {channel}/{persona_name}.")
            return

        # 5. Summarize each segment and embed summaries
        results: List[Tuple[Dict[str, Any], str, bytes]] = []
        for segment in segments:
            summary_result = await self._summarize_segment(segment, embedding_service)
            if summary_result is not None:
                results.append((segment, summary_result[0], summary_result[1]))

        if not results:
            logger.warning(
                f"MemoryAgent: all {len(segments)} segments failed summarization "
                f"for {channel}/{persona_name}, skipping segment write."
            )
            return

        # 6. Store segments and summaries
        with self.memory_manager.transaction() as conn:
            for segment, summary_text, summary_emb in results:
                cursor = conn.execute(
                    """INSERT INTO Memory_Segments
                       (channel, server_id, persona_name,
                        start_interaction_id, end_interaction_id,
                        message_count, created_at)
                       VALUES (?, ?, ?, ?, ?, ?, ?)""",
                    (channel, server_id, persona_name,
                     segment['start_id'], segment['end_id'],
                     segment['count'], now)
                )
                segment_id = cursor.lastrowid

                conn.execute(
                    """INSERT INTO Memory_Summaries
                       (segment_id, content, embedding, model_name, created_at)
                       VALUES (?, ?, ?, ?, ?)""",
                    (segment_id, summary_text, summary_emb, model_name, now)
                )

        logger.info(
            f"MemoryAgent: {channel}/{persona_name} — "
            f"{len(results)} segments+summaries stored."
        )

    def _segment_by_similarity(
        self,
        messages: List[Dict[str, Any]],
        embeddings: List[bytes],
        channel: str,
        persona_name: str,
        server_id: Optional[str],
    ) -> List[Dict[str, Any]]:
        """Segment messages by sliding-window centroid similarity.

        Returns list of dicts with keys: start_id, end_id, count, messages, embeddings.
        """
        if not messages:
            return []

        # Seed centroid from previous segment tail
        centroid = self._seed_centroid_from_previous(
            channel, persona_name, server_id
        )

        segments: List[Dict[str, Any]] = []
        current_msgs: List[Dict[str, Any]] = []
        current_embs: List[bytes] = []
        n = 0  # running count for centroid update

        for i, (msg, emb_blob) in enumerate(zip(messages, embeddings)):
            vec = np.frombuffer(emb_blob, dtype=np.float32).copy()

            if centroid is None:
                # First message — initialize centroid
                centroid = vec.copy()
                n = 1
                current_msgs.append(msg)
                current_embs.append(emb_blob)
                continue

            # Compute similarity to running centroid
            similarity = float(np.dot(centroid, vec))

            if (similarity < self._similarity_threshold
                    and len(current_msgs) >= self._min_segment_size):
                # Cut — save current segment
                segments.append({
                    'start_id': current_msgs[0]['interaction_id'],
                    'end_id': current_msgs[-1]['interaction_id'],
                    'count': len(current_msgs),
                    'messages': current_msgs,
                    'embeddings': current_embs,
                })
                # Reset — clean break
                current_msgs = [msg]
                current_embs = [emb_blob]
                centroid = vec.copy()
                n = 1
            else:
                # Update centroid incrementally and re-normalize
                current_msgs.append(msg)
                current_embs.append(emb_blob)
                n += 1
                centroid = centroid * ((n - 1) / n) + vec / n
                norm = np.linalg.norm(centroid)
                if norm > 0:
                    centroid = centroid / norm

        # Final segment
        if current_msgs:
            segments.append({
                'start_id': current_msgs[0]['interaction_id'],
                'end_id': current_msgs[-1]['interaction_id'],
                'count': len(current_msgs),
                'messages': current_msgs,
                'embeddings': current_embs,
            })

        return segments

    def _seed_centroid_from_previous(
        self,
        channel: str,
        persona_name: str,
        server_id: Optional[str],
    ) -> Optional["np.ndarray[Any, Any]"]:
        """Load tail embeddings from the previous segment and compute mean centroid.

        Returns None if no previous segment or model mismatch.
        """
        model_name = (self._embedding_service.model_name
                      if self._embedding_service else None)

        tail_blobs = self.memory_manager.get_last_segment_tail_embeddings(
            channel=channel,
            persona_name=persona_name,
            server_id=server_id,
            n=3,
            model_name=model_name,
        )

        if tail_blobs is None:
            return None

        # Compute mean of tail embeddings and normalize
        vectors = [np.frombuffer(b, dtype=np.float32) for b in tail_blobs]
        centroid: np.ndarray[Any, Any] = np.mean(vectors, axis=0)
        norm = float(np.linalg.norm(centroid))
        if norm > 0:
            centroid = centroid / norm
            return centroid

        return None

    async def _summarize_segment(
        self,
        segment: Dict[str, Any],
        embedding_service: EmbeddingService,
    ) -> Optional[Tuple[str, bytes]]:
        """Extract facts from a segment via LLM and embed the result.

        Returns (facts_text, summary_embedding) or None on failure.
        """
        persona = self.chat_system.personas.get(self._persona_name)
        if not persona:
            logger.error(
                f"System persona '{self._persona_name}' not found. Cannot summarize."
            )
            return None

        # Build transcript
        lines = []
        for msg in segment['messages']:
            role = msg.get('author_role', 'user')
            name = msg.get('author_name', 'Unknown')
            content = msg.get('content', '')
            lines.append(f"[{role}] {name}: {content}")

        transcript = "\n".join(lines)
        prompt = (
            f"Extract factual information from this conversation segment.\n\n"
            f"TRANSCRIPT:\n{transcript}"
        )

        try:
            response, _ = await self.text_engine.generate_response(
                persona_config=persona.get_config_for_engine(),
                context_object=self._build_llm_context(persona, prompt),
                tools=None,
            )

            if response.get('type') != 'text':
                logger.warning("MemoryAgent: summarizer returned non-text response.")
                return None

            facts_text = response.get('content', '').strip()
            if not facts_text:
                logger.warning("MemoryAgent: summarizer returned empty content.")
                return None

            # Embed the summary
            summary_embedding = await embedding_service.encode_single(facts_text)
            return (facts_text, summary_embedding)

        except Exception as e:
            logger.error(f"MemoryAgent: summarization failed: {e}", exc_info=True)
            return None
