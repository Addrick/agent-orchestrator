# src/agents/memory_agent.py

import logging
import time
import asyncio
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from src.agents.base import Agent
from src.chat_system import ChatSystem
from src.embedding_service import EmbeddingService, GeminiEmbeddingProvider

# --- NEW: Import rate limits and dynamic model names ---
from config.global_config import (
    EMBEDDING_MODEL,
    GEMINI_EMBEDDING_001_RPM,
    GEMINI_EMBEDDING_001_TPM,
    GEMINI_EMBEDDING_001_RPD
)

logger = logging.getLogger(__name__)


class EmbeddingRateLimiter:
    """Proactively tracks item-based Google API quota limits to prevent 429s."""

    def __init__(self, rpm: int, tpm: int, rpd: int):
        self.rpm = rpm
        self.tpm = tpm
        self.rpd = rpd
        self.minute_req_history: List[Tuple[float, int]] = []  # (timestamp, items)
        self.minute_tok_history: List[Tuple[float, int]] = []  # (timestamp, tokens)
        self.day_req_history: List[Tuple[float, int]] = []  # (timestamp, items)
        self._lock = asyncio.Lock()

    async def acquire(self, item_count: int, token_count: int) -> None:
        """Awaits until the payload can safely be sent without hitting a 429."""
        async with self._lock:
            while True:
                now = time.time()

                # Prune out-of-window timestamps
                self.minute_req_history = [(t, c) for t, c in self.minute_req_history if now - t < 60.0]
                self.minute_tok_history = [(t, tok) for t, tok in self.minute_tok_history if now - t < 60.0]
                self.day_req_history = [(t, c) for t, c in self.day_req_history if now - t < 86400.0]

                # 1. Enforce Daily Limit (Hard stop)
                current_day_reqs = sum(c for _, c in self.day_req_history)
                if current_day_reqs + item_count > self.rpd:
                    raise RuntimeError(f"Daily Google AI Studio Quota Exhausted ({self.rpd} items).")

                # 2. Check Minute Limits (Requests & Tokens)
                current_min_reqs = sum(c for _, c in self.minute_req_history)
                current_min_toks = sum(tok for _, tok in self.minute_tok_history)

                if current_min_reqs + item_count <= self.rpm and current_min_toks + token_count <= self.tpm:
                    # Safe to proceed! Log consumption.
                    self.minute_req_history.append((now, item_count))
                    self.minute_tok_history.append((now, token_count))
                    self.day_req_history.append((now, item_count))
                    break

                # 3. Throttle: Calculate exact sleep time until enough items expire
                sleep_time = 0.0
                if current_min_reqs + item_count > self.rpm and self.minute_req_history:
                    sleep_time = max(sleep_time, 60.0 - (now - self.minute_req_history[0][0]))
                if current_min_toks + token_count > self.tpm and self.minute_tok_history:
                    sleep_time = max(sleep_time, 60.0 - (now - self.minute_tok_history[0][0]))

                if sleep_time > 0:
                    logger.info(f"Embedding rate limiter active: waiting {sleep_time:.1f}s for bucket refill...")
                    await asyncio.sleep(sleep_time + 0.1)  # Add 100ms buffer to ensure API clearance


class MemoryAgent(Agent):
    """
    Batch agent that segments conversations by topic, extracts facts via LLM,
    and stores embedded summaries for retrieval-augmented conversation context.
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

        # Initialize the proactive sliding-window rate limiter
        self._rate_limiter = EmbeddingRateLimiter(
            rpm=GEMINI_EMBEDDING_001_RPM,
            tpm=GEMINI_EMBEDDING_001_TPM,
            rpd=GEMINI_EMBEDDING_001_RPD
        )

        # Config with defaults
        self._similarity_threshold: float = float(
            self.agent_config.get("similarity_threshold", 0.3)
        )
        self._min_segment_size: int = int(
            self.agent_config.get("min_segment_size", 3)
        )

        # We cap batches to Gemini's hard 100 item array limit, or whatever is lower
        config_batch_size = int(self.agent_config.get("batch_size", 100))
        self._batch_size: int = min(config_batch_size, 100)

        # Max tokens per chunk defaults to the TPM variable to prevent immediate token exhaustion
        self._max_tokens_per_chunk: int = int(
            self.agent_config.get("max_tokens_per_chunk", GEMINI_EMBEDDING_001_TPM)
        )
        self._persona_name: str = self.agent_config.get("persona", "memory_summarizer")
        self._allowed_channels = self._parse_allowed_channels(
            self.agent_config.get("allowed_channels")
        )

    @staticmethod
    def _parse_allowed_channels(
            raw: Optional[List[Any]],
    ) -> Optional[List[Dict[str, str]]]:
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
        if self._embedding_service is None:
            provider_name = self.agent_config.get("embedding_provider", "gemini")
            if provider_name == "gemini":
                # Ensure the provider respects our global model variable
                provider = GeminiEmbeddingProvider()
            else:
                raise ValueError(f"Unknown embedding provider: {provider_name}")

            self._embedding_service = EmbeddingService(provider)

            # Optionally override or assert the model name if your EmbeddingService supports it
            if hasattr(self._embedding_service, 'model_name') and getattr(self._embedding_service,
                                                                          'model_name') != EMBEDDING_MODEL:
                logger.debug(f"Overriding EmbeddingService model to: {EMBEDDING_MODEL}")
                self._embedding_service.model_name = EMBEDDING_MODEL

            self.chat_system._embedding_service = self._embedding_service
            logger.info(
                f"MemoryAgent initialized EmbeddingService with provider "
                f"'{provider_name}' (model: {EMBEDDING_MODEL})"
            )

        return self._embedding_service

    async def deploy(self) -> None:
        embedding_service = self._get_embedding_service()
        # Ensure we query using the globally configured model name
        model_name = EMBEDDING_MODEL

        all_channels = self.memory_manager.get_active_channels(model_name=model_name)

        if self._allowed_channels is not None:
            channels = [
                (ch, pn, sid) for ch, pn, sid in all_channels
                if any(
                    a["channel"] == ch and a["server_id"] == sid
                    for a in self._allowed_channels
                )
            ]
            if len(channels) != len(all_channels):
                logger.debug(
                    f"MemoryAgent: {len(all_channels)} active channels, "
                    f"{len(channels)} after allowed_channels filter. "
                    f"Active: {[(ch, pn, sid) for ch, pn, sid in all_channels]}"
                )
        else:
            channels = all_channels

        if not channels:
            logger.info("MemoryAgent: no channels with unprocessed messages.")
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
                # Keep error logging quiet without vomiting stack traces
                logger.error(f"MemoryAgent: error processing {channel}/{persona_name}: {str(e)}")

    async def _process_channel(
            self,
            channel: str,
            persona_name: str,
            server_id: Optional[str],
            embedding_service: EmbeddingService,
    ) -> None:
        # --- Phase 1: Embed unembedded messages ---
        try:
            await self._embed_unembedded(
                channel, persona_name, server_id, embedding_service
            )
        except Exception as e:
            logger.warning(f"MemoryAgent: {channel}/{persona_name} embedding phase aborted: {str(e)}")

        # --- Phase 2: Segment and summarize ---
        try:
            await self._segment_and_summarize(
                channel, persona_name, server_id, embedding_service
            )
        except Exception as e:
            logger.warning(f"MemoryAgent: {channel}/{persona_name} summary phase aborted: {str(e)}")

    async def _embed_unembedded(
            self,
            channel: str,
            persona_name: str,
            server_id: Optional[str],
            embedding_service: EmbeddingService,
    ) -> None:
        model_name = EMBEDDING_MODEL

        messages = self.memory_manager.get_unembedded_messages(
            persona_name=persona_name,
            channel=channel,
            server_id=server_id,
            limit=self._batch_size,
            model_name=model_name,
        )

        if not messages:
            logger.debug(f"MemoryAgent: {channel}/{persona_name} — no unembedded messages found.")
            return

        logger.info(f"MemoryAgent: {channel}/{persona_name} — {len(messages)} unembedded messages to process.")

        chunk_size = self._batch_size
        total_stored = 0
        now = datetime.now(timezone.utc)

        chunks = self._chunk_messages(messages, chunk_size, self._max_tokens_per_chunk)
        for chunk_idx, chunk_msgs in enumerate(chunks):
            chunk_texts = [msg['content'] for msg in chunk_msgs]
            est_tokens = sum(len(t) for t in chunk_texts) // 4

            logger.debug(
                f"MemoryAgent: {channel}/{persona_name} — "
                f"embedding chunk {chunk_idx + 1}/{len(chunks)} "
                f"({len(chunk_texts)} messages, ~{est_tokens} tokens)"
            )

            try:
                # Proactively ensure we do not hit API rate limits
                await self._rate_limiter.acquire(item_count=len(chunk_texts), token_count=est_tokens)

                # Execute API call
                chunk_embs = await embedding_service.encode(chunk_texts)

                with self.memory_manager.transaction() as conn:
                    for msg, emb in zip(chunk_msgs, chunk_embs):
                        conn.execute(
                            """INSERT OR REPLACE INTO Message_Embeddings
                               (interaction_id, embedding, model_name, created_at)
                               VALUES (?, ?, ?, ?)""",
                            (msg['interaction_id'], emb, model_name, now)
                        )
                total_stored += len(chunk_embs)

            except RuntimeError as e:
                # Daily limit hit - break loop cleanly
                logger.warning(f"Embedding loop stopped: {str(e)}")
                break
            except Exception as e:
                # Network or Unexpected error - backoff and break to prevent spam
                logger.error(f"API Error during chunk encoding: {str(e)}")
                await asyncio.sleep(60)
                break

        logger.info(f"MemoryAgent: {channel}/{persona_name} — stored {total_stored} embeddings.")

    @staticmethod
    def _chunk_messages(
            messages: List[Dict[str, Any]],
            max_items: int,
            max_tokens: int,
    ) -> List[List[Dict[str, Any]]]:
        chunks: List[List[Dict[str, Any]]] = []
        current: List[Dict[str, Any]] = []
        current_tokens = 0
        for msg in messages:
            msg_tokens = len(msg.get('content', '')) // 4
            would_exceed_tokens = (current and current_tokens + msg_tokens > max_tokens)
            would_exceed_items = len(current) >= max_items
            if would_exceed_items or would_exceed_tokens:
                chunks.append(current)
                current = []
                current_tokens = 0
            current.append(msg)
            current_tokens += msg_tokens
        if current:
            chunks.append(current)
        return chunks

    async def _segment_and_summarize(
            self,
            channel: str,
            persona_name: str,
            server_id: Optional[str],
            embedding_service: EmbeddingService,
    ) -> None:
        model_name = EMBEDDING_MODEL

        rows = self.memory_manager.get_unsegmented_embedded_messages(
            persona_name=persona_name,
            channel=channel,
            server_id=server_id,
            model_name=model_name,
            limit=self._batch_size,
        )

        logger.info(f"MemoryAgent: {channel}/{persona_name} — {len(rows)} unsegmented embedded messages found.")

        if len(rows) < self._min_segment_size:
            return

        messages = [
            {k: r[k] for k in ('interaction_id', 'author_role', 'author_name',
                               'content', 'timestamp')}
            for r in rows
        ]
        embeddings = [r['embedding'] for r in rows]

        segments = self._segment_by_similarity(
            messages, embeddings, channel, persona_name, server_id
        )

        if not segments:
            return

        results: List[Tuple[Dict[str, Any], str, bytes]] = []
        for segment in segments:
            summary_result = await self._summarize_segment(segment, embedding_service)
            if summary_result is not None:
                results.append((segment, summary_result[0], summary_result[1]))

        if not results:
            logger.warning(
                f"MemoryAgent: all {len(segments)} segments failed summarization for {channel}/{persona_name}.")
            return

        now = datetime.now(timezone.utc)
        with self.memory_manager.transaction() as conn:
            for segment, summary_text, summary_emb in results:
                msg_timestamps = [m['timestamp'] for m in segment['messages'] if m.get('timestamp')]
                first_msg_at = min(msg_timestamps) if msg_timestamps else None
                last_msg_at = max(msg_timestamps) if msg_timestamps else None

                cursor = conn.execute(
                    """INSERT INTO Memory_Segments
                       (channel, server_id, persona_name,
                        start_interaction_id, end_interaction_id,
                        message_count, created_at,
                        first_message_at, last_message_at)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (channel, server_id, persona_name,
                     segment['start_id'], segment['end_id'],
                     segment['count'], now,
                     first_msg_at, last_msg_at)
                )
                segment_id = cursor.lastrowid

                conn.execute(
                    """INSERT INTO Memory_Summaries
                       (segment_id, content, embedding, model_name, created_at)
                       VALUES (?, ?, ?, ?, ?)""",
                    (segment_id, summary_text, summary_emb, model_name, now)
                )

        logger.info(f"MemoryAgent: {channel}/{persona_name} — {len(results)} segments+summaries stored.")

    def _segment_by_similarity(
            self,
            messages: List[Dict[str, Any]],
            embeddings: List[bytes],
            channel: str,
            persona_name: str,
            server_id: Optional[str],
    ) -> List[Dict[str, Any]]:
        if not messages:
            return []

        centroid = self._seed_centroid_from_previous(channel, persona_name, server_id)

        segments: List[Dict[str, Any]] = []
        current_msgs: List[Dict[str, Any]] = []
        current_embs: List[bytes] = []
        n = 0

        for i, (msg, emb_blob) in enumerate(zip(messages, embeddings)):
            vec = np.frombuffer(emb_blob, dtype=np.float32).copy()

            if centroid is None:
                centroid = vec.copy()
                n = 1
                current_msgs.append(msg)
                current_embs.append(emb_blob)
                continue

            similarity = float(np.dot(centroid, vec))

            if (similarity < self._similarity_threshold and len(current_msgs) >= self._min_segment_size):
                segments.append({
                    'start_id': current_msgs[0]['interaction_id'],
                    'end_id': current_msgs[-1]['interaction_id'],
                    'count': len(current_msgs),
                    'messages': current_msgs,
                    'embeddings': current_embs,
                })
                current_msgs = [msg]
                current_embs = [emb_blob]
                centroid = vec.copy()
                n = 1
            else:
                current_msgs.append(msg)
                current_embs.append(emb_blob)
                n += 1
                centroid = centroid * ((n - 1) / n) + vec / n
                norm = np.linalg.norm(centroid)
                if norm > 0:
                    centroid = centroid / norm

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
        model_name = EMBEDDING_MODEL

        tail_blobs = self.memory_manager.get_last_segment_tail_embeddings(
            channel=channel,
            persona_name=persona_name,
            server_id=server_id,
            n=3,
            model_name=model_name,
        )

        if tail_blobs is None:
            return None

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
        persona = self.chat_system.personas.get(self._persona_name)
        if not persona:
            logger.error(f"System persona '{self._persona_name}' not found. Cannot summarize.")
            return None

        lines = []
        for msg in segment['messages']:
            role = msg.get('author_role', 'user')
            name = msg.get('author_name', 'Unknown')
            content = msg.get('content', '')
            ts = msg.get('timestamp', '')
            if ts:
                if isinstance(ts, str):
                    ts = datetime.fromisoformat(ts)
                ts_str = ts.strftime('%Y-%m-%d %H:%M')
                lines.append(f"[{ts_str}] [{role}] {name}: {content}")
            else:
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

            # Calculate tokens and await rate limits for the embedding of the summary
            est_tokens = len(facts_text) // 4
            await self._rate_limiter.acquire(item_count=1, token_count=est_tokens)

            summary_embedding = await embedding_service.encode_single(facts_text)
            return (facts_text, summary_embedding)

        except RuntimeError as e:
            logger.warning(f"MemoryAgent: Summarization embedding paused due to Daily Quota: {str(e)}")
            return None
        except Exception as e:
            logger.error(f"MemoryAgent: summarization failed: {str(e)}")
            return None
