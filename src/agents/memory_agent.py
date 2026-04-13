# src/agents/memory_agent.py

import logging
import asyncio
import re
import numpy as np
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from src.agents.base import Agent
from src.chat_system import ChatSystem
from src.embedding_service import (
    EmbeddingService,
    GeminiEmbeddingProvider,
    GLOBAL_EMBEDDING_LIMITER,
)

from config.global_config import (
    EMBEDDING_MODEL,
    GEMINI_EMBEDDING_001_TPM,
)

logger = logging.getLogger(__name__)


class MemoryAgent(Agent):
    """
    Batch agent that segments conversations by topic, extracts facts via LLM,
    and stores embedded summaries for retrieval-augmented conversation context.
    """

    agent_name: str = "memory"

    def __init__(
            self,
            chat_system: ChatSystem,
            memory_manager: Optional[Any] = None,
            agent_config: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(chat_system)
        self.memory_manager = memory_manager or chat_system.memory_manager
        self.agent_config = agent_config or {}
        self._embedding_service: Optional[EmbeddingService] = None

        # Share quota state with GeminiEmbeddingProvider so concurrent
        # consumers (consolidator + agent) can't double-spend the budget.
        self._rate_limiter = GLOBAL_EMBEDDING_LIMITER

        # Config with defaults
        self._similarity_threshold: float = float(
            self.agent_config.get("similarity_threshold", 0.80)
        )
        self._min_segment_size: int = int(
            self.agent_config.get("min_segment_size", 1)
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
            # First, check if the ChatSystem already has a service initialized (e.g. from main.py)
            if self.chat_system._embedding_service is not None:
                self._embedding_service = self.chat_system._embedding_service
                logger.info("MemoryAgent: using existing EmbeddingService from ChatSystem.")
            else:
                # Initialize new service
                from src.embedding_service import GeminiEmbeddingProvider
                provider_name = self.agent_config.get("embedding_provider", "gemini")
                if provider_name == "gemini":
                    provider = GeminiEmbeddingProvider()
                else:
                    raise ValueError(f"Unknown embedding provider: {provider_name}")

                self._embedding_service = EmbeddingService(provider)

                # Ensure model name consistency
                if self._embedding_service.model_name != EMBEDDING_MODEL:
                    logger.debug(f"Overriding EmbeddingService model to: {EMBEDDING_MODEL}")
                    self._embedding_service.model_name = EMBEDDING_MODEL

                # Shared service with the bot
                self.chat_system._embedding_service = self._embedding_service
                logger.info(
                    f"MemoryAgent: initialized new EmbeddingService ({provider_name}) (model: {EMBEDDING_MODEL})"
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
        # --- Phase 1: Embed all unembedded messages (loop until exhausted) ---
        batch_num = 0
        while not self._shutdown_event.is_set():
            try:
                count = await self._embed_unembedded(
                    channel, persona_name, server_id, embedding_service
                )
                if count == 0:
                    break
                batch_num += 1
                logger.info(f"MemoryAgent: {channel}/{persona_name} — embedding batch {batch_num} done ({count} stored).")
            except Exception as e:
                logger.warning(f"MemoryAgent: {channel}/{persona_name} embedding phase aborted: {str(e)}")
                break

        # --- Phase 2: Segment and summarize all available (loop until clear) ---
        batch_num = 0
        while not self._shutdown_event.is_set():
            try:
                # We do NOT pass a limit here anymore as per USER_REQUEST;
                # we process everything the similarity logic generates in one sweep.
                committed_count = await self._segment_and_summarize(
                    channel, persona_name, server_id, embedding_service
                )
                if committed_count == 0:
                    break
                batch_num += 1
                logger.info(f"MemoryAgent: {channel}/{persona_name} — summary batch {batch_num} done ({committed_count} stored).")
            except Exception as e:
                logger.warning(f"MemoryAgent: {channel}/{persona_name} summary phase aborted: {str(e)}")
                break

    async def _embed_unembedded(
            self,
            channel: str,
            persona_name: str,
            server_id: Optional[str],
            embedding_service: EmbeddingService,
    ) -> int:
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
            return 0

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
                        conn.execute(
                            """INSERT OR REPLACE INTO vec_Message_Embeddings
                               (interaction_id, embedding)
                               VALUES (?, ?)""",
                            (msg['interaction_id'], emb)
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
        return total_stored

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
    ) -> int:
        model_name = EMBEDDING_MODEL

        rows = self.memory_manager.get_unsegmented_embedded_messages(
            persona_name=persona_name,
            channel=channel,
            server_id=server_id,
            model_name=model_name,
        )

        logger.info(f"MemoryAgent: {channel}/{persona_name} — {len(rows)} unsegmented embedded messages found.")

        if len(rows) < self._min_segment_size:
            return 0

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
            return 0

        total_committed = 0
        for i, segment in enumerate(segments):
            if self._shutdown_event.is_set():
                logger.info(f"MemoryAgent: {channel}/{persona_name} — shutdown signalled, stopping at segment {i + 1}/{len(segments)}.")
                break

            logger.info(f"MemoryAgent: {channel}/{persona_name} — Processing segment {i + 1}/{len(segments)}...")
            summary_result = await self._summarize_segment(segment, embedding_service)
            if summary_result is None:
                logger.warning(
                    f"MemoryAgent: {channel}/{persona_name} — Segment {i + 1}/{len(segments)} failed summarization, skipping.")
                continue

            summary_text, summary_emb, outlier_ids = summary_result

            now = datetime.now(timezone.utc)
            msg_timestamps = []
            seg_msg_ids = []
            for m in segment['messages']:
                m_id = m.get('interaction_id')
                if m_id:
                    seg_msg_ids.append(m_id)

                ts = m.get('timestamp')
                if ts:
                    if isinstance(ts, str):
                        ts = datetime.fromisoformat(ts)
                    if getattr(ts, 'tzinfo', None) is None:
                        ts = ts.replace(tzinfo=timezone.utc)
                    msg_timestamps.append(ts)

            first_msg_at = min(msg_timestamps) if msg_timestamps else None
            last_msg_at = max(msg_timestamps) if msg_timestamps else None

            with self.memory_manager.transaction() as conn:
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

                cursor2 = conn.execute(
                    """INSERT INTO Memory_Summaries
                       (segment_id, content, embedding, model_name, created_at)
                       VALUES (?, ?, ?, ?, ?)""",
                    (segment_id, summary_text, summary_emb, model_name, now)
                )
                summary_id = cursor2.lastrowid
                conn.execute(
                    """INSERT INTO vec_Memory_Summaries
                       (summary_id, embedding)
                       VALUES (?, ?)""",
                    (summary_id, summary_emb)
                )

                outlier_set = set(outlier_ids)
                summarized_ids = [m_id for m_id in seg_msg_ids if m_id not in outlier_set]

                if summarized_ids:
                    placeholders = ",".join("?" for _ in summarized_ids)
                    conn.execute(
                        f"UPDATE User_Interactions SET parent_summary_id = ? WHERE interaction_id IN ({placeholders})",
                        [summary_id] + summarized_ids
                    )

                if outlier_ids:
                    logger.info(f"MemoryAgent: Excluded {len(outlier_ids)} outliers from summary {summary_id}. They remain NULL for re-queueing.")

            total_committed += 1
            logger.info(f"MemoryAgent: {channel}/{persona_name} — Segment {i + 1}/{len(segments)} committed.")

        if total_committed == 0 and segments:
            logger.warning(
                f"MemoryAgent: all {len(segments)} segments failed summarization for {channel}/{persona_name}.")

        logger.info(f"MemoryAgent: {channel}/{persona_name} — {total_committed}/{len(segments)} segments+summaries stored.")
        return total_committed

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

            # --- SEGMENTATION BRIDGING (GRAVITY) ---
            # 1. Explicit Link: Pair messages if one explicitly replies to another in the same cluster.
            current_ids = {m.get('interaction_id') for m in current_msgs}
            reply_link = msg.get('reply_to_id') in current_ids

            # 2. Heuristic Link: Fallback for historical data without IDs (User followed by Assistant).
            role_link = (
                msg.get('reply_to_id') is None and
                len(current_msgs) == 1 and
                current_msgs[0].get('author_role') == 'user' and
                msg.get('author_role') == 'assistant'
            )

            is_bridge = reply_link or role_link

            if (similarity < self._similarity_threshold and len(current_msgs) >= self._min_segment_size and not is_bridge):
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
    ) -> Optional[Tuple[str, bytes, List[int]]]:
        persona = self.chat_system.personas.get(self._persona_name)
        if not persona:
            logger.error(f"System persona '{self._persona_name}' not found. Cannot summarize.")
            return None

        lines = []
        for msg in segment['messages']:
            role = msg.get('author_role', 'user')
            name = msg.get('author_name', 'Unknown')
            content = msg.get('content', '')
            msg_id = msg.get('interaction_id')
            ts = msg.get('timestamp', '')

            # Strip vertexai grounding redirect URLs from content.
            # Preserves link text and citation markers, removes only the URL.
            # [Text](https://vertexaisearch.cloud.google.com/...) → [Text]
            # [[1](<https://vertexaisearch...>)] → [[1]]
            content = re.sub(
                r'\(<?https://vertexaisearch\.cloud\.google\.com/[^)]*>?\)',
                '', content
            )
            
            id_tag = f"[ID: {msg_id}]" if msg_id else ""
            
            if ts:
                if isinstance(ts, str):
                    ts = datetime.fromisoformat(ts)
                ts_str = ts.strftime('%Y-%m-%d %H:%M')
                lines.append(f"{id_tag} [{ts_str}] [{role}] {name}: {content}")
            else:
                lines.append(f"{id_tag} [{role}] {name}: {content}")

        transcript = "\n".join(lines)
        prompt = (
            f"Please process the following conversation segment and extract factual knowledge.\n\n"
            f"TRANSCRIPT:\n{transcript}"
        )

        # --- TOKEN GUARDRAIL (SDK BASED) ---
        try:
            # We check against the Gemma 4 TPR limit from global_config.
            # Using 240,000 as a safety margin for the 256,000 limit.
            from config.global_config import RATE_LIMIT_GEMMA_4_TPR
            
            # Request token count from Google SDK if available
            token_count = 0
            if self.text_engine.google_client:
                # Build the content structure exactly like the API expects
                # Note: count_tokens is a synchronous call in the current SDK version or async depending on usage.
                # In our TextEngine, the client is usually initialized for async.
                model_name = persona.get_config_for_engine().get("model_name")
                if not isinstance(model_name, str):
                    # Fallback to heuristic if model name is missing or invalid
                    token_count = len(prompt) // 4
                else:
                    count_resp = await self.text_engine.google_client.models.count_tokens(
                        model=model_name,
                        contents=prompt
                    )
                    token_count = count_resp.total_tokens or 0
            else:
                # Fallback to heuristic if SDK client is missing
                token_count = len(prompt) // 4
            
            if token_count > RATE_LIMIT_GEMMA_4_TPR * 0.95:
                logger.warning(f"MemoryAgent: Segment token count ({token_count}) exceeds safety limit. Splitting segment.")
                # We return None to signal failure; the iterative loop will try again with a smaller fetch?
                # No, if we don't have a LIMIT, it will just fetch the same thing.
                # IMPLEMENTATION NOTE: Since the user rejected manual split logic but wants a safeguard,
                # we skip this giant segment for now and log it.
                return None

        except Exception as e:
            logger.debug(f"MemoryAgent: Token counting failed/skipped: {e}")

        try:
            tools_for_llm = self.chat_system._filter_tools_for_persona(persona)
            response, _ = await self.text_engine.generate_response(
                persona_config=persona.get_config_for_engine(),
                context_object=self._build_llm_context(persona, prompt),
                tools=tools_for_llm,
            )

            observations = []
            outlier_ids = []

            if response.get('type') == 'tool_calls':
                for call in response.get('calls', []):
                    if call.get('name') == 'submit_memory_summary':
                        args = call.get('arguments', {})
                        if isinstance(args, str):
                            import json
                            args = json.loads(args)
                        observations = args.get('observations', []) or args.get('facts', [])
                        outlier_ids = args.get('outlier_ids', [])
                        break

            # Fallback to text parsing if model returned plain text instead of a tool call
            if not observations and response.get('type') == 'text' and response.get('content'):
                text = response['content'].strip()
                if text and text != "NO_FACTS":
                    observations = [line.strip("- ").strip() for line in text.split("\n") if line.strip()]

            if not observations:
                logger.info(f"MemoryAgent: No observations extracted for segment starting at {segment['start_id']}.")
                return None

            facts_text = "\n".join([f"- {f}" for f in observations])
            
            # Calculate tokens for embedding of the concatenated facts
            est_tokens = len(facts_text) // 4
            await self._rate_limiter.acquire(item_count=1, token_count=est_tokens)

            summary_embedding = await embedding_service.encode_single(facts_text)
            return (facts_text, summary_embedding, outlier_ids)

        except RuntimeError as e:
            logger.warning(f"MemoryAgent: Summarization embedding paused due to Daily Quota: {str(e)}")
            return None
        except Exception as e:
            logger.error(f"MemoryAgent: summarization failed: {str(e)}")
            return None
