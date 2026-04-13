import logging
import numpy as np
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional, Tuple

from src.database.memory_manager import (
    MemoryManager, LEVEL_EPISODIC, LEVEL_CORE
)
from src.embedding_service import EmbeddingService
from src.engine import TextEngine
from src.persona import Persona

# The system persona used for LLM-based cluster compression
SUMMARIZER_MODEL = 'gemma-4-31b-it'
SUMMARIZER_PERSONA_NAME = 'memory_summarizer'

logger = logging.getLogger(__name__)

class MemoryConsolidator:
    def __init__(self, memory_manager: MemoryManager, text_engine: TextEngine, embedding_service: EmbeddingService):
        self.memory_manager = memory_manager
        self.text_engine = text_engine
        self.embedding_service = embedding_service
        self.similarity_threshold = 0.90  # Strict Topic Isolation (prevents Hardware merging with Wagyu/Jury)

    async def start_daemon(self, check_interval_seconds: int = 3600) -> None:
        """Infinite loop to periodically consolidate memory across all active channels."""
        logger.info(f"MemoryConsolidator daemon started (interval: {check_interval_seconds}s)")
        while True:
            try:
                await self._run_global_consolidation()
            except Exception as e:
                logger.error(f"Error in MemoryConsolidator loop: {e}", exc_info=True)
            import asyncio
            await asyncio.sleep(check_interval_seconds)

    async def _run_global_consolidation(self) -> None:
        with self.memory_manager._lock:
            conn = self.memory_manager._get_connection()
            cursor = conn.cursor()
            cursor.execute('''
                SELECT DISTINCT seg.persona_name, seg.channel, seg.server_id
                FROM Memory_Summaries ms
                JOIN Memory_Segments seg ON ms.segment_id = seg.segment_id
                WHERE ms.summary_level = ?
            ''', (LEVEL_EPISODIC,))
            targets = [dict(r) for r in cursor.fetchall()]

        if not targets:
            return

        for t in targets:
            # Persona for DB lookup (the actual stored persona_name)
            target_persona_name = t['persona_name']
            # Separate LLM persona for text generation — uses the memory_summarizer model
            llm_persona = Persona(
                persona_name=SUMMARIZER_PERSONA_NAME,
                model_name=SUMMARIZER_MODEL,
                prompt="You are a core memory consolidation process. Merge episodic memories into a concise Core Fact Profile.",
            )
            await self.consolidate_memory(
                llm_persona, target_persona_name, t['channel'], t.get('server_id')
            )

    async def consolidate_memory(
            self,
            llm_persona: Persona,
            target_persona_name: str,
            channel: str,
            server_id: Optional[str] = None,
    ) -> None:
        """
        Runs the background optimization pipeline (Mem0-lite).
        Clusters level=0 summaries for target_persona_name/channel and compresses
        them into level=1 'Core Profiles' using llm_persona for the LLM calls.
        """
        with self.memory_manager._lock:
            conn = self.memory_manager._get_connection()
            cursor = conn.cursor()
            # Fetch level 0 summaries for this specific persona + channel
            query = """
                SELECT ms.summary_id, ms.segment_id, ms.content, ms.created_at, v.embedding
                FROM Memory_Summaries ms
                JOIN vec_Memory_Summaries v ON ms.summary_id = v.summary_id
                JOIN Memory_Segments seg ON ms.segment_id = seg.segment_id
                WHERE ms.summary_level = ? AND seg.channel = ? AND seg.persona_name = ?
            """
            params = [LEVEL_EPISODIC, channel, target_persona_name]
            if server_id:
                query += " AND seg.server_id = ?"
                params.append(server_id)
            else:
                query += " AND seg.server_id IS NULL"
                
            cursor.execute(query, params)
            rows = [dict(r) for r in cursor.fetchall()]
            
        if not rows:
            logger.debug(f"consolidate_memory: no L0 rows for {target_persona_name}/{channel}")
            return

        logger.info("consolidate_memory: found %d L0 summaries for %s/%s", len(rows), target_persona_name, channel)

        # Simplified clustering via naive n^2 dot product (or using sqlite-vec natively)
        # We group items together that exceed the similarity threshold
        clusters = []
        assigned = set()
        
        for i, rowA in enumerate(rows):
            if rowA['summary_id'] in assigned:
                continue
            
            cluster = [rowA]
            assigned.add(rowA['summary_id'])
            
            vecA = np.frombuffer(rowA['embedding'], dtype=np.float32)
            for j, rowB in enumerate(rows[i+1:], i+1):
                if rowB['summary_id'] in assigned:
                    continue
                    
                vecB = np.frombuffer(rowB['embedding'], dtype=np.float32)
                sim = float(np.dot(vecA, vecB))
                if sim >= self.similarity_threshold:
                    cluster.append(rowB)
                    assigned.add(rowB['summary_id'])
            
            if len(cluster) > 1:
                logger.info(f"  -> cluster of {len(cluster)} summaries (threshold={self.similarity_threshold})")
                clusters.append(cluster)
            else:
                logger.debug(f"  -> singleton, skipping")

        logger.info(f"consolidate_memory: formed {len(clusters)} cluster(s) for {target_persona_name}/{channel}")
        
        for cluster in clusters:
            try:
                await self._compress_cluster(cluster, llm_persona, channel, server_id)
            except Exception as e:
                logger.error(f"Failed to compress memory cluster: {e}")

    async def _compress_cluster(self, cluster: List[Dict[str, Any]], llm_persona: Persona, channel: str, server_id: Optional[str]) -> None:
        # Provide the LLM with the chronological segments
        cluster.sort(key=lambda x: str(x['created_at']))
        
        lines = []
        for idx, row in enumerate(cluster):
            try:
                # Handle ISO bytes
                if isinstance(row['created_at'], bytes):
                    dt = datetime.fromisoformat(row['created_at'].decode('utf-8'))
                elif isinstance(row['created_at'], str):
                    dt = datetime.fromisoformat(row['created_at'])
                else:
                    dt = row['created_at']
                ts_str = dt.strftime('%Y-%m-%d %H:%M')
            except Exception:
                ts_str = "Unknown Date"
            lines.append(f"[{ts_str}] Memory Segment: {row['content']}")
            
        transcript = "\n".join(lines)

        system_prompt = (
            "You are a core memory consolidation process.\n"
            "Review the following chronological episodic memories about a specific topic/entity.\n"
            "Merge them into a single, comprehensive, up-to-date 'Core Fact Profile'.\n\n"
            "RULES:\n"
            "1. Replace outdated information with the newest state.\n"
            "2. Keep the summary incredibly dense with concrete details.\n"
            "3. Include explicit keywords so an agent can 'drill down' for specifics.\n"
            "4. Respond ONLY with the Core Fact Profile. No preamble."
        )

        user_message = f"MEMORIES TO CONSOLIDATE:\n{transcript}"

        core_profile_text, _ = await self.text_engine.generate_response(
            persona_config=llm_persona.get_config_for_engine(),
            context_object={
                "persona_prompt": system_prompt,
                "history": [{"role": "user", "content": user_message}],
                "current_message": {"text": "", "image_url": None}
            },
            tools=None,
        )
        
        if core_profile_text.get('type') != 'text':
            return
            
        summary_text = core_profile_text.get("content", "").strip()
        if not summary_text:
            return
            
        summary_emb = await self.embedding_service.encode_single(summary_text)
        now = datetime.now(timezone.utc)

        # Pick a segment_id from the cluster to maintain persona/channel context
        target_segment_id = cluster[0].get('segment_id')

        with self.memory_manager.transaction() as conn:
            cursor = conn.execute(
                """INSERT INTO Memory_Summaries
                   (segment_id, content, embedding, model_name, created_at, summary_level)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (target_segment_id, summary_text, summary_emb, self.embedding_service.model_name, now, LEVEL_CORE)
            )
            new_summary_id = cursor.lastrowid
            
            conn.execute(
                """INSERT INTO vec_Memory_Summaries (summary_id, embedding) VALUES (?, ?)""",
                (new_summary_id, summary_emb)
            )
            
            # Map old summaries to this parent. 
            # They stay at LEVEL_EPISODIC but are filtered out of default retrieval 
            # because parent_summary_id is now NOT NULL.
            old_ids = [r['summary_id'] for r in cluster]
            placeholders = ','.join('?' for _ in old_ids)
            conn.execute(
                f"UPDATE Memory_Summaries SET parent_summary_id = ? WHERE summary_id IN ({placeholders})",
                [new_summary_id] + old_ids
            )
            
        logger.info(f"Consolidated {len(cluster)} memories into Core Profile ID: {new_summary_id}")
