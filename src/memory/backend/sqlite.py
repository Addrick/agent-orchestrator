# src/memory/backend/sqlite.py
"""SQLite implementation of MemoryBackend.

Sprint 1 (DP-108) carve-out: this class hosts the SQL bodies that were on
MemoryManager. MemoryManager keeps the public methods (callers unchanged) but
delegates them here. All connection/transaction state is owned by MemoryManager
— this backend reaches back through `mm._lock`, `mm._get_connection()`, and
`mm._suppression_filter()` to share the single connection.

Pure refactor. No behavior change. New-shape methods on the ABC stay as
NotImplementedError stubs / noops; Sprint 2 lands them on HindsightBackend.
"""
from __future__ import annotations

import logging
import sqlite3
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

from config.global_config import EMBEDDING_MODEL
from src.memory.backend.base import MemoryBackend, MemoryHit

if TYPE_CHECKING:
    from src.embedding_service import EmbeddingService
    from src.memory.memory_manager import MemoryManager

logger = logging.getLogger(__name__)


class SqliteSemanticBackend(MemoryBackend):
    """SQLite-backed semantic + episodic store.

    Wraps the existing schema (Memory_Embeddings, Memory_Segments,
    Memory_Summaries, Agent_Actions, Segment_Failures) with no logic change.
    """

    # Failure outcome literals copied from MemoryManager (kept private to backend).
    _FAILURE_OUTCOMES = ("failed", "error", "notification_failed")

    def __init__(
        self,
        memory_manager: "MemoryManager",
        embedding_service: "Optional[EmbeddingService]" = None,
    ) -> None:
        # The backend shares the MemoryManager's connection, lock, and
        # suppression filter. This avoids two parallel SQLite connections to
        # the same DB file (transactions would deadlock) and keeps the
        # transcript layer + semantic layer transactionally consistent.
        self._mm = memory_manager
        # New-shape recall translates a query string → embedding → existing
        # retrieve_relevant_summaries. ChatSystem injects the service after
        # construction via set_embedding_service(); recall returns [] when
        # absent rather than raising — matches HindsightBackend's "fail soft
        # on retrieval" policy.
        self._embedding_service = embedding_service

    def set_embedding_service(self, service: "Optional[EmbeddingService]") -> None:
        self._embedding_service = service

    # ---------- Episodic ----------

    def log_agent_action(
        self,
        agent_name: str,
        action_type: str,
        trigger_context: Optional[str] = None,
        action_payload: Optional[str] = None,
        outcome: Optional[str] = None,
        outcome_payload: Optional[str] = None,
        parent_id: Optional[int] = None,
    ) -> int:
        with self._mm._lock:
            conn = self._mm._get_connection()
            cursor = conn.cursor()
            cursor.execute(
                """INSERT INTO Agent_Actions (parent_id, agent_name, action_type, trigger_context, action_payload, outcome, outcome_payload)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (parent_id, agent_name, action_type, trigger_context, action_payload, outcome, outcome_payload),
            )
            return int(cursor.lastrowid) if cursor.lastrowid is not None else 0

    def update_agent_action_outcome(
        self,
        action_id: int,
        outcome: str,
        outcome_payload: Optional[str] = None,
    ) -> None:
        with self._mm._lock:
            conn = self._mm._get_connection()
            conn.execute(
                "UPDATE Agent_Actions SET outcome = ?, outcome_payload = ? WHERE id = ?",
                (outcome, outcome_payload, action_id),
            )
            conn.commit()

    def add_action_contexts(
        self,
        action_id: int,
        contexts: List[Tuple[str, str]],
    ) -> None:
        with self._mm._lock:
            conn = self._mm._get_connection()
            for context_type, context_value in contexts:
                try:
                    conn.execute(
                        "INSERT OR IGNORE INTO Agent_Action_Contexts (action_id, context_type, context_value) VALUES (?, ?, ?)",
                        (action_id, context_type, context_value),
                    )
                except sqlite3.IntegrityError:
                    pass
            conn.commit()

    def get_relevant_agent_actions(
        self,
        agent_name: str,
        match_contexts: Optional[List[Tuple[str, str]]] = None,
        match_types: Optional[List[str]] = None,
        limit: int = 15,
    ) -> List[Dict[str, Any]]:
        with self._mm._lock:
            conn = self._mm._get_connection()
            cursor = conn.cursor()
            fetch_limit = limit * 3
            query = "SELECT * FROM Agent_Actions WHERE agent_name = ? AND parent_id IS NULL"
            params: List[Any] = [agent_name]

            if match_types:
                placeholders = ",".join("?" for _ in match_types)
                query += f" AND action_type IN ({placeholders})"
                params.extend(match_types)

            query += " ORDER BY timestamp DESC LIMIT ?"
            params.append(fetch_limit)
            cursor.execute(query, params)
            all_actions = [dict(row) for row in cursor.fetchall()]

            matched_ids: set[int] = set()
            if match_contexts:
                for ctx_type, ctx_value in match_contexts:
                    cursor.execute(
                        "SELECT action_id FROM Agent_Action_Contexts WHERE context_type = ? AND context_value = ?",
                        (ctx_type, ctx_value),
                    )
                    matched_ids.update(row["action_id"] for row in cursor.fetchall())

            entity_matches = []
            failures = []
            other = []
            for action in all_actions:
                if action["id"] in matched_ids:
                    entity_matches.append(action)
                elif action.get("outcome") in self._FAILURE_OUTCOMES:
                    failures.append(action)
                else:
                    other.append(action)

            return (entity_matches + failures + other)[:limit]

    def get_action_steps(self, parent_id: int) -> List[Dict[str, Any]]:
        with self._mm._lock:
            conn = self._mm._get_connection()
            cursor = conn.cursor()
            cursor.execute(
                "SELECT * FROM Agent_Actions WHERE parent_id = ? ORDER BY timestamp ASC",
                (parent_id,),
            )
            return [dict(row) for row in cursor.fetchall()]

    # ---------- Semantic ----------

    def store_message_embedding(
        self,
        interaction_id: int,
        embedding: bytes,
        model_name: str,
        created_at: datetime,
    ) -> None:
        with self._mm._lock:
            conn = self._mm._get_connection()
            conn.execute(
                "INSERT OR REPLACE INTO Message_Embeddings (interaction_id, embedding, model_name, created_at) VALUES (?, ?, ?, ?)",
                (interaction_id, embedding, model_name, created_at),
            )
            conn.commit()

    def get_unembedded_messages(
        self,
        persona_name: str,
        channel: str,
        server_id: Optional[str] = None,
        limit: int = 50,
        model_name: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        with self._mm._lock:
            conn = self._mm._get_connection()
            cursor = conn.cursor()

            query = ("SELECT ui.interaction_id, ui.author_role, ui.author_name, ui.content"
                     " FROM User_Interactions ui LEFT JOIN Message_Embeddings me ON ui.interaction_id = me.interaction_id"
                     " WHERE ui.persona_name = ? AND ui.channel = ?")
            params: List[Any] = [persona_name, channel]

            if server_id is not None:
                query += " AND ui.server_id = ?"
                params.append(server_id)
            else:
                query += " AND ui.server_id IS NULL"

            query += self._mm._suppression_filter("ui")
            query += " AND ui.content IS NOT NULL AND ui.content != ''"

            active_model = model_name if model_name else EMBEDDING_MODEL
            query += " AND (me.embedding IS NULL OR me.model_name != ?)"
            params.append(active_model)

            query += " ORDER BY ui.interaction_id ASC LIMIT ?"
            params.append(limit)

            cursor.execute(query, params)
            return [dict(row) for row in cursor.fetchall()]

    def store_segment(
        self,
        channel: str,
        server_id: Optional[str],
        persona_name: str,
        start_id: int,
        end_id: int,
        message_count: int,
        created_at: datetime,
    ) -> int:
        with self._mm._lock:
            conn = self._mm._get_connection()
            cursor = conn.cursor()
            cursor.execute(
                """INSERT INTO Memory_Segments (channel, server_id, persona_name, start_interaction_id, end_interaction_id, message_count, created_at, first_message_at, last_message_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (channel, server_id, persona_name, start_id, end_id, message_count, created_at, None, None),
            )
            conn.commit()
            return int(cursor.lastrowid) if cursor.lastrowid is not None else 0

    def store_summary(
        self,
        segment_id: int,
        content: str,
        embedding: bytes,
        model_name: str,
        created_at: datetime,
        summary_level: Optional[int] = None,
        parent_summary_id: Optional[int] = None,
        untrusted: bool = False,
    ) -> int:
        with self._mm._lock:
            conn = self._mm._get_connection()
            cursor = conn.cursor()

            cols = ["segment_id", "content", "embedding", "model_name", "created_at"]
            vals: List[Any] = [segment_id, content, embedding, model_name, created_at]

            if summary_level is not None:
                cols.append("summary_level")
                vals.append(summary_level)
            if parent_summary_id is not None:
                cols.append("parent_summary_id")
                vals.append(parent_summary_id)
            if untrusted:
                cols.append("untrusted")
                vals.append(1)

            placeholders = ", ".join("?" for _ in vals)
            col_names = ", ".join(cols)

            cursor.execute(
                f"INSERT INTO Memory_Summaries ({col_names}) VALUES ({placeholders})",
                vals,
            )
            summary_id = cursor.lastrowid
            cursor.execute(
                """INSERT INTO vec_Memory_Summaries (summary_id, embedding)
                   VALUES (?, ?)""",
                (summary_id, embedding),
            )
            conn.commit()
            return int(summary_id) if summary_id is not None else 0

    def get_summaries_for_channel(
        self,
        channel: str,
        persona_name: str,
        server_id: Optional[str] = None,
        exclude_after_interaction_id: Optional[int] = None,
        model_name: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        with self._mm._lock:
            conn = self._mm._get_connection()
            cursor = conn.cursor()
            query = (
                "SELECT ms.summary_id, ms.segment_id, ms.content, ms.embedding, ms.model_name, ms.created_at, ms.untrusted, seg.channel, seg.persona_name, seg.start_interaction_id, seg.end_interaction_id"
                " FROM Memory_Summaries ms JOIN Memory_Segments seg ON ms.segment_id = seg.segment_id"
                " WHERE seg.channel = ? AND seg.persona_name = ?")
            params: List[Any] = [channel, persona_name]

            if server_id is not None:
                query += " AND seg.server_id = ?"
                params.append(server_id)
            else:
                query += " AND seg.server_id IS NULL"

            if exclude_after_interaction_id is not None:
                query += " AND seg.start_interaction_id < ?"
                params.append(exclude_after_interaction_id)

            active_model = model_name if model_name else EMBEDDING_MODEL
            query += " AND ms.model_name = ?"
            params.append(active_model)

            query += " ORDER BY seg.start_interaction_id ASC"
            cursor.execute(query, params)
            return [dict(row) for row in cursor.fetchall()]

    def get_unsegmented_embedded_messages(
        self,
        persona_name: str,
        channel: str,
        server_id: Optional[str] = None,
        model_name: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        with self._mm._lock:
            conn = self._mm._get_connection()
            cursor = conn.cursor()

            query = ("SELECT ui.interaction_id, ui.author_role, ui.author_name, ui.content, ui.timestamp, me.embedding, ui.parent_summary_id, ui.reply_to_id"
                     " FROM User_Interactions ui JOIN Message_Embeddings me ON ui.interaction_id = me.interaction_id"
                     " WHERE ui.persona_name = ? AND ui.channel = ? AND ui.parent_summary_id IS NULL")
            params: List[Any] = [persona_name, channel]

            if server_id is not None:
                query += " AND ui.server_id = ?"
                params.append(server_id)
            else:
                query += " AND ui.server_id IS NULL"

            query += self._mm._suppression_filter("ui")
            query += " AND ui.content IS NOT NULL AND ui.content != ''"

            active_model = model_name if model_name else EMBEDDING_MODEL
            query += " AND me.model_name = ?"
            params.append(active_model)

            query += " ORDER BY ui.interaction_id ASC"
            if limit is not None:
                query += " LIMIT ?"
                params.append(limit)

            cursor.execute(query, params)
            return [dict(row) for row in cursor.fetchall()]

    def retrieve_relevant_summaries(
        self,
        persona_name: str,
        channel: str,
        server_id: Optional[str] = None,
        user_identifier: Optional[str] = None,
        memory_mode: str = "channel",
        include_ambient: bool = True,
        exclude_after_interaction_id: Optional[int] = None,
        model_name: Optional[str] = None,
        query_embeddings: Optional[List[bytes]] = None,
        limit: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        import logging
        logger = logging.getLogger(__name__)

        if memory_mode == "ticket":
            return []

        with self._mm._lock:
            conn = self._mm._get_connection()
            cursor = conn.cursor()

            dist_select = ""
            dist_params: List[Any] = []
            if query_embeddings:
                d_exprs = ["vec_distance_cosine(v.embedding, ?)"] * len(query_embeddings)
                if len(d_exprs) > 1:
                    dist_select = f", min({', '.join(d_exprs)}) as dist"
                else:
                    dist_select = f", {d_exprs[0]} as dist"
                dist_params.extend(query_embeddings)

            base_select = (
                f"SELECT ms.summary_id, ms.segment_id, ms.content, ms.embedding,"
                f" ms.model_name, ms.created_at, ms.untrusted, seg.channel, seg.persona_name,"
                f" seg.start_interaction_id, seg.end_interaction_id, seg.last_message_at{dist_select}"
                f" FROM Memory_Summaries ms"
                f" JOIN Memory_Segments seg ON ms.segment_id = seg.segment_id"
            )
            if query_embeddings:
                base_select += " JOIN vec_Memory_Summaries v ON ms.summary_id = v.summary_id"

            build_args = (memory_mode, channel, server_id, user_identifier,
                          exclude_after_interaction_id, model_name)

            # _build_summary_where stays a static helper on MemoryManager so legacy
            # callers + tests that reach for it directly aren't broken.
            from src.memory.memory_manager import MemoryManager
            where_clause, params = MemoryManager._build_summary_where(persona_name, *build_args)
            query = f"{base_select} WHERE {where_clause}"
            final_params = dist_params + params

            if include_ambient and persona_name != "ambient":
                amb_where, amb_params = MemoryManager._build_summary_where("ambient", *build_args)
                query += f" UNION {base_select} WHERE {amb_where}"
                final_params.extend(dist_params + amb_params)

            if query_embeddings:
                query += " ORDER BY dist ASC"

            if limit:
                query += " LIMIT ?"
                final_params.append(limit)

            safe_params = [f"<blob:{len(p)}b>" if isinstance(p, bytes) else p for p in final_params]
            logger.info(f"MemoryManager.retrieve: Executing query for {persona_name} (mode: {memory_mode}). Params: {safe_params}")

            cursor.execute(query, final_params)
            rows = [dict(row) for row in cursor.fetchall()]

            if query_embeddings:
                if not rows:
                    logger.info(f"MemoryManager.retrieve: 0 summaries found for {persona_name} in {channel} (mode={memory_mode})")
                else:
                    best_dist = rows[0].get("dist", 1.0)
                    logger.info(f"MemoryManager.retrieve: {len(rows)} summaries found (best similarity dist: {best_dist:.4f})")

            return rows

    def record_segment_failure(
        self,
        channel: str,
        server_id: Optional[str],
        persona_name: str,
        start_id: int,
        end_id: int,
        message_count: int,
        error_reason: Optional[str] = None,
    ) -> None:
        now = datetime.now(timezone.utc)
        with self._mm._lock:
            conn = self._mm._get_connection()
            if server_id is not None:
                existing = conn.execute(
                    "SELECT failure_id, attempts FROM Segment_Failures"
                    " WHERE channel = ? AND persona_name = ? AND server_id = ?"
                    " AND start_interaction_id = ? AND end_interaction_id = ?",
                    (channel, persona_name, server_id, start_id, end_id),
                ).fetchone()
            else:
                existing = conn.execute(
                    "SELECT failure_id, attempts FROM Segment_Failures"
                    " WHERE channel = ? AND persona_name = ? AND server_id IS NULL"
                    " AND start_interaction_id = ? AND end_interaction_id = ?",
                    (channel, persona_name, start_id, end_id),
                ).fetchone()

            if existing:
                conn.execute(
                    "UPDATE Segment_Failures SET attempts = ?, last_attempt_at = ?, error_reason = ? WHERE failure_id = ?",
                    (existing["attempts"] + 1, now, error_reason, existing["failure_id"]),
                )
            else:
                conn.execute(
                    "INSERT INTO Segment_Failures"
                    " (channel, server_id, persona_name, start_interaction_id, end_interaction_id,"
                    "  message_count, attempts, last_attempt_at, error_reason)"
                    " VALUES (?, ?, ?, ?, ?, ?, 1, ?, ?)",
                    (channel, server_id, persona_name, start_id, end_id, message_count, now, error_reason),
                )
            conn.commit()

    def get_failed_segment_ranges(
        self,
        channel: str,
        persona_name: str,
        server_id: Optional[str] = None,
        max_attempts: int = 3,
        cooldown_hours: float = 24.0,
    ) -> List[Dict[str, Any]]:
        with self._mm._lock:
            conn = self._mm._get_connection()
            cutoff = datetime.now(timezone.utc).timestamp() - (cooldown_hours * 3600)
            query = (
                "SELECT start_interaction_id, end_interaction_id, attempts, last_attempt_at, error_reason"
                " FROM Segment_Failures"
                " WHERE channel = ? AND persona_name = ?"
            )
            params: List[Any] = [channel, persona_name]
            if server_id is not None:
                query += " AND server_id = ?"
                params.append(server_id)
            else:
                query += " AND server_id IS NULL"
            query += " AND (attempts >= ? OR last_attempt_at > ?)"
            params.extend([max_attempts, datetime.fromtimestamp(cutoff, tz=timezone.utc)])
            cursor = conn.cursor()
            cursor.execute(query, params)
            return [dict(row) for row in cursor.fetchall()]

    def clear_segment_failure(
        self,
        channel: str,
        persona_name: str,
        server_id: Optional[str],
        start_id: int,
        end_id: int,
    ) -> None:
        with self._mm._lock:
            conn = self._mm._get_connection()
            if server_id is not None:
                conn.execute(
                    "DELETE FROM Segment_Failures"
                    " WHERE channel = ? AND persona_name = ? AND server_id = ?"
                    " AND start_interaction_id = ? AND end_interaction_id = ?",
                    (channel, persona_name, server_id, start_id, end_id),
                )
            else:
                conn.execute(
                    "DELETE FROM Segment_Failures"
                    " WHERE channel = ? AND persona_name = ? AND server_id IS NULL"
                    " AND start_interaction_id = ? AND end_interaction_id = ?",
                    (channel, persona_name, start_id, end_id),
                )
            conn.commit()

    def get_active_channels(
        self,
        model_name: Optional[str] = None,
    ) -> List[Tuple[str, str, Optional[str]]]:
        with self._mm._lock:
            conn = self._mm._get_connection()
            cursor = conn.cursor()
            active_model = model_name if model_name else EMBEDDING_MODEL

            q1 = ("SELECT DISTINCT ui.channel, ui.persona_name, ui.server_id FROM User_Interactions ui"
                  " LEFT JOIN Message_Embeddings me ON ui.interaction_id = me.interaction_id"
                  " WHERE ui.content IS NOT NULL AND ui.content != ''" + self._mm._suppression_filter("ui"))
            params: List[Any] = [active_model]
            q1 += " AND (me.embedding IS NULL OR me.model_name != ?)"

            q2 = ("SELECT DISTINCT ui.channel, ui.persona_name, ui.server_id FROM User_Interactions ui"
                  " JOIN Message_Embeddings me ON ui.interaction_id = me.interaction_id"
                  " WHERE ui.content IS NOT NULL AND ui.content != '' AND ui.interaction_id > COALESCE("
                  " (SELECT MAX(seg.end_interaction_id) FROM Memory_Segments seg WHERE seg.channel = ui.channel"
                  " AND seg.persona_name = ui.persona_name AND (seg.server_id = ui.server_id OR (seg.server_id IS NULL AND ui.server_id IS NULL))), 0)"
                  + self._mm._suppression_filter("ui"))

            q3 = ("SELECT DISTINCT ui.channel, ui.persona_name, ui.server_id FROM User_Interactions ui"
                  " JOIN Message_Embeddings me ON ui.interaction_id = me.interaction_id"
                  " WHERE ui.content IS NOT NULL AND ui.content != ''"
                  " AND ui.parent_summary_id IS NULL AND me.model_name = ?"
                  + self._mm._suppression_filter("ui"))
            params.append(active_model)

            query = q1 + " UNION " + q2 + " UNION " + q3
            cursor.execute(query, params)
            return [(row["channel"], row["persona_name"], row["server_id"]) for row in cursor.fetchall()]

    def get_last_segment_tail_embeddings(
        self,
        channel: str,
        persona_name: str,
        server_id: Optional[str] = None,
        n: int = 3,
        model_name: Optional[str] = None,
    ) -> Optional[List[bytes]]:
        with self._mm._lock:
            conn = self._mm._get_connection()
            cursor = conn.cursor()
            seg_query = "SELECT segment_id, start_interaction_id, end_interaction_id FROM Memory_Segments WHERE channel = ? AND persona_name = ?"
            seg_params: List[Any] = [channel, persona_name]

            if server_id is not None:
                seg_query += " AND server_id = ?"
                seg_params.append(server_id)
            else:
                seg_query += " AND server_id IS NULL"

            seg_query += " ORDER BY end_interaction_id DESC LIMIT 1"
            cursor.execute(seg_query, seg_params)
            seg_row = cursor.fetchone()
            if seg_row is None:
                return None

            emb_query = (
                "SELECT me.embedding FROM Message_Embeddings me JOIN User_Interactions ui ON me.interaction_id = ui.interaction_id"
                " WHERE ui.channel = ? AND ui.persona_name = ? AND ui.interaction_id BETWEEN ? AND ?")
            emb_params: List[Any] = [channel, persona_name, seg_row["start_interaction_id"],
                                     seg_row["end_interaction_id"]]

            if server_id is not None:
                emb_query += " AND ui.server_id = ?"
                emb_params.append(server_id)
            else:
                emb_query += " AND ui.server_id IS NULL"

            active_model = model_name if model_name else EMBEDDING_MODEL
            emb_query += " AND me.model_name = ?"
            emb_params.append(active_model)

            emb_query += " ORDER BY me.interaction_id DESC LIMIT ?"
            emb_params.append(n)

            cursor.execute(emb_query, emb_params)
            rows = cursor.fetchall()
            if not rows:
                return None
            return [row["embedding"] for row in reversed(rows)]

    # ===== New Hindsight-shape (DP-113) =====
    #
    # `retain_turn` is intentionally a noop on SQLite: the legacy
    # MemoryAgent batch loop continues to drive embedding/segmentation/
    # summarization off `User_Interactions` rows under the `sqlite_legacy`
    # selector. Per-turn inline writes would either duplicate that work or
    # double the embedding-API cost for no behaviour change. See
    # tasks/DP-113.md §"Consolidation trigger location".
    #
    # `recall` translates the new-shape API to the existing
    # `retrieve_relevant_summaries` path so ChatSystem can be rewired to the
    # backend boundary while still riding the legacy summary index.

    @staticmethod
    def _parse_scope_tags(tag_filter: Optional[List[str]]) -> Dict[str, Optional[str]]:
        # `exclude_after` carries the sliding-window cutoff so the legacy
        # summary index doesn't surface memories that are still in the
        # visible history window (see ChatSystem._retrieve_memory_block).
        scope: Dict[str, Optional[str]] = {
            "channel": None, "server": None, "user": None,
            "interface": None, "exclude_after": None,
        }
        if not tag_filter:
            return scope
        for tag in tag_filter:
            if ":" not in tag:
                continue
            key, value = tag.split(":", 1)
            if key in scope:
                scope[key] = value
        return scope

    async def retain_turn(
        self,
        bank_id: str,
        role: str,
        content: str,
        *,
        timestamp: datetime,
        scope_tags: List[str],
        source_persona: str,
        untrusted: bool = False,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        # Noop under sqlite_legacy — see class-level note above.
        return ""

    async def recall(
        self,
        bank_id: str,
        query: str,
        *,
        k: int = 10,
        types: Optional[List[str]] = None,
        tag_filter: Optional[List[str]] = None,
        max_tokens: Optional[int] = None,
        budget: Optional[str] = None,
    ) -> List[MemoryHit]:
        if self._embedding_service is None:
            logger.info("SqliteSemanticBackend.recall: no embedding service set; returning empty.")
            return []

        scope = self._parse_scope_tags(tag_filter)
        channel = scope["channel"]
        server_id = scope["server"]
        user_identifier = scope["user"]
        memory_mode = "channel" if channel else "global"
        try:
            exclude_after = int(scope["exclude_after"]) if scope["exclude_after"] else None
        except (TypeError, ValueError):
            exclude_after = None

        try:
            query_embeddings = await self._embedding_service.encode([query])
        except Exception as e:  # noqa: BLE001
            logger.warning("SqliteSemanticBackend.recall: embedding failed: %s", e)
            return []
        if not query_embeddings:
            return []

        rows = self.retrieve_relevant_summaries(
            persona_name=bank_id,
            channel=channel or "",
            server_id=server_id,
            user_identifier=user_identifier,
            memory_mode=memory_mode,
            include_ambient=True,
            exclude_after_interaction_id=exclude_after,
            model_name=self._embedding_service.model_name,
            query_embeddings=query_embeddings,
            limit=k,
        )

        hits: List[MemoryHit] = []
        for row in rows:
            # `retrieve_relevant_summaries` returns cosine *distance* (smaller
            # is closer). Map to similarity score in [0, 1] so MemoryHit.score
            # follows "higher is better" like Hindsight's recall results.
            dist = row.get("dist")
            score = max(0.0, 1.0 - float(dist)) if dist is not None else 0.0
            ts = row.get("last_message_at") or row.get("created_at")
            tags = []
            if row.get("channel"):
                tags.append(f"channel:{row['channel']}")
            if row.get("persona_name"):
                tags.append(f"persona:{row['persona_name']}")
            hits.append(MemoryHit(
                id=str(row["summary_id"]),
                content=row.get("content", ""),
                score=score,
                untrusted=bool(row.get("untrusted", 0)),
                tags=tags,
                timestamp=ts if isinstance(ts, datetime) else None,
                metadata={
                    "segment_id": row.get("segment_id"),
                    "start_interaction_id": row.get("start_interaction_id"),
                    "end_interaction_id": row.get("end_interaction_id"),
                    "model_name": row.get("model_name"),
                },
            ))
        return hits
