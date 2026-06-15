# src/turn_persistence.py
"""Turn persistence for the chat pipeline (DP-200 slice B).

The write-side tail of a turn, extracted from ChatSystem: logging the user
row (or archiving the prior assistant on retry), committing/updating the
assistant row, fire-and-forget LTM retain, and the per-user API-request
caches that back `dump_history`. The orchestration kernel decides *when*
these happen; this module owns *how* and holds the cache state.
"""

import logging
from collections import defaultdict
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, cast

from config.global_config import MAX_CACHED_API_REQUESTS
from src.generation_events import ResponseType
from src.memory.backend.base import MemoryBackend
from src.memory.memory_manager import MemoryManager
from src.request_builder import build_scope_tags
from src.security.scrubber import get_scrubber

logger = logging.getLogger(__name__)


class TurnPersistence:
    """Owns turn write-paths (user/assistant rows, retain) + request caches."""

    def __init__(self, memory_manager: MemoryManager,
                 memory_backend: MemoryBackend) -> None:
        self.memory_manager = memory_manager
        self.memory_backend = memory_backend
        self.last_api_requests: Dict[str, Dict[str, Optional[Dict[str, Any]]]] = defaultdict(dict)
        # Per-turn list of every LLM-call payload in the tool loop (reset at the
        # first iteration of each turn). last_api_requests keeps only the final
        # payload for back-compat; this preserves the whole loop for dump_history.
        self.last_api_iterations: Dict[str, Dict[str, List[Dict[str, Any]]]] = defaultdict(dict)

    def store_api_request(self, user_identifier: str, persona_name: str,
                          payload: Dict[str, Any],
                          tools_for_llm: Optional[List[Dict[str, Any]]] = None,
                          is_first_iteration: bool = False) -> None:
        """Stores the last API request payload, evicting the oldest user entry if over capacity.

        `is_first_iteration` marks the opening LLM call of a turn; it resets the
        per-turn iteration list so dump_history shows the whole tool loop (one
        payload per LLM call) rather than only the final iteration.
        """
        # Egress scrub (DP-225 boundary 3): the cached payload is surfaced by
        # the /assemble inspector and the portal, so redact any registered
        # secret before it enters last_api_requests / last_api_iterations.
        payload = cast(Dict[str, Any], get_scrubber().scrub(payload))

        if tools_for_llm is not None:
            payload["_tools_for_llm"] = tools_for_llm
        else:
            existing = self.last_api_requests.get(user_identifier, {}).get(persona_name)
            if existing and "_tools_for_llm" in existing:
                payload["_tools_for_llm"] = existing["_tools_for_llm"]

        # LRU touch: re-storing a user's payload moves them to the most-recent
        # end so eviction drops a genuinely idle user, not whoever was seen
        # first. dict preserves insertion order; pop+reinsert is the move.
        if user_identifier in self.last_api_requests:
            self.last_api_requests[user_identifier] = self.last_api_requests.pop(user_identifier)
            if user_identifier in self.last_api_iterations:
                self.last_api_iterations[user_identifier] = self.last_api_iterations.pop(user_identifier)
        self.last_api_requests[user_identifier][persona_name] = payload

        if is_first_iteration:
            self.last_api_iterations[user_identifier][persona_name] = []
        self.last_api_iterations[user_identifier].setdefault(persona_name, []).append(payload)

        if len(self.last_api_requests) > MAX_CACHED_API_REQUESTS:
            # Evict the least-recently-used user (front of insertion order).
            oldest_key = next(iter(self.last_api_requests))
            del self.last_api_requests[oldest_key]
            self.last_api_iterations.pop(oldest_key, None)

    def log_user_turn(
            self,
            *,
            is_retry: bool,
            persona_name: str,
            user_identifier: str,
            channel: str,
            user_display_name: Optional[str],
            message: str,
            server_id: Optional[str],
            platform_message_id: Optional[str],
            timestamp: datetime,
    ) -> Tuple[Optional[int], Optional[int]]:
        """Log the new user turn or archive prior assistant for retry.

        Returns `(user_interaction_id, retry_assistant_id)`. Exactly one of
        the two will be set on success: retries archive the prior assistant
        and skip user-row insertion; non-retries log a fresh user row.
        """
        if is_retry:
            try:
                retry_assistant_id = self.memory_manager.handle_portal_retry(
                    persona_name=persona_name,
                    user_identifier=user_identifier,
                    channel=channel,
                )
            except Exception as e:
                logger.error(f"handle_portal_retry failed: {e}", exc_info=True)
                retry_assistant_id = None
            return None, retry_assistant_id

        # Symmetric with the assistant-side guard in commit_or_update_assistant:
        # an empty / whitespace-only user message must never land a phantom row.
        # kobold-lite continue/prefetch calls without a user message would
        # otherwise leave zero-length user_interaction rows between turns,
        # polluting context and confusing the model.
        if not message or not message.strip():
            return None, None

        try:
            user_interaction_id = self.memory_manager.log_message(
                user_identifier=user_identifier, persona_name=persona_name,
                channel=channel, author_role='user',
                author_name=user_display_name, content=message,
                timestamp=timestamp, server_id=server_id,
                platform_message_id=platform_message_id,
            )
        except Exception as e:
            logger.error(f"User log_message failed: {e}", exc_info=True)
            user_interaction_id = None
        return user_interaction_id, None

    def commit_or_update_assistant(
            self,
            *,
            persona_name: str,
            user_identifier: str,
            channel: str,
            server_id: Optional[str],
            final_text: str,
            response_type: ResponseType,
            user_interaction_id: Optional[int],
            retry_assistant_id: Optional[int],
            tool_context_json: Optional[str],
    ) -> Optional[int]:
        """Persist the assistant turn (UPDATE on retry, INSERT otherwise).

        Returns the canonical assistant interaction_id, or None when the
        text is empty or the response_type isn't a normal LLM generation.
        """
        if (not final_text or not final_text.strip()) and not tool_context_json:
            return None

        if retry_assistant_id is not None:
            if response_type == ResponseType.PENDING_CONFIRMATION:
                # A retried turn that parked for confirmation must not overwrite
                # the archived assistant row with the ephemeral confirmation text
                # (DP-130: the park renders as an unpersisted chunk; the resumed
                # continuation commits the real text).
                return None
            try:
                # Forward tool_context so the regenerated row's stored tool calls
                # stay paired with its new content — a retried turn may use a
                # different (or no) set of tools than the failed attempt.
                # Explicit reasoning_content=None clears the stale `<think>` of
                # the prior attempt: it no longer matches the regenerated text
                # (DP-141 sentinel contract — omitting would now *preserve* it).
                self.memory_manager.update_interaction_content(
                    retry_assistant_id, final_text,
                    reasoning_content=None, tool_context=tool_context_json,
                )
                return retry_assistant_id
            except Exception as e:
                logger.error(f"Retry update_interaction_content failed: {e}")
                return None

        if response_type != ResponseType.LLM_GENERATION:
            return None

        try:
            assistant_id: Optional[int] = self.memory_manager.log_message(
                user_identifier=user_identifier, persona_name=persona_name,
                channel=channel, author_role='assistant',
                author_name=persona_name, content=final_text,
                timestamp=datetime.now(), server_id=server_id,
                tool_context=tool_context_json,
                reply_to_id=user_interaction_id,
            )
            return assistant_id
        except Exception as e:
            logger.error(f"Assistant log_message failed: {e}", exc_info=True)
            return None

    async def retain_turn_safe(
        self,
        *,
        persona_name: str,
        role: str,
        content: str,
        user_identifier: str,
        channel: str,
        server_id: Optional[str],
        timestamp: datetime,
        interaction_id: int,
        untrusted: bool,
    ) -> None:
        """Fire-and-forget wrapper around backend.retain_turn.

        The Hindsight backend's retain_turn enqueues into a per-bank
        asyncio.Queue and returns immediately; sqlite_legacy is a noop. We
        still wrap in try/except so a backend hiccup never derails the user
        turn — alpha system, retain failures are logged + dropped.
        """
        try:
            await self.memory_backend.retain_turn(
                bank_id=persona_name,
                role=role,
                content=content,
                timestamp=timestamp,
                scope_tags=build_scope_tags(
                    channel=channel, server_id=server_id, user_identifier=user_identifier,
                ),
                source_persona=persona_name,
                untrusted=untrusted,
                metadata={"interaction_id": str(interaction_id)},
            )
        except Exception as e:  # noqa: BLE001
            logger.warning(f"retain_turn dropped ({role} turn): {e}")
