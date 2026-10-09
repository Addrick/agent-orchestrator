# src/turn_persistence.py
"""Turn persistence for the chat pipeline (DP-200 slice B).

The write-side tail of a turn, extracted from ChatSystem: logging the user
row (or archiving the prior assistant on retry), committing/updating the
assistant row, LTM retain (deferred until a turn leaves the history window,
DP-423), and the per-user API-request
caches that back `dump_history`. The orchestration kernel decides *when*
these happen; this module owns *how* and holds the cache state.
"""

import logging
import re
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple, cast

from config.global_config import MAX_CACHED_API_REQUESTS
from src.generation_events import ResponseType
from src.memory.backend.base import SESSION_GAP_SECONDS, MemoryBackend
from src.memory.memory_manager import MemoryManager
from src.memory.scope_tags import build_scope_tags
from src.persona import MemoryMode
from src.security.scrubber import get_scrubber
from src.utils.timeutil import to_local, to_utc

logger = logging.getLogger(__name__)


def format_retained_turn(role: str, speaker: str, content: str,
                         timestamp: datetime) -> str:
    """The text Hindsight extracts from for one turn (DP-402).

    Turns in a channel append into one document, so each carries its speaker.
    Only user turns carry a time, rendered in LOCAL_TZ: timestamps beside the
    model's own words trip it up, and the user stamp already dates the
    exchange.
    """
    if role == "user":
        local = to_local(timestamp)
        return f"[{local:%Y-%m-%d %H:%M}] {speaker}: {content}"
    # Reasoning is never retained (DP-252 contract). A model that leaves its
    # `<think>` inline in the reply bypasses `reasoning_content`, and the
    # extractor then files the model's self-talk as facts about the user (DP-409).
    return f"{speaker}: {_THINK_BLOCK.sub('', content).strip()}"


_THINK_BLOCK = re.compile(r"<think>.*?</think>\s*", re.DOTALL)


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
        # Continue/prefetch-style calls without a user message would
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
                #
                # DP-296: the *tool context* still has to land, or "retry a turn
                # → model proposes a write → operator never answers" leaves no
                # trace and the model re-proposes the action next turn. Attach
                # it to the archived row without touching its content, and
                # report that row so the park can clear it on resume.
                if tool_context_json:
                    try:
                        self.memory_manager.set_tool_context(
                            retry_assistant_id, tool_context_json,
                        )
                        return retry_assistant_id
                    except Exception as e:
                        logger.error(f"Retry park set_tool_context failed: {e}")
                return None
            if not final_text or not final_text.strip():
                # The turn died before producing prose (the error path commits
                # whatever accumulated, which is "" whenever the model emitted
                # tool calls and nothing else). Overwriting here would blank the
                # canonical row: empty content + NULL reasoning fails
                # `_is_renderable`, so the message *and* its version chevron
                # drop out of the transcript and the archived original becomes
                # unreachable through the UI. Keep the text, land the context.
                if tool_context_json:
                    try:
                        self.memory_manager.set_tool_context(
                            retry_assistant_id, tool_context_json,
                        )
                        return retry_assistant_id
                    except Exception as e:
                        logger.error(f"Retry set_tool_context failed: {e}")
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
            # DP-296: a turn that gated writes still has to leave a trace of the
            # actions it took and proposed, or an operator who never answers
            # makes the whole turn invisible to the model — it then re-proposes
            # or hallucinates the action on the next turn.
            #
            # The condition is `tool_context_json`, NOT a response_type: it used
            # to key off PENDING_CONFIRMATION, which DP-297 stopped producing
            # anywhere, so the rescue was dead and the max-iterations exit
            # (DEV_COMMAND + a sealed span holding every awaiting_human_approval
            # entry) returned None here. That dropped the row the parks are
            # patched through, and `_register_parks` then bound them to
            # `parked_assistant_id=None` — approving one executed a real write
            # with no record of it anywhere in history.
            if not tool_context_json:
                return None
            if response_type == ResponseType.PENDING_CONFIRMATION:
                # DP-130 keeps the *confirmation text* ephemeral (re-rendered
                # from the park), so only this type blanks its content. Every
                # other type reaching here — max_iterations especially — has
                # real text the user already saw and must keep it.
                final_text = ""

        try:
            assistant_id: Optional[int] = self.memory_manager.log_message(
                user_identifier=user_identifier, persona_name=persona_name,
                channel=channel, author_role='assistant',
                author_name=persona_name, content=final_text,
                timestamp=datetime.now(timezone.utc), server_id=server_id,
                tool_context=tool_context_json,
                reply_to_id=user_interaction_id,
            )
            return assistant_id
        except Exception as e:
            logger.error(f"Assistant log_message failed: {e}", exc_info=True)
            return None

    async def rebuild_session_safe(
        self,
        *,
        persona_name: str,
        channel: str,
        user_identifier: str,
        server_id: Optional[str],
        retried_id: int,
        retried_text: str,
        untrusted: bool,
    ) -> None:
        """Replace the session's Hindsight document with the canonical
        conversation after a retry (DP-409).

        Appending the regenerated reply would leave the discarded attempt's
        facts recallable. The DB row already holds the new version, so the
        transcript is re-rendered from it — except the retried turn itself,
        which takes `retried_text` so DP-335's footer stays out exactly as on
        the normal retain path. No open document → plain append, as before.
        """
        scope_tags = build_scope_tags(
            channel=channel, server_id=server_id, user_identifier=user_identifier,
        )
        try:
            since = await self.memory_backend.session_start(persona_name, scope_tags)
            turns = (
                self.memory_manager.get_session_turns(persona_name, channel, since)
                if since is not None else []
            )
            if turns:
                blocks = []
                for t in turns:
                    role = (t["author_role"] or "").lower()
                    content = retried_text if t["interaction_id"] == retried_id else t["content"]
                    speaker = t["author_name"] or t["user_identifier"] or "Unknown"
                    blocks.append(format_retained_turn(role, speaker, content, t["timestamp"]))
                if await self.memory_backend.replace_session(
                    persona_name, "\n\n".join(blocks),
                    scope_tags=scope_tags, source_persona=persona_name,
                    untrusted=untrusted, timestamp=datetime.now(timezone.utc),
                    metadata={"interaction_id": str(retried_id)},
                ):
                    return
        except Exception as e:  # noqa: BLE001
            logger.warning(f"Session rebuild failed, appending retried turn: {e}")
        await self.retain_turn_safe(
            persona_name=persona_name, role="assistant", speaker=persona_name,
            content=retried_text, user_identifier=user_identifier, channel=channel,
            server_id=server_id, timestamp=datetime.now(timezone.utc),
            interaction_id=retried_id, untrusted=untrusted,
        )

    def queue_retain_safe(
        self,
        *,
        interaction_id: int,
        persona_name: str,
        channel: str,
        untrusted: bool,
        retain_text: Optional[str] = None,
        source_content: Optional[str] = None,
    ) -> None:
        """Queue a turn for retain once it leaves the history window (DP-423).

        While a turn is in the window the model reads it verbatim, so
        retaining it too would make recall hand it back a second time. The
        queue is durable; `flush_evicted` and `flush_aged_out` drain it.
        `retain_text` (DP-335) replaces the row's content at flush time only
        while the row still holds `source_content`.
        """
        try:
            self.memory_manager.queue_retain(
                interaction_id, persona_name, channel, untrusted=untrusted,
                retain_text=retain_text, source_content=source_content,
            )
        except Exception as e:  # noqa: BLE001
            logger.warning(f"queue_retain dropped (interaction {interaction_id}): {e}")

    def is_retain_pending(self, interaction_id: int) -> bool:
        try:
            return self.memory_manager.is_retain_pending(interaction_id)
        except Exception as e:  # noqa: BLE001
            logger.warning(f"is_retain_pending failed (interaction {interaction_id}): {e}")
            return False

    async def flush_retains(self, **filters: Any) -> int:
        """Retain queued turns matching `filters` (see
        `MemoryManager.take_pending_retains`), oldest first; returns how many.

        Each turn keeps its own timestamp, so the doc-scope store cuts
        documents at the same idle gaps it would have cut them live.
        """
        try:
            rows = self.memory_manager.take_pending_retains(**filters)
        except Exception as e:  # noqa: BLE001
            logger.warning(f"Pending retain flush failed ({filters}): {e}")
            return 0
        sent = 0
        for r in rows:
            if r["skip"]:
                continue
            persona_name = r["persona_name"]
            role = (r["author_role"] or "").lower()
            speaker = (
                persona_name if role == "assistant"
                else r["author_name"] or r["user_identifier"] or "Unknown"
            )
            await self.retain_turn_safe(
                persona_name=persona_name, role=role, speaker=speaker,
                content=r["text"], user_identifier=r["user_identifier"],
                channel=r["channel"], server_id=r["server_id"],
                timestamp=r["timestamp"], interaction_id=r["interaction_id"],
                untrusted=r["untrusted"],
            )
            sent += 1
        return sent

    async def flush_aged_out(self, now: Optional[datetime] = None) -> int:
        """Retain every queued turn older than SESSION_GAP_SECONDS, in any
        scope — the age at which the history window drops it
        (`RequestBuilder.build_conversation_history`), and the gap that
        opens a new Hindsight document. Without this a conversation that
        simply stops never leaves the window and is never remembered.

        Lazy sweep, run on every turn and once at startup (the DP-319 pairing:
        the sweep reads the durable queue, the boot pass covers a process
        that comes up and sees no traffic).
        """
        now = now or datetime.now(timezone.utc)
        return await self.flush_retains(
            older_than=now - timedelta(seconds=SESSION_GAP_SECONDS),
        )

    async def flush_evicted(
        self,
        *,
        persona_name: str,
        memory_mode: MemoryMode,
        channel: str,
        user_identifier: str,
        server_id: Optional[str],
        oldest_window_id: Optional[int],
        current_id: Optional[int],
    ) -> int:
        """Retain the queued turns this request's history window pushed out.

        Interaction ids are insertion-ordered, so every turn the window
        covers that is older than its oldest turn is out of it. "Covers" is
        the memory mode's scope, mirroring `RequestBuilder.fetch_raw_history`
        — a narrower flush strands turns another channel evicted, a wider
        one retains turns still in someone else's window. An empty window
        (`ticket` mode, a window of 0, everything aged out) evicts
        everything before the current turn.
        """
        before_id = oldest_window_id if oldest_window_id is not None else current_id
        if before_id is None:
            return 0
        if memory_mode == MemoryMode.GLOBAL:
            scope: Dict[str, Any] = {}
        elif memory_mode == MemoryMode.PERSONAL:
            scope = {"user_identifier": user_identifier}
        elif memory_mode == MemoryMode.SERVER_WIDE:
            scope = {"server_id": server_id}
        elif memory_mode == MemoryMode.TICKET_ISOLATED:
            scope = {"channel": channel}
        else:
            # get_channel_history matches a falsy server_id as IS NULL.
            scope = {"channel": channel, "server_id": server_id or None}
        return await self.flush_retains(
            persona_name=persona_name, before_id=before_id, **scope,
        )

    async def retain_turn_safe(
        self,
        *,
        persona_name: str,
        role: str,
        speaker: str,
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

        `content` is the bare turn; the speaker/time header is added here so
        every caller retains the same shape (DP-402). The DB row and prompt
        history keep the bare text.
        """
        try:
            await self.memory_backend.retain_turn(
                bank_id=persona_name,
                role=role,
                content=format_retained_turn(role, speaker, content, timestamp),
                timestamp=to_utc(timestamp),
                scope_tags=build_scope_tags(
                    channel=channel, server_id=server_id, user_identifier=user_identifier,
                ),
                source_persona=persona_name,
                untrusted=untrusted,
                metadata={"interaction_id": str(interaction_id)},
            )
        except Exception as e:  # noqa: BLE001
            logger.warning(f"retain_turn dropped ({role} turn): {e}")
