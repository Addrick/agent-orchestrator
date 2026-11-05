# src/turn_persistence.py
"""Turn persistence for the chat pipeline (DP-200 slice B).

The write-side tail of a turn, extracted from ChatSystem: logging the user
row (or archiving the prior assistant on retry), committing/updating the
assistant row, LTM retain (deferred until a turn leaves the history window,
DP-423), and the per-user API-request
caches that back `dump_history`. The orchestration kernel decides *when*
these happen; this module owns *how* and holds the cache state.
"""

import asyncio
import json
import logging
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Set, Tuple, cast

from config.global_config import (
    HINDSIGHT_RETAIN_CHUNK_SIZE, MAX_CACHED_API_REQUESTS, RETAIN_CONTEXT_TAIL_CHARS,
)
from src.generation_events import ResponseType
from src.memory.backend.base import MemoryBackend, RetainRefused
from src.memory.memory_manager import MemoryManager
from src.memory.retain_blocks import block_document_id, scope_from_row, split_block
from src.memory.scope_tags import build_scope_tags
from src.security.scrubber import get_scrubber
from src.utils.timeutil import to_local, to_utc

# Turns read back for a block's leading `context`; RETAIN_CONTEXT_TAIL_CHARS
# then keeps the last of them that fit.
_CONTEXT_TURNS = 12

logger = logging.getLogger(__name__)


def format_retained_turn(role: str, speaker: str, content: str,
                         timestamp: datetime) -> str:
    """The text Hindsight extracts from for one turn (DP-402).

    A block holds many turns in one document (DP-495), so each carries its speaker.
    Only user turns carry a time, rendered in LOCAL_TZ: timestamps beside the
    model's own words trip it up, and the user stamp already dates the
    exchange.
    """
    if role == "user":
        local = to_local(timestamp)
        return f"[{local:%Y-%m-%d %H:%M}] {speaker}: {content}"
    # Reasoning is never retained (DP-252 contract): it lives in
    # `reasoning_content`, never in `content` (DP-434).
    return f"{speaker}: {content.strip()}"


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
        # DP-495 block sends and completion polls: serialized, and run as
        # tasks so a slow backend never holds up a turn.
        self._flush_lock = asyncio.Lock()
        self._flush_tasks: Set["asyncio.Task[None]"] = set()
        # A per-turn flush is waiting for the lock: another turn adds nothing.
        self._flush_waiting = False
        # The backend was unreachable on the last try. Turns stop sending;
        # the maintenance loop's pass (once a minute) is the probe.
        self._backend_down = False

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
            reasoning_content: Optional[str] = None,
    ) -> Optional[int]:
        """Persist the assistant turn (UPDATE on retry, INSERT otherwise).

        Returns the canonical assistant interaction_id, or None when the
        turn produced nothing (no text, reasoning or tool context) or the
        response_type isn't a normal LLM generation.

        `reasoning_content` is the turn's thinking, stored in its own column
        so history replay never sends it back to the model (DP-434).
        """
        has_output = bool((final_text and final_text.strip()) or reasoning_content)
        if not has_output and not tool_context_json:
            return None
        if response_type == ResponseType.REACTION:
            # A mark on the user's message, never a reply (DP-449). Checked
            # before the retry branch: a regenerate answered with a reaction
            # must not overwrite the archived reply with the emoji.
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
            # A retry that stopped or errored after its tool steps and before
            # any reply text (tool context, no prose) overwrites the row like
            # any other: keeping the old attempt's text under the new attempt's
            # steps would show the user, and send the model, an answer to calls
            # it never saw. The row stays visible on its tool cards alone
            # (DP-444 `_is_renderable`), so the archived original stays
            # reachable through its version chevron.
            try:
                # Forward tool_context so the regenerated row's stored tool calls
                # stay paired with its new content — a retried turn may use a
                # different (or no) set of tools than the failed attempt.
                # Always pass reasoning_content, even None: the prior attempt's
                # reasoning no longer matches the regenerated text (DP-141
                # sentinel contract — omitting would *preserve* it).
                self.memory_manager.update_interaction_content(
                    retry_assistant_id, final_text,
                    reasoning_content=reasoning_content, tool_context=tool_context_json,
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
                reasoning_content=reasoning_content,
            )
            return assistant_id
        except Exception as e:
            logger.error(f"Assistant log_message failed: {e}", exc_info=True)
            return None

    def upsert_keyed_row(
            self,
            *,
            platform_message_id: str,
            persona_name: str,
            user_identifier: str,
            channel: str,
            role: str,
            author_name: Optional[str],
            content: str,
            timestamp: datetime,
            reasoning_content: Optional[str] = None,
            tool_context_json: Optional[str] = None,
    ) -> Tuple[Optional[int], bool]:
        """A row copied from somewhere else, keyed by its unique
        `platform_message_id` (DP-484: a codr ticket's opencode session
        mirrored into its channel): inserted the first time, rewritten in
        place when its content changed since, left alone otherwise. Not
        retained: the source keeps its own record. Returns (row id, written)."""
        existing = self.memory_manager.get_interaction_by_platform_id(platform_message_id)
        if existing is None:
            row_id = self.memory_manager.log_message(
                user_identifier=user_identifier, persona_name=persona_name, channel=channel,
                author_role=role, author_name=author_name, content=content, timestamp=timestamp,
                platform_message_id=platform_message_id, tool_context=tool_context_json,
                reasoning_content=reasoning_content,
            )
            return row_id, True
        row_id = int(existing["interaction_id"])
        if (existing.get("content"), existing.get("reasoning_content"), existing.get("tool_context")) == (
                content, reasoning_content, tool_context_json):
            return row_id, False
        self.memory_manager.update_interaction_content(
            row_id, content, reasoning_content=reasoning_content, tool_context=tool_context_json,
        )
        return row_id, True

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
        """Queue a turn for retain with the block that cuts it (DP-423, DP-495).

        While a turn is in the prompt the model reads it verbatim, so
        retaining it too would make recall hand it back a second time. The
        queue is durable; a block cut collects it and `schedule_flush`
        sends the block. Re-queueing a turn whose block was already sent (a
        retried reply) rebuilds that block's document (DP-409).
        `retain_text` (DP-335) replaces the row's content at send time only
        while the row still holds `source_content`.
        """
        try:
            self.memory_manager.queue_retain(
                interaction_id, persona_name, channel, untrusted=untrusted,
                retain_text=retain_text, source_content=source_content,
            )
        except Exception as e:  # noqa: BLE001
            logger.warning(f"queue_retain dropped (interaction {interaction_id}): {e}")

    def discard_overwritten_retry(self, row_id: Optional[int],
                                  retry_assistant_id: Optional[int],
                                  written_text: str) -> None:
        """Unqueue a retried reply whose row this attempt overwrote with text
        that is never retained (errored, cancelled, footer-only, not an LLM
        reply). A fresh turn like that is never queued; left queued, the send
        would retain the new text under the discarded attempt's trust bit. A
        block already sent with it is rebuilt without it.
        No-op unless `row_id` is the retried row and `written_text` (what
        `commit_or_update_assistant` wrote over it) is non-empty."""
        if row_id is None or row_id != retry_assistant_id or not written_text.strip():
            return
        try:
            self.memory_manager.discard_pending_retain(row_id)
        except Exception as e:  # noqa: BLE001
            logger.warning(f"discard_pending_retain failed (interaction {row_id}): {e}")

    def schedule_flush(self) -> None:
        """Send queued blocks and poll sent ones, in the background (DP-495).

        Background because delivery is confirmed: a down Hindsight would
        otherwise stall the user's turn for the HTTP timeout. Runs after every
        turn, so a block cut by this request goes out at once; ChatSystem's
        maintenance loop runs the same pass once a minute. At most one waits
        behind the running pass, and none is started while the backend is
        down — the maintenance pass probes it.
        """
        if self._backend_down or self._flush_waiting:
            return
        self._flush_waiting = True

        async def run() -> None:
            async with self._flush_lock:
                self._flush_waiting = False
                await self._flush_locked()
                await self._poll_locked()

        task = asyncio.get_running_loop().create_task(run(), name="dp495-retain-flush")
        self._flush_tasks.add(task)
        task.add_done_callback(self._flush_tasks.discard)

    async def drain_flushes(self) -> None:
        """Await every scheduled flush (tests). Shutdown does not drain:
        the queue is durable, so an interrupted send is redone later."""
        while self._flush_tasks:
            await asyncio.gather(*list(self._flush_tasks), return_exceptions=True)

    async def flush_blocks(self) -> int:
        """Send every queued block that is due, oldest first; returns how many
        the backend accepted. Serialized by `_flush_lock`. A block leaves the
        queue only once the backend accepted it; none is ever given up on."""
        async with self._flush_lock:
            return await self._flush_locked()

    def _set_backend_down(self, down: bool, why: str = "") -> None:
        if down and not self._backend_down:
            logger.warning(f"Hindsight unreachable ({why}); retain sends pause, probing once a minute")
        elif not down and self._backend_down:
            logger.info("Hindsight reachable again; retain sends resume")
        self._backend_down = down

    async def _flush_locked(self) -> int:
        """`flush_blocks` under the lock. Stops at the first block the
        backend could not take (it is down: the rest would only wait out the
        same timeout). A block it refused backs off alone."""
        now = datetime.now(timezone.utc)
        try:
            blocks = self.memory_manager.retain_blocks(statuses=("queued",), due_at=now)
        except Exception as e:  # noqa: BLE001
            logger.warning(f"Retain block flush failed: {e}")
            return 0
        sent = 0
        for block in blocks:
            try:
                ok = await self._deliver_block(block)
            except RetainRefused as e:
                self._set_backend_down(False)
                delay = self.memory_manager.mark_retain_block_refused(block, now=now)
                logger.warning(
                    f"Retain block {block['block_id']} ({block['persona_name']}:"
                    f"{block['scope_key']}) refused by Hindsight, retrying in "
                    f"{delay / 60:.0f} min: {e}"
                )
                continue
            except Exception as e:  # noqa: BLE001
                logger.warning(f"Retain block {block['block_id']} not sent: {e}")
                continue
            if not ok:
                self._set_backend_down(True, f"block {block['block_id']} not accepted")
                break
            self._set_backend_down(False)
            sent += 1
        return sent

    def _render(self, persona_name: str, rows: List[Dict[str, Any]],
                text_key: str) -> List[str]:
        out = []
        for r in rows:
            role = (r["author_role"] or "").lower()
            speaker = (
                persona_name if role == "assistant"
                else r["author_name"] or r["user_identifier"] or "Unknown"
            )
            out.append(format_retained_turn(role, speaker, r[text_key] or "", r["timestamp"]))
        return out

    async def _deliver_block(self, block: Dict[str, Any]) -> bool:
        """Send one block as its Hindsight document(s), `replace` mode.

        The turns are read as they stand now (edits and deletes honoured),
        each with its DP-402 header, concatenated in order. A block larger
        than the bank's chunk size goes as consecutive parts; every part (and
        a whole block) carries the turns before it as labelled background.
        Documents a rebuild no longer produces are deleted.
        """
        persona = block["persona_name"]
        scope = scope_from_row(block)
        members = self.memory_manager.retain_block_members(int(block["block_id"]))
        sendable = [m for m in members if not m["skip"]]
        old_ids: List[str] = json.loads(block["document_ids"] or "[]")

        new_ids: List[str] = []
        ops: List[str] = []
        first_ts: Optional[datetime] = None
        untrusted = bool(block["untrusted"]) or any(m["untrusted"] for m in sendable)
        if sendable:
            stored = self.memory_manager.parse_stored_ts(block["first_ts"]) \
                if block["first_ts"] else None
            first_ts = stored or sendable[0]["timestamp"]
            turns = self._render(persona, sendable, "text")
            prior = self._render(
                persona,
                self.memory_manager.turns_before(
                    scope, sendable[0]["interaction_id"], _CONTEXT_TURNS,
                ),
                "content",
            )
            parts = split_block(
                turns, HINDSIGHT_RETAIN_CHUNK_SIZE, prior=prior,
                tail_chars=RETAIN_CONTEXT_TAIL_CHARS,
            )
            seq = int(block["seq"] or 0)
            tags = self._block_tags(persona, sendable)
            items = []
            for n, part in enumerate(parts):
                doc_id = block_document_id(
                    persona, block["scope_key"], seq, first_ts,
                    part=n + 1 if len(parts) > 1 else None,
                )
                new_ids.append(doc_id)
                items.append({
                    "document_id": doc_id,
                    "content": part["content"],
                    "context": part["context"],
                    "tags": tags,
                    "untrusted": untrusted,
                    "timestamp": to_utc(sendable[part["start"]]["timestamp"]),
                    "metadata": {
                        "block": f"b{seq:04d}",
                        "interaction_ids": f"{sendable[0]['interaction_id']}-"
                                           f"{sendable[-1]['interaction_id']}",
                    },
                })
            accepted = await self.memory_backend.retain_block(
                persona, items, request_id=str(block["request_id"]),
            )
            if accepted is None:
                logger.info(
                    f"Retain block {block['block_id']} ({persona}:{block['scope_key']}) "
                    "not accepted (Hindsight unreachable); it stays queued and in the prompt"
                )
                return False
            ops = accepted
        for stale in (d for d in old_ids if d not in new_ids):
            try:
                await self.memory_backend.delete_document(persona, stale)
            except Exception as e:  # noqa: BLE001
                logger.warning(f"Could not delete superseded block document {stale}: {e}")
        self.memory_manager.mark_retain_block_sent(
            block, members=members, op_ids=ops, document_ids=new_ids,
            first_ts=first_ts, untrusted=untrusted,
        )
        return True

    @staticmethod
    def _block_tags(persona: str, rows: List[Dict[str, Any]]) -> List[str]:
        """Scope tags of every turn in the block (a block can span users,
        channels or servers, depending on the memory mode), so recall scoped
        to any of them still finds it."""
        tags: List[str] = []
        for r in rows:
            for t in build_scope_tags(channel=r["channel"], server_id=r["server_id"],
                                      user_identifier=r["user_identifier"]):
                if t not in tags:
                    tags.append(t)
        return tags + [f"persona:{persona}"]

    async def poll_blocks(self) -> int:
        """Ask the backend how each sent block's extraction stands; a
        completed block is marked done and leaves the prompt on the next
        request. Returns how many completed. Stops at the first answer
        that says the backend is unreachable."""
        async with self._flush_lock:
            return await self._poll_locked()

    async def _poll_locked(self) -> int:
        try:
            blocks = self.memory_manager.retain_blocks(statuses=("sent",))
        except Exception as e:  # noqa: BLE001
            logger.warning(f"Retain block poll failed: {e}")
            return 0
        done = 0
        for block in blocks:
            block_id = int(block["block_id"])
            try:
                status = await self.memory_backend.retain_status(
                    block["persona_name"], json.loads(block["op_ids"] or "[]"),
                    json.loads(block["document_ids"] or "[]"),
                )
            except Exception as e:  # noqa: BLE001
                logger.info(f"Retain block {block_id} poll failed: {e}")
                continue
            if status == "unreachable":
                self._set_backend_down(True, f"poll of block {block_id}")
                break
            self._set_backend_down(False)
            if status == "completed":
                self.memory_manager.set_retain_block_status(block_id, "done", expect="sent")
                done += 1
            elif status == "failed":
                # Hindsight gave up (its own retries exhausted) or an
                # operator cancelled it. Leave the prompt: holding it
                # forever would only push the prompt to the fallback cut.
                logger.warning(
                    f"Retain block {block_id} ({block['persona_name']}:{block['scope_key']}) "
                    "failed or was cancelled in Hindsight; it leaves the prompt unretained"
                )
                self.memory_manager.set_retain_block_status(block_id, "failed", expect="sent")
            elif status == "lost":
                logger.warning(f"Retain block {block_id} unknown to Hindsight; sending it again")
                self.memory_manager.requeue_retain_block(block_id)
        return done
