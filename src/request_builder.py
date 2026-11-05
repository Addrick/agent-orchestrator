# src/request_builder.py
"""Request assembly for the chat pipeline (DP-200 slice B).

Everything between "a message arrived" and "the wire payload the engine
sends" lives here: history retrieval + formatting per memory mode, long-term
memory recall + <memory> block injection, persona tool filtering, generation
param resolution, the context-token budget, and the `/assemble` dry-run
projection. ChatSystem's kernel calls through thin delegates so live submits
and the S5 parity inspector share this single code path by construction.
"""

import logging
import re
from collections import OrderedDict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, Tuple

from config.global_config import (
    MEMORY_RETRIEVAL_ENABLED, RETAIN_TRIM_HIGH, RETAIN_TRIM_LOW,
)
from src.embedding_service import EmbeddingService
from src.generation_params import GenerationParams
from src.memory.backend.base import MemoryBackend, MemoryHit
from src.memory.context_budget import (
    drop_orphaned_tool_head, message_tokens, plan_block_cut, truncate_messages_to_budget,
)
from src.memory.memory_manager import MemoryManager
from src.memory.retain_blocks import RetainScope, legacy_window, retain_scope
from src.memory.scope_tags import recall_scope_tags
from src.persona import Persona, MemoryMode
from src.tools.definitions import MODEL_INCOMPATIBLE_TOOLS
from src.tools.tool_loop import build_wire_messages, replayable_tool_context
from src.tools.tool_manager import ToolManager
from src.tools.turn_context import TurnContext, turn_scope
from src.utils.message_utils import strip_vertex_links
from src.utils.model_utils import get_model_prefix
from src.utils.timeutil import prompt_stamp, prompt_time

logger = logging.getLogger(__name__)

# Upper bound on the sticky per-conversation taint map. The taint bit is keyed
# by (user, persona, channel, server); a long-running deployment serving many
# distinct conversations would otherwise grow this map without bound. We evict
# the least-recently-touched entry once over capacity (taint is a soft cache —
# a re-derived False on a cold key is safe, and recall re-taints if needed).
MAX_CONVERSATION_TAINTS = 10000


_PROMPT_STAMP = re.compile(r"^\[[A-Z][a-z]{2} \d{4}-\d{2}-\d{2} \d{2}:\d{2}\] ")


def _stamped(content: str, ts: Any) -> str:
    """`content` behind its turn's `prompt_stamp` (DP-431). A row with no
    readable stored time passes through bare rather than being stamped "now",
    which would change on every request and break the cached prefix."""
    if ts is None:
        return content
    try:
        return f"{prompt_stamp(ts)} {content}"
    except (TypeError, ValueError):
        return content


@dataclass
class RequestContext:
    """Bundles resolved pipeline state flowing through generate_response phases."""
    persona: Persona
    persona_name: str
    user_identifier: str
    channel: str
    message: str
    server_id: Optional[str] = None
    image_url: Optional[str] = None
    history_limit: Optional[int] = None
    user_display_name: Optional[str] = None
    # When the new user turn was sent; stored with the row and used for its
    # prompt stamp so the two agree (DP-431). None = now.
    timestamp: Optional[datetime] = None
    # Populated during prepare_request
    conversation_history: List[Dict[str, Any]] = field(default_factory=list)
    tools_for_llm: List[Dict[str, Any]] = field(default_factory=list)
    oldest_interaction_id: Optional[int] = None
    local_inference_config: Optional[Dict[str, Any]] = None
    turn_tainted: bool = False
    taint_sources: List[str] = field(default_factory=list)
    # Optional: OAI-format messages from the client (non-streaming OAI callers).
    # Used as a fallback when the DB returns no history for this channel.
    client_messages: Optional[List[Dict[str, Any]]] = None
    # DP-447 continue: the trailing assistant message the model resumes from
    # (`render_continue_prefix`), and the stored reply of the row it extends.
    continue_prefix: Optional[str] = None
    continue_row_content: str = ""
    # DP-495: the history window this request's prompt was built from.
    history_window: Optional["HistoryWindow"] = None


@dataclass
class HistoryWindow:
    """The prompt's history and where it came from (DP-495).

    `rows` are the DB rows shown, in prompt order; `messages` their formatted
    messages, `sources[i]` the interaction id `messages[i]` came from. Rows at
    or above `start_id` are the live window (cuttable); rows from `floor_id`
    up to it belong to cut blocks Hindsight has not finished with yet.
    `count_cap` is the message-count ceiling (0: no history at all);
    `over_cap` says the scope holds more rows than the cap shows.
    """
    scope: RetainScope
    start_id: int
    floor_id: int
    count_cap: int
    over_cap: bool = False
    rows: List[Dict[str, Any]] = field(default_factory=list)
    messages: List[Dict[str, Any]] = field(default_factory=list)
    sources: List[Optional[int]] = field(default_factory=list)

    @property
    def oldest_id(self) -> Optional[int]:
        """The smallest interaction id shown, not rows[0]'s: the window is
        ordered by timestamp, ids by insertion, and they disagree when a
        platform timestamp (Discord `created_at`) lands out of insertion order."""
        ids = [r['interaction_id'] for r in self.rows if r.get('interaction_id') is not None]
        return min(ids) if ids else None


@dataclass
class ContinueTurn:
    """A request to extend the trailing assistant reply in place (DP-447).

    `think_open` overrides what the row implies about its thinking — needed
    only for closed thinking with no reply yet, which the stored columns
    cannot tell apart from an open block. None = decide from the row.

    `row_id` is the reply the caller means to extend. The row continue writes
    to is this session's own latest (retry's lookup), but a GLOBAL/SERVER
    persona's view — and its prompt — can end on another channel's row; a
    mismatch is refused instead of extending a row the screen does not end
    on. None = no check (non-portal callers).
    """
    think_open: Optional[bool] = None
    row_id: Optional[int] = None


@dataclass
class AssembledRequest:
    """Pure dry-run projection of the request the engine would send.

    Produced by `RequestBuilder.assemble_request` via the *same* helpers the
    live `_orchestrate` path uses — `prepare_request` (history + LTM +
    truncation), `resolve_generation_params`, and `build_wire_messages` — so
    the `/assemble` inspector cannot drift from a live submit. No inference,
    DB writes, or turn logging happen during assembly.

    `messages` are the raw wire dicts (system + history + new user turn, with
    any replayed tool_calls/tool rows); `sources[i]` is a provenance tag for
    `messages[i]` (`persona.prompt`, `ltm_block`, `history`, `composer`,
    `tool_call`, `tool_result`).
    """
    persona_name: str
    model_name: str
    route: str
    params: GenerationParams
    messages: List[Dict[str, Any]]
    sources: List[str]
    oldest_interaction_id: Optional[int] = None


class RequestBuilder:
    """Builds the per-turn request state the orchestration kernel submits.

    Owns the sticky per-conversation taint map (read during prepare, written
    back by the kernel after the tool loop) — the taint a request *starts*
    with is request-assembly state.
    """

    def __init__(
        self,
        memory_manager: MemoryManager,
        memory_backend: MemoryBackend,
        tool_manager_lookup: Callable[[], ToolManager],
        persona_lookup: Callable[[str], Optional[Persona]],
        embedding_service: Optional[EmbeddingService] = None,
    ) -> None:
        self.memory_manager = memory_manager
        self.memory_backend = memory_backend
        # Lookup closure (mirrors ConfirmationManager): ToolLoop reads
        # chat_system.tool_manager per call, so a post-init swap must be
        # visible to tool filtering too or the request would offer the model
        # tools from the stale manager.
        self._tool_manager_lookup = tool_manager_lookup
        # Callable (not a dict reference) so callers that rebind their persona
        # map post-construction (tests, live persona reloads) stay visible.
        self.persona_lookup = persona_lookup
        self.embedding_service = embedding_service
        # OrderedDict (not defaultdict) so eviction order is well-defined and
        # the map stays bounded — see set_conversation_taint.
        self.conversation_taints: "OrderedDict[Tuple[str, str, str, Optional[str]], bool]" = OrderedDict()

    def set_conversation_taint(
            self, key: Tuple[str, str, str, Optional[str]], value: bool
    ) -> None:
        """Set the sticky taint bit for a conversation, bounding the map.

        The newest key moves to the MRU end; once over MAX_CONVERSATION_TAINTS
        the least-recently-touched entry is evicted. Taint is a soft cache — a
        cold key re-derives as False and memory recall re-taints if warranted —
        so eviction is safe and keeps the map from growing without bound.
        """
        if key in self.conversation_taints:
            self.conversation_taints.move_to_end(key)
        self.conversation_taints[key] = value
        while len(self.conversation_taints) > MAX_CONVERSATION_TAINTS:
            self.conversation_taints.popitem(last=False)

    def format_raw_history_for_llm(self, raw_history: List[Dict[str, Any]], memory_mode: str,
                                   persona_name: str, server_id: Optional[str],
                                   stamp: bool = False,
                                   sources: Optional[List[Optional[int]]] = None,
                                   ) -> List[Dict[str, Any]]:
        """Formats database history records into a list of messages for the LLM.

        ``stamp`` (the persona's ``inject_timestamp``) prefixes every user-role
        message with its stored time (DP-431). The persona's own assistant
        turns are never stamped: a stamp beside the model's own words gets
        imitated in its replies.

        ``sources``, when given, receives the interaction id each returned
        message came from (DP-495: per-row token costs for the block cut).
        """
        final_history: List[Dict[str, Any]] = []
        is_group_chat = memory_mode in ("server", "global", "ticket") or (
                    memory_mode == "channel" and server_id is not None)

        has_user_turn = False
        for msg in raw_history:
            emitted = len(final_history)
            has_user_turn = self._format_row(
                msg, final_history, persona_name, is_group_chat, stamp, has_user_turn
            )
            if sources is not None:
                sources.extend([msg.get('interaction_id')] * (len(final_history) - emitted))
        return final_history

    @staticmethod
    def _format_row(msg: Dict[str, Any], final_history: List[Dict[str, Any]],
                    persona_name: str, is_group_chat: bool, stamp: bool,
                    has_user_turn: bool = False) -> bool:
        """Append one history row's messages to `final_history`.

        Returns True if `final_history` contains at least one user-role message
        (either from a prior row or emitted by this row).
        """
        author_role = msg.get('author_role')
        author_name = msg.get('author_name')
        content = msg.get('content', '')
        # Noise reduction: Strip vertexai grounding redirect URLs from history before sending to LLM.
        content_clean = strip_vertex_links(content) if content else content

        if author_role == 'user':
            if is_group_chat and author_name:
                formatted_content = f"{author_name}: {content_clean}"
            else:
                formatted_content = content_clean
            if stamp:
                formatted_content = _stamped(formatted_content, msg.get('timestamp'))
            final_history.append({'role': 'user', 'content': formatted_content})
            has_user_turn = True
        elif author_role == 'assistant':
            if author_name == persona_name:
                tool_context_json = msg.get('tool_context')
                if tool_context_json:
                    # DP-296 / DP-474: A replayed tool block must follow a user
                    # turn in the history window. The history limit slices raw
                    # rows, so an assistant row carrying tool_context can end
                    # up oldest — replaying it then opens the wire array with
                    # a function call (or an assistant text turn followed by a
                    # function call, which build_google_history merges into a
                    # single model Content turn at index 0) and Gemini rejects
                    # the request outright ("function call turn must come
                    # immediately after a user turn or after a function response
                    # turn").
                    #
                    # Conversely, once at least one user turn precedes it in
                    # the window, consecutive assistant rows from DP-297
                    # non-blocking parks / approval-resume continuations
                    # (assistant(text) followed by assistant(tool_calls)) are
                    # legal: OpenAI accepts consecutive assistant messages, and
                    # build_google_history merges them into a single model turn
                    # following the user (or tool) turn. Drop tool_context only
                    # when no preceding user turn survived the limit.
                    if has_user_turn:
                        final_history.extend(replayable_tool_context(tool_context_json))
                    else:
                        logger.debug(
                            "Dropping orphaned tool_context for %s: no "
                            "preceding user turn survived the history limit.",
                            persona_name,
                        )
                if content_clean:
                    final_history.append({'role': 'assistant', 'content': content_clean})
            elif content_clean:
                # In a group chat, messages from other personas are treated as user messages.
                # A reply that was only reasoning has nothing to show (DP-434).
                formatted_content = f"{author_name}: {content_clean}"
                if stamp:
                    formatted_content = _stamped(formatted_content, msg.get('timestamp'))
                final_history.append({'role': 'user', 'content': formatted_content})
                has_user_turn = True
        return has_user_turn

    def fetch_raw_history(
            self,
            mode: MemoryMode,
            persona_name: str,
            user_identifier: str,
            channel: str,
            server_id: Optional[str],
            effective_limit: int,
            min_id: Optional[int] = None,
    ) -> Tuple[List[Dict[str, Any]], str]:
        """Dispatches history retrieval based on memory mode. Returns (raw_history, mode_label).

        `min_id` is the DP-495 window floor: rows below it were cut into a
        block Hindsight has finished with. None = no floor (transcript view).
        The scope each branch covers is `retain_blocks.retain_scope`'s.
        """
        mm = self.memory_manager
        if mode == MemoryMode.TICKET_ISOLATED:
            return [], "ticket"
        elif mode == MemoryMode.SERVER_WIDE:
            # SERVER_WIDE returns history for the specific server.
            # If server_id is None (Web UI/local), we still query, as the DB
            # stores these rows with server_id=NULL. MemoryManager handles
            # the NULL check via IS NULL in its queries.
            return mm.get_server_history(server_id, persona_name, effective_limit, min_id=min_id), "server"
        elif mode == MemoryMode.PERSONAL:
            return mm.get_personal_history(user_identifier, persona_name, effective_limit, min_id=min_id), "personal"
        elif mode == MemoryMode.GLOBAL:
            return mm.get_global_history(persona_name, effective_limit, min_id=min_id), "global"
        else:
            return mm.get_channel_history(channel, persona_name, server_id, effective_limit, min_id=min_id), "channel"

    def get_view_history(
            self,
            persona_name: str,
            user_identifier: str,
            channel: Optional[str],
            server_id: Optional[str] = None,
            limit: Optional[int] = None,
    ) -> Tuple[List[Dict[str, Any]], str]:
        """Raw history the way the engine would see it, for the transcript view.

        DP-136 / handoff §10. The bespoke portal's transcript must mirror what
        the engine would actually feed the model for a given (persona, channel).
        That depends on the persona's `memory_mode`:
          - GLOBAL  → all channels merge (the `channel` arg is irrelevant)
          - CHANNEL → only the supplied channel's rows
          - PERSONAL/SERVER/TICKET → scoped by user/server/ticket respectively
        So we dispatch through the SAME `fetch_raw_history` the live path uses,
        keyed on the persona's mode — guaranteeing the rendered transcript and a
        live submit see the same isolation behavior. When `channel` is None we
        preserve the legacy behavior (global history regardless of mode) so
        existing single-channel callers are unchanged.

        Returns (raw_history, memory_mode_label). Rows are NOT formatted for the
        LLM (the transcript projection wants the raw columns).
        """
        persona = self.persona_lookup(persona_name)
        if persona is None:
            return [], "global"
        if channel is None:
            return self.memory_manager.get_global_history(persona_name, limit), "global"
        # DP-142: transcript view is read-only — peek the window without
        # advancing the hello override.
        effective_limit = persona.get_history_messages(advance=False)
        if limit is not None:
            effective_limit = min(effective_limit, limit)
        return self.fetch_raw_history(
            persona.get_memory_mode(), persona_name,
            user_identifier, channel, server_id, effective_limit,
        )

    def build_conversation_history(
            self,
            persona: Persona,
            user_identifier: str,
            channel: str,
            server_id: Optional[str],
            history_limit: Optional[int],
            advance: bool = True,
    ) -> Tuple[List[Dict[str, Any]], Optional[int]]:
        """Retrieves and formats conversation history based on the persona's memory mode.

        Returns (formatted_history, oldest_interaction_id).
        oldest_interaction_id is the smallest interaction_id in the window,
        used for the memory recency filter.

        DP-142: ``advance`` gates the hello-override side effect. Live turns pass
        True; read-only / dry-run callers pass False. It also gates storing a
        first-seen scope's window start (DP-495): a dry run writes nothing.
        """
        window = self.load_history_window(
            persona, user_identifier, channel, server_id, history_limit, advance=advance,
        )
        return window.messages, window.oldest_id

    def load_history_window(
            self,
            persona: Persona,
            user_identifier: str,
            channel: str,
            server_id: Optional[str],
            history_limit: Optional[int],
            advance: bool = True,
            scope: Optional[RetainScope] = None,
    ) -> HistoryWindow:
        """The prompt's history window for a request (DP-495).

        The window starts at the scope's pinned start (`Retain_Window`), or
        lower while a cut block is still waiting for Hindsight, so the prompt
        prefix stays the same from one request to the next until a block is
        cut or finishes. `context` / `history_limit` cap it by message count
        on top. `scope` overrides the request's own (the idle sweep).
        """
        persona_name = persona.get_name()
        effective_limit: int = persona.get_history_messages(advance=advance)
        if history_limit is not None:
            effective_limit = min(effective_limit, history_limit)
        if scope is None:
            scope = retain_scope(
                persona.get_memory_mode(), persona_name=persona_name, channel=channel,
                user_identifier=user_identifier, server_id=server_id,
                legacy=legacy_window(persona.get_base_history_messages()),
            )
        start_id, floor_id = self.memory_manager.retain_window(scope, persist=advance)
        cap = max(effective_limit, 0) if scope.has_window else 0
        window = HistoryWindow(scope=scope, start_id=start_id, floor_id=floor_id, count_cap=cap)
        if cap == 0:
            return window
        # One row past the cap tells whether the cap is cutting anything off.
        raw_history, memory_mode_used = self.fetch_raw_history(
            MemoryMode[scope.mode], persona_name,
            scope.user_identifier or user_identifier, scope.channel or channel,
            scope.server_id if scope.filter_server else server_id,
            cap + 1, min_id=floor_id,
        )
        if len(raw_history) > cap:
            # Over the cap only when what it hides is unsent live history; a
            # block already cut and in flight may fall off the cap freely.
            window.over_cap = any(
                (r.get('interaction_id') or 0) >= start_id for r in raw_history[:-cap]
            )
            raw_history = raw_history[-cap:]
        window.rows = raw_history
        window.messages = self.format_raw_history_for_llm(
            raw_history, memory_mode_used, persona_name,
            scope.server_id if scope.filter_server else server_id,
            stamp=persona.get_inject_timestamp(), sources=window.sources,
        )
        return window

    @staticmethod
    def prompt_budget(persona: Persona,
                      local_inference_config: Optional[Dict[str, Any]] = None) -> int:
        """Tokens the prompt may use: the context minus the reply's reserve.
        A portal's per-request `max_context_length` overrides the persona's."""
        ctx_limit = (local_inference_config or {}).get("max_context_length") \
            or persona.get_max_context_tokens()
        return int(ctx_limit) - persona.get_response_token_limit()

    def plan_window_cut(self, persona: Persona, window: HistoryWindow, *, budget: int,
                        fixed_tokens: int, trigger: bool) -> Optional[int]:
        """The window start after a block cut (DP-495), or None for no cut.

        One function for both triggers: a new user message (`trigger`, the
        90 % watermark) and the 48 h idle trim (no watermark, cut to 50 % if
        over it). A count cap that is cutting rows off (`over_cap`) cuts the
        window to half the cap, so a binding cap trims in blocks too and no
        row slides out of the prompt unsent. Rows are cut whole — a row's tool
        calls and results are one row, so a cut never splits them — and the
        window left behind starts on a turn (a user message).
        """
        if window.count_cap == 0:
            # No history shown (`ticket` mode, `context 0`): every turn is its
            # own block, cut on the next one.
            if not trigger:
                return None
            top = self.memory_manager.scope_max_id(window.scope)
            return top + 1 if top is not None and top >= window.start_id else None
        tokens: Dict[int, int] = {}
        first_role: Dict[int, str] = {}
        for msg, src in zip(window.messages, window.sources):
            if src is None:
                continue
            tokens[src] = tokens.get(src, 0) + message_tokens(msg)
            first_role.setdefault(src, msg.get("role") or "")
        live = sorted(
            (r for r in window.rows
             if r.get("interaction_id") is not None and r["interaction_id"] >= window.start_id),
            key=lambda r: r["interaction_id"],
        )
        rows = [(first_role.get(r["interaction_id"]) == "user", tokens.get(r["interaction_id"], 0))
                for r in live]
        k = plan_block_cut(
            rows, budget=budget, fixed_tokens=fixed_tokens,
            keep_at_most=window.count_cap // 2 if window.over_cap else None,
            high=RETAIN_TRIM_HIGH, low=RETAIN_TRIM_LOW,
            trigger=trigger,
        )
        if k is None:
            return None
        return int(live[k]["interaction_id"]) if k < len(live) \
            else int(live[-1]["interaction_id"]) + 1

    def cut_window(self, window: HistoryWindow, new_start: Optional[int]) -> Optional[int]:
        """Persist a planned cut; returns the new block's id (None: no cut,
        or a concurrent request cut first)."""
        if new_start is None:
            return None
        block_id = self.memory_manager.cut_retain_block(
            window.scope, expected_start=window.start_id, new_start=new_start,
        )
        if block_id is not None:
            logger.info(
                "Block cut: %s:%s history below %s → block %s",
                window.scope.persona_name, window.scope.key, new_start, block_id,
            )
        return block_id

    def _cut_on_new_message(self, ctx: RequestContext, budget: int, window_tokens: int) -> None:
        """The request-time block cut, for a fresh user message only — not a
        continue, nor a client that supplies its own history."""
        if ctx.history_window is None or ctx.continue_prefix is not None \
                or ctx.client_messages or not (ctx.message and ctx.message.strip()):
            return
        total = sum(message_tokens(m) for m in ctx.conversation_history)
        try:
            self.cut_window(ctx.history_window, self.plan_window_cut(
                ctx.persona, ctx.history_window, budget=budget,
                fixed_tokens=total - window_tokens, trigger=True,
            ))
        except Exception as e:  # noqa: BLE001 — a failed cut never fails the turn
            logger.warning(f"Block cut failed for {ctx.persona_name}: {e}")

    def idle_trim(self, persona: Persona, scope: RetainScope) -> Optional[int]:
        """The 48 h idle trim (DP-495): cut the scope's quiet conversation as
        close to 50 % of the prompt budget as a turn boundary allows. Every
        row is from before the silence. Under 50 %: no cut."""
        window = self.load_history_window(
            persona, scope.user_identifier or "", scope.channel or "",
            scope.server_id, None, advance=False, scope=scope,
        )
        new_start = self.plan_window_cut(
            persona, window, budget=self.prompt_budget(persona), fixed_tokens=0, trigger=False,
        )
        return self.cut_window(window, new_start)

    async def retrieve_memory_block(
            self,
            persona: Persona,
            user_identifier: str,
            channel: str,
            server_id: Optional[str],
            conversation_history: List[Dict[str, Any]],
            current_message: Optional[str] = None,
            oldest_interaction_id: Optional[int] = None,
    ) -> Tuple[Optional[str], bool]:
        """Retrieve and format relevant long-term memory summaries for injection.

        Returns (formatted_block, has_untrusted) where has_untrusted is True
        when any retrieved summary carried the untrusted flag (Phase 5 taint).
        """
        logger.warning(f"### RETRIEVAL_DIAGNOSTIC: Entering retrieve_memory_block for {persona.get_name()} (Enabled: {MEMORY_RETRIEVAL_ENABLED}, Service: {'YES' if self.embedding_service else 'NO'})")

        if not MEMORY_RETRIEVAL_ENABLED or not persona.get_long_term_memory():
            return None, False

        # Build the recall query string: prefer the current user message;
        # fall back to the most recent user turn in the formatted history.
        query_text: Optional[str] = None
        if current_message and current_message.strip():
            query_text = strip_vertex_links(current_message)
        if not query_text and conversation_history:
            for msg in reversed(conversation_history):
                if msg.get('role') == 'user':
                    content = msg.get('content', '')
                    if isinstance(content, str) and content.strip():
                        # Drop the DP-431 stamp: recall reads dates in a
                        # query as a time filter.
                        query_text = strip_vertex_links(_PROMPT_STAMP.sub('', content, count=1))
                        break
        if not query_text:
            logger.warning(f"### RequestBuilder: Skipping retrieval for {persona.get_name()} (no text content to embed)")
            return None, False

        # DP-253: scope the recall by the persona's isolation mode, mirroring
        # `fetch_raw_history`. Previously this always emitted a `channel:` tag,
        # which pinned recall to the caller's channel even for GLOBAL personas —
        # so a GLOBAL persona queried from the portal (channel=web_ui) never saw
        # memory ingested under any other channel. `recall_scope_tags` drops the
        # channel tag for non-channel modes and returns the explicit mode label.
        tag_filter, mode_label = recall_scope_tags(
            persona.get_memory_mode(),
            channel=channel, server_id=server_id, user_identifier=user_identifier,
        )
        # Carry the sliding-window cutoff through the backend boundary as a
        # tag predicate. SqliteSemanticBackend.recall translates this back into
        # `exclude_after_interaction_id` so the legacy summary index doesn't
        # surface memories that are still in the visible window. Hindsight
        # ignores unknown tag prefixes — no-op there.
        if oldest_interaction_id is not None:
            tag_filter.append(f"exclude_after:{oldest_interaction_id}")

        try:
            hits = await self.memory_backend.recall(
                bank_id=persona.get_name(),
                query=query_text,
                # DP-491: per-persona recall knobs; None = backend default.
                k=persona.get_effective_ltm_max_items(),
                max_tokens=persona.get_ltm_max_tokens(),
                budget=persona.get_ltm_budget(),
                tag_filter=tag_filter,
                memory_mode=mode_label,
                include_ambient=persona.get_include_ambient_memory(),
            )
        except Exception as e:  # noqa: BLE001
            logger.warning(f"Memory recall failed: {e}")
            return None, False

        if not hits:
            logger.warning(f"### RequestBuilder: No relevant memories returned from backend for {persona.get_name()}")
            return None, False

        has_untrusted = any(h.untrusted for h in hits)
        memory_block = self.format_memory_block(hits)
        if memory_block:
            logger.warning(f"### RequestBuilder: Injected memory block for {persona.get_name()} ({len(hits)} hits, untrusted={has_untrusted})")
        return memory_block, has_untrusted

    @staticmethod
    def _attach_memory_block(history: List[Dict[str, Any]], block: str) -> None:
        """Prepend the LTM block to the newest user turn (DP-431).

        Not at the head of the window: it is re-recalled every turn, so there
        it changed the prompt right after the system prompt and a local
        server's prefix cache reprocessed the whole history each time. Beside
        the newest turn only the last exchange reprocesses, and the facts sit
        next to the message they were recalled for. Folded *into* that turn
        rather than added as its own message, so roles still alternate (chat
        templates such as Gemma's reject two user messages in a row) and the
        token pruner, which always keeps the newest user turn, keeps it too.
        """
        for msg in reversed(history):
            if msg.get("role") != "user":
                continue
            content = msg.get("content")
            if isinstance(content, str):
                msg["content"] = f"{block}\n\n{content}"
            elif isinstance(content, list):
                msg["content"] = [{"type": "text", "text": block}] + content
            return
        logger.debug("LTM block dropped: no user turn to attach it to.")

    async def get_session_memory_block(
            self,
            persona_name: str,
            user_identifier: str,
            channel: str,
            server_id: Optional[str],
            query: Optional[str] = None,
    ) -> Optional[str]:
        """Public LTM seam for interfaces that bypass generate_response (portal).

        Builds the persona's sliding-window history, then returns a formatted
        memory block or None when retrieval is disabled / yields no matches.
        """
        persona = self.persona_lookup(persona_name)
        if persona is None:
            return None
        # DP-142: read-only LTM seam — peek the window, never advance the
        # hello override (and avoid the prior double-advance from calling the
        # getter here AND inside build_conversation_history).
        history, oldest_id = self.build_conversation_history(
            persona, user_identifier, channel, server_id,
            persona.get_history_messages(advance=False), advance=False,
        )
        block, _has_untrusted = await self.retrieve_memory_block(
            persona=persona,
            user_identifier=user_identifier,
            channel=channel,
            server_id=server_id,
            conversation_history=history,
            current_message=query or None,
            oldest_interaction_id=oldest_id,
        )
        return block

    @staticmethod
    def resolve_generation_params(
        persona: Persona, local_inference_config: Optional[Dict[str, Any]],
    ) -> GenerationParams:
        """Resolve the per-request generation params: persona defaults with the
        optional per-call inference overrides merged in.

        Single source of truth for param resolution — both `_orchestrate` (live
        submit) and `assemble_request` (dry-run) call this, so the params the
        inspector shows are exactly the params a live submit would forward.
        """
        params = persona.get_generation_params().copy()
        if local_inference_config:
            params.merge_inference_config(local_inference_config)
        return params

    @staticmethod
    def derive_message_sources(
        messages: List[Dict[str, Any]], *, is_retry: bool,
    ) -> List[str]:
        """Tag each assembled wire message with its provenance for the inspector.

        Structural derivation (the flat wire dicts carry no provenance): index 0
        is the persona system prompt; replayed tool turns are tagged
        tool_call/tool_result; on a non-retry turn the final user row is the
        composer's new message; everything else is sliding-window history. The
        user row the LTM block rides on (DP-431) gets a `+ltm` suffix.
        """
        sources: List[str] = []
        for i, m in enumerate(messages):
            if i == 0:
                sources.append("persona.prompt")
                continue
            role = m.get("role")
            if role == "tool":
                sources.append("tool_result")
            elif role == "assistant" and m.get("tool_calls"):
                sources.append("tool_call")
            else:
                sources.append("history")
        # The new user turn is appended last on a non-retry submit; tag it so the
        # inspector distinguishes "what I'm about to send" from prior history.
        if not is_retry:
            for i in range(len(messages) - 1, 0, -1):
                if sources[i] == "history" and messages[i].get("role") == "user":
                    sources[i] = "composer"
                    break
        for i, m in enumerate(messages):
            content = m.get("content")
            if i and m.get("role") == "user" and isinstance(content, str) \
                    and content.startswith("<memory>"):
                sources[i] += "+ltm"
        return sources

    async def assemble_request(
            self,
            persona_name: str,
            user_identifier: str,
            channel: str,
            message: str,
            *,
            server_id: Optional[str] = None,
            image_url: Optional[str] = None,
            history_limit: Optional[int] = None,
            local_inference_config: Optional[Dict[str, Any]] = None,
            is_retry: bool = False,
            client_messages: Optional[List[Dict[str, Any]]] = None,
    ) -> Optional[AssembledRequest]:
        """Dry-run assembler — the parity primitive behind the Raw-req inspector.

        Produces the exact `{route, model_name, params, messages}` the engine
        would send for this turn, with **inference disabled and no DB writes /
        turn logging**. Parity with a live submit is true by construction: this
        reuses `prepare_request` (history rebuild + LTM injection + token
        truncation), `resolve_generation_params`, and `build_wire_messages` —
        the identical helpers `_orchestrate` and `ToolLoop` drive on a live call.

        Returns None when the persona is unknown. Note LTM injection follows the
        persona's own `long_term_memory` setting (the engine path has no client
        LTM toggle), so the dry-run matches what a live submit actually assembles.
        """
        persona = self.persona_lookup(persona_name)
        if persona is None:
            return None

        ctx = RequestContext(
            persona=persona, persona_name=persona_name,
            user_identifier=user_identifier, channel=channel, message=message,
            server_id=server_id, image_url=image_url,
            history_limit=history_limit,
            local_inference_config=local_inference_config,
            client_messages=client_messages,
        )

        # turn_scope so a scoped recall during retrieve_memory_block inherits
        # persona/channel/user/server. Read-only: the dry-run never writes the
        # sticky taint bit back (that happens only on _LoopFinishedEvent).
        with turn_scope(TurnContext(
            persona_name=persona_name,
            user_identifier=user_identifier,
            channel=channel,
            server_id=server_id,
            memory_mode=persona.get_memory_mode(),
        )):
            # DP-142: dry-run — never advance the hello override.
            await self.prepare_request(ctx, is_retry=is_retry, advance=False)

        params = self.resolve_generation_params(persona, local_inference_config)
        messages = build_wire_messages(persona, ctx.conversation_history)
        sources = self.derive_message_sources(messages, is_retry=is_retry)
        return AssembledRequest(
            persona_name=persona_name,
            model_name=persona.get_model_name(),
            route="engine · POST /v1/chat/completions",
            params=params,
            messages=messages,
            sources=sources,
            oldest_interaction_id=ctx.oldest_interaction_id,
        )

    @staticmethod
    def format_memory_block(hits: List[MemoryHit]) -> Optional[str]:
        """Format MemoryHit list into a <memory> block for injection.

        The block rides at the head of the newest user turn (DP-431), so its
        preamble has to say it is not the user's words, and that it is
        background to draw on rather than something to answer. Hit times are
        absolute (`prompt_time`), not "2 minutes ago": the same hits then
        render the same text on every request, and they compare directly with
        the turn stamps.
        """
        if not hits:
            return None

        lines = [
            "<memory>",
            "Recalled from long-term memory for the message below. This is "
            "background added by the system, not part of what the user wrote: "
            "use whatever is relevant, ignore the rest, and do not reply to or "
            "mention this block itself.",
            "",
        ]

        for hit in hits:
            tag_map: Dict[str, str] = {}
            for tag in hit.tags or []:
                if ":" in tag:
                    k, v = tag.split(":", 1)
                    tag_map[k] = v
            channel = tag_map.get("channel", "unknown")
            persona = tag_map.get("persona", "")

            label_parts = [f"#{channel}"]
            if hit.id:
                label_parts.append(f"ID:{hit.id}")
            if persona == "ambient":
                label_parts.append("ambient")
            if hit.timestamp:
                label_parts.append(prompt_time(hit.timestamp))

            lines.append(f"[{', '.join(label_parts)}]")
            for fact_line in hit.content.strip().split('\n'):
                if fact_line.strip():
                    lines.append(fact_line)
            lines.append("")

        lines.append("</memory>")
        return "\n".join(lines)

    def filter_tools_for_persona(self, persona: Persona) -> List[Dict[str, Any]]:
        """Filters available tools by persona policy, service bindings, and model compatibility."""
        all_tools = self._tool_manager_lookup().get_tool_definitions()

        # 1. Primary filtering via ToolPolicy
        tools_for_llm = persona.get_tool_policy().filter_tools(all_tools)

        # 2. Filter out tools whose service_binding isn't in the persona's bindings
        bindings = set(persona.get_service_bindings())
        tools_for_llm = [t for t in tools_for_llm
                         if not t.get('service_binding') or t.get('service_binding') in bindings]

        # 3. Model compatibility check
        model_prefix = get_model_prefix(persona.get_model_name())
        tools_for_llm = [t for t in tools_for_llm
                         if model_prefix not in MODEL_INCOMPATIBLE_TOOLS.get(
                             t.get('function', {}).get('name'), set())]

        return tools_for_llm

    async def prepare_request(self, ctx: RequestContext, is_retry: bool = False,
                              advance: bool = True) -> None:
        """Build history, inject long-term memory, filter tools, append user message.

        On `is_retry=True`: the prior user turn is already in DB, and the
        prior assistant turn is the row we are about to UPDATE in place.
        Pop the trailing assistant so the model regenerates from the user
        instead of continuing its own discarded response, and skip appending
        a fresh user turn (DB already terminates with the matching user row).

        When `ctx.client_messages` is provided and the DB returns no history,
        the client-side message array (a non-streaming OAI caller's) is used
        as a fallback so sessions with rich UI state are not hollow on the
        first engine-routed turn.  System/trailing-assistant/trailing-user
        messages are stripped — the engine re-injects them from persona config
        and ctx.message.
        """
        ctx.history_window = self.load_history_window(
            ctx.persona, ctx.user_identifier, ctx.channel,
            ctx.server_id, ctx.history_limit, advance=advance
        )
        ctx.conversation_history = list(ctx.history_window.messages)
        ctx.oldest_interaction_id = ctx.history_window.oldest_id
        # Taken before the new turn and the memory block are added: every
        # other token in the prompt is "fixed" for the block cut below.
        window_tokens = sum(message_tokens(m) for m in ctx.conversation_history)

        # Load sticky taint bit for this conversation (touch for LRU recency)
        taint_key = (ctx.user_identifier, ctx.persona_name, ctx.channel, ctx.server_id)
        ctx.turn_tainted = self.conversation_taints.get(taint_key, False)
        if taint_key in self.conversation_taints:
            self.conversation_taints.move_to_end(taint_key)

        # If the client supplied its own message array (the non-streaming OAI
        # route passes it through), prefer it over the DB result — that caller
        # owns its window. DB history is used when no client messages are
        # present (the /derpr portal, Discord, Gmail, other bot callers).
        if ctx.client_messages:
            fallback = []
            for m in ctx.client_messages:
                m_copy = dict(m)
                content = m_copy.get("content", "")
                if isinstance(content, str):
                    # Strip KoboldCPP instruct placeholders a client may echo
                    # back; the engine does its own templating.
                    content = content.replace("{{[INPUT]}}", "").replace("{{[OUTPUT]}}", "").strip()
                    # Noise reduction for client-supplied history
                    content = strip_vertex_links(content)
                    m_copy["content"] = content
                fallback.append(m_copy)

            # Strip leading system message — engine re-injects from persona.
            if fallback and fallback[0].get("role") == "system":
                fallback.pop(0)
            # Strip trailing assistant continuation prefix (continue_assistant_turn).
            if fallback and fallback[-1].get("role") == "assistant" and not fallback[-1].get("content"):
                fallback.pop()
            # Strip trailing user message — engine will re-append from ctx.message.
            if fallback and fallback[-1].get("role") == "user":
                # Ensure the fallback's last user matches current message to avoid double-appending
                # if the client already included it in the array. Content may be
                # None or an OAI multimodal list — only a plain string can match.
                last_content = fallback[-1].get("content")
                if isinstance(last_content, str) and last_content.strip() == ctx.message.strip():
                    fallback.pop()

            # Capture the discarded DB row count *before* reassigning, so the
            # diagnostic distinguishes "DB returned N rows we ignored" from
            # "DB returned 0 rows" — the fallback size is a different number.
            db_row_count = len(ctx.conversation_history)
            ctx.conversation_history = fallback
            logger.info(
                "prepare_request: using %d client messages (cleaned client history) "
                "for %s / %s — DB result (%d rows) discarded",
                len(ctx.conversation_history), ctx.persona_name, ctx.channel,
                db_row_count,
            )

        # Recalled before the new turn is appended: on a retry the query falls
        # back to the last history user turn, which must not carry the block.
        memory_block, has_untrusted = await self.retrieve_memory_block(
            ctx.persona, ctx.user_identifier, ctx.channel,
            ctx.server_id, ctx.conversation_history,
            current_message=ctx.message,
            oldest_interaction_id=ctx.oldest_interaction_id,
        )
        if has_untrusted:
            ctx.turn_tainted = True
            if "memory_recall" not in ctx.taint_sources:
                ctx.taint_sources.append("memory_recall")

        ctx.tools_for_llm = self.filter_tools_for_persona(ctx.persona)

        if ctx.continue_prefix is not None:
            # DP-447: end the wire array on the reply being continued, as the
            # prefix the local server appends verbatim after the generation
            # prompt. History replays the row's reply alone (a reasoning-only
            # row not at all), so swap that message for the prefix. No tools:
            # a prefilled turn that calls one has no defined shape.
            ctx.tools_for_llm = []
            last = ctx.conversation_history[-1] if ctx.conversation_history else None
            if last is not None and last.get("role") == "assistant" \
                    and not last.get("tool_calls") \
                    and last.get("content") == strip_vertex_links(ctx.continue_row_content):
                ctx.conversation_history.pop()
            ctx.conversation_history.append(
                {"role": "assistant", "content": ctx.continue_prefix},
            )
        elif is_retry:
            if ctx.conversation_history and ctx.conversation_history[-1].get("role") == "assistant":
                ctx.conversation_history.pop()
        elif ctx.message and ctx.message.strip():
            # Symmetric with the DB-side guard in TurnPersistence.log_user_turn: an empty /
            # whitespace-only message (a continue/prefetch-style call) must not
            # land a phantom `{'role':'user','content':''}` turn in the prompt,
            # which would otherwise make the model generate off a blank turn.
            # Noise reduction: Strip vertexai grounding redirect URLs from user message for LLM context.
            content = strip_vertex_links(ctx.message)
            if ctx.persona.get_inject_timestamp():
                # The stamp the row will carry as history next turn, so the
                # prompt prefix is unchanged when it gets there (DP-431).
                content = _stamped(content, ctx.timestamp or datetime.now(timezone.utc))
            ctx.conversation_history.append({"role": "user", "content": content})

        if memory_block:
            self._attach_memory_block(ctx.conversation_history, memory_block)

        # Respect per-request context overrides from the portal for history truncation.
        prompt_budget = self.prompt_budget(ctx.persona, ctx.local_inference_config)

        # DP-495: a new user message may cut the oldest history as one block
        # (90 % watermark → ~50 %, or the count cap). The block stays in this
        # prompt — and the next ones — until Hindsight has finished with it,
        # so the cut changes nothing here; it only moves the pinned start.
        # Only a live, fresh turn cuts: never a dry run, retry, continue, or a
        # client that supplies its own history.
        if advance and not is_retry:
            self._cut_on_new_message(ctx, prompt_budget, window_tokens)

        # Fallback: a prompt over the full budget (blocks still in flight,
        # Hindsight down) drops its oldest messages for this request only.
        ctx.conversation_history, dropped = truncate_messages_to_budget(
            ctx.conversation_history, prompt_budget,
        )
        if dropped:
            logger.info(
                f"Token-prune: dropped {dropped} oldest messages to fit "
                f"max_context_tokens={ctx.persona.get_max_context_tokens()} "
                f"(prompt_budget={prompt_budget}) for persona={ctx.persona_name}"
            )

        # Last stop before the wire: a history that STARTS on an
        # assistant-with-tool_calls / tool message whose user turn is gone is
        # rejected outright by Google. DP-296 closed the two DB-side producers
        # (atomic tool spans in the pruner, the tool_context replay guard in
        # build_conversation_history); the client_messages path above has
        # neither guard, and this also fails safe for any future producer.
        # Runs after the token-prune.
        ctx.conversation_history, orphaned = drop_orphaned_tool_head(
            ctx.conversation_history,
        )
        if orphaned:
            logger.info(
                "Dropped %d orphaned tool-sequence message(s) from the head of the "
                "history for persona=%s (window opened mid-turn)",
                orphaned, ctx.persona_name,
            )
