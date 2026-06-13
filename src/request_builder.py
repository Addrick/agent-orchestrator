# src/request_builder.py
"""Request assembly for the chat pipeline (DP-200 slice B).

Everything between "a message arrived" and "the wire payload the engine
sends" lives here: history retrieval + formatting per memory mode, long-term
memory recall + <memory> block injection, persona tool filtering, generation
param resolution, the context-token budget, and the `/assemble` dry-run
projection. ChatSystem's kernel calls through thin delegates so live submits
and the S5 parity inspector share this single code path by construction.
"""

import json
import logging
from collections import OrderedDict
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Tuple

from config.global_config import (
    MEMORY_MAX_SUMMARIES_IN_CONTEXT, MEMORY_RETRIEVAL_ENABLED,
)
from src.embedding_service import EmbeddingService
from src.generation_params import GenerationParams
from src.memory.backend.base import MemoryBackend, MemoryHit
from src.memory.context_budget import truncate_messages_to_budget
from src.memory.memory_manager import MemoryManager
from src.persona import Persona, MemoryMode
from src.tools.definitions import MODEL_INCOMPATIBLE_TOOLS
from src.tools.tool_loop import build_wire_messages
from src.tools.tool_manager import ToolManager
from src.tools.turn_context import TurnContext, turn_scope
from src.utils.message_utils import strip_vertex_links
from src.utils.model_utils import get_model_prefix

logger = logging.getLogger(__name__)

# Upper bound on the sticky per-conversation taint map. The taint bit is keyed
# by (user, persona, channel, server); a long-running deployment serving many
# distinct conversations would otherwise grow this map without bound. We evict
# the least-recently-touched entry once over capacity (taint is a soft cache —
# a re-derived False on a cold key is safe, and recall re-taints if needed).
MAX_CONVERSATION_TAINTS = 10000


def _relative_time(dt: datetime) -> str:
    """Format a datetime as a relative time string (e.g., '2 days ago')."""
    now = datetime.now(dt.tzinfo) if dt.tzinfo else datetime.now()
    delta = now - dt
    seconds = int(delta.total_seconds())
    if seconds < 0:
        return "just now"
    if seconds < 60:
        return "just now"
    minutes = seconds // 60
    if minutes < 60:
        return f"{minutes} minute{'s' if minutes != 1 else ''} ago"
    hours = minutes // 60
    if hours < 24:
        return f"{hours} hour{'s' if hours != 1 else ''} ago"
    days = hours // 24
    if days < 7:
        return f"{days} day{'s' if days != 1 else ''} ago"
    weeks = days // 7
    if weeks < 5:
        return f"{weeks} week{'s' if weeks != 1 else ''} ago"
    months = days // 30
    if months < 12:
        return f"{months} month{'s' if months != 1 else ''} ago"
    years = days // 365
    return f"{years} year{'s' if years != 1 else ''} ago"


def build_scope_tags(
    *,
    channel: Optional[str],
    server_id: Optional[str],
    user_identifier: Optional[str],
    interface: Optional[str] = None,
) -> List[str]:
    """Plan §1.4 scope tags. Used for retain + recall tag predicates."""
    tags: List[str] = []
    if channel:
        tags.append(f"channel:{channel}")
    if user_identifier:
        tags.append(f"user:{user_identifier}")
    if server_id:
        tags.append(f"server:{server_id}")
    if interface:
        tags.append(f"interface:{interface}")
    return tags


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
    # Populated during prepare_request
    conversation_history: List[Dict[str, Any]] = field(default_factory=list)
    tools_for_llm: List[Dict[str, Any]] = field(default_factory=list)
    oldest_interaction_id: Optional[int] = None
    local_inference_config: Optional[Dict[str, Any]] = None
    turn_tainted: bool = False
    taint_sources: List[str] = field(default_factory=list)
    # Optional: OAI-format messages from the client (e.g. kobold-lite jinja history).
    # Used as a fallback when the DB returns no history for this channel.
    client_messages: Optional[List[Dict[str, Any]]] = None


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
                                   persona_name: str, server_id: Optional[str]) -> List[Dict[str, Any]]:
        """Formats database history records into a list of messages for the LLM."""
        final_history: List[Dict[str, Any]] = []
        is_group_chat = memory_mode in ("server", "global", "ticket") or (
                    memory_mode == "channel" and server_id is not None)

        for msg in raw_history:
            author_role = msg.get('author_role')
            author_name = msg.get('author_name')
            content = msg.get('content', '')
            # Noise reduction: Strip vertexai grounding redirect URLs from history before sending to LLM.
            content_clean = strip_vertex_links(content) if content else content

            if author_role == 'user':
                if is_group_chat and author_name:
                    formatted_content = f"{author_name}: {content_clean}"
                    final_history.append({'role': 'user', 'content': formatted_content})
                else:
                    final_history.append({'role': 'user', 'content': content_clean})
            elif author_role == 'assistant':
                if author_name == persona_name:
                    tool_context_json = msg.get('tool_context')
                    if tool_context_json:
                        final_history.extend(json.loads(tool_context_json))
                    final_history.append({'role': 'assistant', 'content': content_clean})
                else:
                    # In a group chat, messages from other personas are treated as user messages
                    formatted_content = f"{author_name}: {content_clean}"
                    final_history.append({'role': 'user', 'content': formatted_content})
        return final_history

    def fetch_raw_history(
            self,
            mode: MemoryMode,
            persona_name: str,
            user_identifier: str,
            channel: str,
            server_id: Optional[str],
            effective_limit: int
    ) -> Tuple[List[Dict[str, Any]], str]:
        """Dispatches history retrieval based on memory mode. Returns (raw_history, mode_label)."""
        if mode == MemoryMode.TICKET_ISOLATED:
            return [], "ticket"
        elif mode == MemoryMode.SERVER_WIDE:
            # SERVER_WIDE returns history for the specific server.
            # If server_id is None (Web UI/local), we still query, as the DB
            # stores these rows with server_id=NULL. MemoryManager handles
            # the NULL check via IS NULL in its queries.
            return self.memory_manager.get_server_history(server_id, persona_name, effective_limit), "server"
        elif mode == MemoryMode.PERSONAL:
            return self.memory_manager.get_personal_history(user_identifier, persona_name, effective_limit), "personal"
        elif mode == MemoryMode.GLOBAL:
            return self.memory_manager.get_global_history(persona_name, effective_limit), "global"
        else:
            return self.memory_manager.get_channel_history(channel, persona_name, server_id, effective_limit), "channel"

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
        oldest_interaction_id is the interaction_id of the oldest message in the
        sliding window, used for the memory recency filter.

        DP-142: ``advance`` gates the hello-override side effect. Live turns pass
        True; read-only / dry-run callers pass False.
        """
        persona_name = persona.get_name()

        effective_limit: int = persona.get_history_messages(advance=advance)
        if history_limit is not None:
            effective_limit = min(effective_limit, history_limit)

        raw_history, memory_mode_used = self.fetch_raw_history(
            persona.get_memory_mode(), persona_name,
            user_identifier, channel, server_id, effective_limit
        )

        oldest_interaction_id = None
        if raw_history:
            oldest_interaction_id = raw_history[0].get('interaction_id')

        formatted = self.format_raw_history_for_llm(raw_history, memory_mode_used, persona_name, server_id)
        return formatted, oldest_interaction_id

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
                        query_text = strip_vertex_links(content)
                        break
        if not query_text:
            logger.warning(f"### RequestBuilder: Skipping retrieval for {persona.get_name()} (no text content to embed)")
            return None, False

        tag_filter = build_scope_tags(
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
                k=MEMORY_MAX_SUMMARIES_IN_CONTEXT,
                tag_filter=tag_filter,
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
        is the persona system prompt; a leading `<memory>` user row is the LTM
        block; replayed tool turns are tagged tool_call/tool_result; on a
        non-retry turn the final non-LTM user row is the composer's new message;
        everything else is sliding-window history.
        """
        sources: List[str] = []
        for i, m in enumerate(messages):
            if i == 0:
                sources.append("persona.prompt")
                continue
            role = m.get("role")
            content = m.get("content")
            if role == "user" and isinstance(content, str) and content.startswith("<memory>"):
                sources.append("ltm_block")
            elif role == "tool":
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
        """Format MemoryHit list into a <memory> block for injection."""
        if not hits:
            return None

        lines = ["<memory>", "The following are relevant facts from previous conversations:", ""]

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
                label_parts.append(_relative_time(hit.timestamp))

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
        the client-side message array (e.g. kobold-lite jinja history) is used
        as a fallback so sessions with rich UI state are not hollow on the
        first engine-routed turn.  System/trailing-assistant/trailing-user
        messages are stripped — the engine re-injects them from persona config
        and ctx.message.
        """
        ctx.conversation_history, ctx.oldest_interaction_id = self.build_conversation_history(
            ctx.persona, ctx.user_identifier, ctx.channel,
            ctx.server_id, ctx.history_limit, advance=advance
        )

        # Load sticky taint bit for this conversation (touch for LRU recency)
        taint_key = (ctx.user_identifier, ctx.persona_name, ctx.channel, ctx.server_id)
        ctx.turn_tainted = self.conversation_taints.get(taint_key, False)
        if taint_key in self.conversation_taints:
            self.conversation_taints.move_to_end(taint_key)

        # If the client supplied its own message array (kobold-lite jinja mode),
        # prefer it over the DB result. The portal export already uses global
        # history (all channels), so kobold-lite's in-memory state is the
        # correct, authoritative window — re-querying a narrower channel filter
        # here would miss cross-channel turns. DB history is still used when no
        # client messages are present (Discord, Gmail, other bot callers).
        if ctx.client_messages:
            fallback = []
            for m in ctx.client_messages:
                m_copy = dict(m)
                content = m_copy.get("content", "")
                if isinstance(content, str):
                    # Strip kobold-lite internal placeholders used for dynamic templating.
                    # These are injected by kobold_export.py but redundant for engine-side templating.
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
                "prepare_request: using %d client messages (cleaned kobold-lite history) "
                "for %s / %s — DB result (%d rows) discarded",
                len(ctx.conversation_history), ctx.persona_name, ctx.channel,
                db_row_count,
            )

        # Inject long-term memory block before the sliding window
        memory_block, has_untrusted = await self.retrieve_memory_block(
            ctx.persona, ctx.user_identifier, ctx.channel,
            ctx.server_id, ctx.conversation_history,
            current_message=ctx.message,
            oldest_interaction_id=ctx.oldest_interaction_id,
        )
        if memory_block:
            ctx.conversation_history.insert(0, {"role": "user", "content": memory_block})
        if has_untrusted:
            ctx.turn_tainted = True
            if "memory_recall" not in ctx.taint_sources:
                ctx.taint_sources.append("memory_recall")

        ctx.tools_for_llm = self.filter_tools_for_persona(ctx.persona)

        if is_retry:
            if ctx.conversation_history and ctx.conversation_history[-1].get("role") == "assistant":
                ctx.conversation_history.pop()
        elif ctx.message and ctx.message.strip():
            # Symmetric with the DB-side guard in TurnPersistence.log_user_turn: an empty /
            # whitespace-only message (kobold-lite continue/prefetch) must not
            # land a phantom `{'role':'user','content':''}` turn in the prompt,
            # which would otherwise make the model generate off a blank turn.
            # Noise reduction: Strip vertexai grounding redirect URLs from user message for LLM context.
            ctx.conversation_history.append({"role": "user", "content": strip_vertex_links(ctx.message)})

        # Respect per-request context overrides from the portal for history truncation.
        ctx_limit = (ctx.local_inference_config or {}).get("max_context_length") or ctx.persona.get_max_context_tokens()
        prompt_budget = ctx_limit - ctx.persona.get_response_token_limit()
        ctx.conversation_history, dropped = truncate_messages_to_budget(
            ctx.conversation_history, prompt_budget,
        )
        if dropped:
            logger.info(
                f"Token-prune: dropped {dropped} oldest messages to fit "
                f"max_context_tokens={ctx.persona.get_max_context_tokens()} "
                f"(prompt_budget={prompt_budget}) for persona={ctx.persona_name}"
            )
