# src/chat_system.py

import asyncio
import json
import logging
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, AsyncIterator, Coroutine, Dict, List, Optional, Set, Tuple

from config.global_config import MAX_CACHED_API_REQUESTS, \
    PENDING_CONFIRMATION_TIMEOUT, MEMORY_RETRIEVAL_ENABLED, MEMORY_MAX_SUMMARIES_IN_CONTEXT
from src.memory.context_budget import truncate_messages_to_budget
from src.embedding_service import EmbeddingService
from src.clients.service_integration import ServiceIntegration
from src.memory.memory_manager import MemoryManager
from src.engine import TextEngine
from src.generation_events import (
    DoneEvent as DoneEvent,
    ErrorEvent as ErrorEvent,
    GenerationEvent as GenerationEvent,
    ResponseType as ResponseType,
    TokenEvent as TokenEvent,
    ToolCallResultEvent as ToolCallResultEvent,
    ToolCallStartEvent as ToolCallStartEvent,
)
from src.stream_engine import StreamEngine
from src.message_handler import BotLogic
from src.persona import Persona, MemoryMode
from src.tools.definitions import MODEL_INCOMPATIBLE_TOOLS
from src.tools.tool_loop import ToolLoop, _ApiPayloadEvent, _LoopFinishedEvent
from src.tools.tool_manager import ToolManager, WebSearchHandler
from src.utils.model_utils import get_model_list, get_model_prefix
from src.utils.save_utils import load_personas_from_file, save_personas_to_file
from src.utils.message_utils import strip_vertex_links

logger = logging.getLogger(__name__)


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


@dataclass
class PendingConfirmation:
    """Stores state for a tool call awaiting user approval in CONFIRM mode."""
    write_calls: List[Dict[str, Any]]
    conversation_history: List[Dict[str, Any]]
    persona_name: str
    tools_for_llm: List[Dict[str, Any]]
    image_url: Optional[str]
    created_at: float = field(default_factory=time.time)


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
    # Populated during _prepare_request
    conversation_history: List[Dict[str, Any]] = field(default_factory=list)
    tools_for_llm: List[Dict[str, Any]] = field(default_factory=list)
    oldest_interaction_id: Optional[int] = None
    local_inference_config: Optional[Dict[str, Any]] = None
    # Optional: OAI-format messages from the client (e.g. kobold-lite jinja history).
    # Used as a fallback when the DB returns no history for this channel.
    client_messages: Optional[List[Dict[str, Any]]] = None


class ChatSystem:
    def __init__(self, memory_manager: MemoryManager, text_engine: TextEngine,
                 embedding_service: Optional[EmbeddingService] = None,
                 stream_engine: Optional[StreamEngine] = None) -> None:
        self.personas: Dict[str, Persona] = load_personas_from_file() or {}
        self.memory_manager: MemoryManager = memory_manager
        self.text_engine: TextEngine = text_engine
        self.stream_engine: Optional[StreamEngine] = stream_engine
        self.tool_manager: ToolManager = ToolManager()
        WebSearchHandler().register(self.tool_manager)
        
        from src.tools.tool_manager import MemoryToolHandler
        MemoryToolHandler(self.memory_manager).register(self.tool_manager)

        self.bot_logic: BotLogic = BotLogic(self)
        self.last_api_requests: Dict[str, Dict[str, Optional[Dict[str, Any]]]] = defaultdict(dict)
        self.models_available: Dict[str, Any] = get_model_list() or {}
        self.background_tasks: Set[Coroutine[Any, Any, Any]] = set()
        self._pending_confirmations: Dict[Tuple[str, str], PendingConfirmation] = {}
        self._services: Dict[str, ServiceIntegration] = {}
        self._embedding_service: Optional[EmbeddingService] = embedding_service

    def register_service(self, service: ServiceIntegration) -> None:
        """Register a service integration and its tools."""
        self._services[service.name] = service
        service.register_tools(self.tool_manager)
        logger.info(f"Registered service integration: {service.name}")

    def _store_api_request(self, user_identifier: str, persona_name: str,
                           payload: Dict[str, Any],
                           tools_for_llm: Optional[List[Dict[str, Any]]] = None) -> None:
        """Stores the last API request payload, evicting the oldest user entry if over capacity."""
        if tools_for_llm is not None:
            payload["_tools_for_llm"] = tools_for_llm
        else:
            existing = self.last_api_requests.get(user_identifier, {}).get(persona_name)
            if existing and "_tools_for_llm" in existing:
                payload["_tools_for_llm"] = existing["_tools_for_llm"]
        self.last_api_requests[user_identifier][persona_name] = payload
        if len(self.last_api_requests) > MAX_CACHED_API_REQUESTS:
            oldest_key = next(iter(self.last_api_requests))
            del self.last_api_requests[oldest_key]

    def _format_raw_history_for_llm(self, raw_history: List[Dict[str, Any]], memory_mode: str,
                                    persona_name: str, server_id: Optional[str]) -> List[Dict[str, Any]]:
        """Formats database history records into a list of messages for the LLM."""
        final_history: List[Dict[str, Any]] = []
        is_group_chat = memory_mode in ("server", "global", "ticket") or (
                    memory_mode == "channel" and server_id is not None)

        for msg in raw_history:
            author_role = msg.get('author_role')
            author_name = msg.get('author_name')
            content = msg.get('content', '')

            if author_role == 'user':
                if is_group_chat and author_name:
                    formatted_content = f"{author_name}: {content}"
                    final_history.append({'role': 'user', 'content': formatted_content})
                else:
                    final_history.append({'role': 'user', 'content': content})
            elif author_role == 'assistant':
                if author_name == persona_name:
                    tool_context_json = msg.get('tool_context')
                    if tool_context_json:
                        final_history.extend(json.loads(tool_context_json))
                    final_history.append({'role': 'assistant', 'content': content})
                else:
                    # In a group chat, messages from other personas are treated as user messages
                    formatted_content = f"{author_name}: {content}"
                    final_history.append({'role': 'user', 'content': formatted_content})
        return final_history

    async def _execute_write_calls(
            self,
            write_calls: List[Dict[str, Any]],
            conversation_history: List[Dict[str, Any]]
    ) -> None:
        """Execute write tool calls and append results to history."""
        for call_item in write_calls:
            tool_name: str = call_item.get("name", "")
            tool_args = call_item.get("arguments", {})
            tool_result = await self.tool_manager.execute_tool(tool_name, **tool_args)
            conversation_history.append({
                "role": "tool",
                "tool_call_id": call_item.get("id"),
                "name": tool_name,
                "content": json.dumps(tool_result)
            })

    @staticmethod
    def _append_denied_tool_results(
            write_calls: List[Dict[str, Any]],
            conversation_history: List[Dict[str, Any]]
    ) -> None:
        """Appends denial results for write tools the user rejected."""
        for call_item in write_calls:
            conversation_history.append({
                "role": "tool",
                "tool_call_id": call_item.get("id"),
                "name": call_item.get("name"),
                "content": json.dumps({"error": "Tool call denied by user"})
            })

    def _fetch_raw_history(
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

    def _build_conversation_history(
            self,
            persona: Persona,
            user_identifier: str,
            channel: str,
            server_id: Optional[str],
            history_limit: Optional[int]
    ) -> Tuple[List[Dict[str, Any]], Optional[int]]:
        """Retrieves and formats conversation history based on the persona's memory mode.

        Returns (formatted_history, oldest_interaction_id).
        oldest_interaction_id is the interaction_id of the oldest message in the
        sliding window, used for the memory recency filter.
        """
        persona_name = persona.get_name()

        effective_limit: int = persona.get_history_messages()
        if history_limit is not None:
            effective_limit = min(effective_limit, history_limit)

        raw_history, memory_mode_used = self._fetch_raw_history(
            persona.get_memory_mode(), persona_name,
            user_identifier, channel, server_id, effective_limit
        )

        oldest_interaction_id = None
        if raw_history:
            oldest_interaction_id = raw_history[0].get('interaction_id')

        formatted = self._format_raw_history_for_llm(raw_history, memory_mode_used, persona_name, server_id)
        return formatted, oldest_interaction_id

    async def _retrieve_memory_block(
            self,
            persona: Persona,
            user_identifier: str,
             channel: str,
            server_id: Optional[str],
            conversation_history: List[Dict[str, Any]],
            current_message: Optional[str] = None,
            oldest_interaction_id: Optional[int] = None,
    ) -> Optional[str]:
        """Retrieve and format relevant long-term memory summaries for injection.

        Returns a formatted <memory> block string, or None if no relevant memories
        or the feature is disabled.
        """
        logger.warning(f"### RETRIEVAL_DIAGNOSTIC: Entering _retrieve_memory_block for {persona.get_name()} (Enabled: {MEMORY_RETRIEVAL_ENABLED}, Service: {'YES' if self._embedding_service else 'NO'})")

        if not MEMORY_RETRIEVAL_ENABLED or not persona.get_long_term_memory() or self._embedding_service is None:
            return None

        # Map MemoryMode enum to string for retrieve_relevant_summaries
        mode_map = {
            MemoryMode.CHANNEL_ISOLATED: "channel",
            MemoryMode.SERVER_WIDE: "server",
            MemoryMode.PERSONAL: "personal",
            MemoryMode.GLOBAL: "global",
            MemoryMode.TICKET_ISOLATED: "ticket",
        }
        memory_mode = mode_map.get(persona.get_memory_mode(), "channel")

        # THE FIX: Only include the current message in the search query
        # to avoid broad query averaging/multi-matching that dilutes relevance.
        query_texts = []
        if current_message and current_message.strip():
            query_texts.append(current_message)

        # Fallback: if current_message is empty, use the most recent USER message from history
        if not query_texts and conversation_history:
            for msg in reversed(conversation_history):
                # We specifically look for 'user' role to honor the intent of 
                # searching for the user's latest query/intent.
                if msg.get('role') == 'user':
                    content = msg.get('content', '')
                    if isinstance(content, str) and content.strip():
                        query_texts.append(content)
                        break

        # Enhanced Diagnostic Logging
        logger.debug(
            f"### RETRIEVAL_TRACE: persona={persona.get_name()}, "
            f"current_msg_len={len(current_message) if current_message else 0}, "
            f"history_len={len(conversation_history)}, "
            f"query_texts={query_texts}"
        )

        if not query_texts:
            logger.warning(f"### ChatSystem: Skipping retrieval for {persona.get_name()} (no text content to embed)")
            return None

        # Embed the query messages (1 batched API call)
        try:
            query_embeddings = await self._embedding_service.encode(query_texts)
        except Exception as e:
            logger.warning(f"Memory retrieval: embedding failed: {e}")
            return None

        if not query_embeddings:
            logger.warning("Memory retrieval: encode returned empty results")
            return None

        # Fetch candidate summaries natively in sqlite using sqlite-vec nearest neighbor
        summaries = self.memory_manager.retrieve_relevant_summaries(
            persona_name=persona.get_name(),
            channel=channel,
            server_id=server_id,
            user_identifier=user_identifier,
            memory_mode=memory_mode,
            include_ambient=persona.get_include_ambient_memory(),
            exclude_after_interaction_id=oldest_interaction_id,
            model_name=self._embedding_service.model_name,
            query_embeddings=query_embeddings,
            limit=MEMORY_MAX_SUMMARIES_IN_CONTEXT,
        )

        if not summaries:
            logger.warning(f"### ChatSystem: No relevant memories returned from database for {persona.get_name()}")
            return None

        # Pack into the expected format matching (score, summary)
        scored_summaries = [(summary.get('dist', 1.0), summary) for summary in summaries]

        # Format memory block
        memory_block = self._format_memory_block(scored_summaries)
        if memory_block:
            logger.warning(f"### ChatSystem: Injected memory block for {persona.get_name()} ({len(scored_summaries)} summaries)")
        else:
            logger.warning(f"### ChatSystem: No relevant memories found for {persona.get_name()}")
        return memory_block

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
        Wraps the two private helpers so callers do not depend on private API.
        """
        persona = self.personas.get(persona_name)
        if persona is None:
            return None
        history, oldest_id = self._build_conversation_history(
            persona, user_identifier, channel, server_id, persona.get_history_messages(),
        )
        return await self._retrieve_memory_block(
            persona=persona,
            user_identifier=user_identifier,
            channel=channel,
            server_id=server_id,
            conversation_history=history,
            current_message=query or None,
            oldest_interaction_id=oldest_id,
        )

    @staticmethod
    def _format_memory_block(scored_summaries: List[Tuple[float, Dict[str, Any]]]) -> Optional[str]:
        """Format scored summaries into a <memory> block for injection."""
        if not scored_summaries:
            return None

        lines = ["<memory>", "The following are relevant facts from previous conversations:", ""]

        for _score, summary in scored_summaries:
            channel = summary.get('channel', 'unknown')
            persona = summary.get('persona_name', '')
            # Prefer the actual time the memory occurred over when it was summarized
            memory_timestamp = summary.get('last_message_at') or summary.get('created_at')

            # Build label
            label_parts = [f"#{channel}"]
            if persona == 'ambient':
                label_parts.append("ambient")
            if memory_timestamp:
                label_parts.append(_relative_time(memory_timestamp))

            label = ", ".join(label_parts)
            lines.append(f"[{label}]")

            content = summary.get('content', '')
            for fact_line in content.strip().split('\n'):
                if fact_line.strip():
                    lines.append(fact_line)
            lines.append("")

        lines.append("</memory>")
        return "\n".join(lines)

    def _filter_tools_for_persona(self, persona: Persona) -> List[Dict[str, Any]]:
        """Filters available tools by persona config, service bindings, and model compatibility."""
        all_tools = self.tool_manager.get_tool_definitions()
        enabled_tool_names = persona.get_enabled_tools()

        if enabled_tool_names == ['*']:
            tools_for_llm = all_tools
        elif enabled_tool_names:
            tools_for_llm = [tool for tool in all_tools if
                             tool.get('function', {}).get('name') in enabled_tool_names]
        else:
            tools_for_llm = []

        # Filter out tools whose service_binding isn't in the persona's bindings
        bindings = set(persona.get_service_bindings())
        tools_for_llm = [t for t in tools_for_llm
                         if not t.get('service_binding') or t.get('service_binding') in bindings]

        model_prefix = get_model_prefix(persona.get_model_name())
        tools_for_llm = [t for t in tools_for_llm
                         if model_prefix not in MODEL_INCOMPATIBLE_TOOLS.get(
                             t.get('function', {}).get('name'), set())]

        return tools_for_llm

    async def _execute_read_calls(
            self,
            read_calls: List[Dict[str, Any]],
            conversation_history: List[Dict[str, Any]]
    ) -> None:
        """Execute read-only tool calls and append results to history."""
        for call_item in read_calls:
            tool_name: str = call_item.get("name", "")
            tool_args = call_item.get("arguments", {})
            tool_result = await self.tool_manager.execute_tool(tool_name, **tool_args)
            conversation_history.append({
                "role": "tool",
                "tool_call_id": call_item.get("id"),
                "name": tool_name,
                "content": json.dumps(tool_result)
            })

    async def _prepare_request(self, ctx: RequestContext, is_retry: bool = False) -> None:
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
        ctx.conversation_history, ctx.oldest_interaction_id = self._build_conversation_history(
            ctx.persona, ctx.user_identifier, ctx.channel,
            ctx.server_id, ctx.history_limit
        )

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
                # if the client already included it in the array.
                last_content = fallback[-1].get("content", "").strip()
                if last_content == ctx.message.strip():
                    fallback.pop()
            
            ctx.conversation_history = fallback
            logger.info(
                "_prepare_request: using %d client messages (cleaned kobold-lite history) "
                "for %s / %s — DB result (%d rows) discarded",
                len(ctx.conversation_history), ctx.persona_name, ctx.channel,
                len(ctx.conversation_history),
            )

        # Inject long-term memory block before the sliding window
        memory_block = await self._retrieve_memory_block(
            ctx.persona, ctx.user_identifier, ctx.channel,
            ctx.server_id, ctx.conversation_history,
            current_message=ctx.message,
            oldest_interaction_id=ctx.oldest_interaction_id,
        )
        if memory_block:
            ctx.conversation_history.insert(0, {"role": "user", "content": memory_block})

        ctx.tools_for_llm = self._filter_tools_for_persona(ctx.persona)

        if is_retry:
            if ctx.conversation_history and ctx.conversation_history[-1].get("role") == "assistant":
                ctx.conversation_history.pop()
        else:
            ctx.conversation_history.append({"role": "user", "content": ctx.message})

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

    def _log_user_turn(
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

        user_message_clean = strip_vertex_links(message) if message else message
        try:
            user_interaction_id = self.memory_manager.log_message(
                user_identifier=user_identifier, persona_name=persona_name,
                channel=channel, author_role='user',
                author_name=user_display_name, content=user_message_clean,
                timestamp=timestamp, server_id=server_id,
                platform_message_id=platform_message_id,
            )
        except Exception as e:
            logger.error(f"User log_message failed: {e}", exc_info=True)
            user_interaction_id = None
        return user_interaction_id, None

    def _commit_or_update_assistant(
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
        final_text_clean = strip_vertex_links(final_text) if final_text else final_text
        if not final_text_clean or not final_text_clean.strip():
            return None

        if retry_assistant_id is not None:
            try:
                self.memory_manager.update_interaction_content(
                    retry_assistant_id, final_text_clean,
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
                author_name=persona_name, content=final_text_clean,
                timestamp=datetime.now(), server_id=server_id,
                tool_context=tool_context_json,
                reply_to_id=user_interaction_id,
            )
            return assistant_id
        except Exception as e:
            logger.error(f"Assistant log_message failed: {e}", exc_info=True)
            return None

    async def _orchestrate(
            self,
            persona_name: str,
            user_identifier: str,
            channel: str,
            message: str,
            *,
            server_id: Optional[str] = None,
            image_url: Optional[str] = None,
            history_limit: Optional[int] = None,
            user_display_name: Optional[str] = None,
            platform_message_id: Optional[str] = None,
            timestamp: Optional[datetime] = None,
            local_inference_config: Optional[Dict[str, Any]] = None,
            is_retry: bool = False,
            client_messages: Optional[List[Dict[str, Any]]] = None,
    ) -> AsyncIterator[GenerationEvent]:
        """Shared streaming kernel — single source of truth for the request
        pipeline. Yields TokenEvent for each text delta, terminal DoneEvent
        with final ids, or ErrorEvent on failure. Phase C kernel; both
        `generate_response` (collect-stream wrapper) and `stream_response`
        (portal entry) delegate here.
        """
        # 1. Dev command preprocessing — short-circuits before any LLM call.
        command_result: Optional[Dict[str, Any]] = await self.bot_logic.preprocess_message(
            persona_name, user_identifier, message
        )
        if command_result:
            if command_result.get("mutated", False):
                save_personas_to_file(self.personas)
            yield DoneEvent(
                text=command_result["response"],
                response_type=ResponseType.DEV_COMMAND,
            )
            return

        persona: Optional[Persona] = self.personas.get(persona_name)
        if persona is None:
            yield DoneEvent(
                text="Error: Persona not found.",
                response_type=ResponseType.DEV_COMMAND,
            )
            return

        ctx = RequestContext(
            persona=persona, persona_name=persona_name,
            user_identifier=user_identifier, channel=channel, message=message,
            server_id=server_id, image_url=image_url,
            history_limit=history_limit, user_display_name=user_display_name,
            local_inference_config=local_inference_config,
            client_messages=client_messages,
        )

        try:
            await self._prepare_request(ctx, is_retry=is_retry)
        except Exception as e:
            logger.error(
                f"_prepare_request failed for {user_identifier}: {e}", exc_info=True,
            )
            yield ErrorEvent(message="An internal error occurred while processing your request.")
            return

        # 2. Log user turn (or archive for retry). Done after history is built
        #    (so the freshly-inserted row doesn't show up twice) but before
        #    the LLM call so the user row is always pinned even if the model
        #    errors mid-flight.
        user_ts = timestamp or datetime.now()
        user_interaction_id, retry_assistant_id = self._log_user_turn(
            is_retry=is_retry, persona_name=persona_name,
            user_identifier=user_identifier, channel=channel,
            user_display_name=user_display_name, message=message,
            server_id=server_id, platform_message_id=platform_message_id,
            timestamp=user_ts,
        )

        # 3. Tool loop. ToolLoop owns iteration + tool dispatch; this
        #    forwards Token / ToolCallStart / ToolCallResult events,
        #    siphons api_payload into the request cache, and unpacks the
        #    terminal _LoopFinishedEvent to drive CONFIRM-mode parking +
        #    assistant persistence.
        params = ctx.persona.get_generation_params().copy()
        if ctx.local_inference_config:
            params.merge_inference_config(ctx.local_inference_config)
        params_first_iter = True
        final_text = ""
        response_type = ResponseType.LLM_GENERATION
        tool_context_json: Optional[str] = None
        accumulated_parts: List[str] = []
        pending_writes: Optional[List[Dict[str, Any]]] = None

        # Construct per-call so tests that swap `chat_system.text_engine`
        # post-init still see the new engine; ToolLoop is stateless.
        tool_loop = ToolLoop(self.text_engine, self.tool_manager)
        try:
            async for ev in tool_loop.run(
                persona=ctx.persona,
                conversation_history=ctx.conversation_history,
                params=params,
                tools=ctx.tools_for_llm,
                local_inference_config=ctx.local_inference_config,
                image_url=ctx.image_url,
            ):
                if isinstance(ev, _ApiPayloadEvent):
                    self._store_api_request(
                        user_identifier, persona_name, ev.payload,
                        tools_for_llm=ctx.tools_for_llm if params_first_iter else None,
                    )
                    params_first_iter = False
                elif isinstance(ev, TokenEvent):
                    accumulated_parts.append(ev.delta)
                    yield ev
                elif isinstance(ev, (ToolCallStartEvent, ToolCallResultEvent)):
                    yield ev
                elif isinstance(ev, ErrorEvent):
                    yield ev
                    return
                elif isinstance(ev, _LoopFinishedEvent):
                    final_text = ev.final_text
                    response_type = ev.response_type
                    tool_context_json = ev.tool_context_json
                    pending_writes = ev.pending_writes
        except asyncio.CancelledError:
            # Client disconnect / abort. Flush whatever assistant text has
            # accumulated so the row reflects what the user actually saw,
            # then re-raise so the surrounding StreamingResponse aborts.
            partial = "".join(accumulated_parts)
            if partial.strip():
                self._commit_or_update_assistant(
                    persona_name=persona_name, user_identifier=user_identifier,
                    channel=channel, server_id=server_id,
                    final_text=partial,
                    response_type=ResponseType.LLM_GENERATION,
                    user_interaction_id=user_interaction_id,
                    retry_assistant_id=retry_assistant_id,
                    tool_context_json=None,
                )
            raise

        if pending_writes is not None:
            self._pending_confirmations[(ctx.user_identifier, ctx.persona_name)] = (
                PendingConfirmation(
                    write_calls=pending_writes,
                    conversation_history=ctx.conversation_history,
                    persona_name=ctx.persona_name,
                    tools_for_llm=ctx.tools_for_llm,
                    image_url=ctx.image_url,
                )
            )

        # 4. Strip vertex links from the resolved assistant text, log/update.
        assistant_id = self._commit_or_update_assistant(
            persona_name=persona_name, user_identifier=user_identifier,
            channel=channel, server_id=server_id,
            final_text=final_text, response_type=response_type,
            user_interaction_id=user_interaction_id,
            retry_assistant_id=retry_assistant_id,
            tool_context_json=tool_context_json,
        )
        final_text_clean = strip_vertex_links(final_text) if final_text else final_text

        yield DoneEvent(
            text=final_text_clean if final_text_clean else "",
            response_type=response_type,
            assistant_id=assistant_id,
            user_interaction_id=user_interaction_id,
        )

    async def stream_response(
            self,
            persona_name: str,
            user_identifier: str,
            channel: str,
            message: str,
            *,
            is_retry: bool = False,
            server_id: Optional[str] = None,
            image_url: Optional[str] = None,
            history_limit: Optional[int] = None,
            user_display_name: Optional[str] = None,
            platform_message_id: Optional[str] = None,
            timestamp: Optional[datetime] = None,
            local_inference_config: Optional[Dict[str, Any]] = None,
            client_messages: Optional[List[Dict[str, Any]]] = None,
    ) -> AsyncIterator[GenerationEvent]:
        """Portal-facing streaming entry. Yields TokenEvent /
        ToolCallStartEvent / ToolCallResultEvent / DoneEvent / ErrorEvent.
        Tool-enabled personas are supported as of tool_revamp_v1 — the
        ToolLoop interleaves tool lifecycle events with token deltas in a
        single linear stream.
        """
        async for ev in self._orchestrate(
            persona_name=persona_name,
            user_identifier=user_identifier,
            channel=channel,
            message=message,
            is_retry=is_retry,
            server_id=server_id,
            image_url=image_url,
            history_limit=history_limit,
            user_display_name=user_display_name,
            platform_message_id=platform_message_id,
            timestamp=timestamp,
            local_inference_config=local_inference_config,
            client_messages=client_messages,
        ):
            yield ev

    async def generate_response(
            self,
            persona_name: str,
            user_identifier: str,
            channel: str,
            message: str,
            server_id: Optional[str] = None,
            image_url: Optional[str] = None,
            history_limit: Optional[int] = None,
            user_display_name: Optional[str] = None,
            platform_message_id: Optional[str] = None,
            timestamp: Optional[datetime] = None,
            local_inference_config: Optional[Dict[str, Any]] = None
    ) -> Tuple[str, ResponseType, Optional[int], Optional[int]]:
        """Non-streaming surface — drains the orchestration kernel into the
        existing 4-tuple. Phase C made this a collect-stream wrapper so
        Discord/Gmail/agents share a single pipeline with the portal.
        """
        logger.warning(
            f"### ChatSystem.generate_response: Received message from {user_identifier} for {persona_name}"
        )
        final_text = ""
        response_type = ResponseType.DEV_COMMAND
        assistant_id: Optional[int] = None
        user_interaction_id: Optional[int] = None
        async for ev in self._orchestrate(
            persona_name=persona_name,
            user_identifier=user_identifier,
            channel=channel,
            message=message,
            server_id=server_id,
            image_url=image_url,
            history_limit=history_limit,
            user_display_name=user_display_name,
            platform_message_id=platform_message_id,
            timestamp=timestamp,
            local_inference_config=local_inference_config,
        ):
            if isinstance(ev, TokenEvent):
                continue
            if isinstance(ev, DoneEvent):
                final_text = ev.text
                response_type = ev.response_type
                assistant_id = ev.assistant_id
                user_interaction_id = ev.user_interaction_id
            elif isinstance(ev, ErrorEvent):
                final_text = ev.message
                response_type = ResponseType.DEV_COMMAND
                assistant_id = None
                user_interaction_id = None
        return final_text, response_type, assistant_id, user_interaction_id

    async def resume_pending_confirmation(
            self, user_identifier: str, persona_name: str, approved: bool
    ) -> Tuple[str, ResponseType, Optional[int], Optional[int]]:
        """Resumes a tool execution that was paused for user confirmation."""
        key = (user_identifier, persona_name)
        pending = self._pending_confirmations.pop(key, None)

        if not pending:
            return "No pending confirmation found.", ResponseType.DEV_COMMAND, None, None

        if time.time() - pending.created_at > PENDING_CONFIRMATION_TIMEOUT:
            return "Confirmation expired. Please try again.", ResponseType.DEV_COMMAND, None, None

        persona = self.personas.get(pending.persona_name)
        if not persona:
            return "Error: Persona not found.", ResponseType.DEV_COMMAND, None, None

        conversation_history = pending.conversation_history

        try:
            if approved:
                await self._execute_write_calls(pending.write_calls, conversation_history)
            else:
                self._append_denied_tool_results(pending.write_calls, conversation_history)

            # Continue the LLM conversation with the tool results
            history_object = {
                "persona_prompt": persona.get_prompt(),
                "message_history": pending.conversation_history,
                "history": pending.conversation_history,  # Legacy key
                "current_message": {"text": "", "image_url": pending.image_url}
            }
            llm_response, api_payload = await self.text_engine.generate_response(
                persona.get_config_for_engine(), history_object, tools=pending.tools_for_llm
            )
            if api_payload:
                self._store_api_request(
                    user_identifier, persona_name, api_payload,
                    tools_for_llm=pending.tools_for_llm
                )

            final_text = llm_response.get("content", "")

            assistant_id: Optional[int] = None
            if final_text and final_text.strip():
                assistant_id = self.memory_manager.log_message(
                    user_identifier=user_identifier, persona_name=persona_name,
                    channel="",  # Not available in pending context
                    author_role='assistant', author_name=persona_name,
                    content=final_text, timestamp=datetime.now(),
                )

            return final_text, ResponseType.LLM_GENERATION, assistant_id, None

        except Exception as e:
            logger.error(f"Error resuming pending confirmation for {user_identifier}: {e}", exc_info=True)
            return ("An error occurred while processing the confirmed action.",
                    ResponseType.DEV_COMMAND, None, None)
