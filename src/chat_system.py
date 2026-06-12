# src/chat_system.py

import asyncio
import logging
import time
from contextlib import aclosing
from dataclasses import dataclass
from datetime import datetime
from typing import Any, AsyncGenerator, AsyncIterator, Coroutine, Dict, List, Optional, Set, Tuple

from config.global_config import PENDING_CONFIRMATION_TIMEOUT
from config.global_config import MAX_CACHED_API_REQUESTS  # noqa: F401  (re-export for tests)
from src.embedding_service import EmbeddingService
from src.clients.service_integration import ServiceIntegration
from src.confirmations import ConfirmationManager, PendingConfirmation as PendingConfirmation
from src.memory.backend.base import MemoryBackend, MemoryHit
from src.memory.memory_manager import MemoryManager
from src.engine import TextEngine
from src.generation_events import (
    DoneEvent as DoneEvent,
    ErrorEvent as ErrorEvent,
    GenerationEvent as GenerationEvent,
    PendingConfirmationEvent as PendingConfirmationEvent,
    ResponseType as ResponseType,
    TokenEvent as TokenEvent,
    ToolCallResultEvent as ToolCallResultEvent,
    ToolCallStartEvent as ToolCallStartEvent,
)
from src.message_handler import BotLogic
from src.persona import Persona
from src.request_builder import (
    AssembledRequest as AssembledRequest,
    RequestBuilder,
    RequestContext as RequestContext,
)
from src.request_builder import (  # noqa: F401  (re-exports for tests/back-compat)
    MAX_CONVERSATION_TAINTS as MAX_CONVERSATION_TAINTS,
    _relative_time as _relative_time,
)
from src.tools.tool_loop import ToolLoop, _ApiPayloadEvent, _LoopFinishedEvent
from src.turn_persistence import TurnPersistence
from src.tools.tool_manager import ToolManager
from src.tools.turn_context import TurnContext, turn_scope
from src.utils.save_utils import save_personas_to_file

logger = logging.getLogger(__name__)


@dataclass
class _ResumeState:
    """Carries a resumed write-confirmation into the orchestration kernel.

    `_orchestrate(resume=...)` re-enters with the parked turn instead of a
    fresh request: it skips dev-command preprocessing, history build, and
    user-turn logging, applies the operator's approve/deny decision to the
    parked history, then drives the tool loop + persistence tail through the
    same code path as a normal turn (DP-124).
    """
    pending: PendingConfirmation
    approved: bool


class ChatSystem:
    def __init__(self, memory_manager: MemoryManager, text_engine: TextEngine,
                 embedding_service: Optional[EmbeddingService] = None, *,
                 personas: Dict[str, Persona],
                 system_persona_names: Set[str],
                 tool_manager: ToolManager,
                 models_available: Optional[Dict[str, Any]] = None) -> None:
        # DP-200 slice B: persona loading and tool-handler registration live in
        # src/bootstrap (the composition root). ChatSystem receives its real
        # dependencies instead of locating them itself.
        self.personas: Dict[str, Persona] = personas
        self.system_persona_names: Set[str] = system_persona_names

        self.memory_manager: MemoryManager = memory_manager
        # DP-113: backend boundary for new-shape recall/retain_turn. The
        # MemoryManager owns construction (selector lives in global_config);
        # ChatSystem just borrows the reference + pushes the embedding service
        # into it so SqliteSemanticBackend.recall can translate query → embed.
        self.memory_backend: MemoryBackend = memory_manager.backend
        if embedding_service is not None and hasattr(self.memory_backend, "set_embedding_service"):
            self.memory_backend.set_embedding_service(embedding_service)
        self.text_engine: TextEngine = text_engine
        self.tool_manager: ToolManager = tool_manager

        self.bot_logic: BotLogic = BotLogic(self)
        self.turn_persistence: TurnPersistence = TurnPersistence(
            memory_manager, self.memory_backend,
        )
        # Injected by the composition root (src/bootstrap) so construction
        # stays filesystem-free; `update_models` rebinds it at runtime.
        self.models_available: Dict[str, Any] = models_available if models_available is not None else {}
        self.background_tasks: Set[Coroutine[Any, Any, Any]] = set()
        # Lookup closure over self (like request_builder's persona_lookup) so
        # post-init rebinds of `self.tool_manager` stay visible to resumes.
        self.confirmations: ConfirmationManager = ConfirmationManager(
            lambda: self.tool_manager, memory_manager,
        )
        # persona_lookup is a closure over self (not a dict reference) so
        # tests/admin paths that rebind `self.personas` stay visible.
        self.request_builder: RequestBuilder = RequestBuilder(
            memory_manager=memory_manager,
            memory_backend=self.memory_backend,
            tool_manager_lookup=lambda: self.tool_manager,
            persona_lookup=lambda name: self.personas.get(name),
            embedding_service=embedding_service,
        )
        self._services: Dict[str, ServiceIntegration] = {}
        self._embedding_service: Optional[EmbeddingService] = embedding_service

    @property
    def _pending_confirmations(self) -> Dict[Tuple[str, str], PendingConfirmation]:
        """Back-compat view of the confirmation store.

        The store itself lives on `self.confirmations` (DP-200 slice B);
        existing tests and the portal's transcript projection still address
        the map through this name.
        """
        return self.confirmations.pending

    def visible_personas(self) -> Dict[str, Persona]:
        """Personas safe to expose in user-facing listings (dropdowns, status text).

        System personas remain in `self.personas` so they are still routable when
        addressed by name, but are excluded from discovery surfaces — they are
        background workers, not user-selectable assistants.
        """
        return {
            name: persona
            for name, persona in self.personas.items()
            if name not in self.system_persona_names
        }

    def register_service(self, service: ServiceIntegration) -> None:
        """Register a service integration and its tools."""
        self._services[service.name] = service
        service.register_tools(self.tool_manager)
        logger.info(f"Registered service integration: {service.name}")

    def get_service(self, name: str) -> Optional[ServiceIntegration]:
        """Look up a registered service integration by name."""
        return self._services.get(name)

    @property
    def embedding_service(self) -> Optional[EmbeddingService]:
        """Shared embedding service injected at construction.

        None only in minimal setups (e.g. unit tests) that build ChatSystem
        without one; main.py always supplies it. Consumers that can fall back
        to constructing their own (SqliteConsolidator) must not write back —
        the backend only learns about the service at ChatSystem construction.
        """
        return self._embedding_service

    # ------------------------------------------------------------------
    # RequestBuilder delegation (DP-200 slice B). Request assembly lives in
    # src/request_builder.py; these thin seams keep the kernel's call sites
    # (and the tests that monkeypatch/address them) going through ChatSystem,
    # so live submits and the dry-run inspector share one code path.
    # ------------------------------------------------------------------

    @property
    def _conversation_taints(self) -> Dict[Tuple[str, str, str, Optional[str]], bool]:
        """Back-compat view of the sticky taint map (owned by RequestBuilder)."""
        return self.request_builder.conversation_taints

    def _set_conversation_taint(
            self, key: Tuple[str, str, str, Optional[str]], value: bool
    ) -> None:
        self.request_builder.set_conversation_taint(key, value)

    def _format_raw_history_for_llm(self, raw_history: List[Dict[str, Any]], memory_mode: str,
                                    persona_name: str, server_id: Optional[str]) -> List[Dict[str, Any]]:
        return self.request_builder.format_raw_history_for_llm(
            raw_history, memory_mode, persona_name, server_id,
        )

    def _build_conversation_history(
            self,
            persona: Persona,
            user_identifier: str,
            channel: str,
            server_id: Optional[str],
            history_limit: Optional[int]
    ) -> Tuple[List[Dict[str, Any]], Optional[int]]:
        return self.request_builder.build_conversation_history(
            persona, user_identifier, channel, server_id, history_limit,
        )

    async def _retrieve_memory_block(
            self,
            persona: Persona,
            user_identifier: str,
            channel: str,
            server_id: Optional[str],
            conversation_history: List[Dict[str, Any]],
            current_message: Optional[str] = None,
            oldest_interaction_id: Optional[int] = None,
    ) -> Tuple[Optional[str], bool]:
        return await self.request_builder.retrieve_memory_block(
            persona, user_identifier, channel, server_id, conversation_history,
            current_message=current_message,
            oldest_interaction_id=oldest_interaction_id,
        )

    @staticmethod
    def _format_memory_block(hits: List[MemoryHit]) -> Optional[str]:
        return RequestBuilder.format_memory_block(hits)

    def _filter_tools_for_persona(self, persona: Persona) -> List[Dict[str, Any]]:
        return self.request_builder.filter_tools_for_persona(persona)

    async def _prepare_request(self, ctx: RequestContext, is_retry: bool = False) -> None:
        await self.request_builder.prepare_request(ctx, is_retry=is_retry)

    def get_view_history(
            self,
            persona_name: str,
            user_identifier: str,
            channel: Optional[str],
            server_id: Optional[str] = None,
            limit: Optional[int] = None,
    ) -> Tuple[List[Dict[str, Any]], str]:
        """Raw history the way the engine would see it (DP-136 transcript seam)."""
        return self.request_builder.get_view_history(
            persona_name, user_identifier, channel, server_id=server_id, limit=limit,
        )

    async def get_session_memory_block(
            self,
            persona_name: str,
            user_identifier: str,
            channel: str,
            server_id: Optional[str],
            query: Optional[str] = None,
    ) -> Optional[str]:
        """Public LTM seam for interfaces that bypass generate_response (portal)."""
        return await self.request_builder.get_session_memory_block(
            persona_name, user_identifier, channel, server_id, query=query,
        )

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
        """Dry-run assembler (S5 parity seam) — see RequestBuilder.assemble_request."""
        return await self.request_builder.assemble_request(
            persona_name, user_identifier, channel, message,
            server_id=server_id, image_url=image_url,
            history_limit=history_limit,
            local_inference_config=local_inference_config,
            is_retry=is_retry, client_messages=client_messages,
        )

    # ------------------------------------------------------------------
    # TurnPersistence delegation (DP-200 slice B). The turn write-paths and
    # dump_history's request caches live in src/turn_persistence.py; these
    # seams keep the kernel call sites (and monkeypatching tests) stable.
    # ------------------------------------------------------------------

    @property
    def last_api_requests(self) -> Dict[str, Dict[str, Optional[Dict[str, Any]]]]:
        return self.turn_persistence.last_api_requests

    @last_api_requests.setter
    def last_api_requests(self, value: Dict[str, Dict[str, Optional[Dict[str, Any]]]]) -> None:
        self.turn_persistence.last_api_requests = value

    @property
    def last_api_iterations(self) -> Dict[str, Dict[str, List[Dict[str, Any]]]]:
        return self.turn_persistence.last_api_iterations

    @last_api_iterations.setter
    def last_api_iterations(self, value: Dict[str, Dict[str, List[Dict[str, Any]]]]) -> None:
        self.turn_persistence.last_api_iterations = value

    def _store_api_request(self, user_identifier: str, persona_name: str,
                           payload: Dict[str, Any],
                           tools_for_llm: Optional[List[Dict[str, Any]]] = None,
                           is_first_iteration: bool = False) -> None:
        self.turn_persistence.store_api_request(
            user_identifier, persona_name, payload,
            tools_for_llm=tools_for_llm, is_first_iteration=is_first_iteration,
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
        return self.turn_persistence.log_user_turn(
            is_retry=is_retry, persona_name=persona_name,
            user_identifier=user_identifier, channel=channel,
            user_display_name=user_display_name, message=message,
            server_id=server_id, platform_message_id=platform_message_id,
            timestamp=timestamp,
        )

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
        return self.turn_persistence.commit_or_update_assistant(
            persona_name=persona_name, user_identifier=user_identifier,
            channel=channel, server_id=server_id, final_text=final_text,
            response_type=response_type,
            user_interaction_id=user_interaction_id,
            retry_assistant_id=retry_assistant_id,
            tool_context_json=tool_context_json,
        )

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
            resume: Optional[_ResumeState] = None,
    ) -> AsyncGenerator[GenerationEvent, None]:
        """Shared streaming kernel — single source of truth for the request
        pipeline. Yields TokenEvent for each text delta, terminal DoneEvent
        with final ids, or ErrorEvent on failure. Phase C kernel; both
        `generate_response` (collect-stream wrapper) and `stream_response`
        (portal entry) delegate here.

        `resume` (DP-124) re-enters this kernel to continue a write that was
        parked for confirmation: dev-command preprocessing, history build, and
        user-turn logging are skipped; the parked history carries the turn and
        the approve/deny decision is applied before the tool loop runs.
        """
        # 1. Dev command preprocessing — short-circuits before any LLM call.
        #    Skipped on resume: there is no fresh user message to interpret.
        if resume is None:
            command_result: Optional[Dict[str, Any]] = await self.bot_logic.preprocess_message(
                persona_name, user_identifier, message
            )
            if command_result:
                if command_result.get("mutated", False):
                    save_personas_to_file(self.personas, self.system_persona_names)
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

        # DP-128: a persona quarantined for an insecure tool composition is
        # refused here — no LLM call, no tools — until its tools are fixed live.
        # Dev commands (e.g. `set tools ...`) are handled above before this gate,
        # so the operator can repair the persona in-band without a restart.
        if persona.is_security_blocked():
            reasons = persona.get_security_block_reasons()
            detail = "\n".join(f" - {r}" for r in reasons)
            yield DoneEvent(
                text=(
                    f"⚠️ Persona '{persona_name}' is quarantined (insecure tool "
                    f"composition):\n{detail}\n"
                    "Fix its tools in persona config to enable it."
                ),
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

        # On resume the parked turn replaces the freshly-built request state:
        # the conversation history already terminates in the assistant
        # tool_calls message (+ any read results), and the tool/taint context
        # carries over from when the write was parked.
        if resume is not None:
            ctx.conversation_history = resume.pending.conversation_history
            ctx.tools_for_llm = resume.pending.tools_for_llm
            ctx.turn_tainted = resume.pending.turn_tainted

        # DP-113: pin the active turn's scope so engine-side tools (e.g.
        # `recall_memory`) inherit persona/channel/user/server without those
        # showing up as model-callable args. turn_scope guarantees the
        # ContextVar is reset on *every* exit — post-loop exception, an
        # early-breaking consumer (GeneratorExit at a suspended yield), or
        # normal completion — so a stale scope never leaks into the next turn
        # sharing the event-loop context.
        with turn_scope(TurnContext(
            persona_name=persona_name,
            user_identifier=user_identifier,
            channel=channel,
            server_id=server_id,
        )):
            if resume is None:
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

                # DP-113: retain user turn through the backend boundary. Sqlite_legacy
                # is a noop (batch SqliteConsolidator still drives consolidation); Hindsight
                # enqueues fire-and-forget. Either way, retain_turn returns quickly
                # and does not block the LLM call below.
                if user_interaction_id is not None and message and message.strip():
                    await self.turn_persistence.retain_turn_safe(
                        persona_name=persona_name, role="user", content=message,
                        user_identifier=user_identifier, channel=channel,
                        server_id=server_id, timestamp=user_ts,
                        interaction_id=user_interaction_id, untrusted=False,
                    )
            else:
                # 2'. Resume: no fresh user turn. Apply the operator's
                #     approve/deny decision to the parked history (write results
                #     land *before* the continuation) and log the audit
                #     decision. Runs inside turn_scope so write-tool execution
                #     inherits the persona/channel/user scope.
                user_interaction_id = None
                # Carry the park's retry linkage so a retried turn's resumed
                # continuation UPDATEs the archived assistant row instead of
                # INSERTing a fresh one beside it.
                retry_assistant_id = resume.pending.retry_assistant_id
                try:
                    ctx.turn_tainted = await self.confirmations.apply_resume_decision(
                        resume.pending, resume.approved, ctx.conversation_history,
                        operator_id=ctx.user_identifier,
                        turn_tainted=ctx.turn_tainted,
                    )
                except Exception as e:
                    logger.error(
                        f"Error resuming pending confirmation for {user_identifier}: {e}",
                        exc_info=True,
                    )
                    yield ErrorEvent(
                        message="An error occurred while processing the confirmed action.",
                    )
                    return

            # 3. Tool loop. ToolLoop owns iteration + tool dispatch; this
            #    forwards Token / ToolCallStart / ToolCallResult events,
            #    siphons api_payload into the request cache, and unpacks the
            #    terminal _LoopFinishedEvent to drive CONFIRM-mode parking +
            #    assistant persistence.
            params = self.request_builder.resolve_generation_params(
                ctx.persona, ctx.local_inference_config,
            )
            params_first_iter = True
            final_text = ""
            response_type = ResponseType.LLM_GENERATION
            tool_context_json: Optional[str] = None
            accumulated_parts: List[str] = []
            pending_writes: Optional[List[Dict[str, Any]]] = None
            audit_info: Optional[Dict[str, Any]] = None
            tool_context_start: int = 0

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
                    turn_tainted=ctx.turn_tainted,
                    initial_taint_sources=ctx.taint_sources,
                    history_start_override=(
                        resume.pending.tool_context_start if resume is not None else None
                    ),
                ):
                    if isinstance(ev, _ApiPayloadEvent):
                        self._store_api_request(
                            user_identifier, persona_name, ev.payload,
                            tools_for_llm=ctx.tools_for_llm if params_first_iter else None,
                            is_first_iteration=params_first_iter,
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
                        audit_info = ev.audit_info
                        tool_context_start = ev.tool_context_start
                        ctx.turn_tainted = ev.turn_tainted
                        # Persist back to the conversation cache for stickiness
                        taint_key = (ctx.user_identifier, ctx.persona_name, ctx.channel, ctx.server_id)
                        self._set_conversation_taint(taint_key, ev.turn_tainted)
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

            # DP-130 history contract: the ephemeral_chunk_id for this turn's
            # rendered confirmation chunk. Non-None only on a parked turn — it is
            # the parked PendingConfirmation's correlation token. Carried on the
            # terminal DoneEvent so the id-frame can address the unpersisted chunk.
            ephemeral_chunk_id: Optional[str] = None
            if pending_writes is not None:
                parked = PendingConfirmation(
                    write_calls=pending_writes,
                    conversation_history=ctx.conversation_history,
                    persona_name=ctx.persona_name,
                    tools_for_llm=ctx.tools_for_llm,
                    image_url=ctx.image_url,
                    channel=ctx.channel,
                    server_id=ctx.server_id,
                    turn_tainted=ctx.turn_tainted,
                    audit_info=audit_info,
                    tool_context_start=tool_context_start,
                    confirmation_text=final_text if final_text else "",
                    retry_assistant_id=retry_assistant_id,
                )
                # Parking (incl. supersede-eviction + audit logging) is the
                # ConfirmationManager's job; the kernel only decides *when*.
                self.confirmations.park(ctx.user_identifier, ctx.persona_name, parked)
                ephemeral_chunk_id = parked.token
                # Surface the park to interactive consumers (the portal) so they
                # can render an approve/deny affordance and resume via the token.
                # Emitted before the terminal DoneEvent; non-interactive callers
                # (generate_response, the non-streaming resume drain) ignore it
                # and rely on the DoneEvent text instead.
                yield PendingConfirmationEvent(
                    text=final_text,
                    write_calls=pending_writes,
                    persona_name=ctx.persona_name,
                    token=parked.token,
                    audit_info=audit_info,
                )

            # 4. Log/update assistant turn. Original text (including links) is preserved.
            assistant_id = self._commit_or_update_assistant(
                persona_name=persona_name, user_identifier=user_identifier,
                channel=channel, server_id=server_id,
                final_text=final_text, response_type=response_type,
                user_interaction_id=user_interaction_id,
                retry_assistant_id=retry_assistant_id,
                tool_context_json=tool_context_json,
            )

            # DP-113: retain assistant turn through the backend boundary.
            # Inherit ctx.turn_tainted so the untrusted bit reaches the
            # store when the LLM consumed attacker-influenced tool output.
            if assistant_id is not None and final_text and final_text.strip() \
                    and response_type == ResponseType.LLM_GENERATION:
                await self.turn_persistence.retain_turn_safe(
                    persona_name=persona_name, role="assistant", content=final_text,
                    user_identifier=user_identifier, channel=channel,
                    server_id=server_id, timestamp=datetime.now(),
                    interaction_id=assistant_id, untrusted=ctx.turn_tainted,
                )

            yield DoneEvent(
                text=final_text if final_text else "",
                response_type=response_type,
                assistant_id=assistant_id,
                user_interaction_id=user_interaction_id,
                ephemeral_chunk_id=ephemeral_chunk_id,
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
        # aclosing: if the consumer stops early (client disconnect, break),
        # tearing down this generator must propagate aclose() into the inner
        # _orchestrate so its turn_scope finally runs — a plain `async for`
        # delegation leaves the sub-generator suspended and leaks the scope.
        async with aclosing(self._orchestrate(
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
        )) as agen:
            async for ev in agen:
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
        async with aclosing(self._orchestrate(
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
        )) as agen:
            async for ev in agen:
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

    async def stream_resume_confirmation(
            self, user_identifier: str, persona_name: str, approved: bool,
            *, expected_token: Optional[str] = None,
    ) -> AsyncGenerator[GenerationEvent, None]:
        """Streaming resume of a parked write — single source of truth for the
        approve/deny continuation.

        DP-124 re-enters the `_orchestrate` kernel with the parked turn so the
        continuation runs the full tool loop, persists the assistant row on the
        correct channel, and shares the single turn lifecycle (scope, taint
        write-back, retain, terminal event). DP-127 makes this streaming so the
        portal can render the continuation (and any *chained* confirmation) just
        like a normal turn. The not-found / expired / persona-missing guards are
        surfaced as a terminal DEV_COMMAND DoneEvent so every consumer renders
        them uniformly.

        `expected_token` (the portal path) rejects a resume whose token no longer
        matches the live park — i.e. the model proposed a different write since.
        It is validated *before* the park is consumed, so a stale request leaves
        the real pending confirmation intact. Discord passes None and resumes by
        (user, persona) alone.
        """
        key = (user_identifier, persona_name)
        pending = self._pending_confirmations.get(key)

        if not pending:
            yield DoneEvent(
                text="No pending confirmation found.",
                response_type=ResponseType.DEV_COMMAND,
            )
            return

        if expected_token is not None and pending.token != expected_token:
            yield DoneEvent(
                text="This confirmation is no longer valid. Please try again.",
                response_type=ResponseType.DEV_COMMAND,
            )
            return

        # Commit to acting on this park: remove it so a duplicate approve/deny
        # (double-click, retried POST) can't execute the writes twice.
        self._pending_confirmations.pop(key, None)

        if time.time() - pending.created_at > PENDING_CONFIRMATION_TIMEOUT:
            yield DoneEvent(
                text="Confirmation expired. Please try again.",
                response_type=ResponseType.DEV_COMMAND,
            )
            return

        if pending.persona_name not in self.personas:
            yield DoneEvent(
                text="Error: Persona not found.",
                response_type=ResponseType.DEV_COMMAND,
            )
            return

        async with aclosing(self._orchestrate(
            persona_name=pending.persona_name,
            user_identifier=user_identifier,
            channel=pending.channel,
            message="",
            server_id=pending.server_id,
            image_url=pending.image_url,
            resume=_ResumeState(pending=pending, approved=approved),
        )) as agen:
            async for ev in agen:
                yield ev

    async def resume_pending_confirmation(
            self, user_identifier: str, persona_name: str, approved: bool
    ) -> Tuple[str, ResponseType, Optional[int], Optional[int]]:
        """Non-streaming resume — drains `stream_resume_confirmation` into the
        4-tuple Discord expects. Token validation is skipped (Discord keys the
        confirmation by reaction on a specific message, so a stale token can't
        arise the way it can over HTTP).
        """
        final_text = ""
        response_type = ResponseType.DEV_COMMAND
        assistant_id: Optional[int] = None
        async with aclosing(self.stream_resume_confirmation(
            user_identifier, persona_name, approved,
        )) as agen:
            async for ev in agen:
                if isinstance(ev, (TokenEvent, ToolCallStartEvent,
                                   ToolCallResultEvent, PendingConfirmationEvent)):
                    continue
                if isinstance(ev, DoneEvent):
                    final_text = ev.text
                    response_type = ev.response_type
                    assistant_id = ev.assistant_id
                elif isinstance(ev, ErrorEvent):
                    final_text = ev.message
                    response_type = ResponseType.DEV_COMMAND
                    assistant_id = None
        return final_text, response_type, assistant_id, None

    async def startup(self) -> None:
        """Post-init async startup tasks (e.g. Hindsight memory bank provisioning)."""
        from src.memory.backend import HindsightBackend
        if not isinstance(self.memory_backend, HindsightBackend):
            return
        # Only personas that converse with users get a bank; system personas
        # (model_selector, triage_*, etc.) are single-shot pipeline workers
        # with no accumulating chat history — provisioning would just create
        # empty banks. Gate on `long_term_memory`.
        targets = [n for n, p in self.personas.items() if p.get_long_term_memory()]
        if not targets:
            return
        logger.info(f"Initializing Hindsight memory banks for {len(targets)} persona(s)...")

        async def _ensure(name: str) -> None:
            try:
                await self.memory_backend.ensure_bank(bank_id=name)
            except Exception as e:
                logger.warning(f"Could not ensure Hindsight bank for {name}: {e}")

        await asyncio.gather(*(_ensure(n) for n in targets))
