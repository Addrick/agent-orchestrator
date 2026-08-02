# src/chat_system.py

import asyncio
import logging
from contextlib import aclosing
from dataclasses import dataclass
from datetime import datetime
from typing import Any, AsyncGenerator, AsyncIterator, Coroutine, Dict, List, Optional, Set, Tuple

from src.embedding_service import EmbeddingService
from src.clients.service_integration import ServiceIntegration
from src.confirmations import ConfirmationManager, Decision, ParkedWrite
from src.memory.backend.base import MemoryBackend
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
    format_internal_error as format_internal_error,
)
from src.message_handler import BotLogic
from src.origin import ANONYMOUS, Origin
from src.persona import Persona
from src.request_builder import AssembledRequest, RequestBuilder, RequestContext
from src.security.scrubber import get_scrubber
from src.tools.tool_loop import (
    ToolLoop, WriteParkedEvent, _ApiPayloadEvent, _LoopFinishedEvent,
    _ToolContextEvent, write_call_identity,
)
from src.turn_persistence import TurnPersistence
from src.tools.tool_manager import ToolManager
from src.tools.turn_context import TurnContext, turn_scope
from src.personas.store import save_personas_to_file

logger = logging.getLogger(__name__)


@dataclass
class _ContinuationState:
    """Carries a batch of just-resolved parks into the orchestration kernel.

    `_orchestrate(continuation=...)` runs an ordinary turn that happens to log
    no user row: history is rebuilt LIVE from the DB (where each resolved
    write's entry has already been patched with its real outcome), so the model
    reads what actually happened and summarizes it.

    This replaced DP-124's `_ResumeState`, which instead replayed the parked
    turn's *snapshot* of history. A snapshot cannot survive DP-297's bursts —
    with several parks resolvable in any order, every snapshot predates its
    siblings, so replaying one forks the conversation.
    """
    batch: List[Decision]


def _render_resolution_nudge(batch: List[Decision]) -> str:
    """The synthetic user turn that opens a continuation.

    Exists for a wire-level reason, not a prompt-engineering one: without it the
    message array would end on the parked turn's assistant message, which
    Anthropic treats as a prefill to continue rather than a turn to answer — the
    model would resume its old sentence instead of reporting the outcome.

    Deliberately NOT persisted. `prepare_request` appends it to the in-memory
    history only; the continuation skips `log_user_turn`, so the durable record
    of what happened stays the patched tool entries rather than words the
    operator never typed.
    """
    lines = []
    for decision in batch:
        tool_name = decision.park.write_call.get("name") or "action"
        if not decision.approved:
            lines.append(f"- denied: {tool_name}")
        elif decision.ok:
            lines.append(f"- approved and executed: {tool_name}")
        else:
            lines.append(f"- approved but FAILED: {tool_name}")
    verb = "action" if len(lines) == 1 else "actions"
    return (
        f"[The operator reviewed {len(lines)} pending {verb}:]\n"
        + "\n".join(lines)
        + "\n[Results are in the tool context above. Report the outcome "
          "briefly. Do not re-propose an action that was already decided, "
          "whether it was approved or denied.]"
    )


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

        self.turn_persistence: TurnPersistence = TurnPersistence(
            memory_manager, self.memory_backend,
        )
        # Injected by the composition root (src/bootstrap) so construction
        # stays filesystem-free; `update_models` (BotLogic) and main.py's
        # refresh loop rebind it at runtime.
        self.models_available: Dict[str, Any] = models_available if models_available is not None else {}
        # DP-202: BotLogic takes explicit deps instead of the whole ChatSystem.
        # Rebindable collaborators go in as closures over self so post-init
        # swaps (tests, admin paths) stay visible to the command layer.
        self.bot_logic: BotLogic = BotLogic(
            personas=lambda: self.personas,
            visible_personas=self.visible_personas,
            text_engine=lambda: self.text_engine,
            tool_manager=lambda: self.tool_manager,
            turn_persistence=self.turn_persistence,
            memory_manager=memory_manager,
            get_models_available=lambda: self.models_available,
            set_models_available=lambda models: setattr(self, "models_available", models),
        )
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
    # Public request-assembly API. Request assembly lives in
    # src/request_builder.py; these delegates are the supported external
    # surface (portal/transcript/dry-run inspector) so live submits and the
    # inspector share one code path. Internal callers address
    # `self.request_builder` directly (DP-201b removed the private seams).
    # ------------------------------------------------------------------

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
            continuation: Optional[_ContinuationState] = None,
            origin: Optional[Origin] = None,
    ) -> AsyncGenerator[GenerationEvent, None]:
        """Shared streaming kernel — single source of truth for the request
        pipeline. Yields TokenEvent for each text delta, terminal DoneEvent
        with final ids, or ErrorEvent on failure. Phase C kernel; both
        `generate_response` (collect-stream wrapper) and `stream_response`
        (portal entry) delegate here.

        `continuation` (DP-297) runs the summary turn after an operator
        resolved one or more gated writes: dev-command preprocessing and
        user-turn logging are skipped, but history is built normally — the
        resolved writes are already patched into it.
        """
        # 1. Dev command preprocessing — short-circuits before any LLM call.
        #    Skipped on a continuation: there is no fresh user message to
        #    interpret, only the synthetic nudge built by the caller.
        #    DP-277: callers that don't assert an authenticated origin get
        #    ANONYMOUS (operator=False) — control-plane commands are refused.
        if continuation is None:
            command_result: Optional[Dict[str, Any]] = await self.bot_logic.preprocess_message(
                origin or ANONYMOUS, persona_name, user_identifier, message
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
            try:
                await self.request_builder.prepare_request(
                    ctx, is_retry=is_retry and continuation is None,
                )
            except Exception as e:
                err_id, err_msg = format_internal_error(e, scrub=get_scrubber().scrub)
                logger.error(
                    f"[err {err_id}] prepare_request failed for "
                    f"{user_identifier}: {e}", exc_info=True,
                )
                yield ErrorEvent(message=err_msg)
                return

            if continuation is None:
                # 2. Log user turn (or archive for retry). Done after history is built
                #    (so the freshly-inserted row doesn't show up twice) but before
                #    the LLM call so the user row is always pinned even if the model
                #    errors mid-flight.
                user_ts = timestamp or datetime.now()
                user_interaction_id, retry_assistant_id = self.turn_persistence.log_user_turn(
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
                # 2'. Continuation: the operator's decisions were already
                #     executed and patched into history by the caller, so there
                #     is nothing to apply here and no user row to log. The
                #     synthetic nudge in `message` rode into the wire array via
                #     prepare_request above but is deliberately NOT persisted —
                #     the patched tool entries are the durable record.
                user_interaction_id = None
                retry_assistant_id = None
                # Inherit taint from any approved write that produced untrusted
                # output, so the summary turn is marked like the turn that
                # proposed it.
                if any(d.park.turn_tainted for d in continuation.batch):
                    ctx.turn_tainted = True

            # 3. Tool loop. ToolLoop owns iteration + tool dispatch; this
            #    forwards Token / ToolCallStart / ToolCallResult events,
            #    siphons api_payload into the request cache, collects gated
            #    writes, and unpacks the terminal _LoopFinishedEvent to drive
            #    assistant persistence.
            params = self.request_builder.resolve_generation_params(
                ctx.persona, ctx.local_inference_config,
            )
            params_first_iter = True
            final_text = ""
            response_type = ResponseType.LLM_GENERATION
            tool_context_json: Optional[str] = None
            accumulated_parts: List[str] = []
            # Writes this turn gated for approval. Registered in the store only
            # after the assistant row commits, since each needs that row's id to
            # patch later — see the park-registration block below.
            parks_this_turn: List[ParkedWrite] = []

            def _already_pending(write_call: Dict[str, Any]) -> Optional[str]:
                """Token of an identical proposal still awaiting the operator.

                Spans both scopes on purpose: parks made earlier in THIS turn
                (not yet in the store — they register only after the assistant
                row commits) and parks still live from earlier turns. The
                cross-turn case is the one that matters most: a continuation
                exists because something was decided, and the model re-reading
                its own still-pending siblings is exactly when it re-proposes.
                """
                identity = write_call_identity(write_call)
                for park in parks_this_turn:
                    if write_call_identity(park.write_call) == identity:
                        return park.token
                for park in self.confirmations.list_for(
                        ctx.user_identifier, ctx.persona_name):
                    if write_call_identity(park.write_call) == identity:
                        return park.token
                return None

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
                    pending_lookup=_already_pending,
                ):
                    if isinstance(ev, _ApiPayloadEvent):
                        self.turn_persistence.store_api_request(
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
                    elif isinstance(ev, _ToolContextEvent):
                        tool_context_json = ev.tool_context_json
                    elif isinstance(ev, WriteParkedEvent):
                        # DP-297: a gated write, mid-turn. Hold it — the store
                        # registration needs the assistant row id that does not
                        # exist until this turn commits — but surface it now so
                        # an interactive client can render the affordance in
                        # stream order.
                        parks_this_turn.append(ParkedWrite(
                            token=ev.token,
                            write_call=ev.write_call,
                            audit_info=ev.audit_info,
                            confirmation_text=ev.confirmation_text,
                            user_identifier=ctx.user_identifier,
                            persona_name=ctx.persona_name,
                            channel=ctx.channel,
                            server_id=ctx.server_id,
                            turn_tainted=ev.turn_tainted,
                        ))
                        yield PendingConfirmationEvent(
                            text=ev.confirmation_text,
                            write_calls=[ev.write_call],
                            persona_name=ctx.persona_name,
                            token=ev.token,
                            audit_info=ev.audit_info,
                        )
                    elif isinstance(ev, ErrorEvent):
                        # The loop died mid-turn. Persist whatever tool calls it
                        # made first — otherwise the next turn shows the model
                        # its own prose with no trace of the call that failed,
                        # and it re-proposes or hallucinates the action.
                        if tool_context_json:
                            errored_id = self.turn_persistence.commit_or_update_assistant(
                                persona_name=persona_name,
                                user_identifier=user_identifier,
                                channel=channel, server_id=server_id,
                                final_text="".join(accumulated_parts),
                                response_type=ResponseType.LLM_GENERATION,
                                user_interaction_id=user_interaction_id,
                                retry_assistant_id=retry_assistant_id,
                                tool_context_json=tool_context_json,
                            )
                            # Writes gated before the loop died are still real
                            # proposals — register them against the row that
                            # just captured their `awaiting_human_approval`
                            # entries, or the operator sees affordances that
                            # resolve to nothing.
                            self._register_parks(parks_this_turn, errored_id)
                        yield ev
                        return
                    elif isinstance(ev, _LoopFinishedEvent):
                        final_text = ev.final_text
                        response_type = ev.response_type
                        tool_context_json = ev.tool_context_json
                        ctx.turn_tainted = ev.turn_tainted
                        # Persist back to the conversation cache for stickiness
                        taint_key = (ctx.user_identifier, ctx.persona_name, ctx.channel, ctx.server_id)
                        self.request_builder.set_conversation_taint(taint_key, ev.turn_tainted)
            except asyncio.CancelledError:
                # Client disconnect / abort. Flush whatever assistant text has
                # accumulated so the row reflects what the user actually saw,
                # then re-raise so the surrounding StreamingResponse aborts.
                partial = "".join(accumulated_parts)
                if partial.strip():
                    self.turn_persistence.commit_or_update_assistant(
                        persona_name=persona_name, user_identifier=user_identifier,
                        channel=channel, server_id=server_id,
                        final_text=partial,
                        response_type=ResponseType.LLM_GENERATION,
                        user_interaction_id=user_interaction_id,
                        retry_assistant_id=retry_assistant_id,
                        tool_context_json=None,
                    )
                raise

            # 4. Log/update assistant turn. Original text (including links)
            #    is preserved. Since DP-297 a gated write no longer diverts this
            #    into a separate park-only row: a turn that proposed writes still
            #    ends with real text, so it persists like any other turn and the
            #    proposals hang off that row's tool_context.
            assistant_id = self.turn_persistence.commit_or_update_assistant(
                persona_name=persona_name, user_identifier=user_identifier,
                channel=channel, server_id=server_id,
                final_text=final_text, response_type=response_type,
                user_interaction_id=user_interaction_id,
                retry_assistant_id=retry_assistant_id,
                tool_context_json=tool_context_json,
            )
            self._register_parks(parks_this_turn, assistant_id)

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
                # No ephemeral chunk: since DP-297 the turn's own text is
                # persisted normally and each proposal carries its own token on
                # its PendingConfirmationEvent instead.
                ephemeral_chunk_id=None,
            )

    def _register_parks(self, parks: List[ParkedWrite],
                        assistant_id: Optional[int]) -> None:
        """Make this turn's gated writes resolvable, bound to the row that
        holds their `awaiting_human_approval` entries.

        Registration is deliberately deferred to here rather than done when the
        loop emits each park: `parked_assistant_id` is the row this turn is
        only now committing, and a park registered without it cannot be patched
        when it resolves — the operator would approve a write whose history
        entry says "pending" forever.

        The cost is a short window between the surface rendering an affordance
        (mid-stream) and the token becoming resolvable (here). A click inside it
        is refused with "no such pending action" and the operator clicks again;
        it fails closed and never executes the wrong thing.
        """
        for parked in parks:
            parked.parked_assistant_id = assistant_id
            self.confirmations.park(parked)

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
            origin: Optional[Origin] = None,
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
            origin=origin,
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
            local_inference_config: Optional[Dict[str, Any]] = None,
            origin: Optional[Origin] = None,
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
            origin=origin,
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

    async def stream_resolve_park(
            self, user_identifier: str, persona_name: str, token: str,
            approved: bool, *, note: Optional[str] = None,
    ) -> AsyncGenerator[GenerationEvent, None]:
        """Approve or deny ONE gated write, then summarize (DP-297).

        Single entry point for every surface. The token is mandatory: with
        several writes resolvable per conversation, `(user, persona)` no longer
        identifies one. (It was optional before, on the reasoning that Discord
        keyed off a specific message so a stale token could not arise — true
        only while at most one park existed.)

        Sequence, and why it is this order:

        1. `take()` the park — synchronous, so exactly one caller can win it and
           a double-click cannot execute the write twice.
        2. Acquire the conversation lock, then drain. Whoever holds the lock
           folds in every decision that arrived while it waited, so a flurry of
           approvals yields one summary rather than N racing tool loops over the
           same history.
        3. Execute + patch history for each decision, in approval order.
        4. Run ONE continuation turn on the freshly-rebuilt history.
        """
        key = (user_identifier, persona_name)
        parked = self.confirmations.take(token)

        if parked is None:
            yield DoneEvent(
                text="No such pending action — it was already resolved or it expired.",
                response_type=ResponseType.DEV_COMMAND,
            )
            return

        if parked.key != key:
            # A token belonging to another conversation must not be resolvable
            # from this one even if the caller somehow knows the hex.
            self.confirmations.restore(parked)
            yield DoneEvent(
                text="No such pending action.",
                response_type=ResponseType.DEV_COMMAND,
            )
            return

        if self.confirmations.is_expired(parked):
            self.confirmations.patch_parked_entry(
                parked, "expired", {"reason": "expired before review"},
            )
            yield DoneEvent(
                text="That action expired before it was reviewed.",
                response_type=ResponseType.DEV_COMMAND,
            )
            return

        if parked.persona_name not in self.personas:
            yield DoneEvent(
                text="Error: Persona not found.",
                response_type=ResponseType.DEV_COMMAND,
            )
            return

        self.confirmations.enqueue(
            Decision(park=parked, approved=approved, note=note),
        )

        applied: List[Decision] = []
        async with self.confirmations.lock_for(key):
            batch = self.confirmations.drain(key)
            if not batch:
                # A continuation that held the lock before us already folded
                # this decision in and acted on it. Nothing left to do.
                return

            try:
                # Re-drain after each round. Acquiring an uncontended
                # asyncio.Lock does not suspend, so the winner of a race gets
                # here before the loser has even enqueued — draining once would
                # leave the loser to run a second continuation over the same
                # history. Looping until the queue is empty is what actually
                # folds a flurry of approvals into one summary.
                while batch:
                    for decision in batch:
                        await self.confirmations.apply(decision)
                        applied.append(decision)
                    batch = self.confirmations.drain(key)
            except Exception as e:
                err_id, err_msg = format_internal_error(
                    e, scrub=get_scrubber().scrub,
                )
                logger.error(
                    f"[err {err_id}] Error applying approval decision for "
                    f"{user_identifier}: {e}", exc_info=True,
                )
                yield ErrorEvent(
                    message=f"Error processing the confirmed action. {err_msg}",
                )
                return

            async with aclosing(self._orchestrate(
                persona_name=parked.persona_name,
                user_identifier=user_identifier,
                channel=parked.channel,
                message=_render_resolution_nudge(applied),
                server_id=parked.server_id,
                continuation=_ContinuationState(batch=applied),
            )) as agen:
                async for ev in agen:
                    yield ev

    async def resolve_park(
            self, user_identifier: str, persona_name: str, token: str,
            approved: bool, *, note: Optional[str] = None,
    ) -> Tuple[str, ResponseType, Optional[int], Optional[int]]:
        """Non-streaming resolve — drains `stream_resolve_park` into the
        4-tuple Discord expects.
        """
        final_text = ""
        response_type = ResponseType.DEV_COMMAND
        assistant_id: Optional[int] = None
        async with aclosing(self.stream_resolve_park(
            user_identifier, persona_name, token, approved, note=note,
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

    async def provision_persona_memory(self, name: str) -> None:
        """Provision the Hindsight memory bank for a specific persona."""
        from src.memory.backend import HindsightBackend
        if not isinstance(self.memory_backend, HindsightBackend):
            return
        backend: HindsightBackend = self.memory_backend
        if name not in self.personas:
            return
        persona = self.personas[name]
        if not persona.get_long_term_memory():
            return

        try:
            # retain_mission / reflect_mission are honoured only at bank
            # CREATION (ensure_bank → acreate_bank, 409-noop if it exists);
            # observations_mission / enable_observations are seeded here too.
            # DP-255.
            await backend.ensure_bank(
                bank_id=name,
                retain_mission=persona.get_retain_mission(),
                reflect_mission=persona.get_reflect_mission(),
                enable_observations=persona.get_enable_observations(),
                observations_mission=persona.get_observations_mission(),
            )
        except Exception as e:
            logger.warning(f"Could not ensure Hindsight bank for {name}: {e}")
            return

        # disposition is LIVE-patchable (apatch_bank_config) so it applies to
        # existing banks without a rebuild, unlike the retain mission. DP-255.
        disposition = persona.get_disposition()
        if not disposition:
            return
        patch = {f"disposition_{k}": v for k, v in disposition.items()}
        try:
            await backend._get_client().apatch_bank_config(name, patch)
        except Exception as e:
            logger.warning(f"Could not patch disposition for Hindsight bank {name}: {e}")

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

        await asyncio.gather(*(self.provision_persona_memory(n) for n in targets))
