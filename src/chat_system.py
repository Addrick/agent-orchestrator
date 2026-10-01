# src/chat_system.py

import asyncio
import logging
from contextlib import aclosing
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import (
    Any, AsyncGenerator, AsyncIterator, Coroutine, Dict, FrozenSet, List, Optional,
    Set, Tuple,
)

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
from src.deferral_kinds import DEFERRAL_KIND_APPROVAL
from src.request_builder import AssembledRequest, RequestBuilder, RequestContext
from src.security.scrubber import get_scrubber
from src.tools.tool_loop import (
    PARK_STATUS_EXPIRED, ToolDeferredEvent, ToolLoop, _ApiPayloadEvent,
    _LoopFinishedEvent, _ToolContextEvent, _WriteDuplicateEvent,
    write_call_identity,
)
from src.turn_persistence import TurnPersistence
from src.tool_policy import callable_tool_names
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

    That non-persistence is the property both re-derived wake paths lost. fixr
    and DP-343 entered through `generate_response`, where `continuation is None`
    and `_orchestrate` therefore calls `log_user_turn` — so their whole
    synthetic wake text, instruction block included, was written to durable
    history as a **user row**, under the operator's real Discord id for DP-343.
    Every later turn in that channel then re-read "Nobody asked you this…" as
    something the operator had typed. The rendering is per-kind; the refusal to
    persist it is not, and that is why there is one of these.
    """
    lines = []
    for decision in batch:
        park = decision.park
        tool_name = park.write_call.get("name") or "action"
        if park.kind != DEFERRAL_KIND_APPROVAL:
            # Nobody decided anything here — the work finished somewhere else.
            # Saying "approved" would credit the operator with a judgement they
            # were never asked for, and the model would report it that way.
            lines.append(f"- {tool_name} ({park.kind}): {decision.status}")
        elif not decision.approved:
            lines.append(f"- denied: {tool_name}")
        elif decision.ok:
            lines.append(f"- approved and executed: {tool_name}")
        else:
            lines.append(f"- approved but FAILED: {tool_name}")
        if not decision.patched:
            # History still reads the `awaiting` placeholder for this call, so
            # the model is about to see its own proposal as pending and would
            # otherwise report the wrong thing. The nudge is ephemeral, but it
            # is the only channel left once the durable one has failed.
            lines[-1] += (" (its history entry could not be updated — trust "
                          "this line, not the tool context)")

    verb = "action" if len(lines) == 1 else "actions"
    if all(d.park.kind == DEFERRAL_KIND_APPROVAL for d in batch):
        header = f"[The operator reviewed {len(lines)} pending {verb}:]"
    elif any(d.park.kind == DEFERRAL_KIND_APPROVAL for d in batch):
        # A mixed batch: an operator click and a job finishing folded into one
        # continuation by the conversation lock. Attributing all of it to the
        # operator would be false for half the list.
        header = f"[{len(lines)} pending {verb} resolved:]"
    else:
        header = f"[{len(lines)} deferred {verb} finished:]"
    return (
        header + "\n"
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
            lambda: self.tool_manager, memory_manager, self._offered_tool_names,
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

    def _offered_tool_names(self, persona_name: str) -> FrozenSet[str]:
        """What `persona_name` is offered right now (DP-404) — the same
        filter a turn uses, so an approval re-check and a live turn cannot
        disagree. An unknown persona is offered nothing."""
        persona = self.personas.get(persona_name)
        if persona is None:
            return frozenset()
        return callable_tool_names(self.request_builder.filter_tools_for_persona(persona))

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
        #    DP-330: the same call also applies the persona origin allowlist —
        #    a disallowed origin gets a refusal here, before the LLM and before
        #    any dev command runs. The gate lives in `preprocess_message`
        #    rather than in this kernel because the Discord and portal adapters
        #    resolve dev commands through that seam WITHOUT entering the
        #    kernel; putting it here gated only one of the three surfaces.
        #    Skipping preprocessing on a continuation therefore also exempts
        #    the resumed turn from the addressing check, which is intended:
        #    the decision was made on the turn that raised the park, and the
        #    approved write has already executed by the time we get here.
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
            memory_mode=persona.get_memory_mode(),
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
                        persona_name=persona_name, role="user",
                        speaker=user_display_name or user_identifier, content=message,
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
            # What goes to the memory bank when that is not the whole reply.
            # Only the DP-335 exhaustion exit sets it — see `_LoopFinishedEvent`.
            retain_text: Optional[str] = None
            accumulated_parts: List[str] = []
            # Writes this turn gated for approval. Registered in the store only
            # after the assistant row commits, since each needs that row's id to
            # patch later — see the park-registration block below.
            parks_this_turn: List[ParkedWrite] = []
            # Writes the duplicate guard suppressed this turn. Registered
            # against the committed row alongside the parks, so their
            # `duplicate_of_pending` entries get corrected when the original
            # proposal resolves.
            dups_this_turn: List[_WriteDuplicateEvent] = []

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

            def _already_resolved(
                    write_call: Dict[str, Any]) -> Optional[Dict[str, Any]]:
                """The decided twin of this proposal, if there is a recent one.

                Covers what `_already_pending` structurally cannot: on a
                continuation turn the park being resolved has already been
                taken, so it is in neither scope above — and that is the turn
                where the model is re-reading its own tool span and most likely
                to re-propose. Durable since DP-319, which is why this lookup
                can exist at all.

                CONTINUATION TURNS ONLY, and that scoping is the whole
                correctness argument. Unlike the pending guard, this one
                suppresses with no affordance at all: no park, no
                `PendingConfirmationEvent`, nothing on Discord or the portal.
                Left running on ordinary turns it would answer "restart that
                service again" — four minutes after the first restart hung —
                with "that already happened", silently, for fifteen minutes,
                breaking the property `user_guide.md` states for the pending
                guard ("a persona can legitimately propose the same action
                again later"). A continuation is the only turn nobody asked
                for, so it is the only turn where a re-proposal can be assumed
                to be the model re-reading itself rather than a human meaning
                it.
                """
                if continuation is None:
                    return None
                return self.confirmations.already_resolved(
                    (ctx.user_identifier, ctx.persona_name), write_call,
                )

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
                    resolved_lookup=_already_resolved,
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
                    elif isinstance(ev, ToolDeferredEvent):
                        # DP-297: a deferred call, mid-turn. Hold it — the store
                        # registration needs the assistant row id that does not
                        # exist until this turn commits — but surface it now so
                        # an interactive client can render the affordance in
                        # stream order.
                        #
                        # This is where a deferral gets its coordinates, and
                        # they come from `ctx` — the turn that raised it — not
                        # from configuration. Every kind is bound here, so no
                        # kind ever needs to be told where to answer.
                        parks_this_turn.append(ParkedWrite(
                            token=ev.token,
                            write_call=ev.write_call,
                            audit_info=ev.audit_info,
                            confirmation_text=ev.confirmation_text,
                            user_identifier=ctx.user_identifier,
                            persona_name=ctx.persona_name,
                            channel=ctx.channel,
                            server_id=ctx.server_id,
                            kind=ev.kind,
                            turn_tainted=ev.turn_tainted,
                        ))
                        if ev.kind == DEFERRAL_KIND_APPROVAL:
                            # Only an approval has something for a human to
                            # decide. Emitting this for every kind would put an
                            # approve/deny affordance in front of the operator
                            # for a node job that nobody is being asked about,
                            # and clicking it would resolve the deferral early.
                            yield PendingConfirmationEvent(
                                text=ev.confirmation_text,
                                write_calls=[ev.write_call],
                                persona_name=ctx.persona_name,
                                token=ev.token,
                                audit_info=ev.audit_info,
                            )
                    elif isinstance(ev, _WriteDuplicateEvent):
                        # Suppressed by the pending-duplicate guard: no
                        # affordance, no audit row — but its history entry will
                        # need correcting when the original resolves.
                        dups_this_turn.append(ev)
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
                            # After the parks: a duplicate can point at a park
                            # created earlier in THIS turn, which only became
                            # findable by token on the line above.
                            self._register_duplicates(dups_this_turn, errored_id)
                        yield ev
                        return
                    elif isinstance(ev, _LoopFinishedEvent):
                        final_text = ev.final_text
                        response_type = ev.response_type
                        tool_context_json = ev.tool_context_json
                        retain_text = ev.retain_text
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
            self._register_duplicates(dups_this_turn, assistant_id)

            # DP-113: retain assistant turn through the backend boundary.
            # Inherit ctx.turn_tainted so the untrusted bit reaches the
            # store when the LLM consumed attacker-influenced tool output.
            #
            # What is PERSISTED is the whole reply; what is EMBEDDED can be
            # less. `retain_text` is the loop's opt-out for machine-generated
            # text appended to a real answer — today only DP-335's tool-call
            # footer, which is ground truth for the reader but would become a
            # recallable "memory" of tool names and arguments if embedded.
            to_retain = retain_text if retain_text is not None else final_text
            if assistant_id is not None and to_retain and to_retain.strip() \
                    and response_type == ResponseType.LLM_GENERATION \
                    and assistant_id == retry_assistant_id:
                # DP-409: a retry replaced this row in place, so Hindsight gets
                # the rebuilt canonical session, not the new attempt appended
                # beside the discarded one.
                await self.turn_persistence.rebuild_session_safe(
                    persona_name=persona_name, channel=channel,
                    user_identifier=user_identifier, server_id=server_id,
                    retried_id=assistant_id, retried_text=to_retain,
                    untrusted=ctx.turn_tainted,
                )
            elif assistant_id is not None and to_retain and to_retain.strip() \
                    and response_type == ResponseType.LLM_GENERATION:
                await self.turn_persistence.retain_turn_safe(
                    persona_name=persona_name, role="assistant",
                    speaker=persona_name, content=to_retain,
                    user_identifier=user_identifier, channel=channel,
                    server_id=server_id, timestamp=datetime.now(timezone.utc),
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

        A `None` row id fails closed the same way, and for a stronger reason:
        without it there is nothing to patch when the write resolves, so
        registering anyway would let an operator approve an irreversible action
        whose only record — the `awaiting_human_approval` entry — was never
        committed. Dropping the park costs the operator a refused click; keeping
        it costs a write that happened and that history never mentions.
        """
        if assistant_id is None:
            logger.error(
                "Assistant row missing for %d gated write(s); dropping them "
                "rather than making them resolvable against no history entry. "
                "Tokens: %s", len(parks), [p.token for p in parks],
            )
            return
        for parked in parks:
            parked.parked_assistant_id = assistant_id
            self.confirmations.park(parked)

    def _register_duplicates(self, dups: List[_WriteDuplicateEvent],
                             assistant_id: Optional[int]) -> None:
        """Point each suppressed duplicate at the park it was folded into.

        Same deferral as `_register_parks`, and for the same reason: the row id
        does not exist until the turn commits. Without this the duplicate's
        `duplicate_of_pending` entry — which says the action is still awaiting
        the operator — would never be corrected once the original resolves.

        The target park may be in a DIFFERENT row than this one (a re-proposal
        usually arrives a turn later), so the reference is stored on the park
        rather than resolved by scanning this row.
        """
        if assistant_id is None:
            return
        for dup in dups:
            if dup.call_id is None:
                continue
            parked = self.confirmations.pending.get(dup.token)
            if parked is None:
                # Resolved between the guard firing and this turn committing.
                # Its entry is already stale, but there is nothing left to hang
                # the reference on — and the model was told the truth at the
                # time, which is the part that mattered.
                logger.debug(
                    "duplicate for token %s: park already resolved, "
                    "leaving its entry as-is", dup.token,
                )
                continue
            # Through the manager, not by appending to the list directly: the
            # reference has to reach the durable row too, or a restart reloads
            # the park without it and the duplicate's history entry keeps
            # claiming the action is still awaiting an operator forever.
            self.confirmations.note_duplicate_ref(
                parked, assistant_id, dup.call_id,
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

        if parked.kind != DEFERRAL_KIND_APPROVAL:
            # An approve/deny click can only answer a deferral that asked a
            # human a question. Every other kind is answered by the authority
            # that owns the work, and `apply()` would execute the deferred call
            # a SECOND time here — the tool already ran, which is why it is
            # waiting on a job at all. `list_for` does not surface these, so
            # reaching this branch means a caller supplied a token by some other
            # route; it fails closed and restores the deferral.
            self.confirmations.restore(parked)
            logger.warning(
                "park %s: refusing an approve/deny on a %r deferral — that kind "
                "is resolved by its own authority, not by an operator click",
                token, parked.kind,
            )
            yield DoneEvent(
                text="No such pending action.",
                response_type=ResponseType.DEV_COMMAND,
            )
            return

        if self.confirmations.is_expired(parked):
            self.confirmations.expire(
                parked, PARK_STATUS_EXPIRED,
                "Operator answered after the TTL had passed",
            )
            yield DoneEvent(
                text="That action expired before it was reviewed.",
                response_type=ResponseType.DEV_COMMAND,
            )
            return

        if parked.persona_name not in self.personas:
            # Same restore as the wrong-conversation branch above: `take()` has
            # already removed it, and dropping it here would destroy both the
            # proposal and the operator's decision with no audit trail, leaving
            # its history entry reading `awaiting_human_approval` forever with
            # no park left for the duplicate guard to match against.
            self.confirmations.restore(parked)
            yield DoneEvent(
                text="Error: Persona not found.",
                response_type=ResponseType.DEV_COMMAND,
            )
            return

        async with aclosing(self._stream_settle(
            Decision(park=parked, approved=approved, note=note),
        )) as agen:
            async for ev in agen:
                yield ev

    async def stream_resolve_deferral(
            self, token: str, *, kind: str, status: str, result: Any,
            note: Optional[str] = None,
    ) -> AsyncGenerator[GenerationEvent, None]:
        """Settle a non-approval deferral, then summarize (DP-345).

        The twin of `stream_resolve_park` for the kinds whose answer comes from
        an authority rather than a human: a node job finishing, an agent
        emitting `done`. Everything after the claim is literally the same code —
        see `_stream_settle` — because folding, patching and continuing were
        never approval-specific, and rebuilding them per subsystem is what this
        ticket exists to undo.

        Addressed by TOKEN, like every other kind. The authority is not handed
        some identifier of its own that we then have to map back: derpr mints
        the park token and passes that same string outward as the job id, so
        the thing the node names in its callback IS the park. A second
        namespace plus an index to join it was state this call already had.

        `kind` is what the CALLER EXPECTS to be answering, not a lookup key —
        the claim is `take(token)` and the kind is checked against the row
        afterwards. Both directions fail closed, because a token that resolves
        the wrong mechanism is the failure this whole ticket is about:

        * an `approval` park reached here would execute a human-gated write
          with nobody in the loop;
        * a mismatch against `kind` means the caller believes it is answering
          a different authority than the one that parked this, so its `status`
          and `result` describe some other piece of work.

        Both restore the park rather than dropping it, exactly as the persona
        check below does.

        `status` and `result` are the *re-read* outcome, per DP-343's rule that
        the trigger carries an identifier and never facts: the caller asks the
        authority what happened and passes that in. Nothing here trusts the
        ping's own payload.
        """
        parked = self.confirmations.take(token)
        if parked is None:
            # Already settled (the node retries its ping ~6s apart), expired, or
            # never registered. Not an error, and deliberately silent: there is
            # no conversation to interrupt with a message about a job the model
            # has already been told about.
            logger.info(
                "deferral %s (%s): nothing pending to settle — already "
                "resolved, expired, or never registered", token, kind,
            )
            return

        if parked.kind == DEFERRAL_KIND_APPROVAL:
            # The mirror of `stream_resolve_park`'s non-approval refusal, and
            # the more dangerous half: an approval park gates an irreversible
            # write behind a human, and settling it here would run it on an
            # authority's say-so with no human having decided anything.
            self.confirmations.restore(parked)
            logger.error(
                "deferral %s: refusing to settle an approval park as a %r "
                "deferral — a gated write is answered by a human, never by an "
                "authority ping", token, kind,
            )
            return

        if parked.kind != kind:
            # The caller is answering a different authority than the one that
            # parked this, so its `status` and `result` are about other work.
            self.confirmations.restore(parked)
            logger.error(
                "deferral %s: caller expected kind %r but the park is %r; "
                "leaving it pending rather than settling it with another "
                "authority's outcome", token, kind, parked.kind,
            )
            return

        if parked.persona_name not in self.personas:
            # Restored rather than dropped, exactly as the approval path does:
            # the work really happened, and discarding the deferral would leave
            # its history entry reading `awaiting:<kind>` forever with nothing
            # left to patch it.
            self.confirmations.restore(parked)
            logger.error(
                "deferral %s (%s): persona %r no longer exists; leaving it "
                "pending rather than settling into nowhere",
                token, kind, parked.persona_name,
            )
            return

        async with aclosing(self._stream_settle(Decision(
            park=parked, approved=True, note=note,
            result=result, outcome_status=status,
        ))) as agen:
            async for ev in agen:
                yield ev

    async def _stream_settle(
            self, decision: Decision,
    ) -> AsyncGenerator[GenerationEvent, None]:
        """Fold, apply and continue — the half of a resolve that has no kind.

        Split out of `stream_resolve_park` when DP-345 gave the store a second
        kind. Both entry points reach here holding a park they have already
        claimed, and from this point on nothing cares which external event did
        the claiming: the per-conversation lock folds whatever else arrived, each
        decision is applied in isolation, and exactly one continuation turn
        reports the batch.

        Folding across kinds is the part that comes free. A node-job ping that
        lands while an operator is approving something in the same conversation
        joins that batch instead of racing a second tool loop over the same
        history — which the two re-derived wake paths could not do, because they
        entered through `generate_response` and never touched this lock.
        """
        parked = decision.park
        key = parked.key
        self.confirmations.enqueue(decision)

        applied: List[Decision] = []
        async with self.confirmations.lock_for(key):
            batch = self.confirmations.drain(key)
            if not batch:
                # A continuation that held the lock before us already folded
                # this decision in and acted on it. Nothing left to do.
                return

            # Re-drain after each round. Acquiring an uncontended asyncio.Lock
            # does not suspend, so the winner of a race gets here before the
            # loser has even enqueued — draining once would leave the loser to
            # run a second continuation over the same history. Looping until
            # the queue is empty is what actually folds a flurry of approvals
            # into one summary.
            failed: List[str] = []
            while batch:
                # `queued`, not `decision`: this loop settles everything the
                # drain folded in, which is not necessarily the decision this
                # call arrived with — and shadowing the parameter here made the
                # continuation's coordinates look like they came from whichever
                # park happened to be last in the batch.
                for queued in batch:
                    # Isolated per decision. A single try around the whole loop
                    # abandoned every un-applied sibling: drain() had already
                    # popped them from _queued and take() had already removed
                    # their parks, so they existed in no structure at all and
                    # the operator's approval simply evaporated. One decision
                    # that cannot be applied must not silently discard the rest.
                    try:
                        await self.confirmations.apply(queued)
                    except Exception as e:
                        err_id, _ = format_internal_error(
                            e, scrub=get_scrubber().scrub,
                        )
                        logger.error(
                            f"[err {err_id}] Error settling a "
                            f"{queued.park.kind} deferral for "
                            f"{queued.park.user_identifier}: {e}", exc_info=True,
                        )
                        queued.patched = False
                        failed.append(
                            queued.park.write_call.get("name") or "action")
                    applied.append(queued)
                batch = self.confirmations.drain(key)

            if failed and not any(d.patched for d in applied):
                # Nothing survived — there is no outcome for a continuation to
                # report, so say so directly rather than asking the model to
                # summarize an empty result set.
                yield ErrorEvent(
                    message="Error processing the confirmed action"
                            f"{'s' if len(failed) > 1 else ''}: "
                            f"{', '.join(failed)}.",
                )
                return

            # Every coordinate comes off the park — the turn that raised the
            # deferral — and none from configuration. That is the DP-345
            # invariant: a resume that has to be told where to answer is one
            # whose originating turn was thrown away.
            async with aclosing(self._orchestrate(
                persona_name=parked.persona_name,
                user_identifier=parked.user_identifier,
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
