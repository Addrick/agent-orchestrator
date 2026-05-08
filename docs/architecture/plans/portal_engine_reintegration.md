---
name: Portal engine reintegration plan
description: Migrate the kobold-lite portal from passthrough into chat_system's generate_response flow while preserving kobold-owned templating; phased plan with resolved design decisions
type: project
---

## Goal

Bring the kobold-lite portal under the same `chat_system.generate_response` orchestration that Discord, Gmail, and agents use, so:

- Logging, retry archival, prune-to-budget, LTM retrieval, and BotLogic dev commands live in **one** place (the engine), not duplicated in `kobold_adapter.py`.
- New providers gain portal access automatically (engine routes by persona model).
- Other interfaces benefit from richer persona params (kcpp samplers, etc.) without per-interface plumbing.

Constraint that survives unchanged: **kobold-lite owns templating** (decision `2026-04-19-portal-phase2-approach.md`). The engine never rewraps kobold's rendered prompt.

## Why now

Adapter today (~738 lines) mirrors orchestration logic the engine already does for other interfaces:

- `_log_interaction` / `_commit_assistant` mirror `chat_system` user/assistant logging
- `handle_portal_retry` archive + UPDATE-in-place mirrors retry semantics
- `truncate_messages_to_budget` callsite duplicated in adapter
- LTM retrieval was a private-method leak (fixed 2026-04-26 with `get_session_memory_block`); should not exist as a separate seam at all

Two forks of the same work means every future change is 2x. Reintegration consolidates.

## Design — resolved

### Param model: `GenerationParams`

Replace persona's flat sampler fields with a structured object.

**Universal** (consistent semantics across providers):
`temperature`, `top_p`, `top_k`, `max_tokens`, `stop_sequences`, `seed`

**Per-provider escape hatch:** `provider_extras: Dict[str, Dict[str, Any]]` keyed by provider id (`"kobold"`, `"openai"`, `"anthropic"`, `"gemini"`). Each provider's `build_payload` consumes universal + its own extras block.

Anything with cross-provider semantic drift (`repetition_penalty`, `min_p`) starts in `provider_extras["kobold"]`. **Future** (post-v1): `harmonize_params(params, provider) → params` hook for one-logical-knob-many-wire-mappings.

### Engine streaming surface

Two provider-level entry points:

```python
class TextProvider(ABC):
    async def stream_messages(persona, messages, params) -> AsyncIterator[str]
    async def stream_prompt(persona, rendered_prompt, params) -> AsyncIterator[str]
```

Providers implement what they support: OpenAI/Anthropic/Gemini = messages-only; Kobold = both; raw llama.cpp = prompt-only. Engine routes by persona's model.

Non-streaming consumers (Discord, Gmail, agents) go through a collect-stream wrapper — single streaming impl, no duplicate code paths.

### ChatSystem orchestration

```python
class ChatSystem:
    async def generate_response(...) -> Tuple[str, ResponseType, ...]
        # Existing signature — collect-stream wrapper around _orchestrate
    
    async def stream_response(...) -> AsyncIterator[GenerationEvent]
        # Portal entry point — yields events
    
    async def _orchestrate(...) -> AsyncIterator[GenerationEvent]
        # Shared kernel:
        # 1. BotLogic.preprocess (commands) → DoneEvent + return early on hit
        # 2. Log user turn (with retry archive when is_retry=True)
        # 3. Map persona → GenerationParams
        # 4. Call provider stream_messages / stream_prompt
        # 5. Yield TokenEvent per delta
        # 6. On close: log assistant, yield DoneEvent(assistant_id)
        # 7. On error: yield ErrorEvent
```

Event types:
- `TokenEvent(delta: str)` — incremental text
- `DoneEvent(assistant_id: int, full_text: str)` — final commit signal (replaces today's `event: derpr` SSE frame)
- `ErrorEvent(message: str)`

### Tool loop + streaming policy

**v1 = refuse-on-conflict.** If `persona.has_tools_enabled` and stream=True, return error. Tool revamp is upcoming work; speculative tool-call-in-stream design parked.

**Future** (post-tool-revamp): emit tool calls inline as additional events:
```python
class ToolCallStartEvent: tool_name: str; args: Dict
class ToolCallResultEvent: tool_name: str; result: str
```
Portal renders these as `<details>` thought-process blocks inline between TokenEvent runs. Linear stream, no drain-and-restart. Don't build until tool revamp is in motion.

### Portal adapter shape after migration

Adapter collapses to thin SSE transcoder over `stream_response`:

```python
@self.app.post("/v1/chat/completions")
async def oai_chat_completions(request):
    data = await request.json()
    async def relay():
        async for ev in self.chat_system.stream_response(
            persona_name=self._get_current_persona_name(),
            messages=data["messages"],
            user_text=data.get("derpr_user_text"),
            user_identifier="portal", channel="web_ui",
            is_retry=bool(data.get("derpr_retry")),
        ):
            if isinstance(ev, TokenEvent):    yield format_oai_sse(ev.delta)
            elif isinstance(ev, DoneEvent):   yield format_derpr_frame(ev.assistant_id); yield b"data: [DONE]\n\n"
            elif isinstance(ev, ErrorEvent):  yield format_oai_error(ev.message)
    return StreamingResponse(relay(), media_type="text/event-stream", ...)
```

Same shape for `/api/extra/generate/stream` route, but using `prompt` + `stream_prompt` path instead of messages.

Routes that **stay in adapter** (HTTP boundary, not orchestration):
- Persona CRUD (`GET/PATCH /api/v1/persona/...`)
- Version chevrons (`/versions`, `/select_version/{k}`, `PATCH/DELETE /api/v1/interaction/...`)
- Kobold export (`/api/v1/session/{persona}/kobold_export`)
- Forwarders (`/api/extra/version`, `/api/v1/abort`, etc.)
- LTM block (`/api/v1/session/{persona}/ltm_block` — already wraps `get_session_memory_block`)

Estimate: adapter ~738 → ~200 lines.

### Persona panel UX

Existing portal persona config panel **stays as-is**, minor extensions only. Kobold-lite's own sliders handle the bulk of llama.cpp-specific params; once saved through kobold's UI those values persist on the persona via the existing PATCH route. BotLogic gets a fallback dotted-path setter (`set kobold.mirostat 2`) for occasional CLI edits — nice to have, not critical. No JSON-editor or per-provider form schema work needed.

## Phased migration

| Phase | Scope | Depends on |
|---|---|---|
| **A** ✅ shipped 2026-04-27 | `GenerationParams` model (`src/generation_params.py`). Persona stores `_params: GenerationParams`, legacy flat getters/setters facade unchanged, new `get_generation_params()` seam. `save_utils.to_dict` writes nested `params` block; load is dual-shape (prefers `params`, falls back to legacy flat keys). 635/635 pass; mypy delta 0. | — |
| **B** ✅ shipped 2026-04-27 | Provider streaming surface. `StreamEngine.stream_messages` / `stream_prompt` typed entries (kobold), shared `_kobold_stream` core; legacy `stream_local` is now a back-compat wrapper. `TextEngine.stream_messages(persona_config, messages, params, tools)` dispatches by model — local routes to `StreamEngine` for real SSE, OpenAI/Anthropic/Google wrap `generate_response` and emit a single `text_delta` (Phase C will flip ownership). `TextEngine.stream_prompt` is local-only and refuses non-local. `TextEngine.collect_stream` drains the unified event stream into `(result_dict, api_payload)` matching `generate_response`. 654/654 pass; mypy delta 0. | A |
| **C** ✅ shipped 2026-04-28 | `ChatSystem.stream_response` + `_orchestrate` kernel landed. Event types (`TokenEvent` / `DoneEvent` / `ErrorEvent`) are dataclasses; `generate_response` is now a thin collect-stream wrapper. Kernel: preprocess → user-log (or `handle_portal_retry` when `is_retry=True`) → tool loop via `text_engine.stream_messages` → assistant log / UPDATE-in-place on retry. `stream_response` refuses tool-enabled personas (tool-call-in-stream parked until tool revamp). `_run_tool_loop` / `_execute_request` inlined and deleted. User-row now logged *before* the LLM call so it survives mid-flight failures — matches the portal's existing pattern. `_messages_to_history_object` strips a leading system message into `persona_prompt` and exposes a legacy `history` alias so engine handlers and `call_args.args[1]['history']` test contracts keep working. Test fixtures swapped from `MagicMock(spec=TextEngine)` to a real `TextEngine` with `generate_response` patched — every existing `text_engine.generate_response.*` assertion keeps firing through `stream_messages`'s wrap. 660/660 unit+integration pass; mypy delta 0. | B |
| **D** ✅ shipped 2026-04-28 | Portal adapter migration — **OAI route only**. `/v1/chat/completions` is now a thin SSE transcoder over `stream_response`. Engine rebuilds messages from DB; client `data["messages"]` is discarded; `derpr_user_text` (or last-user fallback for non-portal clients) drives the user turn. Native `/api/extra/generate/stream` and `/api/v1/generate` left untouched — they still call `_log_interaction`/`_commit_assistant`. Kernel changes: `_log_user_turn` + `_commit_or_update_assistant` helpers extracted; `_orchestrate` wraps the stream loop in `try/except asyncio.CancelledError` and flushes a partial assistant commit before re-raising; `_prepare_request(is_retry=True)` pops the trailing assistant from DB-built history and skips the user-append (DB already terminates with the matching user row from the prior turn). Adapter side: deleted `_strip_envelope` (now dead), removed `truncate_messages_to_budget` import, removed `handle_portal_retry`/`_log_interaction`/`_commit_assistant` from the OAI handler, dropped the `httpx` upstream call. Reasoning_content separation lost — engine routes through `StreamEngine` → kobold native (`/api/extra/generate/stream`), which delivers `<think>` blocks inline rather than as a structured field. Tests rewritten with a `_make_real_adapter` fixture that wires real ChatSystem + in-memory MemoryManager + stubbed `text_engine.stream_messages`; dropped tests for `_strip_envelope`, budget prune (now engine-side, covered in `tests/test_chat_system.py`), and reasoning_content separation. 660/660 unit+integration pass; mypy delta 0 (errors net -4, no new files). Decision rationale: `decisions/2026-04-28-portal-engine-as-source-of-truth.md`. | C |
| **E** ✅ shipped 2026-04-29 | BotLogic dotted-path setter landed. `set <provider>.<key> <value>` routes through a fallback in `_handle_set` when the sub_command isn't in `set_handlers` and contains a `.`; value is coerced int → float → bool → str, with `none`/`null`/`clear` removing the key. Symmetric `what <provider>.<key>` lookup added. Storage lives in `Persona.set_provider_extra` / `get_provider_extra` / `clear_provider_extra`, which hit `params.provider_extras[provider][key]`; emptying a provider block prunes it. The setter-coverage and getter-coverage tests in `tests/test_message_handler.py` got `set_provider_extra` / `get_provider_extra` exception entries since the dotted path is the entrypoint, not a dedicated command. Help text and `docs/user_guide.md` updated. 676/676 unit+integration pass; mypy delta 0. Persona panel extensions deferred — kobold-lite's own UI covers the bulk and no gaps surfaced. | C |

A and B both ship green before touching ChatSystem. Suggested PR cadence: one PR per phase.

### Pre-Phase-A coverage prep (landed 2026-04-27)

Before kicking off Phase A, three gaps were closed so each phase can be verified one-for-one:

- **Spec drift fixed.** `docs/user_guide.md` claimed `/api/v1/generate` and `/api/extra/generate/stream` were removed; both are still served by the adapter (and Phase D migrates the latter via `stream_prompt`). Spec now describes both routes accurately.
- **`StreamEngine.stream_local` end-to-end tests** in `tests/test_stream_engine.py`: api_payload-first ordering, text_delta order, finish_reason termination, mid-stream `<tool_call>` extraction, non-200 → `LLMCommunicationError` with `api_payload`, transport error path, early-break aborts upstream with the genkey, tool-list folded into system prompt. Closest existing surface to Phase B's `stream_prompt` — now pinned.
- **`/api/extra/generate/stream` adapter tests** in `tests/interfaces/test_kobold_adapter.py`: user-row from `prompt`, assistant assembled from token deltas with `reply_to_id`, empty-prompt skips user log, `model` selector stripped from forwarded body, abort flushes partial assistant. Native route was completely untested before this; Phase D migrates it.

620/620 unit+integration pass; mypy delta 0.

## DB migration deferral

In-place migration of persona schema would bloat the codebase for a one-time op. Strategy: write straight to new shape, accept that existing prod DB breaks until the consolidated deployment migration script runs. Unit tests use `:memory:` so test green is preserved. **Reminder:** any other schema/data migrations landing between now and deploy must be added to the same script.

## Open items deliberately not resolved here

- **Streaming on other interfaces.** Discord token-by-token reply experiments parked indefinitely (UX + rate-limit risk). If reopened, `stream_response` already exists as the seam.
- **Tool-call-in-stream.** Designed enough above to slot in post-tool-revamp; do not pre-build the events.
- **`harmonize_params` cross-provider mapping.** Single-knob-many-wires hook for `repetition_penalty`-style cases. Not v1.
- **Native route (`/api/extra/generate/stream`) deprecation.** Kobold-lite's `useoaichatcompl` toggle is the only reason both wire formats reach us. OAI became the preferred path when upstream shipped jinja chat templates. Investigation needed: does every kobold-lite feature now reach parity over OAI? If yes, force the toggle on (or fork lite config) and delete the adapter native handler entirely. If any sampler/abort/feature only fires on native, those need OAI parity first. Until the audit lands, native stays as adapter passthrough; engine work in Phase D does not touch it. See `decisions/2026-04-28-portal-engine-as-source-of-truth.md`.

## TODO — portal defaults to DERPR innately

A fresh browser tab on the 5003 engine adapter loads with kobold-lite's native (in-memory) history source because `persona_history_source_<name>` in localStorage is unset. Result: persona switch doesn't fetch `/api/v1/session/<persona>/kobold_export`, history stays empty, and the user has to toggle "Use DERPR history" by hand each time they open the portal.

When the portal is being served by `KoboldEngineAdapter` (port 5003 / `kobold_engine_adapter.py`), DERPR should be the implicit history source — no toggle required. Options:

- Server-side flag in `/api/v1/model` GET (or a new `/api/v1/portal/mode`) telling the portal "you're talking to the engine, default to DERPR."
- Portal detects engine adapter via a response header or `/api/extra/version` payload field and treats DERPR-history as the default when present.
- Drop the per-persona localStorage flag entirely on the engine adapter; treat DERPR as the only source (kobold-native history was the legacy 5002 affordance).

Also reset/inject the kobold-lite settings the portal currently leaves blank on first load (Instruct Tag Preset, sampler defaults from persona) so a fresh tab is usable immediately. Today the user has to re-pick a preset, which is also why instruct_tags don't reach the engine on a cold tab.

## Out of scope

- New provider integrations (xAI, etc.) — slot in trivially after B
- Multi-session per-persona threads (Web UI roadmap Tier 3)
- Memory provenance UI (Web UI roadmap Tier 1)

## References

- `decisions/2026-04-19-kobold-portal-passthrough.md` — Stage 1 verbatim passthrough
- `decisions/2026-04-19-portal-phase2-approach.md` — kobold owns templating, author's note for LTM
- `plans/web_ui_roadmap.md` — overall portal phasing, backlog tiers
- `plans/toolloop_extraction.md` — tool revamp track that unlocks tool-call-in-stream
- 2026-04-26 review notes (in `_overview.md`) — diagnosis of duplication driving this plan
