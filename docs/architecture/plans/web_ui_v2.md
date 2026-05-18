# Plan: Web UI v2 — kobold-native tool surface + approval

DP-ID: TBD

## Why

`portal_tool_trace_ui.md` shipped a custom marker scheme
(`[[DERPR_TOOL:<call_id>]]` + `event: derpr-tool-*` SSE frames) to render
tool calls in the portal. It works and looks good, but it is custom in
ways that bite:

- **No reload rehydration.** Markers are plain text in DB; their
  `<details>` rendering depends on a window-local Map populated only
  during the live SSE stream. Page reload → markers drop silently.
  Phase 4 of the original tool_revamp_v1 plan parked this work.
- **No approval UI for portal users.** derpr already builds full
  `audit_info` payloads (`tool_loop.py:206-213`: tool name, args,
  irreversibility, always_confirm, service_binding, sensitivity,
  enrichment) and stages write calls via `_pending_confirmations`
  (`chat_system.py:148`). Discord drives the resume via
  `resume_pending_confirmation`. **The portal has no equivalent route.**
  Today a portal user can't approve a write tool at all — the LLM emits
  a textual "I'd like to perform the following actions:" block and the
  conversation stalls.
- **Custom wire format.** kobold-lite already understands the
  OAI-spec `delta.tool_calls` streaming format
  (`portal.html:7573-7610`), accumulates them, raises an Approve /
  Inspect UI when `tools_auto_exec=false` (`:25800-25804`), and renders
  committed turns natively via `chatunits[i].tool_calls` (`:25787`).
  We are duplicating mechanisms it already has.

This plan migrates the portal to the OAI-native wire format and reuses
kobold's UI surface for approval. The marker scheme retires.

## Goal

Single end-state:

- Tool calls stream as standard OAI `delta.tool_calls` chunks +
  `role: "tool"` result messages. No `[[DERPR_TOOL:*]]` markers, no
  `event: derpr-tool-*` SSE frames.
- Committed turns store structured `tool_calls` on the chatunit, so
  reload renders them without any rehydration map.
- Pending write-tool confirmations surface via kobold's existing
  Approve / Inspect buttons. New HTTP route resumes the server-side
  `_pending_confirmations` queue.

## Non-goals

- Persona / settings UI redesign — separate work
  (`portal_persona_settings_sync.md`).
- Tool-result rich rendering (file attachments, images returned from
  tools). Kobold's native rendering is a one-liner placeholder; richer
  expansion is deferred.
- Multi-tenant approval (different users approving same persona's
  pending call). derpr's `_pending_confirmations` is already keyed
  `(user_identifier, persona_name)`; the portal pins to one user.

## Architecture comparison

| Concern | Today (markers) | v2 (kobold-native) |
|---|---|---|
| Wire shape | `event: derpr-tool-start` + `[[DERPR_TOOL:<id>]]` injected into delta.content | `delta.tool_calls: [{id, type, function:{name,arguments}}]` + `finish_reason: "tool_calls"`, then a `role: "tool"` chunk with the result |
| Live render | `format_streaming_text` regex → `<details>` | `pending_toolcall_objs` accumulator → "Made a tool call to X" placeholder |
| Committed render | `derpr_replace_tool_markers` in `repack_postprocess_turn` | `chatunits[i].tool_calls` rendered by `repact_instruct_turns_beautify_render` |
| Reload | Marker → empty (no rehydration) | Structured field on the turn — automatic |
| Approval | None for portal users | Kobold Approve / Inspect buttons on the pending tool_call chatunit; backend route resumes |
| Result panel | Rich `<details>` with args + body | One-liner "Received tool call results for X" (regression) |

The regression on result rendering is real. Mitigation in Phase D.

## Approval flow — concrete answer to "how easily can we ape this?"

**Today (Discord path):**

```
write call detected
  → tool_loop emits _LoopFinishedEvent(response_type=PENDING_CONFIRMATION,
                                       pending_writes=[...], audit_info={...})
  → chat_system stores in _pending_confirmations[(user_id, persona)]
  → text response: "I'd like to perform the following actions: ..."
discord user replies "yes"/"no"
  → discord_bot calls chat_system.resume_pending_confirmation(user_id, persona, approved)
  → returns (final_text, response_type, assistant_id, user_interaction_id)
```

**v2 approval (portal path):**

```
write call detected → same backend path
  → adapter emits the staged calls as delta.tool_calls chunks with
    finish_reason="tool_calls" (NOT executed yet)
  → kobold accumulates → renders Approve/Inspect buttons (tools_auto_exec=false)
portal user clicks Approve
  → button calls a new derpr override that POSTs to:
    POST /api/v1/persona/{persona}/confirm
    body: {"approved": true|false}
  → endpoint calls chat_system.resume_pending_confirmation
  → reply streams back through the existing /v1/chat/completions SSE channel
    (or returned inline — kobold expects the response to follow naturally)
```

The mapping cost:

- **derpr audit_actions → OAI tool_call shape:** trivial. Each
  `{"tool": name, "arguments": args, ...}` → `{"id": "call_<n>", "type":
  "function", "function": {"name": name, "arguments": json.dumps(args)}}`.
  Extra fields (irreversible, sensitivity, service_binding, enrichment)
  attach as a sibling `derpr_meta` field on the call — Inspect dialog
  surfaces them; kobold ignores unknown keys.
- **Inspect button:** already renders `JSON.stringify(toolcall_waiting_approve)`
  (`portal.html:25844`). Free.
- **Approve hook:** kobold's `execute_waiting_toolcall` calls
  `MCPToolCall(...)` against `localsettings.cached_mcp_tools`
  (`:25849`). We override this for derpr-side calls: if the pending
  call's `id` starts with `call_derpr_*`, hit our endpoint instead of
  MCP. Single conditional.
- **New HTTP route:** `POST /api/v1/persona/{persona}/confirm` —
  ~20 lines. Calls `resume_pending_confirmation`, returns the streamed
  follow-up turn or its DB id for the portal to refetch.

**Verdict: easy.** ~150 LoC backend + ~50 LoC portal override. The
existing `audit_info` shape carries everything kobold's UI needs to
present.

## Phases

| Phase | Scope | Files |
|---|---|---|
| **A** | Add `POST /api/v1/persona/{persona}/confirm` route that wraps `chat_system.resume_pending_confirmation`. Returns the streamed follow-up or its assistant_id for refetch. Unit test against a fake pending confirmation. | `src/interfaces/kobold_engine_adapter.py`, `src/chat_system.py`, `tests/interfaces/test_engine_adapter.py` |
| **B** | Adapter wire-format switch: replace `event: derpr-tool-*` SSE frames with OAI `delta.tool_calls` chunks (with `id`, `function.name`, `function.arguments`) and follow-up `role: "tool"` content chunks. Emit `finish_reason: "tool_calls"` between iterations. Keep `group_id` on a sibling `derpr_meta` field. | `src/interfaces/kobold_engine_adapter.py` |
| **C** | Pending-confirmation emission: when `_LoopFinishedEvent` has `response_type=PENDING_CONFIRMATION`, emit `delta.tool_calls` chunks with `derpr_meta: {pending: true, audit_info: {...}}` + `finish_reason="tool_calls"`. Adapter does NOT execute — sits until `/confirm` arrives. | `src/interfaces/kobold_engine_adapter.py` |
| **D** | Portal-side override for `execute_waiting_toolcall`: if the pending call's id is derpr-tagged, POST to our `/confirm` instead of MCPToolCall. Add a richer Inspect dialog rendering `derpr_meta.audit_info` (irreversibility, enrichment, taint sources) instead of a raw `JSON.stringify`. | `src/interfaces/web_assets/portal.html` |
| **E** | Retire the marker scheme: remove `derpr_replace_tool_markers`, `_derpr_pending_tool_calls` map, `derpr-tool-*` SSE frames, marker injection in WritableStream.write, the legacy regex in `format_streaming_text`. Strip `[[DERPR_TOOL:*]]` from any rows that still carry markers in DB (one-shot migration script). | `src/interfaces/web_assets/portal.html`, `src/interfaces/kobold_engine_adapter.py`, `scripts/migrate_strip_tool_markers.py` |
| **F** | (Optional) Restore the rich `<details>` result panel by post-processing kobold's "Received tool call results for X" placeholder when the matching turn has `tool_context_json`. Toggle via persona setting if controversial. | `src/interfaces/web_assets/portal.html` |

Each phase tested independently. E and F are the dessert; A–D give the
core win.

## Risks / open questions

- **finish_reason semantics.** Kobold's accumulator triggers
  `MCPToolCall` on `finish_reason: "tool_calls"`. If we emit that
  between iter-0 and iter-1 of `ToolLoop`, kobold will try to call MCP
  and hit our override; that's fine for the `/confirm` case, but for
  auto-executed read calls we need a different signal. Probably emit a
  custom `derpr_auto_executed: true` flag on the tool_call so the
  portal override returns immediately without faking a result.

- **Streaming order.** Today read tools execute serially within the
  loop; the adapter emits start → result → start → result. Kobold's
  accumulator groups all `delta.tool_calls` chunks before
  `finish_reason`. Need to verify it tolerates start-result-start-result
  interleaving, or batch the calls per iteration.

- **Persistence shape.** Kobold's chatunit `tool_calls` field is OAI
  shape. derpr's `tool_context_json` is a slightly different JSON
  (messages-style: role=assistant + tool_calls, role=tool + content).
  Need a small adapter on the export side
  (`/api/v1/session/{persona}/kobold_export`) to project one into the
  other.

- **What about the rich UI we just built?** Phase E retires it. If the
  result panel is a hard requirement, Phase F restores something
  similar but keyed off the OAI-native data instead of markers. Decide
  before Phase E.

## References

- `portal_tool_trace_ui.md` — the current marker scheme this plan
  retires.
- `tool_revamp_v1.md` Phase 4 — the original "render tool calls in
  portal" work; this plan supersedes its frontend half.
- `harmony_channel_stop_seq.md` — independent fix for the model
  truncating mid-channel. Orthogonal to wire format.
- Kobold-lite native paths:
  - Accumulator: `portal.html:7573-7610`
  - Approve UI: `portal.html:25800-25814`
  - MCP dispatcher: `portal.html:8580-8650`
  - Committed render: `portal.html:25787-25812`
- derpr backend:
  - Pending writes staged: `chat_system.py:894-906`
  - Resume entry point: `chat_system.py:1039` (`resume_pending_confirmation`)
  - audit_actions shape: `tool_loop.py:206-213`
