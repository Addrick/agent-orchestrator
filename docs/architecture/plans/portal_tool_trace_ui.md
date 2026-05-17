# Plan: Portal tool-trace UI (event-driven)

DP-ID: TBD (was the original ask: "show tool traces in the web ui").

## Why

Today the engine adapter (`kobold_engine_adapter.py:633-678`) emits tool
calls two ways:

1. **Named SSE frames** — `event: derpr-tool-start` / `event: derpr-tool-result`,
   each with a JSON payload (`tool_name`, `arguments`, `call_id`, `result`,
   `error`).
2. **Inline `<think>🔧 …</think>` content chunks** — injected into
   `delta.content` so the portal's existing `format_streaming_text` regex
   (portal.html:5252-5262) renders them as `<details class="tool-process">`
   blocks without any frontend awareness of SSE events.

Phase-1 diagnosis (2026-05-17) showed two real problems with this dual
emission:

- The portal's OAI SSE transform (portal.html:7377) only matches `data:`
  lines. Named `event:` frames leave their `event: …\n` prefix line stuck
  in the parser buffer, which poisons subsequent regex matches and can
  silently drop content chunks. End-to-end symptom: iter=1 post-tool prose
  is sent by the backend but never reaches `synchro_pending_stream`.
- The fallback inline `<think>🔧 …</think>` chunks DO render the tool
  block, masking the SSE-parser issue while breaking the actual response
  text rendering for any turn that has a tool call followed by prose.

Phase 1 also confirmed:

- The tool loop works correctly (multiple iters, multiple calls per iter).
- The model produces post-tool prose when context isn't polluted.
- `web_search` returning empty was a stale `duckduckgo_search` package —
  swapped to `ddgs` (already fixed in this session).

## Goal

Make the portal a first-class consumer of the SSE event channel and stop
relying on inline content injection for tool-trace rendering.

## Non-goals

- Persistence / history rehydration — covered by Phase 4 of the original
  `tool_revamp_v1.md` plan; deferred.
- Hindsight `drill_down_memory` rewrite — separate work (see "Open work"
  below).
- Backwards compatibility with the inline `<think>🔧` shim for legacy
  rendered messages in saved sessions — accept that old messages may show
  the shim text until the persistence pass.

## Design

### Wire shape (backend → frontend)

Keep both SSE event types as today, with one addition (`group_id`) for
forward-compatible parallel calls:

```
event: derpr-tool-start
data: {
  "call_id": "call_web_search_0",
  "group_id": "iter0_<uuid>",
  "tool_name": "web_search",
  "arguments": {"query": "…", "max_results": 5},
  "iter_idx": 0
}

event: derpr-tool-result
data: {
  "call_id": "call_web_search_0",
  "group_id": "iter0_<uuid>",
  "tool_name": "web_search",
  "result": "<JSON-encoded result string>",
  "error": null
}
```

- `group_id`: stable identifier for "the set of tool calls produced by one
  model iteration." Today every group has exactly one call (serial
  `_execute_calls`); future parallel-call work groups multiple calls
  under the same id. Frontend can use it to render concurrent calls
  stacked under a "Tool calls" header. **Add the field now**, populate
  it from `iter_idx + uuid4().hex[:8]`, and persist in `tool_context_json`
  so a later rehydrate pass already has it.
- `call_id`: unique per call; comes from `tool_loop._execute_calls`
  (`call_{name}_{uuid}` or model-provided).
- `arguments` / `result`: rendered as pre-formatted JSON / raw text in the
  details body. Result strings can be long — frontend should respect a
  collapse-by-default rule for any body over N lines.

Content chunks no longer contain `<think>🔧 …</think>` for tool blocks.
The model's own prose continues to stream as ordinary `delta.content`.

### Backend change (`kobold_engine_adapter.py:633-678`)

Drop the inline content emission in both `ToolCallStartEvent` and
`ToolCallResultEvent` branches:

- Keep the `event: derpr-tool-start` / `derpr-tool-result` SSE frames.
- Remove both the `inline_open` and `inline_close` `chat.completion.chunk`
  yields. No new content goes into `delta.content` for tool events.

Add `group_id` to the payload. Generate it in `ToolLoop` (see Loop change
below) or in the adapter (cheaper; just hash `iter_idx + persona + ts`
and stuff it in). Recommend Loop ownership so it lands in the persisted
`tool_context_json` too.

### Loop change (`tool_loop.py:170-184`)

Mint a `group_id` once per iteration that produces tool calls. Pass it on
the events the loop yields:

```python
group_id = f"iter{iter_idx}_{uuid4().hex[:8]}"
for call_item in tool_calls_collected:
    call_item["group_id"] = group_id  # for persistence
async for ev in self._execute_calls(read_calls, conversation_history,
                                    group_id=group_id):
    yield ev
```

Add `group_id` to `ToolCallStartEvent` / `ToolCallResultEvent` dataclasses
(`src/generation_events.py`). Default `None` for backwards-compat with
non-loop callers; the adapter only emits the field when populated.

### Frontend change (`portal.html`)

Three pieces.

#### 1. Event-aware SSE transform (replace lines 7365-7385)

Replace the `data:`-only regex with a proper SSE frame parser. Frames are
separated by `\n\n` and may contain `event:` and/or `data:` lines. Pseudo:

```js
transform(chunk, ctrl) {
    ctrl.buf += chunk;
    let evs = [];
    while (true) {
        const sep = ctrl.buf.indexOf("\n\n");
        if (sep === -1) break;
        const frame = ctrl.buf.slice(0, sep);
        ctrl.buf = ctrl.buf.slice(sep + 2);
        let etype = "message";
        let dataLines = [];
        for (const line of frame.split(/\r?\n/)) {
            if (line.startsWith("event:")) etype = line.slice(6).trim();
            else if (line.startsWith("data:")) dataLines.push(line.slice(5).replace(/^ /, ""));
        }
        if (!dataLines.length) continue;
        const data = dataLines.join("\n");
        if (data === "[DONE]") { evs.push({event: etype, data: "[DONE]"}); continue; }
        try { evs.push({event: etype, data: JSON.parse(data)}); }
        catch (e) { console.log("Cannot parse a chunk: " + data); }
    }
    if (evs.length) ctrl.enqueue(evs);
}
```

This is byte-correct against the SSE spec, and `event:`-typed frames no
longer pollute the buffer.

#### 2. WritableStream.write dispatch (modify around 7388-7600)

In the per-event loop, branch on `event.event` BEFORE the existing
`event.data.choices` handling:

```js
if (event.event === "derpr-tool-start") {
    handle_tool_start(event.data);
    continue;
}
if (event.event === "derpr-tool-result") {
    handle_tool_result(event.data);
    continue;
}
if (event.event === "derpr") { /* existing assistant_id hydration */ continue; }
// fall through to existing `event.data.choices` branch
```

`derpr` (the assistant_id hydration frame) already exists; it's currently
also unrouted but harmless. Consolidate handling here.

#### 3. Per-call_id tool block injector (new top-level helpers)

State:

```js
window._derpr_pending_tool_calls = new Map(); // call_id → {tool_name, args, group_id, status, dom_marker}
```

`handle_tool_start({call_id, group_id, tool_name, arguments})`:

- Build the marker string the renderer will replace with a `<details>`
  block:

    ```
    [[DERPR_TOOL:<call_id>]]
    ```

- Append the marker into `synchro_pending_stream` at the current write
  position (i.e. between whatever prose came before and whatever comes
  after). The marker is a no-op for stop-detection and ordinary text
  rendering — it survives `replaceAll("\n","<br>")` and Markdown.
- Store the metadata under `call_id`.

`handle_tool_result({call_id, tool_name, result, error})`:

- Update the stored metadata with status + result/error.
- Trigger a re-render (call `update_pending_stream_displays()` /
  `render_gametext(false,false)` — whichever path the surrounding code is
  in).

Renderer change (new pass in `format_streaming_text`, replacing the
deleted `toolThinkDone` / `toolThinkOpen` blocks):

```js
input = input.replace(/\[\[DERPR_TOOL:([^\]]+)\]\]/g, (match, callId) => {
    const meta = (window._derpr_pending_tool_calls || new Map()).get(callId);
    if (!meta) return ""; // marker without metadata — drop silently
    const argsBody = `<div class="tool-args"><code>${escapeHtml(JSON.stringify(meta.args))}</code></div>`;
    if (meta.status === "running") {
        return `<details class="tool-process" open data-call-id="${callId}">
            <summary><span class="tool-status-icon">⚡</span> Tool: ${meta.tool_name} (running…)</summary>
            <div class="tool-content">${argsBody}</div>
        </details>`;
    }
    const icon = meta.error ? "⚠️" : "🔧";
    const bodyText = meta.error || meta.result || "";
    return `<details class="tool-process" data-call-id="${callId}">
        <summary><span class="tool-status-icon">${icon}</span> Tool: ${meta.tool_name}</summary>
        <div class="tool-content">${argsBody}<pre class="tool-result">${escapeHtml(bodyText)}</pre></div>
    </details>`;
});
```

Marker survives storage (it's plain text), so a future
history-rehydration pass can re-inject metadata into the map from the
DB's `tool_context_json` and the same renderer will repaint trace blocks
on reload.

### Removed code

- `kobold_engine_adapter.py` inline `<think>🔧` injection in both event
  branches (lines 640-657 and 666-678) — keep the JSON SSE frames.
- `portal.html` `format_streaming_text` `toolThinkDone` / `toolThinkOpen`
  regex pair (5247-5262). The legacy regex is no longer needed because
  the backend stops emitting that markup.

### Schema migration / persistence

`tool_context_json` on `User_Interactions` rows already stores the
post-tool history slice (`role:assistant tool_calls=[…]`, `role:tool
content=…`). Add `group_id` to each `tool_call` entry when minted. No DB
schema change required.

History rehydration is out of scope for this plan (Phase 4 of
`tool_revamp_v1.md`). Note: once shipped, the marker-based renderer means
rehydration is "rebuild the `_derpr_pending_tool_calls` map + insert
`[[DERPR_TOOL:<id>]]` markers in the message text at the right positions."

## Phases / tasks

| Phase | Scope | Files |
|---|---|---|
| **A** | Add `group_id` to `ToolCallStartEvent` / `ToolCallResultEvent`. Mint in `ToolLoop`. Plumb through `kobold_engine_adapter` payload. No frontend change yet (frontend ignores unknown fields). | `src/generation_events.py`, `src/tools/tool_loop.py`, `src/interfaces/kobold_engine_adapter.py` |
| **B** | Drop inline `<think>🔧` content chunks in `kobold_engine_adapter`. SSE frames only. | `src/interfaces/kobold_engine_adapter.py` |
| **C** | Frontend: event-aware SSE transform. Verifies content streams without buffer poisoning. | `src/interfaces/web_assets/portal.html` |
| **D** | Frontend: `handle_tool_start` / `handle_tool_result`, `_derpr_pending_tool_calls` map, marker renderer. Remove legacy `toolThinkDone` / `toolThinkOpen` regex. | `src/interfaces/web_assets/portal.html` |
| **E** | Strip channel markers (`<\|tool\|>`, `<\|tool_call\|>`, `<\|tool_response\|>`, `<\|channel\|>…`, harmony tags) from `_ToolCallStreamParser.visible_text`. Stops cosmetic leak + self-poisoning of future contexts. | `src/stream_engine.py` |
| **F** | Revert PHASE1 instrumentation. Recompile `requirements.txt` via pip-compile to lock the `ddgs` swap. | `src/tools/tool_loop.py`, `src/stream_engine.py`, `requirements.txt` |

Each phase tested independently:

- **A:** unit test on `ToolLoop` — assert every emitted ToolCall*Event in a
  multi-call iter shares the same `group_id`, and a second iter has a
  different one.
- **B:** integration test on `chat_system.stream_response` — assert SSE
  output contains `event: derpr-tool-start` frames and NO inline
  `<think>` markup in token deltas.
- **C:** unit test on the JS transform (jsdom or a hand-written runner)
  — feed a mixed `event:`/`data:` stream, assert events emit with correct
  type + parsed JSON, buffer stays bounded across chunk boundaries.
- **D:** manual portal smoke. Trigger a single tool call → see running →
  done. Trigger two sequential calls → both render. Confirm post-tool
  prose visibly renders after the second block.
- **E:** unit test on `_ToolCallStreamParser` — feed
  `<\|tool\|>\n<tool_call>{…}</tool_call>` and assert `visible_text ==
  ""` with 1 call extracted.
- **F:** `pytest`, `flake8`, `mypy src/`. `pip-compile requirements.in`.

## Parallel-call shape (deferred but reserved)

`group_id` is the only forward-compatible plumbing this plan adds. Real
parallel execution requires:

- `tool_loop._execute_calls` running `asyncio.gather` over read-only calls.
- Event interleaving on the wire: starts and results may arrive in any
  order; UI must rely on `call_id` rather than positional ordering. The
  marker approach handles this — markers are inserted at start time,
  status flips when result arrives.
- All-or-nothing taint update after the whole group resolves.
- Writes still block the group and re-enter the audit/confirm path.

None of that is implemented here; the plumbing just doesn't preclude it.

## Open work (out of scope, tracked here for visibility)

- **`drill_down_memory` under Hindsight.** Tool queries SQLite-only
  `Memory_Summaries`; under `SEMANTIC_BACKEND=hindsight` it always
  returns `[]` and the model hallucinates IDs. Either:
  (a) gate it out of the tool list when `SEMANTIC_BACKEND != "sqlite"`,
  (b) implement a Hindsight-native equivalent (wrap `mcp__hindsight__get_memory`
      by UUID, possibly with a sibling-lookup pass), or
  (c) generalize: rename to `get_full_memory(memory_id: str)` and dispatch
      on backend. Recommend (a) first as smallest unblocker.

- **History rehydration of tool blocks.** Phase 4 of `tool_revamp_v1.md`.
  Once this plan ships, the marker-based renderer means rehydration is
  cheap: rebuild `_derpr_pending_tool_calls` from `tool_context_json` and
  inject markers into the message body at load time.

- **Empty-iter handling.** If iter≥1 returns `raw_len=0`, `tool_loop`
  finishes with empty `final_text`. Today the orchestrator yields a
  `DoneEvent` with empty text and the portal renders nothing. Acceptable
  for now; Phase 2 of the original plan proposed a graceful "no more to
  say" fallback. Park.

## References

- `docs/architecture/plans/tool_revamp_v1.md` — original phased plan.
  Phases 1–3 shipped; Phase 4 is the portal frontend (this plan
  supersedes it).
- `docs/architecture/plans/portal_engine_reintegration.md` — Phase D
  (engine path on `/v1/chat/completions`) is the prerequisite.
- Session log (2026-05-17, PHASE1 instrumentation): proved the SSE
  parser bug and the model-produces-post-tool-text behavior.
