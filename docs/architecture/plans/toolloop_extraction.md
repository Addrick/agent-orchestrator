---
name: ToolLoop extraction plan
description: Phase 2 plan to extract tool-use loop from ChatSystem into shared module, enabling agents to use tool loops
type: project
---

## Intent

Extract `_run_tool_loop()` from ChatSystem into a shared `ToolLoop` class in `src/tools/`, so both conversational personas and background agents can use tool loops. This was Phase 2 of a two-phase plan; Phase 1 (service lifecycle removal) is complete.

## Why

- Personas and agents are conceptually similar (both use LLMs + tools) but structurally separate. Adam values conceptual consistency.
- Current agents have hardcoded pipelines (ZammadBot's multi-stage triage), which is correct for reliability — but as agents grow more complex, they need optional tool-use loops (e.g., dispatch agent deciding whether to web search during triage).
- ChatSystem is already large; adding agent workflow code to it would bloat it further. Moving the tool loop out is cleaner.
- The Phase 1 cleanup (removing service lifecycle hooks) simplified ChatSystem's tool execution path, making extraction easier.

## Key design points discussed

- The shared ToolLoop should work for both interactive (persona/ChatSystem) and non-interactive (agent) contexts
- Agents should be able to mix hardcoded pipeline steps with optional tool-use loops
- ToolManager and tool-related code fits naturally in `src/tools/`
- There was a discussed challenge around tool registration (ToolManager currently registers handlers at startup; agents may need different tool subsets) — details were lost to compaction
- The naming discussion (personas vs agents) was deferred but feeds into this: if both share a ToolLoop, the distinction becomes "conversational + interactive" vs "autonomous + scheduled"

## Status

**SHIPPED (reconciled 2026-06-04).** `ToolLoop` extracted into `src/tools/tool_loop.py` with an event-yielding `run()`; `_orchestrate` is now a thin forwarder. Delivered under DP-104, superseded by `tool_revamp_v1.md` (event-yielding shape). Retained as historical reference.
