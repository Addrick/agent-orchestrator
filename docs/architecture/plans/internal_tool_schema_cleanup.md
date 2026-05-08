---
name: Internal tool schema cleanup
description: Fix submit_core_profile/submit_memory_summary living in ALL_TOOL_DEFINITIONS when they are agent-internal schemas, not user-invocable tools
type: project
---

## Context

During the per-persona long-term memory toggle implementation (2026-04-15), a code review found two design issues with internal memory tools:

1. `submit_core_profile` was in `ALL_TOOL_DEFINITIONS` with no registered handler → wiring test failed. A stub handler was added as a stopgap.
2. `submit_memory_summary`'s stub handler has wrong parameter names (`facts` vs `observations`, missing `keywords`).

Both tools share the same pattern: agents (MemoryAgent, MemoryConsolidator) pass them inline to a single TextEngine call and intercept the LLM response directly — ToolManager.execute_tool is never called for them. Their presence in `ALL_TOOL_DEFINITIONS` is an architectural mistake.

## The Problem

`ALL_TOOL_DEFINITIONS` is the global registry of user-facing tools. All tools in it are:
- Exposed to any persona with `enabled_tools: ["*"]` (joy, it-help)
- Visible in `what tools` output
- Subject to the wiring test requiring an executable handler

`submit_memory_summary` and `submit_core_profile` are LLM-facing schemas used as structured output contracts for specific agent LLM calls. They are not user tools. Exposing them to user personas is a bug — if the LLM calls them through a user conversation, the stub handlers either silently no-op or TypeError.

## Plan

### Step 1 — Move `submit_core_profile` inline to MemoryConsolidator

In `src/memory/memory_consolidation.py`, replace:
```python
from src.tools.definitions import ALL_TOOL_DEFINITIONS
l2_tool_def = next(t for t in ALL_TOOL_DEFINITIONS if t.get('function', {}).get('name') == 'submit_core_profile')
```
With a locally-defined dict literal (copy the schema from definitions.py). Remove `submit_core_profile` from `ALL_TOOL_DEFINITIONS`.

### Step 2 — Move `submit_memory_summary` inline to MemoryAgent

In `src/agents/memory_agent.py`, `_summarize_segment` calls:
```python
tools_for_llm = self.chat_system._filter_tools_for_persona(persona)
```
This relies on `submit_memory_summary` being in `ALL_TOOL_DEFINITIONS` and in the `memory_summarizer` persona's `enabled_tools`. Replace with a locally-defined list containing just the `submit_memory_summary` schema (copy from definitions.py). Pass it directly to `text_engine.generate_response()`.

Remove `submit_memory_summary` from `ALL_TOOL_DEFINITIONS`.

### Step 3 — Remove stubs from MemoryToolHandler

Once neither tool is in `ALL_TOOL_DEFINITIONS`, remove:
- `manager.register("submit_memory_summary", ...)` from MemoryToolHandler.register
- `manager.register("submit_core_profile", ...)` from MemoryToolHandler.register
- `_submit_memory_summary` method
- `_submit_core_profile` method

### Step 4 — Update memory_summarizer persona config

`config/system_personas.json`: `memory_summarizer.enabled_tools` currently contains `["submit_memory_summary"]`. This no longer needs to be there since the tool is passed inline. Set to `[]`. (The `_filter_tools_for_persona` call in MemoryAgent will be removed anyway.)

### Step 5 — Update tests

- `tests/integration/test_startup_wiring.py` — no changes needed; the tools are gone from ALL_TOOL_DEFINITIONS so the assertion naturally passes
- `tests/agents/test_memory_agent.py` — `_summarize_segment` tests mock the text engine response; confirm they still pass after the inline tool change
- `tests/memory/test_memory_consolidation.py` — same; check consolidation tests still pass

### Step 6 — Verify `what tools` output

Run `what tools` against a persona with `*` enabled tools and confirm `submit_memory_summary` and `submit_core_profile` no longer appear.

## Secondary issue: `include_ambient_memory` / `long_term_memory` dependency

When `long_term_memory` is off, `include_ambient_memory` has no effect. Currently the `detail` output shows both independently. Low priority — could add a note in the detail output like `Include Ambient Memory: on (inactive — long_term_memory is off)`.

## Secondary issue: `thinking_level` validation

`set thinking_level <value>` stores any string verbatim. If invalid, the engine will fail at call time. Acceptable for now (power-user setting, engine error is informative enough), but could add a known-values list (none/minimal/low/medium/high/max) as a soft warning. Not blocking.

## Why: decisions made this session

- `submit_core_profile` stub was added as a stopgap to unblock the wiring test. Decision: **remove, don't stub**.
- Both internal tools follow the "intercept LLM response directly" pattern. This pattern should be the authoritative signal that a schema is agent-internal, not a user tool.
- `ALL_TOOL_DEFINITIONS` should only contain tools that can be meaningfully executed via ToolManager by a user persona.
