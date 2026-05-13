---
name: model/tool selectors use structured tool-call output
description: model_selector and tool_selector personas now use inline tool schemas with enum constraint, bypassing ChatSystem; plain-text output pattern abandoned
type: project
---

## Decision

The `model_selector` and `tool_selector` internal personas no longer produce plain text. They are invoked through a single-shot tool-call pipeline: `BotLogic._query_llm_with_selection_tool` builds an inline function schema with the choices list embedded as an `enum`, plus a `DEFAULT`/`NONE` sentinel, then calls `text_engine.generate_response` directly. No `chat_system.generate_response`, no channels, no conversation history, no memory retrieval.

## Why

Plain-text selection had three compounding failures on Gemma-4-31b-it:

1. Prompt asked model to "not repeat the list / query / instruction" — the user message carried `### INSTRUCTION:` / `### SELECTED MODEL NAME:` markdown headers, which is the exact pattern instruct-tuned models mirror back. The "don't echo" rule was fighting the user-message formatting.
2. Full `chat_system.generate_response` pipeline was used for a one-shot classifier, including `memory_mode: global` — risked polluting global memory with classifier queries.
3. Recovery code (`if "###" in model_name: substring-match fallback`) was symptom patching and ambiguous on overlapping model names like `gemma-3-27b-it` vs `gemma-3-27b-it-ablated`.

Tool-call with `enum` makes the problem structural: providers emit JSON args, nothing to echo, off-list values filtered by a single case-insensitive guard.

## How to apply

- Same pattern already proven by `submit_memory_summary` (MemoryAgent) and `submit_core_profile` (MemoryConsolidator): inline schema, direct TextEngine call, never register in `ALL_TOOL_DEFINITIONS`.
- When adding any new "classifier" style agent operation (pick-one-of-N, yes/no, structured extract), prefer this pattern over a plain-text persona with parse rules.
- `select_model` and `select_tool` are defined inline in `message_handler.py`, not in `ALL_TOOL_DEFINITIONS` — they are agent-internal schemas, not user tools.
- Persona entries in `system_personas.json` are kept as config carriers only (model name, temperature, token limit). Prompt shrunk to one imperative sentence naming the tool.
- Gemma tool-calling honors `enum` most of the time but not always; keep the `case-insensitive in-choices` guard as a cheap safety net.
