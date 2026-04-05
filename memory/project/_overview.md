---
name: Project overview (L1)
description: Current active work and live plans — read this for what's in-flight right now
type: project
---

## Active Work
- **Long-term memory system** — OpenViking-inspired embedding-based semantic memory. Batch agent segments/summarizes older messages by topic, retrieval injects relevant summaries before sliding window. 4 phases planned, reviewed (5 scrutiny passes, 2026-04-03/04), ready for implementation. See `plans/long_term_memory.md`
- **Agent expansion** — dispatch pipeline complete, next: ReminderAgent, cleanup deprecated flags
- **ToolLoop extraction** — extract `_run_tool_loop()` from ChatSystem into shared `tools/` module so agents can use tool loops. Phase 1 (service lifecycle removal) complete, Phase 2 not started. See `plans/toolloop_extraction.md`
- **Documentation** — `docs/user_guide.md` + `memory/codebase/architecture.md` as living specs

## Live Plans
- `plans/long_term_memory.md` — embedding-based long-term memory (Gemini Embedding, MemoryAgent, fact extraction, retrieval injection)
- `plans/agent_expansion.md` — agent framework next steps (ReminderAgent, cleanup)
- `plans/toolloop_extraction.md` — shared tool loop for personas and agents

## MSP Productionization
This codebase is transitioning from personal research platform to operational MSP backend. Goal: automate most tasks. Current live agents: autotriage (hardcoded tool loop, suggests fixes + context), dispatch (reviews new tickets, has tool access). All Zammad writes require human approval. No customer-facing agents yet — internal reliability must come first.

**Security posture:** Privilege separation (ingest ≠ act), approval gates on all writes, untrusted input treated as adversarial. Main open risk: prompt injection via ticket/email content. No reliable sanitization exists for natural language; architecture is the mitigation. Hallucinations low in practice (structured IT domain), risk increases when customer-specific context (device inventory, ticket history) is added.

## Roadmap
Primary roadmap in `codebase/roadmap.md` — prioritized backlog of all planned work.

## Historical Reference
Completed plans, immutable decisions, and research live in `project-history/`.
Only read when you need rationale behind past choices.
