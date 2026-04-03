# Memory Index (L0)

This memory system uses a tiered L0/L1/L2 structure inspired by OpenViking.
See CLAUDE.md "Memory System" section for navigation and update rules.

## codebase/
Async LLM orchestration engine — ChatSystem pipeline, 4 providers, 3 interfaces,
agent framework, tool system (ServiceIntegration is tool-registration-only), SQLite storage.
Read `_overview.md` for component map, `architecture.md` for full structural detail.

## user/
Developer with ML/robotics background, values correctness and testing discipline.
Key rules: ask before investigating, fix root causes, remind to commit.
Read `_overview.md` for collaboration guide, `feedback.md` for all behavioral rules.

## project/
Active work: long-term memory system (planned, not started), agent expansion, ToolLoop extraction.
Memory plan in `plans/long_term_memory.md`. Read `_overview.md` for current status.

## project-history/
Archive of completed plans, decisions, and research. Not loaded by default.
Read `_overview.md` only when you need rationale behind a past choice.

## external/
GitHub repo: Addrick/llm-orchestrator. Google free-tier rate limits. TeamPCP supply
chain attack (March 2026). Claude Code config (autocompact at 90%, settings scope).
Read `_overview.md` for quick reference.
