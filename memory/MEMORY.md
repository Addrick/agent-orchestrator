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
Active work: long-term memory, agent expansion, ToolLoop extraction, MSP productionization.
Read `_overview.md` for current status and live plans.

## project-history/
Archive of completed plans, decisions, and research. Not loaded by default.
Read `_overview.md` only when you need rationale behind a past choice.

## external/
External system references: repo, API limits, supply chain incidents, Claude Code config.
Read `_overview.md` for quick reference.
