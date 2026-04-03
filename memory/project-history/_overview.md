---
name: Project history overview (L1)
description: Archive of completed plans, immutable decisions, and reference research — only read when you need historical rationale
type: project
---

Historical reference. Only drill into L2 files when you need the *rationale* behind a past decision or the details of completed work.

## decisions/
Immutable records of past architectural choices with rationale:
- **agent_naming.md** — "agents" vs "subagents" naming collision; deferred, prefer "subagent"/"worker" for polling agents
- **service_lifecycle_removal.md** — ServiceIntegration stripped to tool-registration-only; all lifecycle hooks and service_data pipeline removed (2026-03-28)
- **claude_code_as_provider.md** — CLI-as-provider evaluated (2026-03-27); subscription cheaper than API but subagent pattern preferred over TextEngine integration
- **interface_refactor.md** — Discord/Gmail should split into client/service/interface layers like Zammad; not urgent, do when touching those files
- **personas_deployment.md** — default/system personas tracked in git, data/personas.json gitignored for local dev
- **python314_upgrade.md** — Completed 2026-03-20; upstream deprecation warnings from google-genai and discord.py noted
- **test_reorg.md** — 4-tier test system (unit/integration/zammad-live/llm-live) established 2026-03-18
- **viking_memory_system.md** — L0/L1/L2 tiered memory adopted 2026-03-27; soft-implementation of OpenViking patterns

## plans/
Completed plans:
- **tool_context_logging.md** — Complete (commit 4ad4186). Logging moved into ChatSystem, tool_context stored as JSON.

## future-prospects/
Ideas considered but without concrete plans yet — may revisit:
- **viztracer.md** — Runtime call trace visualization; deferred until parallel dev environment exists
- **mcp_strategy.md** — Phased MCP integration (client → Zammad server → engine server → registry); blocked on tool permissions
- **platform_ideas.md** — Feature ideas from OpenClaw/OpenCode analysis with suggested execution order

## research/
Reference material and comparisons (not commitments). Empty — use for future research docs.
