---
name: Platform ideas (from OpenClaw + OpenCode analysis)
description: Prioritized feature ideas for engine evolution — context compaction, failover, skills, subagents, event bus, etc.
type: project
---

**Source:** OpenClaw + OpenCode comparisons (2026-03-24)

## Tier 1 — High Value
1. Context Compaction — structured summary at ~90% of model limit
2. Tool Output Pruning — replace old tool results with stubs
3. Model/Provider Failover — `fallback_model` on Persona; retry on 429/5xx
4. Tool Permission Gating — three-state (allow/ask/deny) per tool, persona-level profiles
5. Layered System Prompt Composition — pipeline instead of monolithic prompt

## Tier 2 — Architectural Improvements
6. Skills / Instruction Loading — markdown files in `skills/`, loaded on demand
7. SubAgent / Task Delegation — independent LLM session with restricted tools
8. Event Bus — async pub/sub for agent events
9. Plugin Hooks — `tool.execute.before`, `tool.execute.after`, etc.
10. Workflow Pipelines — YAML-defined multi-step workflows

## Tier 3 — Future / Watch
11. Named A2A Routing
12. Git-Based State Snapshots
13. Additional Messaging Channels
14. Browser/CDP Tools

**Suggested execution order:** Context compaction -> model failover -> layered prompts -> skills -> tool permission gating -> subagent delegation -> event bus -> plugin hooks -> workflows
