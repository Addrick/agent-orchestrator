---
name: Claude Code CLI as engine provider — cost analysis
description: Subscription likely cheaper than API for Opus-heavy usage; architectural trade-offs documented
type: project
---

**Date:** 2026-03-27

**Context:** Evaluated using Claude Code CLI (`claude -p`) as an LLM provider in TextEngine, piping through the subscription instead of paying per-token API costs.

**Cost conclusion:** For Opus at high effort (software engineering tasks), the Max subscription ($100-200/mo) is almost certainly cheaper than API pricing ($15/$75 per MTok input/output). A single meaty conversation can easily cost $5-10+ on the API.

**Architectural trade-offs:**
- Every CLI invocation carries overhead (Claude Code's system prompt, tool definitions, memory) — double-wrapping since TextEngine already manages context
- Shelling out to CLI is fragile (process management, parsing, error handling, latency)
- The interesting use case is Claude Code as a *subagent* for tasks needing environment access (code analysis, test running), not as a general LLM provider
- For standard persona conversations where full context control is needed, the direct Anthropic API is cleaner

**Decision:** Not implemented yet. Worth revisiting if API costs become a concern. The subagent pattern (dispatching to Claude Code for environment-aware tasks) maps more naturally to AgentLoop than to TextEngine.
