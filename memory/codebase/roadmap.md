---
name: Project roadmap
description: Prioritized derpr-python roadmap — grounded in actual needs, informed by OpenClaw/OpenCode comparisons and design conversations
type: project
---

## Cleanup (quick wins)

- [ ] Remove `DISPATCH_ENABLED` / `ZAMMAD_BOT_ENABLED` from `global_config.py` — `agents.json` `auto_start` replaces them
- [ ] Remove `apscheduler` from `requirements.txt` — already gone from `requirements.in`

## Near-term

- [ ] **Uptime Kuma heartbeat** — add async background task that pushes a heartbeat to Uptime Kuma on an interval; only send when Discord gateway is healthy so Kuma reflects actual bot availability
- [ ] **ToolLoop extraction** — extract `_run_tool_loop()` from ChatSystem into shared `tools/` module; enables agents to use tool loops alongside hardcoded pipelines. Phase 1 (service lifecycle removal) complete. See `project/plans/toolloop_extraction.md`
- [ ] **ReminderAgent** — scheduled open-ticket nudges, next agent after dispatch
- [ ] **OpenViking memory management** — in-engine memory for personas/conversations, inspired by OpenViking tiered retrieval
- [ ] **Model/provider failover** — `fallback_model` on Persona, retry on 429/5xx with alternate provider
- [ ] **Tool permission gating expansion** — current: CONFIRM mode gates write tools via emoji reactions. Next: per-tool allow/ask/deny profiles at persona level

## Medium-term

- [ ] **Layered system prompt composition** — pipeline instead of monolithic prompt string
- [ ] **Skills / instruction loading** — markdown files in `skills/`, loaded on demand into system prompt
- [ ] **MCP client (Phase 1)** — consume existing MCP servers, expose their tools to LLM. See `plans/mcp_strategy.md` for full phased plan
- [ ] **Interface refactor** — Discord/Gmail should split into client/service/interface layers like Zammad

## Long-term / watch

- [ ] SubAgent / task delegation — independent LLM sessions with restricted tools
- [ ] Event bus — async pub/sub for agent events
- [ ] MCP Phases 2-4 (Zammad as server, engine as server, registry)
- [ ] Additional messaging channels

## Source documents

- `plans/platform_ideas.md` — OpenClaw/OpenCode feature comparison (inspiration, not commitments)
- `plans/mcp_strategy.md` — phased MCP integration path
- `plans/agent_expansion.md` — agent framework status and next steps
