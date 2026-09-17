---
name: Project roadmap
description: Prioritized agent-orchestrator roadmap — grounded in actual needs, informed by OpenClaw/OpenCode comparisons and design conversations
type: project
---

> ⚠️ **Staleness warning — read before planning from this file.** Reviewed 2026-09-17; it had
> not been substantively touched since 2026-07-03 (`d6bad35`, DP-268). In that window
> DP-277, DP-297/319, DP-306, DP-330, DP-335/338, DP-345, DP-354 and DP-359–365 all shipped
> and **none of them are reflected below.** Unticked here does not mean unbuilt. Check
> `../capability_map.md` and `../mechanism_ledger.md` before concluding anything is missing.
>
> Spot-checked and still accurate as written: **Interface layer refactor** — `src/interfaces/`
> really does still hold monolithic `discord_bot.py` / `gmail_bot.py` while Zammad is split
> across `agents/zammad_bot.py` + `clients/zammad_client.py` + `clients/zammad_service.py`.

## Cleanup (quick wins)

- [x] Remove `DISPATCH_ENABLED` / `ZAMMAD_BOT_ENABLED` from `global_config.py` — `agents.json` `auto_start` replaces them. **DONE** (both gone from `global_config.py`).
- [x] Remove `apscheduler` from `requirements.txt` — already gone from `requirements.in`. **DONE** (gone from both `requirements.txt` and `requirements.in`).

## Near-term

- [ ] **Uptime Kuma heartbeat** — add async background task that pushes a heartbeat to Uptime Kuma on an interval; only send when Discord gateway is healthy so Kuma reflects actual bot availability
- [x] **ToolLoop extraction** — SHIPPED. `ToolLoop` lives in `src/tools/tool_loop.py` (event-yielding `run()`; `_orchestrate` is a thin forwarder). Superseded by `tool_revamp_v1.md`. See `project/plans/toolloop_extraction.md`
- [x] **ReminderAgent** — SHIPPED. `src/agents/reminder_agent.py`, registered in `main.py` when Zammad is available. (Residual: configurable staleness/age threshold not yet landed — DP-102.)
- [ ] **OpenViking memory management** — in-engine memory for personas/conversations, inspired by OpenViking tiered retrieval
- [ ] **Model/provider failover** — `fallback_model` on Persona, retry on 429/5xx with alternate provider
- [ ] **Tool permission gating expansion** — per-tool allow/ask/deny profiles at persona level. ⚠️ **The "current" half of this line was two rewrites out of date and has been corrected:** gating is no longer emoji-reaction CONFIRM mode. Writes are audited **universally, regardless of execution mode**, through a composition of four predicates (DP-306) with `ALWAYS_CONFIRM_TOOLS` and `explicit_overrides`; see the Tool Security section of `../user_guide.md`. The allow/ask/deny *profile* shape is still unbuilt — that part of the item stands.

## Medium-term

- [ ] **Layered system prompt composition** — pipeline instead of monolithic prompt string
- [ ] **Skills / instruction loading** — markdown files in `skills/`, loaded on demand into system prompt
- [x] **MCP client (Phase 1)** — consume external MCP servers, expose their tools to the LLM as `mcp__<server>__<tool>`. See `plans/mcp_strategy.md` for the full phased plan. **MERGED (DP-268, PR #154, `a24996b`):** all phases 0–3 shipped (registry refactor + `MCPClientManager` + add/remove/list management tools + hot-reload maintenance loop).
- [ ] **Interface refactor** — Discord/Gmail should split into client/service/interface layers like Zammad

## Long-term / watch

- [ ] SubAgent / task delegation — independent LLM sessions with restricted tools
- [ ] Meta-Agent security framework — sandboxing and verification for agent-creating agents
- [ ] Event bus — async pub/sub for agent events
- [ ] MCP Phases 2-4 (Zammad as server, engine as server, registry)
- [ ] Additional messaging channels

## Source documents

- `plans/platform_ideas.md` — OpenClaw/OpenCode feature comparison (inspiration, not commitments)
- `plans/mcp_strategy.md` — phased MCP integration path
- `plans/agent_expansion.md` — agent framework status and next steps
