---
name: Agent expansion plan
description: Infrastructure status + next steps for agent framework (dispatch, notifications, lifecycle management)
type: project
---

**Updated:** 2026-03-27

## Completed Infrastructure
- `src/agents/base.py` — Agent base class (renamed from AgentLoop) with async deploy(), persona injection, step logging
- `src/agents/dispatch_agent.py` — Dispatch pipeline: LLM assesses priority/summary, channel/recipient config-driven via agents.json
- `src/agents/zammad_bot.py` — Triage pipeline (moved from src/interfaces/), managed by AgentManager
- `src/agents/agent_manager.py` — Central registry: register, start/stop/restart, config merging, auto_start
- `src/agents/agent_service.py` — AgentServiceIntegration gates tools behind `service_bindings: ["agents"]`
- `src/tools/agent_tool_handler.py` — Tools: get_agent_status, get_agent_history, manage_agent
- `src/clients/notification.py` — NotificationRouter + DiscordNotifier, ZammadNotifier, LogNotifier
- `src/database/memory_manager.py` — Agent_Actions + Agent_Action_Contexts tables
- `config/agents.json` — Agent configs with auto_start, schedule, notification_defaults

## Next Steps
1. **ReminderAgent** — scheduled open-ticket nudges
2. **EmailNotifier** — extract GmailClient for outbound (backburner)
3. **Remove APScheduler from requirements.txt** — already removed from requirements.in
4. **Remove DISPATCH_ENABLED / ZAMMAD_BOT_ENABLED flags** — agents.json auto_start replaces them
