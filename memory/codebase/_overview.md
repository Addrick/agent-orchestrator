---
name: Codebase overview (L1)
description: Component map, relationships, and key patterns — read this before drilling into architecture.md
type: reference
---

Async, provider-agnostic LLM orchestration engine for chatbot automation (IT support, ticketing, conversational AI). Python 3.14, async throughout.

**Pipeline:** Interface -> ChatSystem (DI hub) -> BotLogic (commands) -> TextEngine (LLM) -> ToolManager (tool loop, max 5) -> MemoryManager (SQLite)

**Interfaces:** Discord bot (primary, interactive), Gmail bot (polling, no persistence), ZammadBot (agent-based polling)

**Providers:** OpenAI, Anthropic, Google (Gemini 2.5/3.1, Gemma), local OpenAI-compatible. Rate limiters split by model family.

**Persona system:** Stateful LLM config objects with ExecutionMode (AUTONOMOUS, CONFIRM) and MemoryMode (CHANNEL_ISOLATED, SERVER_WIDE, PERSONAL, GLOBAL, TICKET_ISOLATED — note: TICKET_ISOLATED is dormant since lifecycle hook removal). Runtime-mutable via `set` commands. System personas are read-only configs used by agents for analysis.

**Agent framework:** Agent ABC (src/agents/base.py) → AgentManager lifecycle → AppManager top-level coordinator. Currently: ZammadBot (multi-stage triage via 4 system personas), DispatchAgent (priority assessment + config-driven notification). Autonomous background workers on interval schedules. Config-driven via agents.json. Agents invoke system personas for read-only LLM analysis; do not spawn other agents. Agent tools (status/history/manage) gated behind service_bindings.

**Notification system:** NotificationRouter → Notifier ABC (DiscordNotifier, ZammadNotifier, LogNotifier). Decoupled from agents — channel/recipient config-driven.

**Tool system:** JSON schemas in definitions.py, read-only vs write (confirmation gating in CONFIRM mode), service-binding filtered. ServiceIntegration ABC is a tool-registration-only interface (lifecycle hooks removed 2026-03-28). Agent tools registered via AgentServiceIntegration.

**Storage:** SQLite — User_Interactions (conversations), Suppressed_Interactions, Agent_Actions + Agent_Action_Contexts. All async via to_thread().

**Config:** global_config.py (limits, rate limits), default_personas.json + system_personas.json (git-tracked), data/personas.json (local override), agents.json.

**Testing:** 4-tier (unit, integration, zammad-live, llm-live). No pre-commit test hook — tests run manually. Migration tests via legacy_mem_manager fixture.

**Docs:** `docs/user_guide.md` — user-facing behavior spec (commands, personas, tools, interfaces). Also serves as spec-before-implement target for new features.

**Dead code:** `interfaces/kobold_api.py` (unused), `interfaces/zammad_bot.py` (deprecated stub re-exporting from agents/). Both candidates for removal.

For full detail on any component: read `architecture.md`. For project roadmap: read `roadmap.md`.
