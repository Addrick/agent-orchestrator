---
name: Codebase overview (L1)
description: Component map, relationships, and key patterns — read this before drilling into architecture.md
type: reference
---

Async, provider-agnostic LLM orchestration engine for chatbot automation (IT support, ticketing, conversational AI). Python 3.14, async throughout.

**Pipeline:** Interface -> ChatSystem (DI hub) -> BotLogic (commands) -> TextEngine (LLM) -> ToolManager (tool loop, max 5) -> MemoryManager (SQLite, transcript layer) -> MemoryBackend (semantic + episodic; default `SqliteSemanticBackend`; `HindsightBackend` shipped and merged to master, selectable via `SEMANTIC_BACKEND=hindsight`). Engine-side retrieval/consolidation now flows through `MemoryBackend.recall` / `retain_turn` (DP-113); transcript layer (log_message, suppression, audit) still goes via MemoryManager. Cross-persona fan-out recall available via `src/memory/router.py` `MemoryRouter` (Sprint 4 groundwork; no production caller yet — wired in Sprint 5 metabank). Per-turn scope (persona/channel/user/server) is pinned in `src/tools/turn_context.py` so engine-side tools can inherit context without it appearing as model-callable args.

**Interfaces:** Discord bot (primary, interactive), Gmail bot (polling, no persistence), ZammadBot (agent-based polling), KoboldEngineAdapter (FastAPI; legacy kobold-lite portal at `/portal` + bespoke React/Vite/TS portal at `/derpr` (DP-132–137), both on :5003; OAI route `/v1/chat/completions` is a thin SSE transcoder over `chat_system.stream_response` (Phase D, 2026-04-28); native `/api/v1/generate` + `/api/extra/generate/stream` live on as DB-logging forwarders to local KoboldCPP (the standalone :5002 `kobold_adapter.py` app was retired in DP-200/203); persona CRUD incl. `POST /api/v1/personas` create (DP-231), DB-as-source history export, version-chevron + transcript API)

**Providers:** OpenAI, Anthropic, Google (Gemini 2.5/3.1, Gemma), local kobold-native (StreamEngine, DP-206b), Antigravity (`agy` CLI, DP-127), Claude Code (`cc-*` sandboxed CLI, DP-222 — runs its OWN sandboxed tool loop, DERPR's `tools` arg ignored). Rate limiters split by model family.

**Persona system:** Stateful LLM config objects with ExecutionMode (AUTONOMOUS, CONFIRM) and MemoryMode (CHANNEL_ISOLATED, SERVER_WIDE, PERSONAL, GLOBAL, TICKET_ISOLATED). Note: execution mode still affects UI presentation, but **all write tools park for audit** regardless of mode per the security framework. Runtime-mutable via `set` commands. System personas are read-only configs used by agents for analysis.

**Agent framework:** Agent ABC (src/agents/base.py) → AgentManager lifecycle → AppManager top-level coordinator. Currently: ZammadBot (multi-stage triage via 4 system personas), DispatchAgent (priority assessment + config-driven notification), ReminderAgent (scheduled open-ticket nudges — shipped), and SqliteConsolidator (SQLite memory consolidation, registered only when `SEMANTIC_BACKEND=sqlite`). Autonomous background workers on interval schedules. Config-driven via agents.json. Agents invoke system personas for read-only LLM analysis; do not spawn other agents. Agent tools (status/history/manage) gated behind service_bindings.

**fixr self-improvement supervisor (DP-227):** `src/self_edit/` — event-driven dispatcher above the engine. `dispatch_fix` (WRITE→parked) spawns a detached `claude` coding-agent subprocess per bug in an isolated git worktree; a bridge tails its stream-json log and wakes the `fixr` persona on question/done/error. `FixrIntegration` (ServiceIntegration, `service_bindings:["fixr"]`) registered at startup. Never merges/pushes — a human merges the PR.

**Security (DP-225):** `src/security/` — `CredentialVault` inventories machine secrets (OpenAI/Anthropic/Google/Zammad keys); `SecretScrubber` redacts them from any string bound for the LLM context / audit / inspector. Wired at startup via `bootstrap.register_credentials()` (vault→scrubber), enforced at egress in tool_loop, turn_persistence, engine, zammad_client.

**Notification system:** NotificationRouter → Notifier ABC (DiscordNotifier, ZammadNotifier, LogNotifier). Decoupled from agents — channel/recipient config-driven.

**Tool system:** JSON schemas in definitions.py, read-only vs write (all write tools go through human confirmation — parked for audit regardless of execution mode), service-binding filtered. ServiceIntegration ABC is a tool-registration-only interface (lifecycle hooks removed 2026-03-28). Agent tools registered via AgentServiceIntegration.

**Storage:** SQLite — User_Interactions (conversations), Suppressed_Interactions, Agent_Actions + Agent_Action_Contexts. All async via to_thread().

**Config:** global_config.py (limits, rate limits), default_personas.json + system_personas.json (git-tracked), data/personas.json (local override), agents.json.

**Testing:** 4-tier (unit, integration, zammad-live, llm-live). No pre-commit test hook — tests run manually. Migration tests via legacy_mem_manager fixture.

**Docs:** `docs/user_guide.md` — user-facing behavior spec (commands, personas, tools, interfaces). Also serves as spec-before-implement target for new features.

**Dead code:** none known (the deprecated `interfaces/zammad_bot.py` stub was removed by DP-203).

For full detail on any component: read `architecture.md`. For project roadmap: read `roadmap.md`.
