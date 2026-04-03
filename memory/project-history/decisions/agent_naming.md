---
name: Agent naming collision
description: "Agents" should be "subagents" — conversational personas are also agentic; naming to revisit
type: project
---

The conversational persona system is itself agentic — personas use tools, have execution modes, etc. The polling-based "agent" system (ZammadBot, DispatchAgent) should really be called "subagents" since they're subordinate automated workers.

Two competing paradigms share the term: conversation uses `User_Interactions` DB history, agents use single-shot LLM calls with `Agent_Actions` history.

**Decision:** Defer major rename until agent system has more usage and clearer requirements. When expanding either system, prefer "subagent" or "worker" for the polling agents to disambiguate.
