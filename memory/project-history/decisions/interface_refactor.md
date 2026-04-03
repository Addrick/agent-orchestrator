---
name: Interface layer refactor — split Discord/Gmail like Zammad
description: Discord and Gmail combine client + service integration in one file; should be split for consistency
type: project
---

Discord and Gmail each combine client work (API calls) and service integration work (persona detection, context resolution) in a single interface file. Zammad has these properly separated: ZammadClient, ZammadServiceIntegration, and the agent.

**Target architecture:**

| Layer | Discord | Gmail | Zammad |
|-------|---------|-------|--------|
| Client | `discord_client.py` (maybe skip) | `gmail_client.py` | `zammad_client.py` (done) |
| Service Integration | `discord_service.py` | `gmail_service.py` | `zammad_service.py` (done) |
| Interface | `discord_bot.py` (slimmed) | `gmail_bot.py` (slimmed) | N/A (agent-based) |

**Status:** Not urgent. Extract service integration logic when touching Discord/Gmail for other reasons.
