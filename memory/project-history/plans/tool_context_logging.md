---
name: Tool context logging plan
description: Approved — move logging into ChatSystem + store tool context as JSON on assistant rows
type: project
---

Approved plan saved at `C:\Users\Adam\.claude\plans\woolly-booping-marble.md`. Ready for implementation.

**Why:** Tool calls/results are ephemeral (lost between turns) and logging is owned by the Discord bot instead of ChatSystem, making it hard to attach internal details like tool context.

**Key changes:** Schema migration (tool_context TEXT column), move log_message calls from Discord bot into ChatSystem.generate_response, collect tool messages during tool loop, reconstruct during history formatting. Discord bot retains ambient logging only, plus lightweight update_platform_message_id callback.
