# Capability Map

**What this is for:** the other two docs are indexed by *module*. `architecture.md`
answers "what is `src/foo.py`?" and `user_guide.md` answers "what can I do from
Discord?". Neither can answer **"what already implements capability X?"** — so
when a capability gets built a second time, nothing in either doc looks wrong.
The new module simply gets a new section.

This file is the inverse index: capability → implementations. Its job is to make
a second implementation *visible* at the moment someone adds one.

**Read this before building anything that parks, schedules, stores, notifies,
retries, or spawns.** Search by intent, not by name — a re-derived implementation
never shares the name of the original (DP-302).

**Verdict column:**

- `single` — one owner. Reuse it.
- `by design` — several implementations, deliberately, with a recorded reason.
  Adding a third needs the same bar: a decision record, not a preference.
- `unreviewed` — several implementations, no recorded decision. Not automatically
  wrong; nobody has ruled. These are the audit queue.

Keep rows sorted by capability. When you add a capability, add a row. When you
add a *second* implementation of an existing capability, change the verdict to
`by design` and link the decision — or don't write it.

---

## Human-in-the-loop

| Capability | Implementations | Verdict |
|---|---|---|
| Park a write call for human approval | `confirmations.ConfirmationManager` (chat turns, in-memory, `token`-correlated, 1 pending per user+persona, resumed through the tool loop) · `proposals` queue (agent writes, SQLite `Proposals`, `proposal_id`, approved by LLM tool, run by `ProposalExecutor`) | `by design` — ADR 2026-07-04; the in-memory one is chat-turn-bound and cannot hold an agent's write across restarts. `tools/mcp_bridge.py:192` correctly reuses `create_proposal` rather than adding a third. |
| Approve/deny a parked item | `POST /api/v1/persona/{name}/confirm` (portal/HTTP) · `approve_proposal`/`deny_proposal` LLM tools | `by design` — same split as above, one surface each. |
| Decide whether a call needs approval | `tools/tool_loop.py` universal write-audit (`WRITE_TOOLS` membership) · `mcp_bridge._is_gated` (`is_write` OR `capabilities.irreversible`) · `tools/classifiers.py` argument-aware reversibility | `unreviewed` — three predicates answering "is this dangerous?". They agree today by convention, not by construction. |
| Ask a running subagent a question | `self_edit` `answer_agent` tool + `AgentRecord.status == waiting` | `single` |

## Scheduling and deferred work

| Capability | Implementations | Verdict |
|---|---|---|
| Run work on a repeating interval | `agents/base.Agent._wait_for_next_run` (`schedule.interval` or `daily_at`, respects `_shutdown_event`) · `MemoryConsolidator.start_daemon` (bare `while True` + `asyncio.sleep`, **no shutdown event — cannot be stopped cleanly**) · `mcp_client` maintenance loop (`MCP_RECONNECT_INTERVAL`, own wake/rate-limit logic) · `app_manager.register_task` (hosts the above) | `unreviewed` — three loop implementations with **different shutdown semantics**, which is the concrete cost: only the `Agent` one drains on stop. Verified 2026-07-27. |
| Fire once at a future time | `voice/timer.TimerService` (in-memory) · Zammad `pending reminder` state via `ProposalExecutor` · `ReminderAgent` polling | `unreviewed` |
| Expire/GC stale state | `expire_stale_proposals()` (lazy sweep on read) · `prune_agents` (fixr reaper) · LRU caps (`MAX_CONVERSATION_TAINTS`, `MAX_CACHED_API_REQUESTS`) · Hindsight 24h doc-scope gap | `unreviewed` — four expiry idioms; lazy-sweep-on-read is the only one that survives a restart. |

## State and storage

| Capability | Implementations | Verdict |
|---|---|---|
| Persist structured state | `memory_manager` main SQLite · `self_edit/store.AgentStore` (`CC_FIXR_REGISTRY_DB`) · `_DocScopeStore` (`hindsight_doc_scope.db`) · `_TrustOverrideStore` (`hindsight_overrides.db`) · `data/mcp_servers.json` · `data/personas.json` | `unreviewed` — **six stores, four of them SQLite.** The sibling-DB pattern was chosen per-feature, never as a policy. Highest-value row in this table. |
| Read/write a row with the shared connection | `MemoryManager._get_connection`/`_lock` · `SqliteSemanticBackend` reaching back through `mm._lock` | `by design` — DP-108 layer split; the backend deliberately shares one connection so transactions span both layers. (Note: `arch_audit similar` scores 5 method pairs across these two at 0.57-0.63. That is the *cost* of the design, not a defect.) |
| Record an audit trail | `memory_manager.log_audit_event` · `Agent_Actions` trajectory rows (`_log_task_root`/`_log_step`) · proposal review columns | `unreviewed` |

## LLM invocation

| Capability | Implementations | Verdict |
|---|---|---|
| Call an LLM | `TextEngine.stream_messages` (streaming) · `TextEngine.generate_response` (one-shot) | `single` — everything routes here. |
| Run a tool-calling loop | `tools/tool_loop.ToolLoop` · Claude Code's own loop (`cc-*` provider, DERPR's tools ignored by design) | `by design` — documented in `architecture.md`; `cc` is a clamped text provider, not a peer loop. |
| Get structured output without the tool loop | `sqlite_consolidator` (`submit_memory_summary`) · `ManagrAgent` second planner call (`submit_proposals`) · `DateTagger` / `ContentClassifier` inference agents | `by design` — "consolidator pattern", decision `2026-07-22-single-shot-inference-agents`. |
| Build the message array for a provider | `request_builder.format_raw_history_for_llm` · `stream_engine._build_messages` · `engine/driver._messages_to_history_object` · `engine/providers/_shared.extract_system_prompt` · `providers/google.build_google_history` | `unreviewed` — **five**. `arch_audit similar` pairs three of them at 0.61-0.67. Provider-shape divergence is real, but the *history* half is not provider-specific. |
| Speak the inline `<tool_call>` text protocol | `src/text_tool_protocol.py` | `single` — DP-200 slice C unified agy/cc/stream_engine. **This is the model the rows above should follow.** |

## External processes and hosts

| Capability | Implementations | Verdict |
|---|---|---|
| Spawn a `claude` CLI subprocess | `engine/providers/cc.py` (as an LLM provider) · `self_edit/dispatcher.py` (as a supervised coding agent) | `by design` — different lifetimes (request-scoped vs detached+resumable). The sandbox-settings divergence recorded here before DP-314 is **resolved**: both now delegate to `utils/cc_sandbox.build_sandbox_settings()` and contribute only their own deltas (bridge host, notes path). `tests/utils/test_cc_sandbox.py` pins the parity and fails if they drift again. |
| Build the `claude --settings` sandbox block | `utils/cc_sandbox.build_sandbox_settings` | `single` — unified in DP-314; was two independent copies at 0.68 similarity. |
| Clone a git repo for an agent to work in | `self_edit/clone_manager` (derpr's own source: pristine base + per-dispatch worktree) · `utils/notes_workspace` (the notes repo: ONE shared clone that advances) | `by design` — opposite mutability on purpose. The base clone is never advanced because worktrees hang off it; the notes clone IS the working copy every agent edits. Shared git plumbing lives in `utils/git_support`. |
| Run a command on a remote host | `proxmox/ssh.SSHRunner` (argv-list, never a shell string) | `single` |
| Expose derpr tools to an external caller | `tools/mcp_bridge` (derpr as MCP **server**) | `single` |
| Consume an external tool server | `tools/mcp_client` (derpr as MCP **client**) | `single` |

## Messaging out

| Capability | Implementations | Verdict |
|---|---|---|
| Notify a human | `clients/notification.NotificationRouter` + `Notifier` impls (`DiscordNotifier`, `DiscordChannelNotifier`, `ZammadNotifier`, `LogNotifier`, voice `WebAlarmNotifier`) | `single` — the router is the pattern to copy. |
| Talk to Discord | `interfaces/discord_bot.CustomDiscordBot` (the one real client) · `clients/notification.Discord*Notifier` (wraps it) · `voice/capture.DiscordVoiceCapture` (voice transport) · `self_edit/integration.DiscordThreadClient` (**a `Protocol`**, structurally satisfied by `CustomDiscordBot` — not an implementation) | `single` — one client, three typed views of it. Checked 2026-07-27. |
| Resolve who to notify | `date_tagger._resolve_recipient` · `reminder_agent._resolve_recipient` · `dispatch_agent._resolve_recipient` | **accidental** — three copies of one function, similarity 0.57-0.62, same package. Lift to `agents/base.py`. |

## HTTP surface

| Capability | Implementations | Verdict |
|---|---|---|
| Serve HTTP | `interfaces/kobold_engine_adapter` FastAPI app (owns the lifespan) · `interfaces/portal_render` · `voice/web.attach_web` (mounts onto the adapter app) · `mcp_bridge` (ASGI sub-app on the same) | `by design` — one app, three mounts. |
| Authenticate a caller | `DERPR_CONTROL_TOKEN` control-plane middleware · `BridgeTokenStore` per-dispatch bearer tokens | `by design` — two principals, two credentials, documented in `mcp_bridge`. |

---

## Maintaining this file

The failure mode is not that this file is wrong — it is that it goes stale and
nobody notices, exactly like the import-linter contracts did before DP-302.

- Adding a capability, a `ServiceIntegration`, a store, a loop, or a notifier →
  add or update a row **in the same commit**.
- `python scripts/arch_audit.py similar concepts` surfaces *function*-level
  regeneration mechanically. It cannot see subsystem-level parallelism — two
  subsystems on different primitives share no vocabulary, no shape and no field
  names. That is the gap this file exists to cover, and it can only be filled by
  a human or an agent reading intent.
- Every `unreviewed` row is a question, not an accusation. Resolving one means
  either consolidating, or writing the decision record that makes it `by design`.
