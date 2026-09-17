# Mechanism Ledger

**What this is for:** `architecture.md` is indexed by module, `user_guide.md` by
user-facing behavior, `capability_map.md` by capability-in-English. A re-derived
implementation is invisible to all three, and we have the receipts: the
capability map was created 2026-07-27 (DP-302), and DP-343 shipped a fourth copy
of the deferred-tool-result pipeline on 2026-08-22 — twenty-six days later,
*while filling in a capability-map row and recording `by design`*.

The control fired, produced a verdict, and the verdict was wrong. Two reasons,
and this file exists to fix the second one:

1. **Self-report cannot catch self-blindness.** Every other index is authored at
   write time by the author of the new code. Recognizing your code as a second
   implementation is exactly what you failed to do by writing it.
2. **The index key is wrong.** DP-343's author searched *"wake a persona on a
   non-user event"*, found `_wake_fixr`, compared against it, and ruled. The real
   sibling was filed under *"park a write call for human approval"* — different
   English, same machine. `capability_map.md` records the lesson ("search for the
   implementation with the most shared **mechanism**, not the most similar
   description") but is itself indexed by description, so it cannot obey it.

**English clusters by purpose. Duplicates cluster by mechanism.**

So this file is keyed on mechanism: five mechanical facts per row, all readable
off the source.

| column | question |
|---|---|
| **trigger** | what starts it |
| **durable store** | what survives a restart, and where |
| **idempotency** | what makes it exactly-once |
| **resumes / consumes** | what picks the work back up |
| **gate** | what may refuse it |

Two rows with the same shape and different module names *are* a duplicate. No
judgment call, no English, no author self-report. Read the shape, not the names.

**Regenerate, don't maintain.** Every column is derivable from source, so when
this file disagrees with the code, the code wins and this file gets rebuilt.
Rot here is cheap; rot in a hand-authored index is what produced DP-343.

**Scope:** `src/`. Rows are grouped into families; families are the unit of
comparison. A new subsystem that parks, schedules, stores, notifies, retries, or
spawns belongs in one of these families — if it doesn't fit any, that is itself
worth a second look before writing it.

Verified against `066d08d` (2026-08-23). **Re-checked 2026-09-17 at `7b87083`** — the
tickets merged since (DP-360–365) are infra, eval-harness and error-message changes that do
not touch mechanism families A–F, and `python scripts/arch_audit.py similar concepts`
surfaces no near-duplicate pair that `capability_map.md` has not already classified. ⚠️ That
is a **probe**, not a regeneration — see the caveat at the foot of this file: the ledger calls
itself "regenerated, not maintained" and **no regeneration script exists**. `arch_audit.py`
works at the *name* level that this file's own false-positive section calls the wrong one.

---

## A · Work that outlives the turn that started it

The DP-345 family. A tool call whose real answer arrives after its turn ended.

| trigger | durable store | idempotency | resumes / consumes | gate |
|---|---|---|---|---|
| Model calls a gated write | `Parked_Writes` (`kind='approval'`) | row claim (`take`) | `stream_resolve_park` → `_stream_settle` → one continuation turn | `is_write_tool` / `is_irreversible` / `ALWAYS_CONFIRM_TOOLS`; human approve/deny |
| Model calls `install_model`; node job detaches under `systemd-run` | `Parked_Writes` (`kind='node_job'`), token **is** the job id | row claim (`take`) | `stream_resolve_deferral` → `_stream_settle` → one continuation turn | `HF_TOOLS_ENABLED`; the inbound ping is unauthenticated (DP-355) |
| ⚠️ fixr agent emits an event (`self_edit/integration.py:307`) | **none** — nothing durable | **none** — in-process only | `_wake_fixr` → `ChatSystem.generate_response` (a *new* turn, not a continuation) | none |

**Finding A1 — `_wake_fixr` is the implementation DP-345 did not converge.**
`src/deferral_kinds.py:33` names it explicitly and defers it to future work
("`CC_FIXR_PERSONA` / `CC_FIXR_CHANNEL` are the same defect in fixr and go the
same way *when fixr moves onto this mechanism*"). That is an honest scope call,
but `capability_map.md` records the row's verdict as `single`, which overstates
it: the *mechanism* is single, a second *implementation* is still live. Three
concrete consequences, all visible in the shape above:

- **Coordinates come from config, not the turn** — `global_config.CC_FIXR_PERSONA`
  and `CC_FIXR_CHANNEL`. This is the invariant `deferral_kinds.py` states in bold
  and the exact defect that got `MODEL_JOB_WAKE_PERSONA/_CHANNEL/_USER` deleted.
- **It enters via `generate_response`, so `continuation is None`** — which per
  `capability_map.md` means `_orchestrate` **persists the synthetic wake text as a
  durable USER row**. DP-345 called this out as a live bug and removed it from the
  node-job path. It is still present here.
- **Nothing is durable and nothing is exactly-once**, so a restart mid-event loses
  the wake with no trace.

**Finding A2 — the empty `durable store` / `idempotency` cells are the tell.**
Not "fixr is sloppy" — a row with holes in columns its siblings fill is the
mechanical signature of a re-derivation, and it reads off the table without
anyone having to describe either implementation in English. This is the whole
thesis of the file: the shape found it, not the name.

---

## B · Fire once at a future time

| trigger | durable store | idempotency | resumes / consumes | gate |
|---|---|---|---|---|
| `set_timer` tool (`voice/tools.py:69`) | none — `voice/timer.TimerService`, in-memory | timer id, in-process | `TimerService.schedule` task | voice enabled |
| Proposal sets a Zammad `pending reminder` state | Zammad ticket state (external) | ticket id | Zammad, then `ProposalExecutor` | proposal approval |
| `ReminderAgent` | agent store | polling window | agent's own interval loop | agent schedule |
| ⚠️ fixr question goes unanswered (`self_edit/integration.py:325`) | none — bare `asyncio.create_task` + `asyncio.sleep` | `_idle_timers` dict, in-process | `_idle_fallback` → `_wake_fixr` | none |

**Finding B1 — this family has four implementations; `capability_map.md` lists
three.** `_arm_idle` / `_idle_fallback` (`CC_FIXR_IDLE_MINUTES`) is a fourth
"fire once at a future time", absent from the row. It shares its whole shape with
`TimerService` — a sleep task in a dict keyed by id, lost on restart — and neither
one knows the other exists. The map's row is already marked `unreviewed`, so this
does not contradict a recorded decision; it widens the audit queue by one.

---

## C · Expire / GC stale state

| trigger | durable store | idempotency | resumes / consumes | gate |
|---|---|---|---|---|
| Read of the park store | `Parked_Writes` | `_take_expired` row claim | `sweep_expired` (+ `rebuild_from_store` boot pass, `purge_parked_writes` retention throttled by `PARK_PURGE_INTERVAL`) | — |
| Read of the proposal store | `Proposals` | status transition | `expire_stale_proposals` (lazy sweep on read) | — |
| fixr tool call | `fixr_agents` | — | `_prune_agents` / `dispatcher.prune` | — |
| In-memory growth | none | — | LRU caps (`MAX_CONVERSATION_TAINTS`, `MAX_CACHED_API_REQUESTS`) | — |

**Finding C1 — the park sweep is the only one with a boot pass.** DP-319 learned
that a lazy sweep only walks what is loaded, so a durable row the process never
loaded is one it never expires. `expire_stale_proposals` is the same idiom over
the same kind of durable store and has no boot pass. Whether `Proposals` needs
one is a real question, not a foregone conclusion — but the asymmetry is not
recorded anywhere as deliberate. Same `unreviewed` row in the capability map.

---

## D · Doors to the outside, and their gates

| trigger | outbound call | concurrency cap | error shape | gate |
|---|---|---|---|---|
| pve tool (`proxmox/handler.py:234`) | `SSHRunner.run` | `Semaphore(_MAX_INFLIGHT_SSH=4)` | `{"status": "error", ...}` | `PVE_TOOLS_ENABLED`, checked **inside `_run`** |
| ⚠️ HF tool (`huggingface/handler.py:219`) | `SSHRunner.run` — same class | **none** | same dict shape | `HF_TOOLS_ENABLED`, checked **per tool at each call site**, not in `_run` |
| LLM provider | `aiohttp` / `httpx` per provider | per-provider | provider exceptions | API key |
| Hindsight | `httpx.AsyncClient` | — | `HindsightAPIError` | health probe |
| kcpp adapter | `httpx.AsyncClient` | — | `httpx.RequestError` | — |
| MCP server | `ClientSession` | — | reconnect loop | `MCP_ENABLED` |

**Finding D1 — RESOLVED (DP-348, merged).** There is now **one** node door:
`proxmox.ssh.run_node_command`, imported by both `huggingface/handler.py` and
`proxmox/handler.py`, so the in-flight cap that bounds pve traffic bounds HF
traffic too and the enable check cannot be forgotten per-tool. Pinned by
`tests/integration/test_node_transport_gate.py`.

Kept as a row because the *shape* of the defect is the reusable lesson:
`HFToolHandler._run` had been a near-verbatim copy of `ProxmoxToolHandler._run`
(same try / `SSHError` / `returncode` / result-dict shape) minus the semaphore and
minus the in-`_run` enable check. Two doors through one wall with different gates
is invisible to a capability map — both authors would describe their row honestly
and differently. Putting the check in `_run` is structurally stronger than putting
it at each call site, for the same reason `validate_tool_capabilities` beats a
test.

---

## E · Durable-record identity and outcome

| trigger | store | identity rule | consumer | gate |
|---|---|---|---|---|
| Model proposes a write | pending parks | `tool_loop.write_call_identity` (name + canonicalized args, call id excluded) | `pending_lookup` closure | — |
| Continuation turn re-proposes | `Parked_Writes` history | `write_call_identity_hash` → `identity_digest` | `ConfirmationManager.already_resolved` via `resolved_lookup` | `PARK_REEXECUTION_GUARD_WINDOW`, continuation turns only |
| Any tool returns | — | `tool_manager.tool_error()` | tool-loop card colour **and** `ConfirmationManager.apply` → `Decision.ok` | — |

Healthy family — one canonicalizer, one digest beside it, one failure predicate.
Recorded as the fix for two independent re-derivations of `result.get("error")`
(DP-323). Kept here as the **reference shape**: when a new mechanism needs an
identity rule or a success test, this is the row it should match.

---

## F · Second write door into the memory store

| trigger | store | path | gate |
|---|---|---|---|
| Normal memory write | `Memory_Summaries`, `vec_Memory_Summaries` | `MemoryManager` methods | — |
| Consolidation | same | `memory/backend/sqlite.py`, `memory/memory_consolidation.py`, `agents/sqlite_consolidator.py` | — |
| ⚠️ `update_core_memory` tool (`tools/tool_manager.py:417`) | same | **raw SQL through `memory_manager._lock` + `memory_manager._get_connection()`** | tool registration only |

**Finding F1 — `MemoryToolHandler` reaches through two private members of
`MemoryManager` to write its tables directly.** Both the `UPDATE Memory_Summaries`
and the `INSERT OR REPLACE INTO vec_Memory_Summaries` that keeps the vector index
consistent are re-implemented at the tool layer. Any invariant `MemoryManager`
enforces on that pair — and the embedding/row pairing is exactly such an
invariant — is enforced in two places now, one of which is a tool handler.
Four writers of the same table pair in total.

---

## Known false positive, kept deliberately

`self_edit/integration.py:58` declares `async def send_to_channel(self, channel_id: int, content: str) -> bool`
— byte-identical signature to `interfaces/discord_bot.py:68`. It is a `Protocol`
(`DiscordThreadClient`), i.e. correct structural reuse, not a second
implementation.

Kept in the file because it is the counterexample that justifies the method:
**name-and-signature matching flags this; mechanism-shape matching does not.**
The Protocol has no store, no idempotency rule, and no gate — every mechanism
column is empty because it is a type, not a machine. `scripts/arch_audit.py similar`
works at exactly the level that produces this false positive, which is part of
why it never caught DP-343.

---

## Open rows, worst first

1. **A1** — `_wake_fixr`: third live deferral implementation; config-sourced
   coordinates, no durable row, no exactly-once, and it persists its synthetic
   wake text as a durable user row. Named as future work in `deferral_kinds.py`;
   `capability_map.md`'s `single` verdict should read `single mechanism, one
   implementation outstanding` until it lands.
2. **F1** — tool-layer raw SQL through `MemoryManager` privates.
   `MemoryToolHandler._update_core_memory` (`tools/tool_manager.py`) calls
   `memory_manager._get_connection()` and issues its own `UPDATE Memory_Summaries`
   + `INSERT OR REPLACE INTO vec_Memory_Summaries`. Four writers of that table pair.
3. **B1** — fourth "fire once at a future time"; capability-map row lists three.
   `_arm_idle` / `_idle_fallback` (`self_edit/integration.py:325`).
4. **C1** — proposal expiry has no boot pass where the structurally identical
   park expiry does. `confirmations.rebuild_from_store` runs at boot
   (`bootstrap/__init__.py:129`); `expire_stale_proposals` is driven only from
   `ManagrAgent` (`agents/managr_agent.py:624`).

**Closed:** **D1** — one node door as of DP-348 (see §D).

> ⚠️ **Re-verify before acting on any row here.** This file is *regenerated*, not
> maintained, so a row states what was true at its last regeneration. D1 sat in this
> list claiming "fixed, unmerged" for ten days after DP-348 merged. A1, B1, C1 and F1
> were each re-confirmed live in `origin/master` on 2026-09-03; D1 was not.
