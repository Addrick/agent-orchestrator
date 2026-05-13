---
name: Tool security framework — universal write-audit with taint annotations
description: All write tools park for human audit before execution. Taint tracking annotates the audit surface with untrusted-content warnings. Evolved from the minimum-viable "one bit, one rule" design after discovering that taint-gated-only is insufficient for LTM-equipped agents.
type: project
---

## Status

Sibling to `plans/tool_revamp_v1.md` (ships first, leaves policy permissive). This plan ships the full security model.

**Phases 1–3 complete on `feature/tool-security` branch.** Phases 1–2 landed the tool flags and ToolPolicy engine. Phase 3 originally implemented taint-gated irreversible blocking, then evolved to the universal write-audit model described below.

**Design simplified 2026-05-03.** Earlier draft had 5 capability axes, 6-variant origin enum, two-bank memory split, cascade graduation. Collapsed to one bit + one rule per `decisions/2026-05-03-minimum-viable-tool-security.md`.

**Design evolved 2026-05-04.** The "one rule" (taint + irreversible → block) was found insufficient during design review. Replaced with universal write-audit model. See **Design Evolution** section below for full reasoning.

## Goal

Defend against unauthorized or injection-steered write actions by **requiring human audit of all write tool calls before execution**, regardless of execution mode. Taint tracking provides additional context on the audit surface to help operators assess risk.

User-stated invariant: no write action executes without human review. Taint from untrusted sources (web search, ticket bodies, memory hits) is surfaced as a warning annotation, not used as the sole gate.

## Design Evolution

### Phase 1: Multi-axis taxonomy (rejected 2026-05-03)
The original design had 5 capability axes, 6-variant origin enum, two-bank memory split, cascade graduation. Sound but enormous structural complexity for a threat that reduces to a simpler condition.

### Phase 2: "One bit, one rule" (implemented then evolved 2026-05-03 → 2026-05-04)
Collapsed to: `if is_irreversible(tool) and turn_tainted: park()`. Dramatically less code. But design review exposed three problems:

1. **LTM perma-taint.** With Long-Term Memory, old web search results resurface via recall. A single tainted memory hit from weeks ago would permanently brick autonomous mode for that conversation. No clean de-escalation path.
2. **"Reversible" ≠ safe.** Actions like `set_prompt`, `set_tools`, `set_execution_mode` are technically reversible but represent privilege escalation or behavioral pivots that should never be steered by untrusted content.
3. **False security on the clean path.** Un-tainted turns got a free pass, even though the model could be steered by vectors not yet tracked (prompt structure, history manipulation, etc.).

### Phase 3: Universal write-audit (current design, 2026-05-04)
All write tools park for human audit. Taint becomes an annotation ("⚠️ context contains untrusted content from: web_search") rather than the gate. This is:
- **Simpler** — one rule: `if write_calls: park()`
- **Honest** — doesn't pretend un-tainted turns are safe
- **LTM-compatible** — no perma-taint problem; audit happens regardless
- **Future-proof** — console tools, exfil vectors, etc. get the same treatment automatically

## Threat model

Same as before (kept for reference):

1. **Prompt injection via untrusted content** — ticket bodies, email, web pages, memory hits derived from any of the above. Treated adversarial.
2. **Capability composition exploits** — combining read-untrusted with write-irreversible into an exfiltration / damage path.
3. **Memory poisoning** — untrusted content surfaces later via recall, steers a downstream action.
4. **Privilege escalation via persona swap** — partially mitigated by persona-switch being a CLI command, not a tool. Out of scope here.
5. **Manchurian Candidate** (sub-agent / meta-agent) — sub-agent sandbox is a separate plan; not in v1.
6. **Ingest-act privilege blur** — enforced by the irreversible flag (ingest tools are reversible-or-read; act tools are irreversible).

## Design

### Tool flags (static)

Two booleans on every tool definition:

```python
"capabilities": {
    "produces_untrusted": False,   # default
    "irreversible": False,         # default
}
```

- `produces_untrusted=True` for tools whose result can carry attacker-controlled text: `web_search`, `google_grounding_search`, `get_ticket_details`, `search_tickets`, `drill_down_memory` (and future memory-read tools), any email-read or web-fetch tool added later.
- `irreversible=True` for tools whose effect can't be trivially undone by a follow-up call: `delete_user`, `merge_tickets`, customer-visible message sends, payments. Most internal-state changes are reversible.

Optional third field for the 2–3 tools whose reversibility depends on arguments:

```python
"capabilities": {
    "produces_untrusted": False,
    "irreversible": False,
    "irreversible_if": "src.tools.classifiers:add_note_internal_check",
}
```

The classifier is a callable `(args: dict) -> bool`. Resolved at execution time. Used for:
- `add_note_to_ticket` — irreversible if `internal=False` (customer sees, may trigger email)
- `update_ticket` — irreversible if changing customer-visible field on customer-visible queue (TBD whether worth the precision)
- `create_ticket` — irreversible if queue triggers customer notification

Validator at tool registration asserts both bools are present (or, for `irreversible_if`, that the import resolves).

### Runtime taint (per turn)

`turn_tainted: bool`. Owned by ToolLoop (or per-conversation context — exact ownership decided at implementation). Set true when:

- A tool result this turn was from a tool with `produces_untrusted=True`, OR
- A retrieved memory hit had `untrusted=True`, OR
- Conversation history carries the bit (sticky once set).

Stickiness scope (turn vs. conversation vs. session) is an implementation choice. **Default: sticky for the conversation** — once an agent has read untrusted content, every subsequent irreversible action that turn or later in the same session needs approval. Aggressive but sound. Can loosen if it bites in practice.

### The rule

```python
# In ToolLoop.run(), after executing read_calls:
if write_calls:
    audit_info = {
        "actions": [{"tool": name, "arguments": args,
                     "irreversible": is_irreversible(name, args),
                     "always_confirm": name in ALWAYS_CONFIRM_TOOLS} ...],
        "tainted": turn_tainted,
        "taint_sources": ["web_search", ...],  # which tools caused taint
        "model_reasoning": accumulated_text,     # model's reasoning before the call
        "execution_mode": persona.get_execution_mode().name,
    }
    yield _LoopFinishedEvent(
        response_type=PENDING_CONFIRMATION,
        pending_writes=write_calls,
        audit_info=audit_info,
    )
    return
```

All write tools park. Read-only tools execute freely. Taint is tracked and surfaced but does not gate.

### Approval surface

Reuse existing CONFIRM-mode plumbing: emit a `pending_writes` event from ToolLoop, parking the agent until operator approves. The confirmation text now includes:
- Per-tool `[IRREVERSIBLE]` / `[HIGH-IMPACT]` flags
- `⚠️ Context contains untrusted content from: ...` when tainted
- Structured `audit_info` JSON on `PendingConfirmation` for programmatic consumers

Adapter (Discord emoji today, channel-agnostic later) renders.

Iteration-limit "continue for X more loops?" reuses the same primitive — when ToolLoop hits its iteration cap, emit an approval request with reason `loop_limit_exhausted`. No new infra.

### Memory ABC

```python
class MemoryBackend(ABC):
    async def record_turn(
        self, persona, scope_id, role, content,
        untrusted: bool,             # NEW — required
        metadata, timestamp,
    ) -> None: ...

    async def retrieve_context(
        self, persona, scope_id, query, limit,
    ) -> list[MemoryHit]: ...   # MemoryHit.untrusted: bool

    async def mark_trusted(
        self, persona, scope_id, hit_id,
        operator_id: str, reason: str,
    ) -> None: ...

    async def mark_untrusted(
        self, persona, scope_id, hit_id,
        operator_id: str, reason: str,
    ) -> None: ...
```

Backend implementations:

- **NullBackend** — passes the bit through trivially.
- **SqliteLegacyBackend** — adds `untrusted` column to relevant tables. Migration test required (per CLAUDE.md). Default existing rows to `False` (best-effort backfill; older rows are conversational, mostly trusted).
- **HindsightBackend** — stores the bit as a tag (`untrusted:true` or absent). Reflect output retain inherits OR of source bits — implement at the retain-payload level, not relying on Hindsight's reflect to do this for us.

### Reflect propagation (Hindsight)

When background ReflectionAgent runs, the resulting mental models are recorded with `untrusted = OR(source bits)`. If any source chunk had the bit, the mental model gets it. Cascade graduation is automatic without any DAG bookkeeping: when an operator flips a chunk's bit via `mark_trusted`, the next reflect cycle naturally produces clean mental models from the now-clean source set. Old mental models keep their bit unless operator flips them too.

This is the simplification that made the complex two-bank model unnecessary. Reflect is the cascade.

### Operator override

`mark_trusted(hit_id, reason)` flips the bit on a row. Audit log captures: operator, timestamp, hit content (or hash), reason, prior bit state. Symmetric `mark_untrusted` for revocation.

Surfaces (any/all):
- CLI command (BotLogic dotted-command pattern)
- Discord operator-only channel reaction
- Future dashboard if/when it lands

Not LLM-callable.

### Audit log

Append-only table. Records:
- Every approval request: who/what/when/args/origin-bit/decision.
- Every operator override (`mark_trusted` / `mark_untrusted`): row id, operator, reason, timestamp, prior state.
- Every irreversible execution that proceeded auto (untainted path) — for retrospective sanity check.

Lives in same DB as `User_Interactions` for now. Retention TBD; not blocking v1.

PII concern: log will contain ticket bodies / email content for context. Plaintext on disk acceptable for current threat model (operator-only access). Encryption-at-rest is a separate project.

## Phases

| # | Scope | Status | Notes |
|---|---|---|---|
| **1** | Add `produces_untrusted` + `irreversible` (+ optional `irreversible_if`) to all tool definitions. Validator asserts presence. Added `ToolPolicy` engine + `locality`/`sensitivity` tags. | ✅ Complete | `feature/tool-security` branch. |
| **2** | Per-turn `turn_tainted` tracking in ToolLoop. Set on tainted tool result. Sticky via `ChatSystem._conversation_taints`. | ✅ Complete | Taint tracking operational but now used as annotation, not gate. |
| **3** | Universal write-audit: all write tools park for confirmation with structured `audit_info`. Taint annotates the audit surface. | ✅ Complete | Evolved from taint-gated to universal model. |
| **4** | MemoryBackend ABC: add `untrusted` to `record_turn` and `MemoryHit`. Update backends. | ✅ Complete | Coordination with Hindsight migration. |
| **5** | Memory taint contributes to `turn_tainted` — when retrieve_context returns hits with the bit, set `turn_tainted` true for taint annotation. | ✅ Complete | Taint propagates to audit surface. |
| **6** | `mark_trusted` / `mark_untrusted` ABC methods + impls. Operator surface: CLI command first. | ✅ Complete | Operator-facing trust management. |
| **7** | Audit log table + writers at every decision point. | ✅ Complete | Established `Audit_Log` and hooks for parking/decisions. |
| **8** | Reflect propagation: ReflectionAgent computes OR over source bits when retaining mental models. | Pending | Coordinated with Hindsight Phase 3.3. |

Phases 4-8 remain. 4 and 6/7 can land in parallel. 5 after 4. 8 lands when ReflectionAgent does.

## Test strategy

- **Phase 1:** parametrized over `ALL_TOOL_DEFINITIONS` — every entry has both flags; validator raises on missing flags.
- **Phase 3:** unit test the rule — irreversible+tainted parks; irreversible+clean executes; reversible+tainted executes; reversible+clean executes.
- **Phase 4:** memory backend conformance suite (parametrized over Null/SqliteLegacy/Hindsight) — record→retrieve roundtrip preserves the bit; reflect output inherits OR.
- **Phase 4:** SqliteLegacy migration test using `legacy_mem_manager` fixture — old DB without column gets migrated, existing data preserved, default `False`.
- **Phase 5:** integration — agent retrieves untrusted memory, attempts irreversible write, gets parked.
- **Phase 6:** operator override test — flip bit, audit log entry written, recall returns updated bit.
- **Phase 7:** audit log writer fires at every decision point (parametrized).
- **Phase 8:** reflect retain inherits OR (mock ReflectionAgent inputs).

## Growth path (deferred — not foreclosed)

Each can be added later without re-architecting:

- **Static composition deny-list at persona load.** A persona that declares `web_search` in its tool set AND `delete_user` could be refused at load time as a configuration guardrail. Doesn't replace the runtime rule; supplements it.
- **Sensitivity tags for reviewer UI.** Optional fields on tool definitions to drive the approval-surface display. Doesn't change runtime.
- **Allowlist learning.** Audit log already captures the data. Rules engine ("auto-approve persona X calling tool Y with args matching pattern Z") layers on top. Defer until manual approval volume actually appears as a problem.
- **Per-persona memory trust floor.** Persona declares "don't even retrieve untrusted hits." One filter at recall time. Trivial.
- **Dual-LLM scoped use** for fixed-destination tasks (per earlier decision). Narrow utility, can land for one persona at a time.
- **Sub-agent / meta-agent process sandbox.** Separate scope; nothing in this plan blocks it.

## Sub-agent sandboxing (still future work)

When meta-agent or sub-agent spawning lands, the simple model needs an extension: spawned agents run with a stricter default than user-facing personas. Likely shape:

- Sub-agents default to `irreversible_allowed: False` regardless of tool flags.
- Promotion to non-sandbox requires human review of agent config.
- Process-tier isolation (subprocess / container) for agents with `local:write` capability against the FS — though no tool currently writes to FS, so this is genuinely future.

Not blocking v1. Mentioned for completeness.

## MCP integration (still future work)

Per `mcp_strategy.md` decision #1, MCP client is gated behind this framework. With the simplified model, the gate becomes: MCP-imported tools must declare `produces_untrusted` and `irreversible` at wrap time. Default both to `True` for unknown servers (deny-by-default). Adam-curated trusted servers can declare per-tool. Much simpler integration story than the original capability-tag bridging.

## Resolved questions

- **Stickiness scope of `turn_tainted`.** ~~Per-conversation (sticky-once-set).~~ **Resolved:** Stickiness is maintained for annotation purposes, but since all writes park regardless, the scope is less critical. Kept per-conversation for now.
- **Should `WRITE_TOOLS` and `ALWAYS_CONFIRM_TOOLS` survive?** **Resolved:** Both survive. `WRITE_TOOLS` drives the universal audit gate. `ALWAYS_CONFIRM_TOOLS` provides `[HIGH-IMPACT]` annotation on the audit surface. `irreversible` provides `[IRREVERSIBLE]` annotation.
- **Taint-gated vs universal audit.** **Resolved:** Universal. See Design Evolution.

## Open questions

- **Audit log retention.** Days? Months? Forever? Defer; depends on storage budget.
- **Per-tool `produces_untrusted` for memory tools.** `drill_down_memory` should be tagged `True` because it returns past content (which may have been ingested from external sources). The flag is about origin, not network — tag `True` and document that `produces_untrusted` ≠ `network`.
- **Auto-approve rules.** With all writes parking, the approval volume will be higher. Future: allowlist rules ("auto-approve persona X calling tool Y with args matching pattern Z") could reduce friction once audit log data accumulates.
- **Console tools / exfil vectors.** When shell exec or file-write tools land, they need to be tagged as write tools. The universal model handles them automatically.

## Non-goals

- Sanitizing untrusted natural-language content (impossible; structural mitigation only)
- Cryptographic enforcement (signed configs are integrity, not confidentiality)
- Replacing `service_bindings` (stays as the mechanism for grouping service-specific tools)
- Multi-axis capability taxonomy (deferred unless actual need appears)
- Two-bank memory split (rejected; reflect handles cascade naturally)
- Taint-only gating (rejected; insufficient for LTM-equipped agents, see Design Evolution)

## References

- `decisions/2026-05-03-minimum-viable-tool-security.md` — collapses prior multi-axis design to one bit + one rule
- `decisions/2026-05-03-tool-security-hitl-strategy.md` — earlier same-day decision; intent preserved, taxonomy collapsed
- `plans/tool_revamp_v1.md` — sibling, ships streaming mechanics first
- `plans/hindsight_memory_migration.md` — `MemoryBackend` ABC owns the `untrusted` bool contract; Phase 1 work coordinated
- `plans/agent_expansion.md` Future Horizons — sub-agent / Manchurian Candidate concerns (deferred)
- `mcp_strategy.md` — MCP gated; integration simplified by minimal flag set
- `_overview.md` MSP Productionization — high-level security posture this plan formalizes
