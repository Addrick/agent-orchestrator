---
name: Minimum-viable tool security — one bit, two flags, one rule
description: Collapses the multi-axis capability taxonomy and origin enum from earlier 2026-05-03 design discussion to a binary trust bit + irreversibility flag + single runtime rule; supersedes the complex framework while preserving the same threat coverage
type: project
---

## Decision

Tool security ships as the **simplest model that defends the actual threat**, not the most expressive one.

**Static (per tool definition):** two boolean flags — `produces_untrusted` and `irreversible`. Both default `False`.

**Runtime (per turn):** one bit, `turn_tainted`. Set true if any tool result this turn was `produces_untrusted`, OR any retrieved memory hit had its untrusted bit set, OR conversation history carries the bit (sticky once set per session/turn-group — exact stickiness scope deferred to implementation).

**The rule (entire runtime policy):**
```
if tool.irreversible and turn_tainted:
    require_approval()
else:
    execute()
```

**Memory:** one bool column. ABC carries `untrusted: bool` on record/retrieve. Reflect output inherits OR of source bits. Operator can flip the bit on any row via `mark_trusted` / `mark_untrusted`, audit-logged.

## Why

The earlier 2026-05-03 design discussion (see sibling decision `2026-05-03-tool-security-hitl-strategy.md`) accreted a five-axis capability taxonomy, six-variant origin enum, two-bank-per-persona memory split, cascade graduation with DAG walking, and a composition-rule engine. It was sound but outrageously complex for the threat model that actually matters.

The threat is: **untrusted content steers an irreversible action.** Everything else in the complex model was secondary — composition rules caught misconfigurations, sensitivity tags drove reviewer UI, origin variants distinguished forensic detail. None of those were doing the load-bearing security work.

Collapsing to one bit + one rule:

- Defends the same threat path (untrusted-driven irreversible writes blocked at runtime).
- Strict propagation preserved (OR rule on reflect; sticky in turn).
- Hot-swappable backends preserved (one bool in the ABC).
- Operator override works without bank migration or cascade walking — flip the bit, reflect re-derives naturally next cycle, OR-propagation handles "graduation" implicitly.
- No functionality regression: reversible writes never block; memory recall always works; untrusted hits still inject (with label) so LLM uses them for context.

## What was dropped (and what replaces it)

| Dropped | Replaced by |
|---|---|
| 5 capability axes (direction, locality, trust, sensitivity, reversibility) | 2 flags (`produces_untrusted`, `irreversible`) |
| 6-variant `Origin` enum | 1 bool |
| Two banks per persona (trusted/untrusted) | One bank, one bool column |
| `OPERATOR_ATTESTED` vs `HUMAN_TYPED` distinction | Audit log keeps forensic detail; runtime treats both as trusted |
| Cascade graduation + DAG walking on attestation | Reflect re-derives; OR propagation makes graduation automatic |
| Composition-rule engine at persona load | Optional later layer (static deny-list); not needed for v1 |
| `allow`/`ask`/`deny` policy DSL | "Ask if irreversible+tainted, else allow" — no DSL |
| Allowlist-learning rules engine | Defer; add when review burden actually appears |

## What stays

- HITL on the only combination that matters (untrusted → irreversible)
- Strict propagation via OR (sound, no laundering, automatic)
- Hot-swappable memory backend
- Operator override + audit log
- Reversibility reasoning at the tool level
- Iteration-limit "continue for X more loops?" pattern (same approval primitive)
- Argument-level escape valve for the 2–3 tools where reversibility depends on args (`add_note_to_ticket.internal`, `update_ticket` on customer-visible vs internal queues, `create_ticket` on notifying vs internal queues) — one optional callable per tool, no DSL

## Threat coverage check

- **Prompt injection via untrusted content** — blocked at the irreversible boundary. Reversible writes proceed; worst case is a recoverable bad note that operator can read in audit.
- **Capability composition exploits** — the only one runtime needs to catch (`untrusted_read` → `irreversible_write`) is exactly the rule. Static composition rules (refusing to load a persona that combines `web_search` with `delete_user` etc.) can land later as a separate guardrail without restructuring.
- **Memory poisoning** — reflect output inherits taint. Untrusted memory hits surface with the bit, blocking irreversible writes when retrieved.
- **Privilege blur (ingest ≠ act)** — irreversible flag enforces it: ingest tools are reversible-or-read, act tools are irreversible. Same deny.
- **Sub-agent / meta-agent sandbox** — separate concern, not blocked or unblocked by this. Still a future need.
- **Operator override safety** — flipping the bit is logged; reflect re-runs propagate the change naturally; revocation is symmetric.

## Effort

Phase 1 ships in days, not weeks:

1. Add `produces_untrusted` + `irreversible` to ALL_TOOL_DEFINITIONS (17 tools, mechanical).
2. Add `untrusted: bool` to MemoryBackend ABC (`record_turn`, `MemoryHit`); update Null + SqliteLegacy + Hindsight impls.
3. Per-turn taint tracking in ToolLoop (one bool, sticky).
4. Approval gate at tool execution (reuse CONFIRM-mode plumbing initially).
5. Memory column + operator `mark_trusted` / `mark_untrusted` API.
6. Audit log table (every approval decision, every operator override).

## Growth path

If real need appears later, any dropped layer can be added without re-architecting:

- Composition deny-list at persona load → static check, separate from runtime taint
- Sensitivity tags for reviewer UI → additional optional fields on tool definitions
- Allowlist learning → audit log already captures the data; rules engine layers on top
- Argument-level policy DSL → grow from the optional callables, not a from-scratch DSL

## Relationship to prior decisions

- `decisions/2026-05-03-tool-security-hitl-strategy.md` — sibling decision earlier same day. Captured the intent (HITL, reversibility tiers, taint as escalation signal, allowlist learning, dual-LLM bounded). This decision **simplifies the implementation** without abandoning the intent. HITL stays. Reversibility stays. Taint stays (as a bit). Allowlist learning is deferred but not abandoned. Dual-LLM stays a narrow option.

- `plans/tool_security_framework.md` — superseded in scope. The plan is being rewritten to the minimum-viable model alongside this decision; the complex multi-axis design lives in git history if ever needed for reference.

## References

- `plans/tool_security_framework.md` — implementation plan, rewritten to the minimum-viable model
- `plans/hindsight_memory_migration.md` — `MemoryBackend` ABC owns the `untrusted` bool contract
- `plans/tool_revamp_v1.md` — sibling, ships streaming mechanics first
- `decisions/2026-05-03-tool-security-hitl-strategy.md` — earlier decision; intent preserved, taxonomy collapsed
