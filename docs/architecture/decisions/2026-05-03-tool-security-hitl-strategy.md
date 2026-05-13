---
name: HITL + reversibility-tiered approval is the agent security strategy
description: Captures the 2026-05-03 design discussion concluding that taint tracking alone is insufficient, dual-LLM/CaMeL has narrow applicability, and reversibility-tiered HITL with allowlist learning is the path forward
type: project
---

## Decision

The tool security framework will be built around **human-in-the-loop approval gated by reversibility tiers**, not around capability composition rules alone or per-byte taint flow.

Core elements:

1. **Reversibility axis** is the primary policy gate, layered alongside the capability axes already in `tool_security_framework.md`. Reversible writes (drafts, soft-deletes, scheduled-with-delay sends, internal notes) auto-execute and log for batch review. Irreversible writes (outbound email, ticket close, customer-visible state, anything monetary) block on approval.
2. **Argument-level policy.** Same tool can be reversible or not depending on arguments. `update_ticket` against an internal queue is auto; against a customer-visible field is blocking. Policy must inspect call arguments, not just tool name.
3. **Origin-tag taint, passive.** Every tool result carries `source_origin` metadata (`human_typed`, `internal_db`, `external_fetched`, etc.). Used as an *escalation signal* — an otherwise-auto write that touched `external_fetched` data gets promoted to review. Not used to "untaint" via human read; manual untainting is a bottleneck and was rejected.
4. **Channel-agnostic approval queue.** Pending approvals are an orchestrator-level primitive. Surface adapter (Discord channel, Gmail, future portal route) is interchangeable. Reuses `NotificationRouter` shape.
5. **Allowlist learning.** Each approval can include a "remember this" rule ("persona X may auto-update tickets in queue Y when origin is Zammad webhook"). Approval log is the source. Review surface shrinks over time.
6. **Anomaly-only late-stage review.** Once allowlists mature, an outlier check on agent trajectories (statistical, not LLM-judged) escalates novel patterns. Not phase 1.

## Why

Three alternatives evaluated:

- **Capability composition rules alone** (current plan in `tool_security_framework.md`). Catches the worst combos statically (`network:read` + `local:write` = deny). Necessary but insufficient — doesn't handle in-bounds writes that are still bad ideas (correctly-permissioned but wrongly-aimed).
- **Dual LLM / CaMeL structural separation.** Safe for fixed-shape tasks (summarize X, send to Y where Y is pre-authorized). Degrades to capability-rules-only for content-driven workflows where the agent's job is reacting to untrusted input — which is most of derpr's surface (Zammad triage, Gmail handling). Worth using on narrow subset of tasks; not a general solution.
- **Full IFC with manual untainting.** Bottleneck. Human read-and-approve of every byte before it can flow does not scale.

User concluded: review burden is unavoidable for an agent acting on untrusted content with real-world side effects, but **review volume is tunable**. The strategy targets the volume, not the existence.

## Realistic trajectory

- Month 1: everything writeable blocks. Calibrates frequency.
- Month 2-3: build allowlists from approval log. Drop ~70% of prompts.
- Month 6+: only novel + irreversible + high-value blocks. Target single-digits per day.

## Scope of dual-LLM pattern

Considered and bounded — do not relitigate. May be applied to:

- Pre-authorized fixed-destination tasks ("summarize ticket and post to internal note")
- Tool calls where the recipient/destination is fully known before untrusted content is read

Not applicable to:

- Triage workflows where the agent must branch on content
- Any task where the plan itself is content-driven

## What this decision does NOT cover

- Capability composition rules (still in `tool_security_framework.md`, complementary)
- Sub-agent / meta-agent process sandboxing (separate concern, see plan)
- The `blocking: true|false` flag mechanics (see plan)
- Audit log schema and approval-rule DSL (deferred; will be specified during implementation)

## References

- `plans/tool_security_framework.md` — implementation plan, updated to incorporate this decision
- `plans/tool_revamp_v1.md` — sibling, ships first, leaves policy permissive
- 2026-05-03 design discussion (caveman session): evaluated dual-LLM (Willison 2023) and CaMeL (DeepMind 2025); both judged narrow-utility for derpr's content-driven workload
