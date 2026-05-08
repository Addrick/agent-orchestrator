---
name: Operator trust overrides for long-term memory
date: 2026-05-04
decision: Implement manual 'trust/untrust' CLI commands with audit logging.
status: Implemented
---

# Decision: Operator trust overrides for long-term memory

## Context
In Phase 5, we implemented memory taint propagation, where retrieving an "untrusted" memory summary flags the entire turn as tainted. However, this created a potential "perma-taint" deadlock: if a memory was accidentally or correctly flagged as untrusted but was later verified by an operator, there was no way to "clean" the memory in the database without manual SQL intervention.

## Why manual overrides?
1. **False Positives**: Automated taint (e.g., from a web search summary) might be overly broad. An operator can verify that a specific summary is actually safe.
2. **De-escalation Path**: Without a `mark_trusted` mechanism, a tainted memory hit from weeks ago would permanently trigger audit warnings for any related conversation.
3. **Auditability**: Security-relevant state changes (like trust bit flips) MUST be recorded for accountability.

## Implementation Choice: CLI + Audit Log
- **CLI first**: Commands like `trust <id> <reason>` provide immediate utility for developers and power users.
- **Audit Log Table**: A dedicated `Audit_Log` table was added to `user_memory.db` to record:
    - `operator_id`
    - `prior_state` / `new_state`
    - `reason`
    - `timestamp`
- **ID Visibility**: To make the commands usable, the `summary_id` is now injected into the prompt label (e.g., `[#general, ID:123]`) so operators can identify which record to override.

## Alternatives Considered
- **Automatic De-tainting**: Having the model decide if a memory is safe. (Rejected: Insecure; the attacker can steer the model to "believe" a memory is safe).
- **Time-based Expiry**: Letting taint expire. (Rejected: Taint doesn't necessarily decay; an injection in memory is just as dangerous 6 months later).

## Future Work
- **UI Integration**: Add "Trust" buttons to the Portal UI's memory hits.
- **Bulk Overrides**: Allow trusting all memories from a specific trusted session.
