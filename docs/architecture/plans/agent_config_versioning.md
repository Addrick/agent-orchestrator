---
name: Agent config versioning (future)
description: Future-work note on git-backed agent config snapshots for audit + prompt/param performance attribution
type: project
status: deferred
---

# Agent Config Versioning — Future Work

Idea floated 2026-05-04 during tool security framework (DP-107 era) discussion. Not scheduled. Park here so it isn't lost.

## Goal

Know exact agent config (prompt, params, allowed tools, persona) at the moment any given tool call / response was produced. Two payoffs:

1. **Audit** — reconstruct what an agent could do / was told to do at time T. Pairs with tool security framework's runtime taint tracking (config-time vs runtime layers).
2. **Performance attribution** — when prompt or param change ships, correlate live metrics (latency, satisfaction, tool-call success) against the specific config revision. Today changes are tracked but not stamped onto invocation logs.

## Sketch

- Agent / persona configs live as files (already largely true — `agents.json`, `system_personas.json`, `default_personas.json`).
- On each tool invocation or completed response, stamp the git SHA of the config tree (or a content hash of the resolved agent config) onto the log row.
- Reconstruction = `git show <sha>:path/to/config` + log query.
- Optional: separate config repo / branch so app commits don't churn the audit history.

## Why not now

- Heavy plumbing for current scale. Existing change tracking covers "what did we change" well enough.
- Adds a hard coupling: configs must be file-resident and committed before deploy. Breaks any future move to DB-stored agent configs unless designed around.
- Log schema migration to carry SHA / hash on every row.

## Relationship to tool security taint

Does **not** replace taint tracking. Taint = runtime data-flow (untrusted message poisons context → cap downgrade mid-conversation). Git versioning = config-time provenance. Complementary; both needed if full audit story is wanted.

## Revisit when

- Compliance / audit requirement appears.
- Prompt experimentation cadence rises and post-hoc attribution becomes a recurring pain.
- Multi-tenant or shared-agent scenarios where "which config did this user actually hit" is non-obvious.
