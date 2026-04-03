---
name: Working style feedback
description: Behavioral rules from user corrections and confirmations — split into universal (cross-project) and project-specific
type: feedback
---

## Universal (applies across all projects)

**Ask before investigating.** Ask simple clarifying questions BEFORE launching deep investigation. ~70% of a usage quota was once spent exploring a wrong hypothesis that one question would have eliminated.

**Remind to commit.** Proactively remind Adam to commit after completing logical chunks of work, especially during multi-step refactors or feature branches.

**Fix root causes.** Always opt for the correct solution rather than a workaround, even if the workaround is faster. Misclassifications, wrong defaults, and stale assumptions should be fixed at the source.

**Prioritize self-consistent architecture.** Naming, patterns, and layering must be consistent across the codebase. If one subsystem uses a pattern, others should follow it. Inconsistent patterns force re-discovery and lead to misclassifications.

**No trailing summaries.** Don't summarize what was just done at the end of responses — Adam can read the diff.

**Conceptual consistency matters.** Adam cares deeply that similar things are treated similarly in the codebase. If personas and agents both use LLMs and tools, the framework should reflect that shared nature rather than having two separate paradigms. Design discussions about naming and structure are worth having before building.
**Why:** Prompted a multi-phase refactor (service lifecycle removal + ToolLoop extraction) to unify how personas and agents interact with tools.
**How to apply:** When adding new subsystems or abstractions, check if they duplicate patterns that already exist elsewhere. Propose unification when it makes sense.

## Project-Specific (this codebase)

**Use feature branches for experimental work.** When starting work that may involve design iteration or waffling on decisions. Squash-merge into main when done.

**Ground plans in actual code, not abstractions.** The `plans/platform_ideas.md` was written from comparisons with other projects (OpenClaw/OpenCode) and contained items (context compaction, tool output pruning) that don't map to real needs. Verify claims against the codebase before presenting them as project priorities. Comparison docs are inspiration, not commitments.
**Why:** Adam caught hallucinated plan items being presented as real project needs. Plans must be grounded.
**How to apply:** When referencing any plan or roadmap item, verify it reflects actual code behavior. The grounded roadmap lives in `codebase/roadmap.md`.

**Spec before implement.** When a design conversation produces concrete behavior, add it to `docs/user_guide.md` before writing code. This ensures alignment and gives Adam a document to review and correct.
**Why:** Adam wants to confirm design in plain language before diving into implementation.
**How to apply:** During feature planning, write the user-facing description first.

**Keep CLAUDE.md thin.** Only truly global rules belong in CLAUDE.md (commands, test requirements, doc update rules, Viking protocol). Architecture details, component descriptions, and anything session-specific goes in memory files. CLAUDE.md loads every conversation — every line costs context.
**Why:** The architecture section in CLAUDE.md had grown stale (referenced removed enums) and duplicated memory content.
**How to apply:** Before adding to CLAUDE.md, ask: "does every possible conversation need this?" If no, put it in an L1/L2 memory file.
