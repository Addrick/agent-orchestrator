---
name: Viking-inspired memory system adoption
description: Why and how the L0/L1/L2 tiered memory system was implemented — based on OpenViking research
type: project
---

**Date:** 2026-03-27

**What:** Restructured Claude Code's flat file-based memory into a tiered L0/L1/L2 system inspired by OpenViking (volcengine/OpenViking), a context database for AI agents.

**Why OpenViking's approach:**
- L0/L1/L2 progressive disclosure avoids dumping full documents into every conversation
- Hierarchical directories with overview files let the agent decide what's relevant without reading everything
- Explicit mutability rules (immutable decisions, appendable plans, regenerable codebase knowledge) prevent the "constantly overwriting" problem
- Bottom-up update protocol (L2 -> L1 -> L0) keeps summaries consistent

**Why soft-implement instead of using OpenViking directly:**
- OpenViking is a standalone FastAPI + vector DB service — heavy infrastructure for a file-based memory system
- OpenViking depends on litellm, which suffered a major supply chain attack (see external/supply_chain_attacks.md)
- The valuable part is the *patterns*, not the *tool* — tiered loading, mutability rules, and navigation conventions work fine as markdown files + CLAUDE.md instructions

**Enforcement:** Post-commit hook writes marker file -> UserPromptSubmit hook injects reminder. Non-commit updates rely on CLAUDE.md behavioral instructions.

**MCP was considered and rejected** for this use case — unnecessary process boundary overhead for a single-user project with in-process tools.
