---
name: Memory system future work
description: Planned enhancements to the long-term memory system beyond Phase 1-2 stabilization — spreading activation, decay, consolidation improvements, keyword search tool
type: project
---

## Future Work Items (post-stabilization)

Identified during the 2026-04-11 walkthrough review and design discussion.

### 1. Spreading Activation (next priority)
Replace or augment pure cosine-similarity retrieval with a spreading activation model — when a memory is recalled, semantically and structurally linked memories also gain activation. Would address the "chaining" problem in consolidation clustering and improve recall for conceptually related but embeddings-distant memories.
**Status:** No implementation research yet. Adam is interested but needs to research approaches.

### 2. Memory Keyword Search & Drill-down
Allow the LLM to run a direct keyword/text search against the memory DB and "drill down" to raw L0 messages. Focuses on pre-emptive hydration based on token budget.
**Status:** [keyword_search_tool.md](file:///c:/Users/adama/PycharmProjects/derpr-python/memory/project/plans/keyword_search_tool.md) drafted. Focused on FTS5 keyword search as a primary tool.

### 3. Memory Decay
All memories are currently permanent and equally weighted. Implement temporal decay so older, unreinforced memories lose retrieval priority. May be a project of its own given the design complexity.
**Status:** Future work, needs dedicated design pass.

### 4. O(n²) Consolidation → sqlite-vec KNN
Replace the nested Python loop in `MemoryConsolidator` with sqlite-vec native KNN queries. Query each summary's K nearest neighbors from `vec_Memory_Summaries`, then union-find clusters. sqlite-vec uses L2 distance by default but on normalized vectors this is monotonically related to cosine similarity — no practical drawback.
**Status:** Clear win, no known drawbacks. Ready to implement when prioritized.

### 5. Temporal Gating in Consolidation
Add a maturity threshold (e.g., only consolidate episodes older than N days) to prevent premature merging of fresh episodic memories into core profiles. Preserves recent episodic detail.
**Status:** Needs design thought on clean implementation.

### 6. Centroid Embedding Optimization for Retrieval
Current retrieval passes N individual window embeddings and computes `min(distance_1, distance_2, ...)` in SQL. Pre-computing a single centroid from the window would reduce parameter count and improve performance for large context windows.
**Status:** Future optimization. Current approach works but won't scale well.

### 7. Database Schema Evolution: Hybrid Content & Agent Traceability
Transition the `User_Interactions` table to a hybrid storage model to eliminate on-the-fly stripping and improve observability.
- **`content`**: Cleaned, high-quality text for immediate LLM context.
- **`raw_content`**: Original verbatim text (including grounding links, citations, and metadata) for human display and deep-link retrieval.
- **`agent_trace`**: New column/table for storing structured chain-of-thought, tool execution logs, and reasoning traces associated with the interaction.
**Status**: Future design goal identified 2026-04-30 during citation regression fix.
