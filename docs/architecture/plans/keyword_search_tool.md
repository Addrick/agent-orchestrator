---
name: Memory Keyword Search System
description: Implementation plan for a high-speed keyword and full-text search (FTS) tool for the interaction database.
type: project
---

# Memory Keyword Search System

## Context & Problem Statement

The current Long-Term Memory (LTM) system is primarily embedding-based. While good for semantic similarity, it can miss specific, exact-match information (error codes, unique identifiers, specific filenames) that are critical for IT dispatch and debugging tasks.

Furthermore, automatic LTM injection currently uses a single hard threshold for relevance. While not perfect, we are keeping this simple "one-parameter" approach for now to avoid the complexity of weighted scoring or dynamic hydration trade-offs.

## Proposed Solution: Full-Text Search (FTS) Tool

Instead of a complex "drill-down" or "auto-expansion" system, we will provide the agent with a direct **Keyword Search Tool**. This allows the agent to use its own comprehension to find specific L0 data when the automated LTM injection is insufficient.

### 1. `search_interactions(query, [limit])` Tool
*   **Purpose**: Allows the agent to perform a high-speed text search across the entire `User_Interactions` history.
*   **Implementation**: Uses SQLite's **FTS5** (Full-Text Search) extension.
*   **Capabilities**:
    *   Keyword matching (e.g., `OOM`, `timeout`).
    *   Phrase matching (e.g., `"server 504 error"`).
    *   Boolean operators (e.g., `Python AND NOT traceback`).
*   **Benefit**: This serves as the "manual drill-down" fallback without requiring a complex automated hydration engine.

### 2. Implementation Strategy

#### Step 1: SQLite FTS5 Virtual Table
We will create a virtual table `Interaction_Search` that shadows the `User_Interactions` table.
```sql
CREATE VIRTUAL TABLE Interaction_Search USING fts5(
    interaction_id UNINDEXED,
    content,
    persona_name,
    channel,
    content='User_Interactions',
    content_rowid='interaction_id'
);
```
*   **Triggers**: Add `AFTER INSERT` and `AFTER UPDATE` triggers on `User_Interactions` to keep the search index in sync automatically.

#### Step 2: Tool Integration
*   **`MemoryManager.search_fts(query, limit)`**: A new method to execute the FTS query and return formatted results (Interaction ID, Timestamp, Role, Content).
*   **Tool Schema**: Register `search_interactions` in the `ChatSystem` tool registry.

---

## Design Considerations (Current State)

1.  **Thresholds**: We will maintain the **single hard threshold** for automated LTM injection in `ChatSystem`. No dynamic budget or pre-emptive expansion will be implemented in this phase.
2.  **Punted Features**: The specific "drill-down" tool (`inspect_interaction`) and the "dynamic context hydration" logic are punted until the LTM block performance is better understood.
3.  **Performance**: FTS5 is extremely lightweight and will not impact runtime latency for standard messages.

---

## Next Steps

1.  **Migration**: Add the FTS5 virtual table and triggers to `memory_manager.py`.
2.  **Service Integration**: Implement the `search_fts` method.
3.  **Tool Registration**: Expose the tool to the agent.
