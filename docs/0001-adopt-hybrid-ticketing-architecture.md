# ADR 0001: Adopt Hybrid Ticketing & Memory Architecture

**Date:** 2023-10-27

**Status:** Accepted

## Context:
The initial ML-powered ticketing system used a single, self-contained SQLite database for all user and interaction data. As the project's scope grew, this approach revealed several limitations:
1.  It lacks a user interface for human support agents to manage ticket lifecycles (e.g., assigning, prioritizing, closing tickets).
2.  Building these features would mean reinventing a full IT Service Management (ITSM) platform.
3.  The bot's role is evolving to be both a formal IT support agent and a casual, personalized chat companion, requiring a unified memory of all interactions, not just ticketed ones.

## Decision:
We will refactor the system into a hybrid architecture:
1.  **Zammad for Formal Ticketing:** We will adopt the open-source Zammad helpdesk as the backend for all formal support tickets. The bot will interact with Zammad via its REST API. This makes Zammad the "source of truth" for ticket status, assignments, and agent-led interactions.
2.  **Local "User Memory" Database:** We will maintain a local, lightweight SQLite database (`user_memory.db`). Its sole purpose is to store a unified history of every user interaction (both casual and ticketed) to provide long-term, personalized context for the AI.
3.  **"Lazy Sync" for Users:** User accounts will only be created in Zammad on-demand, the first time a user initiates an interaction through a designated "support channel." A link between the local user ID and the Zammad user ID will be stored in our `user_memory.db`.

## Consequences:
*   **Positive:**
    *   Immediately provides a professional, feature-rich UI for human support agents.
    *   Enables advanced ticketing features (SLAs, reporting, etc.) without custom development.
    *   Achieves the goal of a unified AI memory, allowing for highly contextual and personalized responses across different interaction types.
    *   Clearly separates concerns, simplifying the bot's core logic.
*   **Negative:**
    *   Introduces an external dependency (the Zammad instance).
    *   Requires careful management of the user synchronization logic between the local DB and Zammad.