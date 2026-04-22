# User Guide

This document describes the user-facing behavior of the bot. It serves as both a reference for end users and a living spec for new features — describe behavior here before implementing it.

## Interfaces

### Discord

**Persona routing:** Messages are matched to personas in two ways:
1. **Prefix:** Start your message with a persona name followed by a space (e.g., `joy create a ticket for...`). The prefix is stripped before processing.
2. **Channel name:** If no prefix matches, the bot checks if the channel name starts with a persona name (case-insensitive). The full message is sent.

If neither matches, the message is ignored (unless in an ambient logging channel).

**Dev commands:** Commands like `set`, `what`, `detail`, etc. are handled before the LLM is invoked. Responses are sent as threaded replies on the original message (auto-archive after 60 minutes). Mutating commands (e.g., `set model`) also add a checkmark reaction.

**Confirmation flow:** When a persona is in CONFIRM mode and the LLM requests a write tool:
1. The bot presents the tool call details and adds checkmark/X reactions
2. The user reacts to approve or deny
3. On approval, the tool executes and the LLM generates a follow-up response
4. Timeout: 5 minutes (300 seconds)

**Ambient logging:** Messages in configured channels (default: "general", "random", "development") are logged to the database under persona "ambient" without triggering any response. Useful for building conversational context.

**Message deletion:** Deleting a message in Discord automatically suppresses it from future LLM context (the DB row is flagged, not deleted).

**Character limits:** Messages over 2000 characters are automatically split across multiple messages.

**Status:** Bot status shows `as persona1, persona2, ... eye-emoji` (truncated at 128 chars). During generation, status briefly shows the active persona name.

### Gmail (Proof of Concept)

> **Note:** The Gmail interface is a proof of concept and has not been fully designed. The behavior described here reflects current implementation but is subject to significant change.

**Persona routing:** Extracted from the recipient email prefix (e.g., `support-joy@example.com` routes to persona "joy"). Falls back to default persona if unparseable.

**Current behavior:**
- CONFIRM mode is automatically downgraded to AUTONOMOUS (no interactive approval possible via email)
- No conversation persistence (messages are not logged to the database)
- Only processes emails from allowed senders (configurable via `BLOCK_EXTERNAL_SENDER_REPLIES` and `ALLOWED_SENDER_LIST`)
- Uses Google Pub/Sub watch on INBOX for near-instant processing

### Web Portal (kobold-lite)

A FastAPI adapter (`KoboldAdapter`) hosts a customised kobold-lite UI at `/portal` and forwards requests to the local KoboldCPP instance verbatim. KoboldCPP owns prompt rendering / instruct templating; DERPR adds persona management and history sourcing on top.

**Persona controls:** A persona dropdown in the top nav switches the active persona for forwarded requests. The settings cog opens an Inference Matrix popup where you can edit role, system prompt, model, sampling, max tokens, context length, and persist to the backend. The persona's system prompt is pushed into kobold-lite's **Sys. Prompt** field (`instruct_sysprompt`), so the Memory block stays user-owned for free-form notes.

Switching personas while a session has content — or while either persona is in DERPR Database mode — prompts for confirmation through kobold-lite's standard new-session dialog. Cancel reverts the persona selection; unsaved turns can be rescued via kobold's own Save/Load first. The DERPR database itself is never modified by the portal.

**History Source toggle (Phase 2.1):** Each persona has a two-state switch for where session history comes from. **Requires kobold-lite instruct opmode** — chat / adventure / story modes are not supported.

| Side | Behavior |
|------|----------|
| **Kobold Native** (default) | The active session lives entirely in kobold-lite's own state. Nothing is read from DERPR's database. |
| **DERPR Database** | On switch, the portal fetches `GET /api/v1/session/{persona}/kobold_export`, ingests the response via kobold-lite's standard JSON load path, and the persona's stored conversation appears in the chat. From that point onward, requests are still forwarded as plain passthrough — DERPR does not splice or rewrap. |

The toggle state is remembered per persona in `localStorage`. Switching back to **Kobold Native** prompts for confirmation and clears the visible session (it does **not** delete anything from the DB).

The export pulls global history for that persona name (across all channels) up to the persona's configured `context_length` message count. User turns are wrapped with kobold's `{{[INPUT]}}` / `{{[OUTPUT]}}` placeholders so the portal renders them with the active instruct template at submit time. System rows, empty-content rows, and tool-call-only assistant rows are skipped (the count is logged server-side). Tool-call and tool-result rendering in the portal is out of scope for Phase 2 — see the roadmap backlog.

**LTM Generation (Phase 2.2):** A sub-checkbox under the toggle, enabled only when **DERPR Database** is active. When checked, DERPR runs semantic LTM retrieval against your query before each submit and writes the result into kobold-lite's **Author's Note** field; kobold then places the block near the end of the prompt at its normal author's-note position. The author's note textarea is greyed out and labelled "Managed by DERPR LTM" while this is active. Your prior author's note is backed up to `localStorage` and restored when you uncheck.

The **Memory Scope** dropdown in the Inference Matrix sets the persona's `memory_mode` for retrieval:

| Mode | What memories are searched |
|------|---------------------------|
| `CHANNEL_ISOLATED` (default) | Only turns from `channel=web_ui`. Portal turns are logged as of Phase 2.3a, so this returns portal-only history. |
| `PERSONAL` | All turns attributed to the portal user, across all channels for this persona. |
| `SERVER_WIDE` | All turns for this persona in any channel that shares a server context. |
| `GLOBAL` | All turns across all channels and servers for this persona. Use this to surface Discord / email / Zammad history immediately. |

Saving from the Inference Matrix persists the `memory_mode` to the backend. The LTM checkbox state is stored per persona in `localStorage` (not persisted to the backend).

**Portal conversation logging (Phase 2.3a):** Portal turns are persisted to `message_history` with `channel="web_ui"`. Each submit writes a user row before forwarding to KoboldCPP; the streamed assistant reply is written on stream close with `reply_to_id` linking back to the user row. Aborted generations preserve the partial assistant buffer. Clicking **Retry** on the prior response archives the old assistant content into `Interaction_Edit_History` and overwrites the canonical row in place with the new reply — no new user row is created on retry, and `reply_to_id` linkage is preserved. LTM retrieval on subsequent turns therefore surfaces portal-originated content alongside Discord / email / Zammad history.

> The portal only uses the OpenAI-style `/chat/completions` path (KoboldCPP jinja mode). The prior kobold-native `/api/v1/generate` and `/api/extra/generate/stream` routes have been removed from the adapter.

## Commands

All commands are entered as the message body when addressing a persona. Commands are case-insensitive.

### Conversation Control

| Command | Description |
|---------|-------------|
| `hello` | Start a dynamic context conversation. Context window grows by 2 messages per turn. |
| `goodbye` | End dynamic context mode and revert to the persona's static default context length. |

### Querying Persona State

`what <attribute>` — Display the current value of a persona attribute.

| Attribute | Shows |
|-----------|-------|
| `prompt` | Full system prompt text |
| `model` | Current model name |
| `models [vendor]` | Available models, optionally filtered by vendor (OpenAI, Google, Anthropic, Local) |
| `personas` | All loaded persona names |
| `context` | Conversation history limit (message count) |
| `tokens` | Max response token limit |
| `temp` | Temperature parameter |
| `top_p` | Top-p (nucleus sampling) parameter |
| `top_k` | Top-k sampling parameter |
| `execution_mode` | AUTONOMOUS or CONFIRM |
| `tools` | All available tools with enabled/disabled status |
| `memory_mode` | History retrieval scope |
| `service_bindings` | Bound external services |

### Configuring Persona State

`set <attribute> <value>` — Modify a persona attribute at runtime. Changes persist to `data/personas.json`.

| Attribute | Values | Notes |
|-----------|--------|-------|
| `prompt <text>` | Any text | Replaces the entire system prompt |
| `default_prompt` | (no args) | Resets to default system prompt |
| `model <name>` | Model name or description | Supports fuzzy matching via LLM (e.g., `set model claude opus`) |
| `tokens <number>` | Integer >= 100 | Max response length in tokens |
| `context <number\|dynamic> [start]` | Integer or "dynamic" | Static message count, or dynamic growth starting from optional value |
| `temp <float>` | 0.0 - 2.0 | Temperature (randomness) |
| `top_p <float>` | 0.0 - 1.0 | Nucleus sampling threshold |
| `top_k <integer>` | Positive integer | Top-k sampling limit |
| `display_name <on\|off>` | on/off | Whether persona name prefixes chat responses |
| `execution_mode <mode>` | autonomous, confirm | Tool execution approval behavior |
| `tools <spec>` | all, none, tool_name, or `all -excluded` | Enable/disable tools. Supports exclusion syntax: `set tools all -web_search` |
| `memory_mode <mode>` | See Memory Modes below | History retrieval scope |
| `service_bindings <list\|none>` | Comma-separated service names | e.g., `set service_bindings zammad,agents` |

### Persona Management

| Command | Description |
|---------|-------------|
| `add <name> [prompt]` | Create a new persona. Default prompt: "you are in character as {name}" |
| `delete <name>` | Remove a persona permanently |
| `remember <text>` | Append text to the persona's system prompt (cumulative) |
| `detail` | Dump full persona configuration (all parameters, tools, bindings) |

### Debugging

| Command | Description |
|---------|-------------|
| `dump_last` | Summary of the last API request (model, context size, tools, generation params) |
| `dump_context` | Full context dump as downloadable file (config, tools, conversation history) |
| `help` | Show command list and active personas |
| `update_models` | Refresh available model list from configuration |

## Personas

Personas are stateful LLM configuration objects. Each persona has its own model, system prompt, token limits, sampling parameters, tool access, and memory scope. Users interact with personas through the routing mechanisms described above.

### Default Personas

These ship with the bot (defined in `config/default_personas.json`):

| Name | Model | Purpose | Execution Mode | Tools |
|------|-------|---------|----------------|-------|
| arbitr | gemini-2.5-flash | Directive communication, Discord markdown only | AUTONOMOUS | google_grounding_search |
| joy | gemini-2.5-flash | Production Zammad ticket management | CONFIRM | All (zammad + agents bindings) |
| it-help | gemini-2.5-flash | Testing/dev persona for Zammad integration | AUTONOMOUS | All |
| gemini | gemini-2.5-flash | General-purpose Gemini | AUTONOMOUS | google_grounding_search |
| chatgpt | gpt-5 | General-purpose GPT | AUTONOMOUS | None |
| claude | claude-haiku-4-5-20251001 | General-purpose Claude | AUTONOMOUS | None |
| testr | gemini-2.5-flash | Test persona (responds "success") | AUTONOMOUS | None |

### System Personas

Defined in `config/system_personas.json`. Not directly user-accessible — used internally by agents for analysis tasks:

- **model_selector** — Fuzzy model name matching for `set model`
- **tool_selector** — Fuzzy tool name matching for `set tools`
- **triage_analyst** — Ticket analysis and internal note generation
- **triage_scout** — Keyword extraction from tickets for search
- **triage_filter** — Relevance scoring between historical and new tickets
- **triage_summarizer** — Ticket content compression
- **dispatch_analyst** — Priority assignment and dispatch notification generation
- **memory_summarizer** — Extracts observations from conversation segments for long-term recall; used by MemoryAgent

## Execution Modes

Controls how the bot handles LLM-requested tool calls.

| Mode | Behavior |
|------|----------|
| **AUTONOMOUS** | Tools execute immediately. The user sees only the final response. |
| **CONFIRM** | Write tools (create, update, delete) are presented to the user for approval before execution. Read tools execute immediately. On Discord, approval uses reaction buttons with a 5-minute timeout. |

## Memory Modes

Determines which conversation history is loaded into the LLM context window.

| Mode | Scope | Typical Use |
|------|-------|-------------|
| **CHANNEL_ISOLATED** | Messages in the current channel only (server-aware) | Default. Keeps conversations separate per channel. |
| **SERVER_WIDE** | All messages across the Discord server for this persona | Cross-channel awareness within a team. |
| **PERSONAL** | All messages from the current user, across all channels | Per-user continuity regardless of channel. |
| **GLOBAL** | All messages for this persona, all servers/users | System-wide knowledge. |
| **TICKET_ISOLATED** | Messages tied to a specific Zammad ticket | Ticket-focused context without chat history bleed. |

History is always limited by message count (default: 15, hard cap: 30), not by token count.

## Tools

Tools are capabilities the LLM can invoke during a conversation. Available tools depend on the persona's `enabled_tools` list and `service_bindings`.

### General Tools

| Tool | Type | Description |
|------|------|-------------|
| `web_search` | Read | Search the web via DuckDuckGo. Params: `query`, `max_results` (default 5). |
| `google_grounding_search` | Special | Enables Google's native search grounding. Gemini models only. |

### Zammad Tools (requires `service_bindings: ["zammad"]`)

**Read:**
| Tool | Description |
|------|-------------|
| `get_ticket_details` | Fetch full ticket data by user-facing ticket number |
| `search_tickets` | Search using Zammad query syntax |
| `search_user` | Find user by email or name |

**Write (gated by CONFIRM mode):**
| Tool | Description |
|------|-------------|
| `create_ticket` | Create a new support ticket |
| `update_ticket` | Modify ticket state, priority, owner, tags |
| `add_note_to_ticket` | Append internal or public note |
| `create_user` | Register a new customer |
| `update_user` | Modify user details |
| `delete_user` | Remove a user (irreversible) |

### Agent Tools (requires `service_bindings: ["agents"]`)

| Tool | Type | Description |
|------|------|-------------|
| `get_agent_status` | Read | View running state, deploy counts, error rates for agents |
| `get_agent_history` | Read | Recent action log with optional ticket/customer filters |
| `manage_agent` | Write | Start, stop, or restart an agent |

### Memory Tools (no service binding required)

Available to any persona with `enabled_tools: ["*"]` (e.g., `joy`, `it-help`). These tools interact with the long-term memory store built by MemoryAgent.

| Tool | Type | Description |
|------|------|-------------|
| `drill_down_memory` | Read | Fetch raw episodic memories under a specific Core Profile. Use to recover specific details (dates, links, verbatim quotes) that were abstracted away during consolidation. Requires `parent_summary_id`. |
| `update_core_memory` | Write | Modify an existing Core Profile when new information contradicts or extends it. Requires `summary_id` and the revised content. |

**Internal tools** (used by agents/system personas, not by user personas):

| Tool | Used by | Description |
|------|---------|-------------|
| `submit_memory_summary` | memory_summarizer (MemoryAgent) | Records extracted observations and keywords from a conversation segment; identifies thematic outliers for re-queueing |
| `submit_core_profile` | Consolidator | Merges clustered episodic summaries into a structured core profile with nested concepts |

## Agents

Agents are autonomous background workers that run on a schedule without user interaction. They are configured in `config/agents.json`.

### Current Agents

**MemoryAgent** (`auto_start: true`) — Runs every 15 minutes. Segments recent conversations by topic, extracts observations via LLM, and stores embedded summaries for long-term recall. See [Long-term Memory](#long-term-memory) below for the full pipeline description. Config in `agents.json` under `"memory"`.

**ZammadBot (triage)** — Polls for new, untagged Zammad tickets and runs a multi-stage AI triage pipeline:
1. Extracts search keywords from the ticket
2. Searches for related historical tickets (global + per-user)
3. Scores historical tickets for relevance
4. Compresses context if needed
5. Generates an analysis and posts it as an internal note
6. Tags the ticket as triaged

**DispatchAgent** — Polls for triaged tickets and routes notifications:
1. Fetches the ticket and triage note
2. LLM assesses priority and generates a summary
3. Sends notification via configured channel (Discord DM, Zammad note, etc.)
4. Tags the ticket as dispatched

### Managing Agents

Personas with `service_bindings: ["agents"]` and the relevant tools enabled can:
- Check status: Ask the persona to check agent status (invokes `get_agent_status`)
- View history: Ask about recent agent actions (invokes `get_agent_history`)
- Control lifecycle: Ask to start/stop/restart an agent (invokes `manage_agent`, requires confirmation in CONFIRM mode)

There is no user-level permission system. Access to tools is controlled entirely by persona configuration (enabled tools and service bindings). Any user who can message a persona inherits that persona's tool access.

Agent configuration (schedule intervals, notification channels, recipients) is driven by `config/agents.json`, not by LLM decisions.

## Long-term Memory

The bot automatically builds a long-term memory store from conversations in the background. This is separate from the sliding-window conversation history controlled by `context` and `memory_mode`.

### How it works

1. **Embedding** — Each message logged to the database is embedded using the Gemini Embedding API (`gemini-embedding-001`). Embeddings are stored in the `Message_Embeddings` table.

2. **Segmentation** (MemoryAgent, every 15 min) — Unprocessed embedded messages are grouped into topically coherent segments using centroid-based cosine similarity. Q/A pairs (a user message immediately followed by an assistant reply) are never split across segments. Minimum segment size is configurable (default: 2 messages).

3. **Summarization** — Each segment is sent to the `memory_summarizer` system persona, which extracts discrete observations (facts, preferences, decisions, solutions) and thematic keywords via the `submit_memory_summary` tool. Messages that don't fit the segment's theme are flagged as outliers and re-queued for the next batch.

4. **Consolidation** — Periodically, similar episodic summaries (level 1) are clustered by similarity and merged into core profiles (level 2) via `submit_core_profile`. This creates a two-tier hierarchy: detailed episodic records and compressed concept profiles.

5. **Retrieval** — On each LLM request, relevant summaries are retrieved via KNN vector search and injected into the context window *before* the sliding-window history. This gives the LLM access to facts from older conversations that would otherwise have fallen out of the context limit.

### Scope

Long-term memory retrieval is filtered by channel, persona, and embedding model. Memory built in one channel is not surfaced in another (same scoping rules as `CHANNEL_ISOLATED` history). Currently only channels listed under `allowed_channels` in `agents.json` are processed by MemoryAgent.

### User-visible effects

- Personas may reference past conversations that occurred outside the current context window
- The `drill_down_memory` tool lets a persona with `*` tools fetch raw episodic details behind a core profile
- The `update_core_memory` tool lets a persona correct or extend a core profile when new information supersedes it

## System Defaults

| Setting | Value |
|---------|-------|
| Default model | gemini-2.5-flash-lite |
| Default context limit | 15 messages |
| Context hard cap | 30 messages |
| Max tool calls per request | 5 |
| Max response tokens | 4096 |
| Confirmation timeout | 300 seconds (5 min) |
