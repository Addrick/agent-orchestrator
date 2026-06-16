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

The export pulls global history for that persona name (across all channels) up to the persona's configured `history_messages` count. User turns are wrapped with kobold's `{{[INPUT]}}` / `{{[OUTPUT]}}` placeholders so the portal renders them with the active instruct template at submit time. System rows, empty-content rows, and tool-call-only assistant rows are skipped (the count is logged server-side).

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

**Version chevrons (Phase 2.3b):** The `<` / `>` chevrons on the most recent assistant message navigate between regeneration attempts. Every attempt is persisted — retries no longer overwrite history — and the L0 embedding travels with the content so retrieval reflects whichever version is currently canonical. There is **no client-side undo limit**; the full regen history is retained in the database for as long as the interaction exists. The chevrons are inert on the first generation (no regens yet). On each stream, the adapter emits an SSE `event: derpr` frame immediately before `[DONE]` carrying the canonical `assistant_id`; the portal uses it to fetch the version list and rebuild the chevron stacks.

**Editing and deleting messages (Phase 2.4):** Editing a portal turn from the inline edit UI propagates the new content to the DERPR DB via `PATCH /api/v1/interaction/{id}`. The L0 embedding is invalidated on edit so the next batch from `MemoryAgent` re-encodes against the updated text; the row is also re-queued for L1 summarization (`parent_summary_id` is cleared). Saving an empty edit deletes the message: a soft-suppression flag is recorded server-side via `DELETE /api/v1/interaction/{id}`, after which the row no longer appears in subsequent `kobold_export`s, sliding-window history, or LTM retrieval. Reply chains are left intact (no nulling of `reply_to_id`); orphaned assistant turns whose paired user row was deleted still segment cleanly. Toggling the chevrons back and forth between two contents does not grow the archive — repeat-content swaps reuse the existing archive row instead of inserting a duplicate.

**History contract (Phase 4.1):** Every turn on the `/chat/completions` stream now ends with a server-authored `event: derpr` frame carrying `{user_id, assistant_id, response_type, ephemeral_chunk_id}` — emitted even when a turn parks a write for confirmation (`assistant_id` is `null`, with a stable `ephemeral_chunk_id` for the pending-approval text), runs tools only, or produces no text. A companion `GET /api/v1/session/{persona}/transcript` returns the ordered conversation as identity-addressed chunks (each carries an `interaction_id`, or `ephemeral: true` for a not-yet-saved parked confirmation). Together these let a client address each message by its server identity instead of its position in the story, so edit/delete reliably target the right row even after a parked-write or tool-only turn. (This release is server-side only; the kobold-lite portal's own edit/delete still uses the older positional mapping until the Phase 4.2 stopgap re-syncs it from `/transcript`.)

> The portal's normal generation path is the OpenAI-style `/chat/completions` route (KoboldCPP jinja mode). The kobold-native `/api/v1/generate` and `/api/extra/generate/stream` routes are still served by the adapter for clients that prefer per-token SSE; both proxy to KoboldCPP and log user/assistant turns under `channel="web_ui"` the same way the OAI route does. Token-by-token portal usage falls on the native streaming route.

**Tool-enabled personas in the portal (tool revamp v1):** A persona with `enabled_tools` set can run over the portal SSE stream — token deltas and tool calls interleave in a single linear stream with no drain-and-restart. While the model is invoking a tool, the portal renders an inline collapsible block (using kobold-lite's existing `<think>` Reflective-Process pipeline) showing the tool name, JSON arguments, and the result/error. The block is streaming-only — the database stores the resolved assistant text without it, so reload / version-chevron / retry flows stay clean. CONFIRM-mode write-tool gating is unchanged; the portal currently runs autonomous, so write tools execute immediately. The adapter also emits structured `event: derpr-tool-start` / `event: derpr-tool-result` SSE frames carrying `{tool_name, arguments, call_id}` and `{call_id, result, error}` for portal-aware listeners (`window.derprOnToolStart` / `derprOnToolResult` hooks; latest payloads accumulate in `window.derpr_tool_calls[call_id]`).

### Bespoke DERPR portal (`/derpr`)

A React UI served at `GET /derpr` on the same adapter, driving the OpenAI-style `/chat/completions` SSE route with identity-addressed transcript re-syncs (`GET /api/v1/session/{persona}/transcript`).

**Send feedback (DP-214):** your message appears in the transcript immediately on send, tagged `sending…`, and the assistant row shows an animated typing indicator until the model produces its first token or tool call. When the turn completes, both transient rows are replaced by the authoritative transcript rows. On a failed turn the dismissed error re-syncs the transcript, so the user turn (persisted before generation) stays visible. Note: personas on non-local models currently deliver their response in one piece — true token streaming for cloud providers is tracked as DP-215–217.

Personas with `history_messages: 0` always render an empty transcript — the portal mirrors exactly what the engine would feed the model, and a zero-window persona feeds it nothing.

**Follow scroll + drafting (DP-218):** the transcript auto-follows new content (sent messages, stream frames, completed turns) while you're at the bottom; scrolling up to read history releases the follow, and returning to the bottom re-arms it. The composer stays editable during a response so you can draft your next message mid-stream — Enter won't send until the turn finishes (the SEND button is replaced by ■ stop while streaming).

**Persona selection persistence (DP-219):** your last-selected persona is remembered client-side (browser `localStorage`) and restored on page load, surviving engine restarts. The engine's own active-persona slot (`PUT /api/v1/model`) remains runtime routing state only — it is not persisted server-side; on boot the portal pushes the saved selection back to the engine so kobold-native passthrough routes agree. If the saved persona no longer exists, the portal falls back to the engine's default.

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
| `models [vendor]` | Available models, optionally filtered by vendor (OpenAI, Google, Anthropic, Antigravity, Local) |
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
| `max_context_tokens` | Total context budget (prompt + reserved response, kobold-style) |

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
| `max_context_tokens <integer>` | Integer >= 100 | Total context budget — prompt + reserved response (matches kobold-lite's `max_context_length` slider). Effective prompt prune budget = this minus `tokens`. Oldest non-system messages drop until prompt fits; system messages and the latest user message are always preserved. Default 131072. |
| `<provider>.<key> <value>` | Any provider id + scalar value | Fallback dotted-path setter for provider-specific knobs that have no first-class command (e.g. `set kobold.mirostat 2`, `set kobold.rep_pen 1.15`). Stored in `params.provider_extras[<provider>][<key>]`. Value is coerced to int / float / bool when possible, otherwise kept as a string. Use `set <provider>.<key> none` (or `null`/`clear`) to remove the key. Mirror read: `what <provider>.<key>`. |

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

### Antigravity (`agy`) — OAuth-tier provider

`agy-*` models (e.g. `set model agy-flash`) route through Google Antigravity's
local `agy` CLI instead of an API. This runs on the user's authenticated
**OAuth tier** (currently Gemini 3.5 Flash) rather than a metered API key, at the
cost of a subprocess spawn per call (a few seconds of latency) and no image
support.

Each call is executed inside a persistent workspace directory (by default, persona-specific under `data/workspaces/agy_{persona_name}` or fallback to `data/workspaces/agy_global`), preserving `agy` indexing/auth state caches. Persona names are sanitized to a filesystem-safe slug for the directory name, and concurrent calls sharing a workspace are serialized so they can't clobber each other's CLI state. You can configure this behavior in `.env` or `config/global_config.py`:
- `AGY_PERSISTENT_WORKSPACES` (default `True`): Set to `False` to revert to stateless throwaway temporary directories.
- `AGY_WORKSPACE_MODE` (default `"persona"`): Set to `"global"` to share a single derpr-wide workspace.
- `AGY_SANDBOX` (default `True`): Run `agy` under its built-in OS-level sandbox (`--sandbox`; nsjail on Linux, sandbox-exec on macOS). Set to `False` if the sandbox is unavailable in your environment (e.g. a container without the needed privileges).

> **Platform: POSIX only.** The `agy` provider works on Linux/macOS (and WSL or
> Docker). It does **not** work on **native Windows**: `agy` is a TUI that only
> writes its response to a TTY, while the engine captures `stdout` through a
> pipe — on Windows that capture comes back empty (agy renders to the console
> and emits nothing to a non-TTY stdout/file; no flag or env var changes this).
> The engine therefore refuses the `agy` route on native Windows with a clear
> error rather than returning silent empty responses. To test the `agy` route
> from a Windows dev box, run the engine on the POSIX host (e.g. the Docker
> deployment) or under WSL, where it behaves normally.

Tools work via an **inline protocol**: the engine injects the tool descriptions
into the prompt and asks the model to emit a `<tool_call>{…}</tool_call>` block to
request a tool. The engine parses that block and runs the tool through DERPR's
normal tool loop — so persona tool policy, the read/write split, CONFIRM-mode
approval, and untrusted-taint all apply exactly as they do for the other
providers. The model never executes tools itself.

### Claude Code (`cc`) — sandboxed autonomous provider

`cc-*` models (`set model cc-sonnet`, `cc-opus`, `cc-haiku`) route through the
local `claude -p` headless CLI instead of an API, running on the user's Claude
**subscription/OAuth tier**. Like `agy` it is a subprocess-per-call, one-shot,
**POSIX-only** provider with a persistent per-persona workspace and its own rate
limiter — but it differs from every other provider in one important way:

> **Claude Code runs its *own* tools.** The other providers (including `agy`)
> only generate text; DERPR's tool loop executes any tools. The `cc` provider
> instead launches Claude Code as an **autonomous agent** with
> `--dangerously-skip-permissions` (yolo), bounded by Claude Code's built-in OS
> sandbox. It edits files, runs shell commands, and uses its own tools inside
> the workspace, then returns its final text. DERPR's `tools` argument is
> **ignored** for this provider, and DERPR's read/write split, CONFIRM-mode
> approval, and untrusted-taint do **not** gate Claude Code's actions — the OS
> sandbox is the only boundary. Use it for self-contained goals/tasks, e.g.
> pointing a persona at the DERPR checkout to talk to Claude Code about its own
> codebase from any interface. (Approval routing for Claude Code's tool calls is
> deferred to a future MCP-based tool layer.)

The persona's system prompt **replaces** Claude Code's default system prompt
(via `--system-prompt`); the conversation history is rendered into the `-p`
prompt. Configure in `.env` or `config/global_config.py`:
- `RATE_LIMIT_CC_RPM` (default `15`).
- `CC_PERSISTENT_WORKSPACES` (default `True`): `False` reverts to throwaway temp dirs.
- `CC_WORKSPACE_MODE` (default `"persona"`): `"global"` shares one DERPR-wide workspace (`data/workspaces/cc_{persona}` or `cc_global`).
- `CC_WORKSPACE_DIR` (default unset): an explicit absolute path overriding the above — set it to the DERPR checkout to manage that repo from a chat interface.
- `CC_SANDBOX` (default `True`): run bounded by Claude Code's OS sandbox (Seatbelt on macOS, bubblewrap on Linux/WSL2), with `--dangerously-skip-permissions` (yolo) waived inside that boundary. `False` runs unsandboxed: yolo is **dropped** and tools are gated to `CC_ALLOWED_TOOLS` instead — this is the only way to run on native Windows (the sandbox can't), e.g. a smoke test.
- `CC_ALLOWED_TOOLS` (default empty): comma-separated tool allowlist for the unsandboxed path (`CC_SANDBOX=False`), passed via `--allowedTools` (Claude Code's OS-independent permission system, works on Windows). Empty = no tools pre-allowed (default-deny; a headless run can't answer an approval prompt, so tool-needing actions are refused — enough to verify the CLI runs and returns text). Ignored when `CC_SANDBOX=True`.
- `CC_SANDBOX_WEAKER_NESTED` (default `False`): set `True` inside an unprivileged container (e.g. the DERPR Docker deploy) so bubblewrap can start; only safe when the container already provides isolation.
- `CC_SANDBOX_ALLOWED_DOMAINS` (default empty): comma-separated domains the sandboxed Bash tool may reach. Empty = no network (a headless run cannot answer a domain-approval prompt, so network-needing tasks must list domains here).
- `CC_MAX_TURNS` (default `0` = no cap): bound on agentic turns per call (`--max-turns`).

> **Platform: POSIX only when sandboxed.** The Claude Code OS sandbox runs on
> macOS/Linux/WSL2, never native Windows. Because this provider runs yolo, the
> sandbox is the safety boundary, so DERPR refuses the `cc` route on native
> Windows while `CC_SANDBOX` is on. Run the engine on the POSIX host
> (Linux/macOS/WSL/Docker). On Linux/WSL2 the sandbox needs `bubblewrap` and
> `socat` installed; inside an unprivileged container also set
> `CC_SANDBOX_WEAKER_NESTED=True`.
>
> To smoke-test on native Windows, set `CC_SANDBOX=False`: the `claude -p` CLI
> itself is cross-platform and headless, so it runs and returns text; only the
> OS sandbox is POSIX-only. Tools stay gated to `CC_ALLOWED_TOOLS` (no yolo).

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

Determines the autonomy level for a persona's tool-use capabilities.

| Mode | Behavior |
|------|----------|
| **AUTONOMOUS** | **Read-only** tools execute immediately. **Write tools still require audit** (see [Tool Security](#tool-security) below). The user sees the final response after all automated steps. |
| **CONFIRM** | Standard mode. All write tools are presented for approval. Provides a consistent point of review for all state-changing actions. On Discord, approval uses reaction buttons with a 5-minute timeout. |

## Tool Security

The bot implements a comprehensive security framework to prevent prompt injection and unauthorized actions.

### Universal Write-Audit
Regardless of execution mode, **all write tools** (tools that modify state, like creating tickets or deleting users) are parked for human audit before execution. This ensures that no state-changing action is taken without explicit user consent.

### Taint Tracking
The system tracks the "trustworthiness" of the conversation context. If a persona uses a tool that retrieves potentially untrusted content (like `web_search` or `recall_memory` containing past external input), the current turn is marked as **tainted**. 
- Taint is "sticky" for the duration of the conversation.
- When a turn is tainted, any subsequent write-tool approval request will carry a warning: `⚠️ Context contains untrusted content from: [source]`.

### Insecure Composition Blocking
To prevent sophisticated injection attacks, the system refuses to load any persona whose tool configuration creates an inherently insecure path. Common blocked compositions include:
- **`network:read` + `local:write`**: Prevents tools that read from the internet from being used by a persona that can write to local storage/files.
- **`untrusted:read` + `network:write`**: Prevents untrusted data from being exfiltrated to a network endpoint.
- **`pii:read` + `network:*`**: Prevents sensitive Personal Identifiable Information (PII) from being sent over the network.

### Irreversibility Flags
Some tools are marked as **IRREVERSIBLE** (e.g., `delete_user`). Others may be dynamically flagged based on their arguments—for example, `add_note_to_ticket` is flagged as irreversible if the note is visible to a customer (`internal: false`). These flags are surfaced in the approval dialogue to highlight high-stakes actions.

### Credential Scoping
Machine secrets — provider API keys (`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `GOOGLE_GENERATIVEAI_API_KEY`) and the Zammad API token — are kept out of the LLM's reach by design:
- **Central vault.** All secrets are resolved through a single credential vault rather than read ad-hoc from the environment, giving one authoritative inventory of what counts as a secret. The model never receives secrets as tool arguments.
- **Egress scrubbing.** At startup every known secret value is registered with an egress scrubber. Any string headed for a place the model (or operator inspector) can read it back is scrubbed first, with the value replaced by `[REDACTED:<KEY_NAME>]`. Three boundaries are enforced: **tool results** before they re-enter the conversation, **write-audit arguments** before they are stored or shown in an approval dialogue, and the **cached request payload** surfaced by the `/assemble` inspector.
- **Shape-based fallback.** Beyond known values, the scrubber also redacts strings that *look* like secrets (e.g. `sk-…` API keys, `Token token=…` headers, bearer tokens), so an unregistered credential that leaks into a tool result is still caught.

This is defense-in-depth: today's curated tool surface means no tool returns secrets, but the guarantee holds automatically the moment a more powerful tool (e.g. shell execution) or a bring-your-own-credentials mode is added.

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

**Write:**
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

### fixr Tools (requires `service_bindings: ["fixr"]`)

The `fixr` supervisor persona (`claude-opus-4-8`) repairs DERPR's own code. It
does **not** edit code itself — it **dispatches** detached Claude Code coding
agents (one per bug, each in an isolated `git worktree` off a pristine base
clone of the repo) and is **woken by their log events** to coordinate and
report. Report a bug to `fixr`; it dispatches an agent (after you approve), the
agent diagnoses → fixes → tests → opens a PR, and **a human reviews and merges**
— the agent never merges or pushes master.

| Tool | Type | Description |
|------|------|-------------|
| `dispatch_fix` | **Write (parked)** | Spawn a coding agent for one bug in an isolated worktree. The **only** always-gated fixr tool — it parks for your approval before any agent runs. One agent per bug. |
| `inspect_agents` | Read | List dispatched agents + status (running/waiting/done/error/killed), branch, PR url, last event. |
| `answer_agent` | Write (ungated) | Resume a *waiting* agent (one that asked a question) with a decision, headless via `claude --resume`. |
| `kill_agent` | Write (ungated) | Stop a stuck/runaway agent's process + event bridge; optionally drop its worktree. |
| `send_discord` | Write (ungated) | Post a curated report (e.g. "agent opened PR <link>", or a fork needing a human) to the team channel. fixr reports on its own judgment. |

Only `dispatch_fix` is confirmation-gated ("gate every dispatch; dial the rest
later"); the coordination + reporting tools run ungated so the woken supervisor
isn't approval-prompted on every agent event. Autonomy is bounded by the
dispatch gate **and** the human-merge PR boundary.

An agent signals its supervisor through its final message: `FIXR_QUESTION:` (I
need a decision), `FIXR_DONE: <pr-url>` (finished), `FIXR_ERROR:` (blocked). A
per-agent event bridge tails the agent's log, maps it to a common event schema,
and wakes `fixr` on those events.

Config knobs: `CC_FIXR_CLONE_DIR` (base clone path; worktrees live under
`<clone>/worktrees/<bug>`), `CC_FIXR_BASE_REF` (default `origin/master`),
`CC_FIXR_MODEL_ARG` (dispatched-agent model, default `sonnet`),
`CC_FIXR_DISCORD_CHANNEL` (default `send_discord` recipient). A live run also
needs `github.com,api.github.com` in `CC_SANDBOX_ALLOWED_DOMAINS` and a scoped
`GH_TOKEN` in the host env (never in chat).

### Memory Tools (no service binding required)

Available to any persona with `enabled_tools: ["*"]` (e.g., `joy`, `it-help`). These tools interact with the long-term memory store built by MemoryAgent.

| Tool | Type | Description |
|------|------|-------------|
| `recall_memory` | Read | Search the persona's long-term memory bank for facts relevant to a natural-language query. Returns up to `limit` (default 10) hits — short summaries of past conversations or observations. Scope is inherited from the active turn (persona, channel, user, server); the LLM cannot redirect recall to another persona. Marked `produces_untrusted=True` so retrieved hits taint the turn under the tool-security framework. |
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

## Hindsight Backend (alpha)

The semantic memory tier can be backed by [vectorize-io/hindsight](https://github.com/vectorize-io/hindsight) instead of the default SQLite store. Hindsight runs in Docker with an embedded Postgres + pgvector and handles retain/recall/reflect via a REST API. This is alpha — the SQLite backend remains the default.

### Deployment

**Production (since 2026-05-19):** Hindsight runs on `aux-desktop` / `derpr-host` (`10.0.0.70`), bound to `0.0.0.0:8888`; the derpr default `HINDSIGHT_URL` points there. The stack is **maintained out-of-repo** on that host at `C:\Server\Hindsight\` (`docker-compose.hindsight.yml` + `kobold-lb.conf` + engine patches) — this repo no longer ships a Hindsight compose template. For offline/local development, recreate a compose from the host copy (bind `127.0.0.1:8888` and point the kobold-proxy upstream at a reachable kobold).

The stack runs two containers:

- `hindsight-memory` — the API server (`ghcr.io/vectorize-io/hindsight`), bound to `0.0.0.0:8888` on `10.0.0.70`.
- `hindsight-kobold-proxy` — nginx LB sidecar that load-balances `:5001` across LAN koboldcpp instances (`kobold-lb.conf`). The hindsight container itself has **no** internet egress (paranoid mode, see `memory/project/decisions/2026-05-05-hindsight-paranoid-mode.md`).

### Required host services

- **kobold.cpp** with an OpenAI-compatible `/v1` endpoint and a model loaded. The production proxy on `10.0.0.70` routes to the LAN kobold instances configured in `kobold-lb.conf` (live: `10.0.0.69:5001`; `10.0.0.67:5001` is the laptop, intermittent).
- Docker Desktop / Docker Engine.

If kobold is offline, retain operations are silently dropped (see failure modes below) and recall returns the existing corpus.

### Enable in the bot

Set in `.env` (or `config/global_config.py`):

```
SEMANTIC_BACKEND=hindsight
HINDSIGHT_URL=http://10.0.0.70:8888
```

Restart the bot. `MemoryManager.__init__` will instantiate `HindsightBackend` instead of `SqliteSemanticBackend`. Legacy SQLite-shape methods (`store_segment`, `retrieve_relevant_summaries`, …) will raise `NotImplementedError` if called — every caller must migrate to `retain_turn` / `recall` first.

### First-time bank bootstrap

Each persona uses its own bank. Before retaining or recalling for a persona, call:

```python
await backend.ensure_bank(
    bank_id="alice",
    retain_mission="extract decisions, preferences, and durable facts; ignore chitchat",
    reflect_mission="ground answers in stored decisions and rationale; be precise",
    # Optional:
    # enable_observations=True,
    # observations_mission="stable facts about people and projects",
)
```

`ensure_bank` is idempotent — a 409 from upstream is treated as success.

### Backup and restore

Hindsight stores its embedded Postgres at `/home/hindsight/.pg0` inside the container, mapped to the named volume `hindsight-data`.

**Backup** (host shell):

```bash
docker exec hindsight-memory pg_dump -U hindsight hindsight > hindsight.sql
```

**Restore** (into a fresh stack):

```bash
# bring up the Hindsight stack on the host (see Deployment above), then:
docker exec -i hindsight-memory psql -U hindsight hindsight < hindsight.sql
```

Restore-test at least once before relying on backups — bank IDs and tag schemas must round-trip cleanly.

### Failure modes

| Symptom | Cause | Effect |
|---------|-------|--------|
| Retain calls drop, log `Hindsight retain dropped (kobold offline)` | kobold not running on host | New turns aren't consolidated; existing recall still works |
| Container restart | `docker compose restart hindsight` or crash | Recall + retain both unavailable until container is up; queued retains in-flight at shutdown are lost |
| 409 on `ensure_bank` | Bank already exists | Treated as success — safe to call on every startup |

The retain path is fire-and-forget through a per-bank async queue: user turns enqueue and return immediately; one worker per bank drains in FIFO order. There is no DLQ — alpha tolerates dropped retains rather than risk back-pressure on user turns.

### Operator trust overrides

`mark_trusted` / `mark_untrusted` flip the `untrusted` bit on a specific recall hit (per the [tool security framework](../memory/project/plans/tool_security_framework.md)). Overrides live in a parallel SQLite file (`src/memory/hindsight_overrides.db`) — recall post-filters and rewrites the bit. Every flip is audit-logged with operator_id, reason, prior, and new values.

## System Defaults

| Setting | Value |
|---------|-------|
| Default model | gemini-2.5-flash-lite |
| Default context limit | 15 messages |
| Context hard cap | 30 messages |
| Max tool calls per request | 5 |
| Max response tokens | 4096 |
| Confirmation timeout | 300 seconds (5 min) |
