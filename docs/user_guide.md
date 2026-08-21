# User Guide

This document describes the user-facing behavior of the bot. It serves as both a reference for end users and a living spec for new features — describe behavior here before implementing it.

## Interfaces

### Discord

**Persona routing:** Messages are matched to personas in two ways:
1. **Prefix:** Start your message with a persona name followed by a space (e.g., `joy create a ticket for...`). The prefix is stripped before processing.
2. **Channel name:** If no prefix matches, the bot checks if the channel name starts with a persona name (case-insensitive). The full message is sent.

If neither matches, the message is ignored (unless in an ambient logging channel).

**Dev commands:** Commands like `set`, `what`, `detail`, etc. are handled before the LLM is invoked. Responses are sent as threaded replies on the original message (auto-archive after 60 minutes). Mutating commands (e.g., `set model`) also add a checkmark reaction.

**Confirmation flow:** When the LLM requests a write tool:
1. The turn **keeps going** — the model can do more work and propose more writes
2. After the reply, the bot posts one message per proposed action with checkmark/X reactions
3. React to approve or deny **any** of them, in any order — they are independent
4. Each decision executes (or refuses) that one action, then the persona reports what happened
5. Timeout: 24 hours, and it survives a restart of the bot

A turn can therefore end with several proposals waiting at once. Approving one does
not touch the others, and answering the third before the first is fine. The
persona's reply to an approval may itself propose more actions, which appear as
new messages with their own reactions.

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

**Editing and deleting messages (Phase 2.4):** Editing a portal turn from the inline edit UI propagates the new content to the DERPR DB via `PATCH /api/v1/interaction/{id}`. The L0 embedding is invalidated on edit so the next batch from `SqliteConsolidator` re-encodes against the updated text; the row is also re-queued for L1 summarization (`parent_summary_id` is cleared). Saving an empty edit deletes the message: a soft-suppression flag is recorded server-side via `DELETE /api/v1/interaction/{id}`, after which the row no longer appears in subsequent `kobold_export`s, sliding-window history, or LTM retrieval. Reply chains are left intact (no nulling of `reply_to_id`); orphaned assistant turns whose paired user row was deleted still segment cleanly. Toggling the chevrons back and forth between two contents does not grow the archive — repeat-content swaps reuse the existing archive row instead of inserting a duplicate.

**History contract (Phase 4.1):** Every turn on the `/chat/completions` stream now ends with a server-authored `event: derpr` frame carrying `{user_id, assistant_id, response_type, ephemeral_chunk_id}` — emitted even when a turn proposes writes for approval, runs tools only, or produces no text. (Before DP-297 a turn that gated a write had no text of its own, so it reported `assistant_id: null` plus an `ephemeral_chunk_id`; such a turn now persists normally, and each proposal carries its own token on its `derpr-confirm` frame instead.) A companion `GET /api/v1/session/{persona}/transcript` returns the ordered conversation as identity-addressed chunks (each carries an `interaction_id`, or `ephemeral: true` — one such chunk per pending approval, so a reload renders every proposal still awaiting an answer). Together these let a client address each message by its server identity instead of its position in the story, so edit/delete reliably target the right row even after a parked-write or tool-only turn. (This release is server-side only; the kobold-lite portal's own edit/delete still uses the older positional mapping until the Phase 4.2 stopgap re-syncs it from `/transcript`.)

> The portal's normal generation path is the OpenAI-style `/chat/completions` route (KoboldCPP jinja mode). The kobold-native `/api/v1/generate` and `/api/extra/generate/stream` routes are still served by the adapter for clients that prefer per-token SSE; both proxy to KoboldCPP and log user/assistant turns under `channel="web_ui"` the same way the OAI route does. Token-by-token portal usage falls on the native streaming route.

**Tool-enabled personas in the portal (tool revamp v1):** A persona with `enabled_tools` set can run over the portal SSE stream — token deltas and tool calls interleave in a single linear stream with no drain-and-restart. While the model is invoking a tool, the portal renders an inline collapsible block (using kobold-lite's existing `<think>` Reflective-Process pipeline) showing the tool name, JSON arguments, and the result/error. The block is streaming-only — the database stores the resolved assistant text without it, so reload / version-chevron / retry flows stay clean. CONFIRM-mode write-tool gating is honored: each proposed write surfaces in the bespoke portal transcript as its own pending row with an inline **approve & run / deny** bar (resolved via `POST /api/v1/persona/{name}/confirm`, which now **requires** the proposal's `token` and streams the continuation over the same SSE protocol); several can be open at once and are answered independently. The adapter also emits structured `event: derpr-tool-start` / `event: derpr-tool-result` SSE frames carrying `{tool_name, arguments, call_id}` and `{call_id, result, error}` for portal-aware listeners (`window.derprOnToolStart` / `derprOnToolResult` hooks; latest payloads accumulate in `window.derpr_tool_calls[call_id]`).

### Bespoke DERPR portal (`/derpr`)

A React UI served at `GET /derpr` on the same adapter, driving the OpenAI-style `/chat/completions` SSE route with identity-addressed transcript re-syncs (`GET /api/v1/session/{persona}/transcript`).

**Send feedback (DP-214):** your message appears in the transcript immediately on send, tagged `sending…`, and the assistant row shows an animated typing indicator until the model produces its first token or tool call. When the turn completes, both transient rows are replaced by the authoritative transcript rows. On a failed turn the dismissed error re-syncs the transcript, so the user turn (persisted before generation) stays visible. Note: personas on non-local models currently deliver their response in one piece — true token streaming for cloud providers is tracked as DP-215–217.

Personas with `history_messages: 0` always render an empty transcript — the portal mirrors exactly what the engine would feed the model, and a zero-window persona feeds it nothing.

**Follow scroll + drafting (DP-218):** the transcript auto-follows new content (sent messages, stream frames, completed turns) while you're at the bottom; scrolling up to read history releases the follow, and returning to the bottom re-arms it. The composer stays editable during a response so you can draft your next message mid-stream — Enter won't send until the turn finishes (the SEND button is replaced by ■ stop while streaming).

**Persona selection persistence (DP-219):** your last-selected persona is remembered client-side (browser `localStorage`) and restored on page load, surviving engine restarts. The engine's own active-persona slot (`PUT /api/v1/model`) remains runtime routing state only — it is not persisted server-side; on boot the portal pushes the saved selection back to the engine so kobold-native passthrough routes agree. If the saved persona no longer exists, the portal falls back to the engine's default.

**Create a persona (DP-231):** a **`+ new`** button beside the persona picker opens a create dialog. It captures the essentials — name (the routing key, lowercase `[a-z0-9_-]` only, no spaces), system prompt, model, memory mode, temperature, max tokens, and history window. Name is the only required field; a blank prompt/model falls back to the engine defaults (the prompt defaults to `you are in character as <name>`, matching the `add` dev command). On create the persona is persisted (`POST /api/v1/personas` → `personas.json`) and the portal switches to it, so the full **Inspector** — including the Tools tab for service bindings and tool policy — is immediately available to finish configuring it. A duplicate or malformed name is rejected with the engine's error shown inline.

**Bulk tool toggles (DP-231):** the Inspector's **Tools** tab has a *set all tools* row (off · allow · ask) that flips every tool at once, plus per-service-group *off · allow* quick-set buttons in each group header — useful when configuring a freshly created persona that should start with most tools on (or off). Changes still save through the same `set tool_policy` path that revalidates the security composition.

**Channel rail (DP-136 6b):** the left rail lists every channel the active persona has history in, grouped by originating interface (Web UI / Discord / Zammad / Gmail). Clicking a channel re-scopes the transcript and the next submit to it — a CHANNEL-memory-mode persona shows only that channel's history, a GLOBAL one merges all channels regardless. **`+ new`** points the view at a fresh `web_ui_*` tag; no row exists until the first submit, which materializes the channel in the DB (and the rail).

**UI-state persistence (DP-273):** beyond the persona selection above, the portal remembers the rest of your client-side layout across reloads via browser `localStorage` — the three panel collapse toggles (nav rail / channel rail / inspector), the active Inspector tab, the advanced-sampler fold, and the **active channel per persona** (each persona restores its own last-used channel instead of resetting to the default). Every restored value is validated against what currently exists: a saved channel that no longer appears in the persona's channel list (deleted, or a `+ new` channel that was never sent to) resets gracefully to `web_ui`, and a stale Inspector tab falls back to the persona tab — no blank/error state. Sampler values and toggles are *not* stored here; they live on the persona server-side and are re-fetched on load.

**Backend statusline (DP-311):** a thin row under the top bar reports what the KoboldCPP backend is doing, polled from `GET /api/extra/perf` (1s while a generation is in flight, 5s idle). It shows, left to right: **state** — `idle`, or `generating · <elapsed>` with a pulsing marker, plus `queue N` when requests are waiting; **ingest** — prompt tokens the last completed generation ingested, with its prefill time and rate; **gen** — tokens produced, decode rate, and `total` (prefill + decode) wall time for that generation; **stop** — why the last generation ended (`EOS`, `stop sequence`, `out of tokens`, `aborted`) and how long ago, with the exact clock time on hover; and, right-aligned, completed generation count and backend uptime.

The line reflects the *backend*, not your tab: a generation started from Discord, an agent, or another browser shows as busy here too.

**Live ingestion bar (needs the `kcpp-progress` sidecar).** KoboldCPP publishes per-batch prefill progress in **no API** — every `last_*` field on `/api/extra/perf` is frozen for the duration of a run. The counters exist only on KoboldCPP's stdout (`Processing Prompt [BATCH] (8192 / 24310 tokens)`). Where the optional `services/kcpp-progress` sidecar is deployed alongside the model and the engine has `KOBOLD_PROGRESS_URL` set, the statusline shows a live bar — `ingesting · 15.9s   ingest 10.2k/24.3k tok ▓▓▓░░ 42%` — advancing in `blasbatchsize` steps (visible jumps, roughly every 3s on the 40B; that is the true granularity, not a rendering artifact). Decode progress (`gen 17/400 tok`) comes from the same source.

Without the sidecar the line degrades silently to last-completed counters: no bar, no error, and the poll backs off to once a minute after the engine reports it is not configured (it re-probes rather than giving up for good, so a tab left open across an engine restart that adds the sidecar picks the bar up on its own). Setup and the mandatory `stdbuf -o0` unbuffering of KoboldCPP are documented in `services/kcpp-progress/README.md` — without that, progress records (which end in CR, not LF) sit in a stdio buffer and arrive in 20-30 second clumps.

**The stop reason blanks to `…` mid-generation.** KoboldCPP resets `stop_reason` to `0` when a run *starts*, so a running generation would otherwise misreport itself as "out of tokens". The real value appears when the run completes.

If the backend stops answering, the line turns amber and reads `backend unreachable` while keeping the last known numbers — a dropped poll is not evidence that the counters changed. This covers KoboldCPP being down, not just the engine: a backend that cannot be reached is reported as unreachable rather than as an idle backend with empty counters. The live bar and decode chip disappear for the duration — last-known is not live, and a frozen bar next to an "unreachable" label would contradict itself.

**Slash dev-commands:** a portal message starting with `/` is routed to the dev-command endpoint instead of the LLM — e.g. `/set temperature 0.8` or `/what models` runs the same command surface Discord uses, and the response renders as a dismissible inline row (mutating commands also refresh the Inspector). The composer hints this live (`leading / = dev-command`); there is currently no escape for sending a literal chat message that starts with `/` (the draft is trimmed before the check).

**LTM recall toggle:** the composer's **LTM recall** chip controls whether long-term-memory retrieval runs for this persona (it saves through `long_term_memory` on the persona, so it persists). When on, the context panel previews the `ltm_block` the engine would inject — re-fetched as you type (debounced) so the preview mirrors the per-message recall recomputed at submit.

**Voice availability:** the mic (hold-to-talk), *voice auto-send*, and *listen* (always-listening dictation) controls appear only when the browser supports audio capture **and** the engine reports `voice_web` in `GET /api/v1/capabilities` (i.e. `VOICE_WEB_ENABLED` is set and the `/voice/*` routes are mounted). With voice disabled server-side the controls are hidden rather than failing on every press.

### Memory imports panel (DP-292)

The nav rail's **`◈ MEMORY`** dock opens a full-page **Imports** panel for managing the documents in a Hindsight memory bank. It is an **operator tool**, not a chat surface: reads (bank list, document list, operation status) are open, but every mutation (upload, ingest, delete) requires the operator control token set in the top bar — without it the panel is read-only and mutating actions show a "control token required" notice.

- **Bank picker:** select which Hindsight bank to work in (personas map to banks). The dropdown shows each bank's fact count.
- **Add documents — three sources:**
  - **Upload** `.md` / `.txt` files (multiple at once). Other file types and non-UTF-8 content are rejected per-file with a reason; PDF is deferred. Each file is keyed by its filename, so re-uploading the same name **replaces** that document rather than duplicating it.
  - **Fetch URL** — the engine fetches the URL server-side and ingests the body text, keyed by the URL.
  - **Ingest server path** — walk a file or directory on the engine host by glob (default `**/*.md`). Unchanged files are skipped via a per-bank content hash; re-run is idempotent.
- **Documents table:** lists the bank's documents with derived-unit counts and last-updated time; the **✕** button deletes a document and its derived memory units (with a confirm prompt).
- **Operations monitor:** shows recent async ingest/consolidation operations for the bank with status and any error — Hindsight processes uploads asynchronously, so a freshly-added document appears here as a pending operation before its units are extracted. Use **Refresh** to re-poll.

Documents ingested here are **automatically chunked and extracted by Hindsight** (server-side) using the bank's retain mission — the panel just declares the bank and hands over the content. Operator uploads are tagged trusted. When the engine runs on the SQLite memory backend (no Hindsight), the panel reports the backend has no import surface (HTTP 501) rather than silently doing nothing.

#### Content-date anchoring (DP-292 phase 2)

Every extracted memory in Hindsight is stamped with an **anchor date** (the `mentioned_at` / `event_date` that recall uses for recency and time-range queries). Hindsight derives that date **entirely from the `timestamp` the engine sends on the retain request** — the extraction LLM does *not* read dates out of the prose. So for a document to anchor to *when its content is actually about* (rather than the moment you uploaded it), the engine has to find that date itself before handing the document over.

On every ingest (upload, URL fetch, server path) the engine extracts a **single anchor date** from the document body:

1. **Regex pass (always runs).** Scans the body for machine-readable dates — ISO (`2026-03-12`, `2026-03-12T10:00`), `2026/03/12`, and named-month forms (`March 12, 2026`, `12 March 2026`). Locale-ambiguous bare-numeric forms (`03/12/2026`) are **deliberately ignored** — they cause more wrong anchors than they fix. Of the dates found, the engine picks the **latest one that isn't in the future**; future-dated values (e.g. a document that says "as of 2099") are dropped. This is the path for chat logs and dated notes, where the date sits in line headers.
2. **LLM fallback (optional, only when the regex finds nothing).** A single-shot, sealed **date-tagger** reads the body purely as data and returns one ISO date or "none". It exists to catch prose-only dates ("we met last March", "the Q2 review"). Its output is validated and future-clamped exactly like the regex result — it can only *propose* a plausible past date, never inject instructions, reach a tool, or push the anchor past now. Disabled with `DATE_TAGGER_ENABLED=0`, in which case ingest is regex-only.
3. **Fallback.** If neither finds a usable date, the document anchors to its previous default — file mtime for server-path ingest, upload/fetch time for uploads and URLs.

Each ingested document is tagged `date:<YYYY-MM-DD>` and `date_source:<regex|llm|fallback>`, and the same values are stored in its metadata, so you can see in the documents table and in recall which anchor was used and how it was derived. Anchoring is **per document** (one document → one date); Hindsight stamps every memory unit it extracts from that document with that one anchor. A document whose content genuinely spans weeks (a long chat log) anchors to its most recent dated line, which keeps it correctly ranked for recency in recall.

**Format-gap notifications.** Every time the LLM fallback succeeds, it means the regex missed a date format that a real document actually used. To close that gap over time, the date-tagger sends the operator a Discord DM naming the **verbatim date string** it read (e.g. `"3rd quarter '26"`) plus the ISO date it resolved — a prompt to extend the regex so that format stops needing the LLM. Notifications are **deduplicated by format shape** (digit-masked), so bulk-ingesting many documents that share one unmatched format produces a single DM, not one per document. A reported string is only sent if it genuinely appears in the document body (guards against a hallucinated date). The DM target is the `date_tagger` agent's `notification_defaults` in `agents.json` (default: DM to `adrich`); with Discord disabled the report degrades to a log line.

## Commands

All commands are entered as the message body when addressing a persona. Commands are case-insensitive.

**Persona origin allowlist (DP-330):** a persona can additionally declare *which
origins may address it at all*, independent of what any command may do. Set it
with the operator command `set origin_allowlist <entry ...>`, where each entry is
`server_id[/channel_id[/author_id]]` — the same syntax as `OPERATOR_ALLOWLIST`, so
a bare guild id means "that whole Discord server, any channel, any user". Read it
back with `what origin_allowlist`; clear it with `set origin_allowlist none`.

- **Empty (the default) means unrestricted** — every persona that never sets the
  field behaves exactly as before.
- A persona **with** an allowlist is Discord-only: entries are guild ids, and no
  other transport has a gateway-asserted one. DMs, the web portal, email, ticket
  bodies and agent-initiated turns are all refused with `Persona '<name>' is not
  available from this channel.` The refusal never says what the allowlist holds.
- The gate sits above dev-command handling, so a disallowed origin cannot address
  a restricted persona with `what prompt` either — not on Discord, not from the
  portal's dev-command surface. It does **not** cover the portal's unauthenticated
  `GET` routes, which return a restricted persona's prompt and history to anyone
  who can reach `:5003`. That gap and the LAN-only precondition it depends on are
  spelled out in [Portal Control Plane](#portal-control-plane-dp-277) — read it
  before relying on this field to keep a persona private.
- Malformed entries are dropped rather than honored (a wildcard server would
  grant every guild the bot is in). If *nothing* you supply parses, the persona
  becomes **unreachable from every origin** rather than unrestricted — the safe
  direction — and both `set` and `what origin_allowlist` say so in those words.
  This holds for every unusable shape, including a list of blanks (`["", " "]`)
  and one whose entries are not text at all (`[null]`, `[true]`, `[["12345"]]`).
  A file that fails closed is also **saved** as you wrote it, so a later
  `set temp 0.8` cannot quietly rewrite it to `[]` and reopen the persona.

> **⚠️ You can lock yourself out, and `set origin_allowlist` cannot undo it.**
> The command always targets the persona you are addressing — there is no
> cross-persona form, so running it against a *different* persona clears that
> persona's restriction and leaves the locked-out one exactly as it was. And
> because an allowlisted persona is Discord-only, setting one from the portal
> immediately locks the portal's own dev-command route out of it.
>
> Recovery is a **Discord origin the allowlist admits**, or editing that
> persona's `origin_allowlist` in `data/personas.json` and restarting. Set the
> field from a Discord channel that the list itself allows, and this cannot
> happen.
- Guild ids may be written unquoted in a persona file (`"origin_allowlist":
  [347812763093172225]`) — they are numbers, and the file accepts them as such.
- Changing the field is recorded in the audit log (`origin_allowlist_change`,
  with the operator, the previous list and the new one), like `explicit_overrides`.
- This is not the same gate as operator gating: operator gating asks *may this
  command reconfigure things*, the allowlist asks *may you talk to this persona*.
  A persona can be reachable but non-operator, and it can be operator-authenticated
  in a guild that its allowlist excludes.

**Operator gating (DP-277):** commands that reconfigure a persona or the system — `set`, `add`, `delete`, `remember`, `trust`, `untrust`, `update_models` — are **control-plane** and only honored from an authenticated operator origin: an allowlisted Discord server/channel/user (`OPERATOR_ALLOWLIST`, matched against gateway-asserted ids) or the portal's operator-authenticated control surface. From any other origin (unlisted Discord channels, email, ticket bodies, anonymous portal chat) they are refused with `Refused: '<command>' is an operator command…` — this is the structural defense against injected instructions trying to reconfigure an agent. Read and lifecycle commands (`what`, `detail`, `help`, `dump_last`, `dump_history`, `hello`, `goodbye`) stay open to everyone.

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
| `explicit_overrides <spec>` | Override names, JSON list, or `none` | **Privileged (DP-277):** the only way to suppress a tool-composition security rule. Audit-logged; not settable via `set tool_policy` or the persona API. See [Tool Security](#tool-security). |
| `memory_mode <mode>` | See Memory Modes below | History retrieval scope |
| `service_bindings <list\|none>` | Comma-separated service names | e.g., `set service_bindings zammad,agents` |
| `max_context_tokens <integer>` | Integer >= 100 | Total context budget — prompt + reserved response (matches kobold-lite's `max_context_length` slider). Effective prompt prune budget = this minus `tokens`. Oldest non-system messages drop until prompt fits; system messages and the latest user message are always preserved. Default 131072. |
| `chat_template <name>` | An instruct-preset name | Prompt template used by the **local** (kobold-native) engine — e.g. `chatml` (default), `alpaca`, and the other kobold-sourced presets including thinking variants. Unknown names are rejected. `GET /api/v1/chat_templates` lists the valid values, and the persona editor renders them as a dropdown. Only affects `model local`. |
| `thinking_level <value>` | e.g. `minimal` | Extended-thinking level passed through to the provider (e.g. Gemma). Clears when set to a non-value. |
| `long_term_memory <on\|off>` | on/off | Per-persona long-term-memory switch. Off disables both retrieval *and* retain for this persona — it stops contributing to and drawing from the memory store. |
| `include_ambient_memory <on\|off>` | on/off | Whether ambient-channel memories (messages logged under persona `ambient`) are eligible for this persona's retrieval. Default on. |
| `inject_timestamp <on\|off>` | on/off | Prepend a timestamp to each turn. Defaults on for user personas, off for system personas. |
| `ingest_bank <name>` | Bank name | Hindsight bank the `ingest_path` tool targets for this persona when the tool call omits `bank`. |
| `<provider>.<key> <value>` | Any provider id + scalar value | Fallback dotted-path setter for provider-specific knobs that have no first-class command (e.g. `set kobold.mirostat 2`, `set kobold.rep_pen 1.15`). Stored in `params.provider_extras[<provider>][<key>]`. Value is coerced to int / float / bool when possible, otherwise kept as a string. Use `set <provider>.<key> none` (or `null`/`clear`) to remove the key. Mirror read: `what <provider>.<key>`. |

Every `set <name>` above has a matching `what <name>` read. The full settable set is the `src/persona_fields.py` registry — **26 fields, 24 CLI-settable, 16 also exposed on the persona PATCH route**. `set model` is the one bespoke setter (it needs an async fuzzy-match lookup); `model` is therefore PATCH-able but generated from a hand-written CLI handler.

### Configuring by conversation — `configr` (DP-331)

The `set` table above is the exact-syntax path: it is fast, deterministic, and needs
no LLM. `configr` is the fallback for the two cases that path handles badly — you
don't remember the exact field name, or you are speaking rather than typing.

**It activates on a near-miss, not on a keyword.** You do not address `configr`
directly. It is a system persona the command layer hands off to when a control-plane
command is recognized but its arguments are not:

| You send | Today | With configr |
|----------|-------|--------------|
| `set randomness lower` | `Error: Unknown 'set' command: randomness` | reads it as `set temp`, proposes a lower value |
| `set temp` (value omitted) | silently falls through to the persona | asks for a value, or infers one |
| `who won the game` | goes to the persona you addressed | unchanged — never reaches configr |

A message the command layer does not recognize at all is untouched. `configr` only
ever sees input that already looked like configuration and failed.

**It writes commands, not settings.** `configr` translates what you said into ordinary
commands from the table above and runs them through the same path a typed command
takes. It has no private route into a persona. That is deliberate: the operator gate,
the audit log for privileged fields, and the tool-composition quarantine re-check all
behave identically whether you typed `set temp 0.9` or said "make it less random."

It can target a persona other than the one you are talking to — "give managr more
history" works from any conversation.

**Every change it makes is parked for your approval.** One utterance produces one
pending approval carrying the whole change list, not one per field: "make managr less
random and give it more history" is a single decision and takes a single ✅. The
change list is shown as before → after, so you can see what it understood before
agreeing to it. Approve from Discord or the portal the same way you approve any
other parked action; approving by voice is not supported yet.

**It can decide you were not configuring anything.** A near-miss is not proof of
intent — `set aside some time for me` parses as a failed `set`. When `configr`
concludes the message was not a configuration request, it yields the turn and your
message reaches the persona you addressed, unchanged and in full. Nothing is applied
and nothing is parked.

**It gets one attempt.** If what `configr` produces is still not a valid command, you
get the plain error message you would have gotten anyway. It does not retry, and its
own output never re-enters it.

**Operator origins only.** `configr` sits behind the same DP-277 control-plane gate as
`set` itself — a non-operator origin is refused before `configr` is ever consulted.
It cannot be used to reach configuration from a channel that could not already
reach it.

**Cost.** A failed `set` now costs one call to a small local model (temperature 0,
~100 tokens) before you see the error. This is the same trade `set model` already
makes for fuzzy model matching.

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
| `dump_history` | Full context dump as a downloadable file — the exact history array sent to the model on the last request. (This command was previously documented here as `dump_context`, which has never existed.) |
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
- `AGY_SANDBOX` (default `True`): Run `agy` under its built-in OS-level sandbox (`--sandbox`; nsjail on Linux, sandbox-exec on macOS — see the platform note below for Windows). Set to `False` if the sandbox is unavailable in your environment (e.g. a container without the needed privileges).

> **Prompt size.** The `agy` and `cc-*` CLIs take the whole prompt as a single
> command-line argument, and the OS caps that (128 KiB per argument on POSIX;
> 32767 characters for the entire command line on Windows). A very long
> conversation (or a single huge tool result) is therefore trimmed before the
> call: the persona prompt is kept, then whole messages are dropped oldest-first
> and replaced with an `[...older conversation elided...]` marker. Trimming is
> per-message, so a message is never cut in half — you will not see a stray
> fragment of a tool result presented as conversation. If not even one message
> fits, your most recent question is what is kept. The other providers, which
> send the history over HTTP, are unaffected.

> **Platform: any host with the `agy` CLI.** Linux, macOS, WSL, Docker **and
> native Windows** all work. Windows used to be refused: older `agy` builds only
> wrote the response to a TTY, so the engine's piped capture came back empty.
> `agy` 1.1.9 writes to a pipe on Windows too, so that guard is gone. Two
> practical differences on Windows:
> - **Less prompt fits.** Windows caps the *whole* command line at 32767
>   characters (POSIX caps each argument at 128 KiB), so the prompt budget is
>   20 KiB there instead of 96 KiB — long conversations are trimmed sooner, by
>   the same oldest-first rule described above. Windows also measures the
>   command line *after* escaping, and quotes inside the text are escaped, so
>   quote-dense history (JSON tool results) uses up the budget faster than its
>   character count suggests — up to twice as fast in the worst case.
> - **Sandbox enforcement is unverified.** `agy --sandbox` is accepted and the
>   call succeeds, but the documented sandbox backends (nsjail, sandbox-exec) are
>   POSIX; treat the sandbox as defense-in-depth you should not rely on when
>   running the engine on Windows.

Tools work via an **inline protocol**: the engine injects the tool descriptions
into the prompt and asks the model to emit a `<tool_call>{…}</tool_call>` block to
request a tool. The engine parses that block and runs the tool through DERPR's
normal tool loop — so persona tool policy, the read/write split, CONFIRM-mode
approval, and untrusted-taint all apply exactly as they do for the other
providers. The model never executes tools itself.

### Claude Code (`cc`) — sandboxed autonomous provider

`cc-*` models (`set model cc-sonnet`, `cc-opus`, `cc-haiku`) route through the
local `claude -p` headless CLI instead of an API, running on the user's Claude
**subscription/OAuth tier**. Like `agy` it is a subprocess-per-call, one-shot
provider with a persistent per-persona workspace and its own rate limiter —
though unlike `agy` it is **POSIX-only while its sandbox is on** (see the
platform note further down) — but it differs from every other provider in one
important way:

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

#### Project instructions and memory

Every `cc-*` instance — a persona workspace or a dispatched fixr worktree — gets
`CLAUDE.md` and a **writable** `memory/`, so the instructions it reads are ones
it can actually follow.

`CLAUDE.md` was never the problem for a dispatched agent: a worktree is a real
checkout, and Claude Code reads project instructions from the working directory
even though DERPR passes `--system-prompt` (which replaces the system prompt but
does not suppress instruction files). A **persona workspace** is a bare
directory, so it gets a copy, refreshed on every call.

`memory/` is the notes repo, and no checkout has ever carried it — it is
gitignored, so the CI-built production image has no copy. DERPR now keeps ONE
shared clone and links it into each workspace as `memory/`. It is writable
because the memory protocol tells agents to record decisions and root causes;
read-only would make that instruction unfollowable. Two agents running at once
therefore share one working copy and can collide on `main` — accepted, since
memory writes are append-mostly to distinct files.

- `CC_NOTES_ENABLED` (default `True`): master switch. Off = no link, no seeding.
- `CC_NOTES_DIR` (default `data/notes`): the shared clone.
- `CC_NOTES_REPO_URL` (default unset): derived from the running checkout's own
  `memory/` remote when unset. **The container has no `memory/`, so a deploy
  must set this** or its agents run without memory.
- `CC_NOTES_BRANCH` (default `main`): the notes repo uses `main`, not `master`.

To push memory back, the notes repo needs `github.com,api.github.com` in
`CC_SANDBOX_ALLOWED_DOMAINS` and a `GH_TOKEN` in the environment. The notes repo
is private, so unlike the fixr base clone even the initial clone needs the token.
Every failure here degrades loudly rather than breaking the run: the agent
proceeds without memory and the reason is logged at WARNING.

> **The link points out of the workspace.** The sandbox confines writes to the
> working directory, so the clone's real path is added to the sandbox's
> `filesystem.allowWrite`. Without that entry the link exists but every write
> fails with a bare `EACCES`.

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
| dispatchr | gemini-2.5-flash | Zammad ticket management | CONFIRM | Zammad toolset (`zammad` binding) |
| fixr | default | Self-repair supervisor — dispatches Claude Code agents, never edits code itself | AUTONOMOUS | fixr toolset (`fixr` binding) |
| it-help | gemini-2.5-flash | Testing/dev persona for Zammad integration | AUTONOMOUS | All |
| gemini | gemini-2.5-flash | General-purpose Gemini | AUTONOMOUS | google_grounding_search |
| chatgpt | gpt-5 | General-purpose GPT | AUTONOMOUS | None |
| claude | claude-haiku-4-5-20251001 | General-purpose Claude | AUTONOMOUS | None |
| testr | gemini-2.5-flash | Test persona (responds "success") | AUTONOMOUS | None |

#### hypr — the infra operator (opt-in, not seeded)

`hypr` is **not** a default persona and is **not** created on a fresh deploy. It
ships as a standalone definition at `config/optional_personas/hypr.json`, and an
operator adds it to a specific instance on purpose.

That is deliberate. `hypr` is the only persona holding the `proxmox` binding, and
a parked write is approved by whoever raised it — so *who can address `hypr`* is
the authorization boundary for node power operations. Seeding it would put that
surface on every deployment, including ones whose Discord guild has other people
in it.

**To add it to a running instance:** copy the entry from
`config/optional_personas/hypr.json` into that instance's live
`data/personas.json` (`personas` array) and restart, or create it through the
persona-management commands. The shipped prompt carries an **OPERATOR NOTE** with
two `<unset>` placeholders — fill in the guest that hosts derpr and the guest that
serves the model on `:5001` for *your* node. The template deliberately names no
guests, so until you fill those in `hypr` can only warn in general terms about
powering off its own host. Auto-seeding only ever writes `data/personas.json`
when the file does not already exist, so merging a release never adds a persona
to an instance that is already running.

⚠️ **The same applies to prompt *edits*, and it has bitten three tickets.** When
a release changes `hypr`'s shipped prompt — DP-332's `gpu_status`, DP-265's four
`hf_*` tools, DP-335's read-batching and dead-end guidance — the running
instance keeps the prompt in its own `data/personas.json` and never sees the
change. Merging and redeploying the image is **not** enough. Splice the new
paragraphs into the live record by hand (against its existing anchors, not by
copying the template over it, which would discard whatever the operator has
tuned), or the feature ships fully wired and behaves exactly as it did before.

**Restrict who can reach it (recommended).** Because reachability *is* the authz
boundary here, set `hypr`'s [origin allowlist](#commands) to the one Discord
server you administer:

```
hypr set origin_allowlist <your_guild_id>
```

It ships with `origin_allowlist: []` (unrestricted) because the shipped file is
public and the guild id is yours — fill it in on your own instance. The empty
list is kept on save, so the key stays visible in your `data/personas.json` as a
reminder that the knob exists. Note the trade-off: an allowlisted `hypr` is
Discord-only, so its parked actions can only be approved in Discord, not from the
portal.

Once present, `hypr` is the persona to talk to about the box itself, from **any**
interface that routes to a persona (Discord, the web portal, `/derpr`) — unless
you narrow that with an origin allowlist as above. It holds
the `proxmox` binding and nothing else: it can read the node's topology and power/model state,
and it can act on them — but every destructive action is a write tool, so it
**parks for your approval** on whatever surface you asked from. `reboot_node` is
additionally flagged irreversible.

What it can do:

- **Audit** — "what's running on the box?" → `pve_status` returns uptime plus every
  guest with its `vmid`, `name`, `kind`, and `status`. "what models can I load?" →
  `list_models`, read live off the GPU container. "how much VRAM is free?" →
  `gpu_status`.
- **Power** — "restart the GPU container", "stop the idle VM", "reboot the whole box".
  Guests are addressable by name; see [Addressing a guest by name](#addressing-a-guest-by-name).
- **koboldcpp** — "switch :5001 to gemma" → `set_active_model`.

It is `AUTONOMOUS` on purpose: the execution mode is not what protects you here —
the write-audit is, and that fires regardless of mode. Making it `CONFIRM` would
only add a second prompt in front of the same park.

**It operates the node it runs on.** derpr's own container is a guest on that
node, as are the services around it, so `stop_guest`/`reboot_guest` against that
guest — and `reboot_node` always — take derpr down mid-conversation along with
the approval gate that parks these very actions, and `set_active_model` can cut
off derpr's own inference if it is served by the node's koboldcpp. Nothing in the
toolset refuses a self-directed power-off; the persona prompt names the guest it
lives inside so it has to say what an action costs before proposing it. Read the
park prompt with that in mind.

`hypr` is inert unless `PVE_TOOLS_ENABLED=true` and the pve SSH key is mounted;
without those, every tool call returns a "disabled" error rather than attempting
SSH. `set_active_model` only switches between models already installed on the GPU
container. It does not need to be told which those are: the model list is
enumerated from the container's own systemd units on every call, so a model you
install by hand shows up in `list_models` immediately, with no config edit and no
redeploy. Write the unit however you like — `--model <path>` and
`--model=<path>` are both read — and give it a `--port` if it is not meant to
serve `:5001`, which keeps it out of the swap set entirely.

- **Provision (DP-265)** — "find me a smaller gemma quant" → `hf_search` /
  `hf_files`, then `install_model` downloads it onto the model host and writes a
  **disabled** unit for it. See [HuggingFace Model Tools](#huggingface-model-tools-requires-service_bindings-huggingface).
  This is the second binding `hypr` holds, and deliberately the only one: it
  provisions models onto the very node `proxmox` operates, so it is the same
  blast radius rather than a new one. It needs `HF_TOOLS_ENABLED=true` and the
  node-side script from `services/pve/` deployed.

### System Personas

Defined in `config/system_personas.json`. Not directly user-accessible — used internally by agents for analysis tasks:

- **model_selector** — Fuzzy model name matching for `set model`
- **tool_selector** — Fuzzy tool name matching for `set tools`
- **configr** — Interprets a failed `set` as a configuration request and rewrites it as valid commands (DP-331)
- **triage_analyst** — Ticket analysis and internal note generation
- **triage_scout** — Keyword extraction from tickets for search
- **triage_filter** — Relevance scoring between historical and new tickets
- **triage_summarizer** — Ticket content compression
- **dispatch_analyst** — Priority assignment and dispatch notification generation
- **memory_summarizer** — Extracts observations from conversation segments for long-term recall; used by SqliteConsolidator

## Execution Modes

Determines the autonomy level for a persona's tool-use capabilities.

| Mode | Behavior |
|------|----------|
| **AUTONOMOUS** | **Read-only** tools execute immediately. **Write tools still require audit** (see [Tool Security](#tool-security) below). The user sees the final response after all automated steps. |
| **CONFIRM** | Standard mode. All write tools are presented for approval. Provides a consistent point of review for all state-changing actions. On Discord, approval uses reaction buttons; proposals wait up to 24 hours, survive a restart, and are answered independently. |

## Tool Security

The bot implements a comprehensive security framework to prevent prompt injection and unauthorized actions.

### Universal Write-Audit
Regardless of execution mode, **all write tools** (tools that modify state, like creating tickets or deleting users) are parked for human audit before execution. This ensures that no state-changing action is taken without explicit user consent.

**Proposing an action no longer ends the turn (DP-297).** A gated write is queued
and the model keeps working, so one reply can propose several actions — each with
its own approve/deny affordance, each resolvable independently and out of order.
Previously a persona could propose exactly one write per turn and the turn stopped
there; a second proposal made during an approval was rendered as plain text with no
buttons and dangled forever, unanswerable. Each proposal now carries a stable token,
so the surface always knows which action you answered.

What you see in history: a proposed write appears as *awaiting approval* until you
decide, then that same entry is rewritten in place with the real outcome —
`approved` plus the result, `denied`, or `expired` if it was never answered in time.
The persona reads that outcome on its next turn, so it can tell a completed action
from one it merely suggested, and does not re-propose something you already approved.

A denial carries its instruction with it: the entry reads *"Tool call denied by
operator. Wait for corrections or further instruction."* — the verdict and what to
do about it have the same lifetime, so the guidance is still there many turns
later rather than only in the reply immediately after you denied. Note this is
guidance the persona reads, not a hard block: if you deny a write and then ask for
it directly, it will propose it again, which is intended.

**The same action is never queued twice.** If a persona proposes an action
identical to one already waiting for you — same tool, same arguments — it is not
added to your queue a second time; the model is told it is already pending and
which proposal it belongs to. You will not get two checkmarks for one action, and
approving cannot run it twice. Volume is not restricted: proposing six *different*
ticket updates in one reply gives you six proposals, because that is six decisions.
Once you have answered a proposal it is no longer pending, so a persona can
legitimately propose the same action again later — if you denied a write and then
ask for it directly, it reaches you again rather than being silently swallowed.

**Approvals survive a restart (DP-319).** Pending proposals are stored on disk, so
the 24-hour window is the real deadline rather than a ceiling capped by how long
the bot happened to stay up. Four details worth knowing:

- *The portal recovers on its own.* Reloading a conversation re-renders every
  proposal still waiting, with its approve/deny bar intact.
- *Discord re-posts on your next message.* Reaction buttons live on Discord
  messages the restarted bot no longer recognizes, so the old checkmarks go dead.
  The next time you talk to that persona in that channel, anything still pending
  is posted again with fresh buttons. Answer the new message, not the old one. If
  you click a dead button in the meantime the bot tells you so rather than
  ignoring you — it will not silently swallow an approval. That reply only
  appears on an actual proposal message; reacting ✅ to an ordinary answer is
  still just a reaction.
- *A proposal whose stored details cannot be read is not offered.* In the rare
  case that a proposal's record is damaged on disk, it is closed out at startup
  instead of being restored as an approvable-looking button with nothing behind
  it. If an action you expected to still be waiting is gone after a restart,
  ask for it again.
- *An approval interrupted mid-flight is not retried.* If the bot dies in the
  seconds between your click and the action running, it will not guess. The
  proposal is closed as "interrupted", the persona is told the outcome is unknown,
  and it checks the current state before offering to do it again — because the
  alternative is running an irreversible action twice.

**One approval, one execution.** Approving a proposal runs its action exactly
once. If the persona re-proposes the same action while reporting on the one you
just approved — which is when it is most likely to, since it is re-reading its own
work — you are not given a second button for it; it is told the action already ran
and what the outcome was. This applies only while the persona is reporting back on
your decision. **Asking for something again yourself always reaches you**: say
"restart that service again" a minute after the first restart and you get a fresh
proposal to approve, because a request you typed is a request, not the model
repeating itself. A *denied* action can likewise be re-proposed immediately, since
nothing happened the first time.

An action that was approved and then *errored* counts as having run for this
purpose. The tool may have taken effect before it failed — a ticket created just
before the API returned an error — so the persona is told the outcome is unknown
and asked to check the current state rather than being handed a second button that
would create the ticket twice.

**Gated actions stay in the model's memory (DP-296).** A persona remembers the tool calls it made and the writes it proposed, even when the turn did not finish cleanly — a proposal you approved, denied, or simply never answered, a turn that died on a provider error, or one that hit the tool-iteration cap. Previously only turns that ended with the model writing text carried their tool calls forward, so gated actions were invisible on the next turn: ask "did you see the error you got?" after a parked proposal and the persona had no record of ever calling anything, and would re-propose or invent an answer. Unfinished calls are recorded as *not executed* with a reason (`awaiting_approval`, `denied`, `error`), so a persona can tell "I did this" from "I asked to do this and it didn't happen" and stops re-proposing denied actions. This holds for regenerated turns too: retrying a reply that then proposes a write records the proposal against the regenerated message, without disturbing the text that message already shows or the version history behind it.

**When a persona runs out of tool steps, it answers anyway — and shows you what it spent them on (DP-335).** A turn has a fixed budget of tool steps: **15 tool calls**, counted as calls actually made rather than as replies from the model, so the allowance is the same whichever model is answering. If the persona uses all of them without reaching an answer, it no longer replies with just "I seem to be stuck in a loop":

- It writes a **real answer first**, from everything it already found — what it learned, what is still unresolved and why, and the one next step it would take. Hitting the cap usually does not mean anything broke; the common case is a persona that spent its budget searching and re-reading and simply never got to the action you asked for, and by then it generally knows enough to say something useful.
- Under that it lists **every tool call it made**, in order, with the arguments it used and how each one turned out (succeeded, failed with the error, or still waiting for your approval), and marks any call it made twice with the same arguments. The prose can be vague about what it ran; this list is read straight from the record and cannot be, so you can check one against the other.

That answer is a normal reply — it is remembered like any other, so you can follow up on it in the next message. (The prose is what gets remembered; the call list under it is shown to you but not filed away as something the persona said.) If the model is unreachable at that moment, you still get the list on its own.

The 15 is a stopping condition, not a hard ceiling: if the persona asks for several tools in one message, that whole group runs even when it crosses the line, and the turn ends immediately after. Half of a group the persona proposed as one plan is worse than a call or two of overshoot, and the group runs concurrently anyway. So a turn can report having spent, say, 17 of its 15 steps — that is the overshoot, not a miscount. Size rate limits and costs against a small margin above 15 rather than exactly 15.

### Taint Tracking
The system tracks the "trustworthiness" of the conversation context. If a persona uses a tool that retrieves potentially untrusted content (like `web_search` or `recall_memory` containing past external input), the current turn is marked as **tainted**. 
- Taint is "sticky" for the duration of the conversation.
- When a turn is tainted, any subsequent write-tool approval request will carry a warning: `⚠️ Context contains untrusted content from: [source]`.

### Insecure Composition Blocking
To prevent sophisticated injection attacks, the system refuses to load any persona whose tool configuration creates an inherently insecure path. Common blocked compositions include:
- **`network:read` + `local:write`**: Prevents tools that read from the internet from being used by a persona that can write to local storage/files.
- **`untrusted:read` + `network:write`**: Prevents untrusted data from being exfiltrated to a network endpoint.
- **`pii:read` + `network:*`**: Prevents sensitive Personal Identifiable Information (PII) from being sent over the network.

A `network` tool may opt out of the exfiltration rules (the last two above) with `capabilities.exfil_capable: false` when its egress carries no model-controlled payload — e.g. `set_active_model`, whose only argument is a lookup key against the units discovered on the box (the handler sends the discovered unit name, never the caller's string), and `gpu_status`, which takes no arguments at all — so nothing can ride out over their SSH. Such tools can freely combine with `untrusted:read`/`pii:read` tools. This affects only *exfiltration* accounting; any destructive effect is still gated by the write-audit (parked for confirmation). The default is `true`, so every other tool is unchanged.

### Explicit Overrides (privileged, DP-277)
A composition rule can be deliberately suppressed for a persona with an **explicit override** (`network_read_local_write`, `untrusted_read_network_write`, `pii_read_network_any`). Because overrides are the kill switch for the whole composition framework, they are a **privileged field with a single mutation path**:
- `set explicit_overrides <name ...|json list|none>` is the only command that changes them; every change is audit-logged (operator, prior, and new value) and immediately re-runs the security validation.
- `set tool_policy <json>` and the persona PATCH/create API **ignore** `explicit_overrides` inside a policy dict — a caller-supplied policy can never disable the composition rules.
- `what explicit_overrides` shows the active overrides. In the portal's Tools tab, override checkboxes save through the dedicated command automatically.
- Overrides survive `set tools` / `set tool_policy` edits (they are persona-level, not part of the policy dict) and persist as a top-level `explicit_overrides` key in the persona save file; legacy files that stored them inside `tool_policy` are migrated on load.

### Irreversibility Flags
Some tools are marked as **IRREVERSIBLE** (e.g., `delete_user`). Tools can also be dynamically flagged based on their arguments via an `irreversible_if` classifier (currently unused — `add_note_to_ticket` used one while it could write customer-visible notes; it is now internal-only). These flags are surfaced in the approval dialogue to highlight high-stakes actions.

### Credential Scoping
Machine secrets — provider API keys (`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `GOOGLE_GENERATIVEAI_API_KEY`) and the Zammad API token — are kept out of the LLM's reach by design:
- **Central vault.** All secrets are resolved through a single credential vault rather than read ad-hoc from the environment, giving one authoritative inventory of what counts as a secret. The model never receives secrets as tool arguments.
- **Egress scrubbing.** At startup every known secret value is registered with an egress scrubber. Any string headed for a place the model (or operator inspector) can read it back is scrubbed first, with the value replaced by `[REDACTED:<KEY_NAME>]`. Three boundaries are enforced: **tool results** before they re-enter the conversation, **write-audit arguments** before they are stored or shown in an approval dialogue, and the **cached request payload** surfaced by the `/assemble` inspector.
- **Shape-based fallback.** Beyond known values, the scrubber also redacts strings that *look* like secrets (e.g. `sk-…` API keys, `Token token=…` headers, bearer tokens), so an unregistered credential that leaks into a tool result is still caught.

This is defense-in-depth: today's curated tool surface means no tool returns secrets, but the guarantee holds automatically the moment a more powerful tool (e.g. shell execution) or a bring-your-own-credentials mode is added.

### Portal Control Plane (DP-277)
The web portal (`:5003`) is the capability control surface — its persona/tools/policy panes reconfigure agents — so its mutating routes are authenticated:
- **Operator token.** Every non-GET route except the generation/abort/token-count data plane requires the operator token (`DERPR_CONTROL_TOKEN`), presented as `Authorization: Bearer <token>` (or `X-Derpr-Token`). This covers persona create/edit (`POST /personas`, `PATCH /persona/{name}`), `dev_command`, **the `/confirm` HITL-approval endpoint**, active-persona switch (`PUT /model`), history reset, and interaction edits/deletes. Reads stay open. **If `DERPR_CONTROL_TOKEN` is unset the control plane is locked** (every mutating route answers 401) — set it to enable portal-side configuration. The token compares in constant time and is never surfaced to the model, a persona, or any tool result. In the portal UI, paste it once via the **operator** control in the top bar (stored in the browser, sent automatically thereafter).
- **Deny-by-default routing.** New mutating routes are gated the moment they are added — the allowlist enumerates the *data plane* (generation, abort, token count), not the protected surface, so nothing is accidentally left open.
- **Chat elevation.** Commands typed into the portal chat box are refused for anonymous callers (chat is data-plane), but a valid operator token on the request elevates the origin so the operator's own typed dev commands keep working.
- **Network posture.** The adapter binds `KOBOLD_ADAPTER_HOST` (default `0.0.0.0`). In the containerized deploy the app is reached via Docker port publishing behind a Caddy TLS front, so the app's bind address is not the network boundary — the operator token gate is what protects the surface **for writes** (including the plaintext host port that sits behind Caddy). Auto-generated API docs (`/docs`, `/openapi.json`) are disabled, and CORS no longer advertises `allow_credentials` with a wildcard origin (auth is a bearer token, not an ambient browser credential).

> ### ⚠️ Reads are not authenticated — the portal must stay on a trusted network (DP-333)
>
> "Reads stay open" above is literal: the operator-token middleware exempts
> **every** `GET`, so anyone who can reach `:5003` can read, with no credential:
>
> | route | what it returns |
> |---|---|
> | `GET /api/v1/persona/{name}` | the persona's full configuration, **including its system prompt** |
> | `GET /api/v1/session/{p}/assemble` | the exact assembled system prompt, rebuilt history and resolved parameters a live submit would send |
> | `GET /api/v1/session/{p}/transcript` | the conversation transcript, plus any live parked action's confirmation text and token |
> | `GET /api/v1/session/{p}/kobold_export` | the persona's global history as a savefile |
> | `GET /api/v1/session/{p}/ltm_block` | long-term-memory retrieval for a **caller-supplied query** — i.e. arbitrary search over stored memories |
> | `GET /api/v1/interaction/{id}/versions` | per-message edit history |
> | `GET /api/v1/memory/banks…` | Hindsight bank and document listings |
>
> There is no per-persona check on these, so a persona restricted with an
> [origin allowlist](#commands) is as readable as any other. **The allowlist
> gates who may *address* a persona; it does not gate these reads.**
>
> This is an **accepted risk, valid only while the portal is LAN-only.** The
> shipped deployment satisfies that: `Caddyfile` serves `10.0.0.70:5003` /
> `derpr-host:5003` with `tls internal` — a private address and a private CA,
> no public hostname, no ACME certificate, no tunnel. `docker-compose.yml` also
> publishes the app's plaintext port on host `:5004`, which is likewise
> LAN-only and likewise unauthenticated for reads.
>
> **Revisit this before any of the following:** giving the host a public
> interface or a port-forward; putting a public hostname or ACME certificate in
> the `Caddyfile`; running a tunnel (Cloudflare/ngrok/Tailscale Funnel) to
> `:5003` or `:5004`; or hosting a persona whose prompt or history would matter
> to someone on the far side. Reads carrying the same authorization as writes is
> a real change — the portal SPA, `/derpr`, and the kobold-lite integration all
> issue these `GET`s with no token today and would each need to gain one.
>
> Note that a read cannot *change* anything: writes, `dev_command`, and the
> `/confirm` approval endpoint are all `POST`/`PATCH`/`PUT` and remain
> token-gated. Approving a parked action from the portal still requires
> `DERPR_CONTROL_TOKEN`; approving one from Discord still requires the account
> that raised it.

Discord control commands use the separate `OPERATOR_ALLOWLIST` (see [Commands](#commands)); the two operator surfaces feed the same authorization gate, differing only in how each transport authenticates the origin.

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
| `add_note_to_ticket` | Write an internal note (never customer-visible; a customer-facing reply tool will come later, classified as egress) |
| `create_user` | Register a new customer |
| `update_user` | Modify user details |
| `delete_user` | Remove a user (irreversible) |

### Agent Tools (requires `service_bindings: ["agents"]`)

| Tool | Type | Description |
|------|------|-------------|
| `get_agent_status` | Read | View running state, deploy counts, error rates for agents |
| `get_agent_history` | Read | Recent action log with optional ticket/customer filters |
| `lookup_agent_history` | Read | Dereference one action series by `action_id` — returns the root row plus its child steps and context tags. Used to recover a full trajectory after a memory recall surfaces an `action_id:<n>` reference from a bridged agent experience. |
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
| `prune_agents` | Write (ungated) | Reap finished agents: delete the on-disk worktrees of terminal agents (done/error/killed/orphaned) and archive their records (kept for audit, hidden from the default list). Prune one by `agent_id` or bound by `max_age_hours`. Skips any bug with a still-active agent. |
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

#### Talking directly to an agent (DP-230)

When `CC_FIXR_AGENTS_CHANNEL_ID` is set to a Discord channel id, each dispatched
agent gets its **own thread** under that channel. The thread is the agent's live
transcript — progress (coalesced), questions (highlighted), and the final
done/error summary stream into it. **Reply in the thread to talk straight to the
agent**: your message routes to `answer_agent` (`claude --resume`) with **no
`fixr` LLM turn** in the loop — a human↔agent round-trip, not a relayed one. The
thread confirms what happened: a ✅ ack when your answer resumes the agent, or a
⚠️ notice with the reason if it can't (e.g. the agent isn't waiting — only an
agent that asked a `question` and parked is resumable; a reply while it's still
working or after it finished is rejected, not silently spawned as a second run).
A `//` prefix is a note-to-self (gets a 📝 reaction, not sent to the agent).
Only text is forwarded — an attachment-only reply gets a ⚠️ notice to type your
answer, never a silent drop.

If a `question` goes unanswered for `CC_FIXR_IDLE_MINUTES` (default 10), `fixr`
is woken as a fallback to answer or kill the agent. When
`CC_FIXR_AGENTS_CHANNEL_ID` is unset, agent questions wake `fixr` directly as
before (the feature is off). The parent channel must be pre-created. Extra knobs:
`CC_FIXR_PROGRESS_DEBOUNCE_SECONDS` (progress-coalesce window, default 1.5). The
agent's thread "face" is the hidden `fixr-agent` system persona (identity/routing
only — it never generates an LLM reply).

#### Agent lifecycle after the happy path (DP-237)

Agents that leave the happy path are no longer silent or leaky:

- **Restart recovery.** A derpr restart marks every in-flight (running/waiting)
  agent `orphaned` — its detached process + bridge didn't survive, so it can't be
  resumed. On the next startup `fixr` posts a "⚠️ derpr restarted — this agent was
  lost, redispatch?" notice into each orphan's thread and one digest to the fixr
  channel. **Recovery is human-decided** (redispatch with `dispatch_fix`); there
  is no automatic respawn.
- **Worktree + record cleanup.** Terminal agents' worktrees pile up on disk and
  their records clutter `inspect_agents`. Call `prune_agents` to reap the
  worktrees and soft-archive the rows (kept for audit). Archived agents are
  hidden from `inspect_agents` by default — pass `include_archived: true` to see
  them. Pruning a bug that still has an active agent is refused (a re-dispatch
  reuses the same worktree path). When an agent is pruned its thread is locked +
  archived so stale replies visibly hit a closed thread.

### Voice / Timer Tools (requires `service_bindings: ["voice"]`)

Countdown timers, usable two ways with one shared service:

- **Spoken (browser/phone push-to-talk)** — when `VOICE_WEB_ENABLED`, open
  `http://<host>:5003/voice`, hold the button, and say "set a timer for 10
  minutes" (or "…for the pasta"). On release the clip is transcribed locally and,
  if it's a timer command, scheduled; the page echoes back what it heard and the
  timer it set. When it fires it announces in `VOICE_NOTIFY_CHANNEL_ID`, pinging
  whoever set it. A cheap local keyword match handles this — no LLM call per
  utterance. The page uses the browser's native mic API (works on phones too); no
  app to install.
- **Dictation in the portal** — the derpr web UI (`/derpr`) has a hold-to-talk mic
  button next to Send (when `VOICE_WEB_ENABLED`). Holding it records, and on release
  the clip is transcribed and the text dropped into the composer to edit and send —
  so the LLM (which owns the timer tools) acts on it, not a keyword match. A "voice
  auto-send" toggle (off by default, remembered per browser) sends the transcript
  immediately once you trust the transcription. Needs a secure context for mic
  access (localhost / HTTPS / the tailscale-cloudflared path).
- **Always-listening dictation** — the same UI has a "listen" toggle that opens a
  continuous mic stream (WebSocket); the server detects when you stop talking and
  drops each spoken phrase into the composer (or auto-sends it, honouring the same
  toggle). It's a hot mic only while the toggle is on — there's no auth, so it
  trusts the LAN/tailscale network like the rest of the portal. Phrase boundaries
  are silence-based, so a long pause mid-sentence can split a phrase (tunable via
  `VOICE_VAD_SILENCE_MS`).
- **Typed** — the same tools are LLM-callable from any text conversation, so a
  persona can set/list/cancel timers in chat too.

A fired timer announces **back through the channel it was set in**: a timer set
from the derpr portal (by dictation or typing) fires back **in that same portal
conversation** — it appears as a ⏰ chat line and plays a short beep, streamed to
the browser over an SSE back-channel (`GET /voice/alarms`), with no Discord
channel involved. A timer set in a Discord text channel announces there; if a
turn carries no usable channel, it falls back to `VOICE_NOTIFY_CHANNEL_ID`.

> **Why not listen in a Discord voice channel?** It's no longer possible. Discord
> made end-to-end encryption (the DAVE protocol) mandatory on all voice channels
> in 2026, and no Python library can decrypt received audio. The
> `VOICE_ENABLED`/`VOICE_DISCORD_CHANNEL_ID` Discord-capture path is kept behind
> the same internal seam but is inert. Use the push-to-talk page instead.

| Tool | Type | Description |
|------|------|-------------|
| `set_timer` | Read | Start a countdown. `duration` is natural language ("10 minutes", "30 seconds", "1 hour"); optional `label`. Fires back in the channel it was set in (the portal conversation for a web turn, the Discord channel for a Discord turn, else `VOICE_NOTIFY_CHANNEL_ID`). |
| `list_timers` | Read | List pending timers with remaining time and ids. |
| `cancel_timer` | Read | Cancel a pending timer by `timer_id` (from `list_timers`). |

Speech-to-text uses Moonshine on CPU (purpose-built for short voice commands),
so it never contends with the GPU serving the local LLM. Config: `VOICE_WEB_ENABLED`
(push-to-talk page), `VOICE_NOTIFY_CHANNEL_ID` (where a fired timer pings),
`VOICE_STT_MODEL` (`base`/`tiny`), `VOICE_WAKEWORD`. The inert Discord-capture
path also reads `VOICE_ENABLED`, `VOICE_DISCORD_CHANNEL_ID`, `VOICE_VAD_SILENCE_MS`.
All default-off.

### Proxmox Tools (requires `service_bindings: ["proxmox"]`)

Manage the Proxmox host that runs the stack: reboot the node or a guest, start/stop
a guest, and swap which koboldcpp model serves `:5001` on the GPU container. Every
action executes over SSH to the pve node (`pct`/`qm`/`systemctl`/`reboot`); all
**destructive** tools are write tools, so they **park for your approval** before
running (`reboot_node` is additionally flagged irreversible). Read tools run
straight through.

| Tool | Type | Description |
|------|------|-------------|
| `pve_status` | Read | Node uptime plus every guest as **structured data** — `vmid`, `name`, `kind` (`ct`/`vm`), `status`, `lock` — alongside the raw `pct list` / `qm list` text. This is the infra-topology audit: one call answers "what exists and what's up". |
| `list_models` | Read | koboldcpp models for `:5001` that are **immediately available**, and which one is active. The list is **discovered**, not configured — every `koboldcpp-<name>.service` on the GPU container is enumerated per call, so a unit installed since the last deploy appears and a removed one disappears, with no config to keep in sync. Two kinds of unit are omitted: one whose model file is missing from disk (it can't be loaded), and one whose `ExecStart` binds a port other than `:5001` (it isn't a model this tool can swap to — an embedding or draft server, say). What you see here is exactly what `set_active_model` will act on. |
| `gpu_status` | Read | Live VRAM per card on the GPU container, read straight from the card's sysfs: `vram_total_mib`, `vram_used_mib`, `vram_free_mib`. The card index is discovered too. **Mind which budget you are computing.** Free MiB is the headroom beside the model that is already loaded, so it is the right number for growing the *running* model's context. It is the wrong number for a swap: stopping the current unit releases everything it holds, so a `set_active_model` candidate is sized against total minus what stays resident. Size a swap against free MiB and you will rule out models that fit comfortably. Overcommitting does not raise an error — it spills to GTT and decode throughput roughly halves, so a unit that "started fine" can be why the box went slow. |
| `reboot_node` | **Write (parked, irreversible)** | Reboot the metal — takes down every guest on it. Last resort. |
| `reboot_guest` | **Write (parked)** | Reboot one guest, by `name` or by `vmid`. |
| `start_guest` | **Write (parked)** | Start a stopped guest, by `name` or by `vmid`. |
| `stop_guest` | **Write (parked)** | Hard-stop a running guest (power-off, not graceful shutdown), by `name` or by `vmid`. |
| `set_active_model` | **Write (parked)** | Swap the active model on `:5001`: stops every other discovered `koboldcpp-<name>.service` that binds `:5001`, then enables+starts the target (only one may hold the port). Units serving another port are left alone. Pass a `name` from `list_models`. Two guards: if the target's model file isn't on disk it **refuses and leaves the current model running** (never takes `:5001` down); and if a unit that had to be stopped doesn't stop, the swap **aborts before enabling the target** and says so, rather than reporting a success the old model would go on contradicting. |

#### Addressing a guest by name

The three guest tools accept **either** `name` (the guest's hostname as `pve_status`
reports it, case-insensitive) **or** `vmid` (the numeric id). At least one is
required.

- `kind` is optional when addressing by name — the lookup determines whether the
  guest is a container or a VM.
- `kind` is **required** with a bare `vmid`. Proxmox ids are unique across CTs
  and VMs, but *derpr* cannot tell which CLI (`pct` or `qm`) owns an id without
  asking the node — and guessing wrong on a power-off is not worth risking.
- Passing **both** `name` and `vmid` is allowed but they must refer to the same
  guest; if they disagree the call is refused rather than silently acting on the
  `vmid` while the approval prompt shows the `name`.
- An unknown name is refused with the list of names that do exist. A name held by
  both a CT and a VM is refused as ambiguous until you add `kind`.
- If a listing itself failed (`pct list` or `qm list` errored), a name that did
  not match is reported as **not confirmable**, not as absent — on a power path
  "could not tell" and "does not exist" are different answers.
- Names are resolved **locally**, from parsed `pct list` / `qm list` output — a
  name is never sent to the node. Only the resolved digits cross the SSH boundary,
  so the argv/metacharacter guard in `proxmox/ssh.py` still sees nothing but
  integers and config-pinned unit names.
- Resolution happens at **execution** time (after you approve the park), and the
  result echoes the `vmid`, `kind`, and `name` actually acted on, so the audit
  record shows the resolved target rather than only what the model typed.

Disabled by default. Enable with `PVE_TOOLS_ENABLED=true` and mount the pve SSH
key into the container. Config knobs: `PVE_SSH_HOST`/`PVE_SSH_USER`/`PVE_SSH_KEY`/
`PVE_SSH_TIMEOUT` (the node + how to reach it) and `PVE_MODEL_HOST_VMID` (GPU
container id whose systemd units bind `:5001`). When disabled, every tool call
returns a clear "disabled" error instead of attempting SSH.

There is deliberately **no model list to configure**. There used to be — a
`PVE_MODEL_UNITS` map of friendly name → systemd unit — and it was config
asserting what the box contained while the box was the actual authority. It
drifted silently in both directions: a unit listed here but since removed was
quietly omitted, and a unit on the box but missing from the map was never
disabled, so it kept `:5001` and every swap failed to bind with nothing reporting
an error. The units are enumerated from `systemctl list-unit-files` on
`PVE_MODEL_HOST_VMID` on every call instead. Only units matching
`koboldcpp-<name>.service` are recognised — a bare `koboldcpp.service`, or
anything else on the container, is neither listed nor touched.

That name match is where the recognised set *starts*, not where it ends. A unit can be a perfectly real
`koboldcpp-<name>.service` and still have nothing to do with `:5001` — an
embedding server, a draft model for speculative decoding, a helper you wrote by
hand and pointed at another port. The old config map could never name one of
those; discovery reads whatever is on the box. So the swap looks at each unit's
`--port` (koboldcpp's own default is `5001`, so a unit that names no port counts
as competing) and stops only the ones actually holding the endpoint. Anything on
another port is left running, and is left out of `list_models` for the same
reason: the list you choose from and the set that gets stopped are the same set.

If a unit that had to stop doesn't stop — `systemctl disable --now` can hang long
enough on a busy container for the SSH call to be cut short — the swap **stops
there and tells you**, rather than starting the new model into a port the old one
still holds. Systemd would report that second start as a success, and `:5001`
would go on serving the previous model with nothing anywhere saying so. The cost
of stopping is that units already shut down stay shut down, so `:5001` may be
serving nothing until a swap succeeds; the message says as much.

⚠️ **The friendly name is now the unit stem**, so it changed for existing units:
`koboldcpp-fable-q6xl.service` is `fable-q6xl`, not the `fable` the old map
happened to call it. The name a model passes comes from `list_models` in the same
turn, so nothing has to be updated — but if *you* have a name memorised from
before, check `list_models` rather than assuming.

### HuggingFace Model Tools (requires `service_bindings: ["huggingface"]`)

Find a gguf on HuggingFace and provision it onto the model host, so it *becomes*
one of the choices `list_models` offers and `set_active_model` can switch to.
These are the companion to the Proxmox tools above and ride the same SSH key and
the same node — there is no second credential and no second host.

| Tool | Type | Description |
|------|------|-------------|
| `hf_search` | Read | Search HuggingFace for repos that publish gguf, most-downloaded first: repo id, downloads, likes, gated flag, tags. A hit means the *repo* is tagged as containing gguf, not that any particular file exists — follow up with `hf_files`. Results are third-party text, so this tool is flagged as producing untrusted content and taints the turn. |
| `hf_files` | Read | One repo's gguf files with the exact **byte size** and **sha256** the Hub publishes for each. This is what you size a quant against (compare with `gpu_status`) and where the exact filename comes from. A file whose `sha256` is `null` — a non-LFS file, with no published digest — cannot be installed at all. |
| `install_model` | **Write (parked)** | Download one gguf onto the model host and write a koboldcpp systemd unit for it. **The unit lands disabled and is not started.** Takes `repo`, `file`, `name`, and an optional `contextsize`. Returns a `job_id` immediately; the download continues on the node. |
| `install_status` | Read | Poll one install job: `state` (running / done / failed), current step, bytes downloaded, and on failure a short fixed-vocabulary reason. Also reports `n_layer` / `n_kv_head` / `head_dim` read out of the downloaded gguf and, on a finished job, a `note` that **evaluates** the KV budget from them — bytes per token, the cache size at the installed `contextsize`, and what the unit wants in total once the model buffer, ~1010 MiB compute buffer and ~500 MiB margin are added (DP-337). A gguf whose header omits the three numbers says so instead, and asks for the budget to be measured against `gpu_status` rather than estimated. |

#### What the approval card shows

`install_model` parks like every other write, and the card carries **the repo,
the file, the byte size, and the sha256 that derpr read from HuggingFace
itself** — not values the model typed. You are approving specific bytes. Those
same two numbers are what the node then enforces: it refuses if the Hub's digest
does not match, and it deletes the partial file and fails the job if the
downloaded bytes do not hash to it.

The Hub is re-read at execution time (after you approve), so if a repo replaces
the file between park and approval the digest actually enforced can differ from
the one on the card. The result reports the digest it enforced, so that shows up
in the tool result rather than silently.

#### Two approvals, not one

Installing a model **does not** put it on `:5001` and does not disturb whatever
is running. The unit is written `disabled`, `daemon-reload`ed, and left alone.
Making it active is a separate `set_active_model` call with its own approval —
and worth doing only after checking the `contextsize`, because `install_model`
writes a deliberately small default (8192) on the assumption a human will tune it
against `gpu_status` first.

#### What the node refuses

The whole feature runs **one** node-side script
(`/usr/local/sbin/derpr-model-install`, versioned in `services/pve/`), reachable
through exactly one entry in the SSH key's forced-command allowlist. derpr never
gets `curl`, `systemctl`, or file writes as separate verbs. Before any bytes
move, the node refuses:

- a `name` whose `koboldcpp-<name>.service` already exists on the container —
  overwriting one silently repoints a name `list_models` already publishes;
- a destination file that already exists with a **different** sha256 (an
  identical one is reused, so a retry costs nothing);
- insufficient free space — the larger of 2 GiB or 5% of the download is kept
  free. The models volume is a thin LV, and filling it takes `:5001` and every
  other guest's models with it, so this refuses rather than truncating.

And after downloading, a sha256 mismatch deletes the partial file and fails the
job. Size matching is not proof.

#### Long downloads

A multi-GB download vastly outlives any tool timeout, so the node runs it
detached under `systemd-run` and derpr polls. `install_model` returns as soon as
the job starts; ask for progress with `install_status` rather than waiting.
There is no background loop inside derpr — the node supervises its own job.

Disabled by default. Enable with `HF_TOOLS_ENABLED=true` **and** deploy the
node-side artifacts (`services/pve/README.md` has the steps, including the
forced-command allowlist entry). Config knobs: `HF_API_BASE`, `HF_API_TOKEN`
(only needed for gated repos), `HF_HTTP_TIMEOUT`, `HF_SEARCH_LIMIT_MAX`. The
transport settings are the proxmox ones (`PVE_SSH_*`). When disabled, every tool
returns a clear "disabled" error instead of reaching the Hub or the node.

⚠️ **The persona holding this binding reads attacker-authored text and can write
to the node.** That composition is checked, not assumed: the Hub reads and
`install_model` share one egress domain (`huggingface`), which makes it a
same-origin closed loop under the insecure-composition rules — so the protection
stays *armed* for anything added later, rather than being switched off with an
explicit override. Adding a tool with a different egress domain to that persona
will quarantine it, which is the intended behaviour.

### MCP Servers (requires `service_bindings: ["mcp"]` to manage; `["mcp:<server>"]` to use)

Plug external tool servers into the bot via the
[Model Context Protocol](https://modelcontextprotocol.io) (streamable-HTTP
transport). Discovered tools register into the normal tool system — they park,
taint, and composition-validate exactly like native tools — under the name
`mcp__<server>__<tool>`.

**Setup is agent-driven — no config-file wiring before launch.** Ask a persona
that has the management tools to add a server; the add parks for your approval,
then connects, discovers, and registers the server's tools live (no restart).
The config file (`data/mcp_servers.json`) persists what the tool writes.

| Tool | Type | Description |
|------|------|-------------|
| `add_mcp_server` | **Write (parked)** | Connect a new MCP server by `name` + `url`, discover its tools, register them live, persist the config. The approval banner is your chance to reject an unexpected capability install. |
| `remove_mcp_server` | **Write (parked)** | Disconnect a server, unregister all of its tools, delete it from the config. |
| `list_mcp_servers` | Read | Configured servers with connection status and their registered tools. |

Security model for discovered tools:

- **Restrictive defaults.** Every discovered tool starts as `is_write: True`
  (parks for approval) with `produces_untrusted/irreversible` set and
  `sensitivity: "pii"`. The server's own `readOnlyHint`/`destructiveHint`
  annotations are logged but **never** drive policy — an untrusted party
  doesn't get to classify its own tools. To relax a tool (e.g. a genuinely
  read-only sensor query), edit its `tool_overrides` entry in
  `data/mcp_servers.json` and re-add/restart.
- **Never in the wildcard.** `enabled_tools: ["*"]` / `allow: ["*"]` policies
  do NOT include MCP tools. A persona must list each `mcp__<server>__<tool>`
  explicitly in `allow`/`ask` **and** bind `mcp:<server>` in
  `service_bindings`. This means installing a server can never silently widen
  (or quarantine) an unrelated persona.
- **Per-server egress domain.** Each server is its own domain under the
  composition rules: reading and writing the same server is a closed loop;
  combining a server's tools with foreign-domain tools (web search, zammad, a
  second MCP server) re-arms the exfiltration rules.
- Tool descriptions are server-authored text that enters the system prompt
  (a prompt-injection surface); they are length-capped at discovery.

Disabled by default. Enable with `MCP_ENABLED=true`. Config knobs:
`MCP_SERVERS_FILE`, `MCP_CONNECT_TIMEOUT`, `MCP_CALL_TIMEOUT`,
`MCP_RECONNECT_INTERVAL`. When disabled, the management tools return a clear
"disabled" error and no configured server is contacted.

**Hot reload.** A background maintenance pass runs every
`MCP_RECONNECT_INTERVAL` seconds (default 60; `<= 0` disables it):

- A server that is down — at startup or after dying at runtime — is retried
  automatically. While it is down its tools stay registered and return
  per-call errors, so persona configurations don't churn during an outage;
  the tools go live again on reconnect. A dead server never blocks startup
  or other servers.
- When a server announces `tools/list_changed`, its toolset is re-discovered
  immediately (the notification wakes the pass; the interval is only the
  fallback for servers that never signal). New tools appear with the usual
  restrictive defaults, removed tools are unregistered, and every persona is
  re-validated against the new toolset.

### MCP Bridge — derpr tools for dispatched subagents (DP-240)

The mirror image of the section above: instead of derpr *consuming* someone
else's MCP server, derpr *hosts* one, so a dispatched fixr subagent can call
derpr's own tools. This exists so a subagent can investigate problems that code
alone can't explain (its first real use is live-container investigation, landing
in DP-241).

**Off by default.** A normal `dispatch_fix` is completely unchanged — it never
sees the bridge and its `claude` arguments are identical to before this feature
existed. The capable tier is opt-in per dispatch and expected to be rare.

Setup:

| Setting | Meaning |
|---|---|
| `MCP_BRIDGE_ENABLED` | Master switch. Off by default. |
| `MCP_BRIDGE_TOOLS` | Comma-separated tools exposed over the bridge. **Default-deny** — an empty list exposes nothing, so turning the bridge on without choosing tools does nothing. |
| `MCP_BRIDGE_PUBLIC_URL` | The address the *subagent* uses to reach the bridge. Its host is added to the subagent's sandbox allow-list automatically. |
| `MCP_BRIDGE_PATH` | Path the bridge mounts at on the portal app (default `/mcp`). |

How a subagent's tool call is handled:

- **Read-only tool** → runs immediately and returns its result, like any tool call.
- **Write or irreversible tool** → **does not run.** It is queued as a proposal
  for your review and the subagent is told to stop and wait. You approve or deny
  it exactly like any other proposal — `list_proposals`, `approve_proposal`,
  `deny_proposal` — from Discord, the portal, or anywhere else a persona with
  the `proposals` binding can talk to you. On approval the tool runs and the
  agent is resumed with the result.

Two things worth knowing about the approval:

- **Approval is re-checked at execution time, not at queueing time.** If you
  narrow `MCP_BRIDGE_TOOLS` (or the tool stops being registered) while a request
  is sitting in the queue, approving it will refuse rather than run. A queued
  request can never outlive the permission that allowed it.
- **The queue is the only thing standing between a subagent and the tool.**
  Capable dispatches run with Claude Code's own permission prompts disabled, by
  design — the subagent is headless and cannot answer them. Treat approving a
  subagent's request with the same care as running the tool yourself.

Each capable dispatch gets its own bearer token, revoked when the agent
finishes, is killed, or errors. A request with no valid token is rejected before
it reaches any tool.

> **Requires the proposal queue.** The review tools live behind the `proposals`
> binding. It registers whenever *either* backend is present — Zammad (for the
> ticket actions) or the MCP bridge (for subagent tool calls) — so a
> bridge-only deployment still gets `list_proposals` / `approve_proposal` /
> `deny_proposal`. Approving an action whose backend is missing fails with a
> readable reason rather than reporting success.

### Memory Tools (no service binding required)

Available to any persona with `enabled_tools: ["*"]` (e.g., `joy`, `it-help`). These tools interact with the long-term memory store built by SqliteConsolidator.

| Tool | Type | Description |
|------|------|-------------|
| `recall_memory` | Read | Search the persona's long-term memory bank for facts relevant to a natural-language query. Returns up to `limit` (default 10) hits — short summaries of past conversations or observations. Scope is inherited from the active turn (persona, channel, user, server); the LLM cannot redirect recall to another persona. Marked `produces_untrusted=True` so retrieved hits taint the turn under the tool-security framework. |
| `drill_down_memory` | Read | Fetch raw episodic memories under a specific Core Profile. Use to recover specific details (dates, links, verbatim quotes) that were abstracted away during consolidation. Requires `parent_summary_id`. |
| `update_core_memory` | Write | Modify an existing Core Profile when new information contradicts or extends it. Requires `summary_id` and the revised content. |

**Internal tools** (used by agents/system personas, not by user personas):

| Tool | Used by | Description |
|------|---------|-------------|
| `submit_memory_summary` | memory_summarizer (SqliteConsolidator) | Records extracted observations and keywords from a conversation segment; identifies thematic outliers for re-queueing |
| `submit_core_profile` | Consolidator | Merges clustered episodic summaries into a structured core profile with nested concepts |

## Agents

Agents are autonomous background workers that run on a schedule without user interaction. They are configured in `config/agents.json`.

### Current Agents

**SqliteConsolidator (memory)** (`auto_start: true`) — Runs every 15 minutes (only when `SEMANTIC_BACKEND=sqlite`). Segments recent conversations by topic, extracts observations via LLM, and stores embedded summaries for long-term recall. See [Long-term Memory](#long-term-memory) below for the full pipeline description. Config in `agents.json` under `"sqlite_consolidator"`.

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

**ReminderAgent** (`auto_start: true`) — Runs daily at a configured time (`daily_at`). Polls Zammad for open tickets that haven't been updated and posts a summary nudge to the configured Discord channel.

### Managr (DP-280 Phase 0 + DP-282 Phase 1 shipped; Phases 2-3 planned)

Managr is a top-level planning agent for the whole ticket board. Where triage and dispatch each react to a single new ticket, managr periodically reviews *everything* — open tickets, their ages and priorities, what the other agents have done, and what happened to its own past suggestions — and produces a manager's plan. It is deliberately neutered: **managr can never write to Zammad or any other external system.** It only observes, reports, and proposes; a human approves every action before anything executes.

**Managr is not a chattable persona.** "Managr" is the scheduled agent; the `managr_planner` / `managr_*_analyst` entries in `system_personas.json` are internal, tool-less system personas the agent invokes once per cycle — they hold no conversation state, and talking to them would not affect the next cycle. All operator interaction (reviewing proposals, adding/retiring standing orders) happens through a *conversational* persona that has `service_bindings: ["proposals"]` with the proposal tools enabled — intended to be joy, but any persona granted the binding works, including a dedicated user persona named e.g. `managr` if you prefer a "speak to the manager" front desk. Feedback loops back to the agent through data: standing orders and proposal-denial reasons are injected into its next planning cycle.

**Cycle** (`auto_start: true` — one cycle at startup, then daily at `daily_at`, like ReminderAgent):

1. **Observe** — snapshot the board: open tickets with age/state/priority/tags, staleness, recent triage/dispatch/reminder activity, and the outcomes of managr's previous proposals (approved / denied / expired). Quarantined tickets (see below) are flagged in code and their titles withheld. *Article content lands with DP-288 Phase 2.*
2. **Orient** — fan out read-only analysis briefs to specialized system personas (e.g. a stale-ticket investigator, a per-client summarizer, a cross-ticket pattern detector). Each returns a short structured brief.
3. **Decide** — a single planning call over the briefs produces the plan: an assessment of board health, priorities for the day, and a list of proposed actions.
4. **Report & propose** — the plan is posted as a readable digest (Discord channel and/or Zammad internal note). Each proposed action is written to a durable **proposal queue** for human review; nothing executes on its own.

**Proposals.** A proposal is a schema-validated action drawn from a fixed whitelist. The Phase 1 whitelist is internal-only and low-blast: `add_note` (always an internal article, never customer-visible), `set_priority`, and `remind` (park as pending-reminder until a date); richer actions (`draft_reply`, `merge_tickets`, `escalate_to_human`) come with later phases. Free-text intent never becomes a proposal — the planner emits candidates through a structured `submit_proposals` schema, and each one is validated in code against the whitelist before a row is written (invalid actions are dropped and logged, never stored). Each proposal records the proposing agent, the action and its arguments, the rationale, and taint provenance (which ticket content motivated it), and expires unreviewed after 7 days (`MANAGR_PROPOSAL_TTL_DAYS`). At most 10 proposals are queued per cycle (`MANAGR_MAX_PROPOSALS_PER_CYCLE`). Proposal emission is gated by `proposals_enabled` in managr's `agents.json` entry (absent = off, Phase 0 behavior).

**Self-managing queue (DP-290).** The proposal queue maintains itself across cycles instead of accumulating duplicates when a cycle re-runs (e.g. after a container restart):

- *Deduplication is guaranteed in storage, not by model behavior.* The queue holds at most one pending proposal per (agent, action type, ticket). Submitting a proposal whose key matches an existing pending row **supersedes** it: the row keeps its id and creation time but takes the new arguments, rationale, and a fresh expiry. A restart that replays the planning cycle therefore refreshes the queue instead of doubling it.
- *Reflective dispositions.* Each planning cycle, managr is shown its own still-pending proposals (up to `MANAGR_PENDING_PROPOSAL_LIMIT`, injected into the proposal-extraction call — no extra LLM call) and must disposition each one: **reaffirm** (still stands — expiry resets), **revise** (update the arguments/rationale — revised arguments are re-validated against the action whitelist exactly like new proposals), or **withdraw** (no longer needed — the row moves to a terminal `withdrawn` status, distinct from operator denial so denial-learning stays clean). Proposals the model fails to mention are left untouched — a flaky extraction call can never destroy a pending proposal.
- *TTL is garbage collection, not a decision deadline.* A reaffirmed proposal's 7-day clock (`MANAGR_PROPOSAL_TTL_DAYS`) resets every cycle; expiry only collects abandoned rows managr has stopped reaffirming (typically because the ticket resolved and left the board).

The digest's proposals section reports queue maintenance alongside new proposals (reaffirmed / revised / withdrawn counts and ids), so the operator sees the queue state, not just this cycle's additions.

**Approval.** Proposals are reviewed through any persona with `service_bindings: ["proposals"]` (intended: joy): "list proposals", "approve proposal 12", "deny proposal 13 because …" (`list_proposals` / `approve_proposal` / `deny_proposal`). `approve_proposal` is itself a write tool, so it flows through the existing universal write-audit gate — all execution funnels through the one existing approval surface. Approving executes the action immediately via the proposal executor (never managr) and records the result on the proposal; denying requires a reason, which is stored as structured feedback. Every approve/deny/execute lands in the `Audit_Log`.

**Standing orders (DP-281).** Behavioral tuning of managr happens through data, not prompt surgery: the planner persona's prompt stays fixed (format and rules only), and operator guidance lives in a durable **standing-orders store** injected into every planning cycle. Tell joy things like "add a standing order: client Y tickets are always low priority" or "standing order: stop flagging ticket #123" — joy records them via `add_standing_order` (a write tool, so it parks for confirmation like any other write). `list_standing_orders` shows the active set; `retire_standing_order` (also a write) retires one with an optional note — orders are never deleted, only retired, so the history stays auditable. The tools ride the same `proposals` service binding as the review queue: granting joy that binding enables both surfaces.

Every planning cycle, the newest active orders (up to `MANAGR_STANDING_ORDERS_LIMIT`, newest first) are fetched once and injected as a `STANDING ORDERS` section into both the planning call and the proposal-extraction call — deterministic context, not recall-dependent, and both calls are always constrained by the same order set. When orders are present, the planner is instructed to note in the report which orders changed its plan since the last cycle, so you can verify a correction took. Adds/retires land in the `Audit_Log` (`standing_order_added` / `standing_order_retired`). If the store cannot be read, the cycle still runs, but the report opens with an explicit warning that it was produced without operator guidance — a degraded cycle is never silent.

**Trust boundary:** standing orders enter only through the operator's authenticated surfaces (joy's gated write tools) — never derived from ticket content, digest text, or any other model output; otherwise the store would be a prompt-injection lane straight into the planner. This is enforced in depth: the store itself rejects writes from any non-operator source, and the planner additionally refuses to inject any row whose recorded source is not on the operator allowlist. Denied-proposal reasons stay on the proposal rows (injected separately as proposal outcomes) — they inform the operator's standing orders but never become orders automatically.

**Graduated autonomy.** Per-action-type acceptance rates are tracked. When a low-blast action type (e.g. `add_note`) sustains near-100% unmodified approval, it can be explicitly promoted to auto-execute in config — autonomy is earned per action type, with data, never assumed. **Customer-facing actions (outbound replies, emails) are the last tier and remain human-approved indefinitely**; internal-only actions (notes, priorities, reminders) graduate first.

**Hard boundaries (all phases):**
- Managr has no write access to Zammad, agent configs, personas, or tool assignments — proposing is its only output channel.
- Proposal arguments are re-validated against the action schema at execution time, not just when the LLM emits them.
- Ticket content is treated as adversarial input; plans and proposals derived from it carry taint provenance into the audit record.

**Phases:** (0, shipped) read-only manager's report only — no proposal infrastructure; (1, shipped) proposal queue + approval via joy/Discord for internal low-blast actions; (2) acceptance tracking + config-gated auto-execution of proven internal action types; (3) managr can commission deep-dive focus subagents (fixr-pattern) for investigations — research an error across ticket history, draft a KB article, prepare a client-facing summary (which still lands as a proposal).

### Phishing & adversarial content

Tickets often *contain* hostile text — most often a user forwarding a phishing mail they received, with their "this is phishing" note on top and the forwarded lure below. That lure is written to read like a legitimate request, so an automated agent that reads the raw body could be steered by it.

derpr detects reported and suspected phishing automatically and **quarantines** it. Once a ticket is quarantined:

- managr and the other agents are shown a flagged placeholder (`[CONTENT QUARANTINED]`) instead of the content — they know the ticket exists and needs security handling, but never see the bait, so a forwarded lure can't influence the plan.
- The ticket carries a `security-report` (reported phishing) or `phishing-suspect` (suspected) tag plus an internal note. You can rely on those tags in standing orders.
- A human still approves every action, so a misclassification at worst *mislabels* a ticket — it never acts on its own.

Detection runs during triage, before any agent reads the body. With quarantine in place, managr now plans from richer ticket content: alongside the title-lines, the stalest open tickets get a detail tier — their first and last two articles, each clipped — so the planner reasons over the actual request and its current state. Quarantined tickets are excluded from the detail tier entirely, so enriching the snapshot never reopens the bait-exposure path. Design and configuration detail live in the DP-288 task notes.

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

2. **Segmentation** (SqliteConsolidator, every 15 min) — Unprocessed embedded messages are grouped into topically coherent segments using centroid-based cosine similarity. Q/A pairs (a user message immediately followed by an assistant reply) are never split across segments. Minimum segment size is configurable (default: 2 messages).

3. **Summarization** — Each segment is sent to the `memory_summarizer` system persona, which extracts discrete observations (facts, preferences, decisions, solutions) and thematic keywords via the `submit_memory_summary` tool. Messages that don't fit the segment's theme are flagged as outliers and re-queued for the next batch.

4. **Consolidation** — Periodically, similar episodic summaries (level 1) are clustered by similarity and merged into core profiles (level 2) via `submit_core_profile`. This creates a two-tier hierarchy: detailed episodic records and compressed concept profiles.

5. **Retrieval** — On each LLM request, relevant summaries are retrieved via KNN vector search and injected into the context window *before* the sliding-window history. This gives the LLM access to facts from older conversations that would otherwise have fallen out of the context limit.

### Scope

Long-term memory retrieval is filtered by channel, persona, and embedding model. Memory built in one channel is not surfaced in another (same scoping rules as `CHANNEL_ISOLATED` history). Currently only channels listed under `allowed_channels` in `agents.json` are processed by SqliteConsolidator.

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

`mark_trusted` / `mark_untrusted` flip the `untrusted` bit on a specific recall hit (per the [tool security framework](../memory/project/plans/tool_security_framework.md)). Overrides live in a parallel SQLite file (`data/hindsight_overrides.db`, `HINDSIGHT_OVERRIDE_DB`) — recall post-filters and rewrites the bit. Every flip is audit-logged with operator_id, reason, prior, and new values.

## System Defaults

| Setting | Value | Constant |
|---------|-------|----------|
| Default model | `gemini-3.1-flash-lite` | `DEFAULT_MODEL_NAME` |
| Default agent model | `agy-flash` | `DEFAULT_AGENT_MODEL` |
| Default context limit | 15 messages | `DEFAULT_HISTORY_MESSAGES` |
| Context hard cap | 30 messages | `GLOBAL_HISTORY_MESSAGES` |
| Max tool calls per request | 15 | `MAX_TOOL_CALLS` — tool calls **executed**, not LLM round trips. DP-297 raised it 5 → 10 (a parked write costs a step instead of ending the turn); DP-335 moved the counter off iterations, so the number now means the same thing whichever model answers, and sized it for the longest routine any persona runs |
| Max LLM round trips per request | 25 | `MAX_TOOL_ITERATIONS` — runaway guard only. A turn normally ends by spending `MAX_TOOL_CALLS`; this catches a loop that talks to the provider without calling anything |
| Max response tokens | 4096 | `DEFAULT_TOKEN_LIMIT` |
| Default total context budget | 131072 tokens | `DEFAULT_MAX_CONTEXT_TOKENS` |
| Proposal approval window | 24 hours | `PENDING_ACTION_TTL` — DP-297 renamed and raised this from `PENDING_CONFIRMATION_TIMEOUT` (300s), since a park became a queue worked through later rather than a blocking modal. DP-319 made the store durable, so this is now the real deadline rather than min(this, uptime) |
| Decided-proposal retention | 7 days | `PARK_ROW_RETENTION` — how long a resolved park's row is kept so a re-proposal can be recognized |
| Re-execution guard window | 15 minutes | `PARK_REEXECUTION_GUARD_WINDOW` |
