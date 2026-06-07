# Handoff: DERPR Portal — bespoke web UI (Kobold-Lite replacement)

## Overview

DERPR is an async, provider-agnostic LLM orchestration engine. Its current web UI
is a **hacked-together PoC built on Kobold-Lite**, which renders the entire
conversation as one giant text blob and forwards it to the model. That model
fights the engine's real data model — the engine stores **one DB row per message**
(`User_Interactions`), and the single textbox bleeds those rows together, which
has been a recurring source of divergence bugs between the UI and the DB.

This handoff is for building a **bespoke replacement UI** whose entire premise is
that **every message is a discrete, individually-addressable object**, matching
the DB one-to-one. The engine team already built the server-side contract for
this (the **DP-130 transcript contract** + the engine adapter that makes the DB
the single source of truth). The job here is the front end that consumes it.

The replacement is an **internal developer tool**. Core functionality and
**DB↔UI parity** are paramount; visual polish is secondary. The single most
important capability is being able to **see the exact raw request** the engine
will send to the model, and to **trust that what the UI shows cannot diverge**
from what `chat_system.stream_response` actually assembles internally.

> **Scope note:** This is *not* a request to ship the bundled HTML. See
> "About the Design Files." The HTML is a faithful, contract-driven **reference
> prototype**; the task is to recreate it in the DERPR repo's chosen front-end
> environment.

---

## About the Design Files

The files in `design/` are **design references created in HTML** — a working
prototype demonstrating intended layout, behavior, data shapes, and states. They
are **not production code to copy verbatim.**

The DERPR repo currently has **no established front-end framework** (the old UI
was Kobold-Lite's vendored `portal.html` served by `portal_render.py`). So the
implementer should **choose the most appropriate front-end stack** for an
internal, dense, real-time dashboard tool and build the design there. The
prototype is deliberately written in dependency-free vanilla JS so the data flow
and DOM structure are easy to read and port; **do not** treat "vanilla JS" as a
stack recommendation.

Recommended stack characteristics (implementer's call):
- A component model with first-class lists keyed by stable id (chunks are keyed
  by `interaction_id` / `ephemeral_chunk_id`, **never by array position** — this
  is the whole point of the contract).
- Native SSE / `EventSource` (or fetch-stream) handling for the chat stream.
- Lightweight; this runs locally for a handful of internal users.

The prototype's **`design/portal-data.js` is the canonical reference for every
data shape** — its objects mirror the real endpoint payloads exactly and are
annotated with the endpoint each comes from. Read it alongside `API_CONTRACTS.md`.

---

## Fidelity

**Mixed, leaning hi-fi for structure & behavior, lo-fi for final visual polish.**

- **Layout, information architecture, component anatomy, states, and interaction
  behavior are hi-fi** — recreate them faithfully. These encode real product
  decisions (the row-as-object model, base-vs-kobold param split, parity
  inspector, CONFIRM flow, response-type treatments).
- **Exact pixel styling is a starting point, not a mandate.** Colors, spacing,
  and type in `design/portal.css` are a coherent dark IDE theme and are good to
  adopt as-is, but the implementer may align them to any house style. The
  **semantic color coding must be preserved** (see Design Tokens): teal =
  engine/interactive, violet = memory/LTM, amber = write/mutation/kobold-only,
  green = read/approve, red = danger/error.

---

## The core mental model (read this first)

1. **A chunk == a DB row == a UI message object.** The conversation is rendered
   from `GET /api/v1/session/{persona}/transcript`, which returns
   `{ "chunks": [...] }`. Each chunk is addressed by its `interaction_id`
   (or `ephemeral_chunk_id` for a not-yet-persisted parked confirmation). The UI
   **must address chunks by identity, never by position** — this is what prevents
   the drift that plagued the Kobold UI.

2. **The engine is the source of truth, not the client.** The chat endpoint
   (`/v1/chat/completions`) **discards any client-sent message array** and
   rebuilds history from the DB. The client only sends the new user turn
   (`derpr_user_text`) plus flags. So the UI never "owns" the prompt — it
   reflects what the engine assembles.

3. **System prompt and LTM are NOT transcript chunks.** `build_transcript()`
   skips system rows and tool-only rows. The persona system prompt comes from
   `GET /persona/{name}.prompt`; the recalled memory block comes from
   `GET /session/{persona}/ltm_block`. The UI **stitches** these in for display.
   This matters: editing the system prompt is a `PATCH /persona`, not a chunk edit.

4. **Two render modes of the same data:**
   - **RENDERED** — human-readable conversation (system prompt pinned at top, LTM
     shown inline as an injected row, then the chunks).
   - **CONTEXT ↦ LLM / Raw req** — the *assembled request* as it goes to the
     model. This must be sourced from the engine's own assembler (see the
     **proposed `/assemble` dry-run endpoint** in `API_CONTRACTS.md`), so it is
     **true by construction**, not reconstructed in the browser.

5. **Writes can park for approval (CONFIRM mode).** When a persona's
   `tool_policy.mode == "CONFIRM"`, a write tool call **parks** — it surfaces as a
   trailing **ephemeral chunk** (`interaction_id: null`, `ephemeral: true`) and, on
   the live stream, a `derpr-confirm` frame carrying the structured pending calls +
   a resume token. It is **resolved by a dedicated endpoint**,
   `POST /api/v1/persona/{name}/confirm` (`{approved, token}`), which streams the
   continuation back over SSE. The approve/deny buttons POST there directly — they
   do **not** compose a free-text chat turn. (See `API_CONTRACTS.md` §7 — this
   supersedes the earlier Discord-style "no endpoint" description.)

---

## Screens / Views

There is **one primary screen** (the Chat workspace, "Control Room" layout) plus
six **reserved expansion destinations** that are out of scope now but must have a
permanent home in the navigation (do not build them; just leave the rail slots).

### Primary screen: Chat workspace ("Control Room")

A four-region IDE-style layout, full viewport, dark. Left→right:

```
┌────────────────────────────────────────────────────────────────────────┐
│ TOP BAR: brand · persona picker · status chips · 3 collapse toggles       │
├──────┬───────────────┬─────────────────────────────┬─────────────────────┤
│ NAV  │  CHANNELS      │  CONVERSATION               │  INSPECTOR          │
│ RAIL │  (collapsible) │  (always present)           │  (collapsible)      │
│ 60px │  250px         │  flexible                   │  360px              │
│      │                │                             │                     │
│ icon │  grouped by    │  header: persona, model,    │  tabs:              │
│ docks│  source tag    │  memory-mode, tool-policy,  │   Persona | Tools | │
│      │  (web/discord/ │  RENDERED⇄CONTEXT toggle    │   Raw req           │
│      │  zammad/gmail) │  budget bar                 │                     │
│      │                │  transcript (chunks)        │                     │
│      │  + new channel │  composer                   │                     │
└──────┴───────────────┴─────────────────────────────┴─────────────────────┘
```

The three side regions (nav rail, channels, inspector) each **collapse via a
toggle button in the top bar**, animating the grid columns to zero width. The
conversation column is never collapsible.

**Layout specifics** (from `design/portal.css`):
- Body grid: `grid-template-columns: 60px 250px 1fr 360px`. Collapsed states zero
  out the respective column; transition `0.22s cubic-bezier(.4,0,.2,1)`.
- Everything `overflow: hidden` at the column level; only the channel list, the
  transcript scroll region, and the inspector body scroll internally.
- At `max-width: 1100px`, inspector → 300px, channels → 210px.

#### Region: Top bar (height 50px)
- Brand lockup: `DERPR` (700, letter-spacing 2.5px) + `// PORTAL · ENGINE` (faint).
- **Persona picker** — pill button: 22px avatar (persona initials), persona name,
  caret. Opens persona switcher (drives `PUT /api/v1/model`).
- Status chips (right): connection chip (`kcpp · omen:5001`, green dot) and route
  chip (`engine route · /v1/chat/completions`, teal).
- **Collapse toggles** — three 30px buttons: rail (`⊟`), channels (`◧`),
  inspector (`◨`). `aria-pressed` reflects visible state; active = teal.

#### Region: Nav rail (60px)
Vertical icon docks, each 46×48px with icon + 7.5px caps label:
- **CHAT** (active) — the only built destination.
- **MEMORY**, **AGENTS**, **BUDGET**, **STATS**, **PERSONA** — reserved expansion
  slots. Render them disabled/dimmed with a small "soon" dot (amber) and a
  tooltip describing the future area. **Do not build these.**
- **CFG** (settings) pinned to the bottom.
- Hover tooltip slides out to the right of each item.

#### Region: Channels (250px, collapsible)
- Header `Channels` + a `+` icon button (new channel).
- A filter/search field (non-functional placeholder is fine for S1).
- List **grouped by the channel's source tag**. `channel` is a
  **source-agnostic string** on the engine side; the UI groups by source prefix
  (`web_ui`, `discord`, `zammad`, `gmail`). Each item: 28px avatar, name,
  a small colored source badge (`web`/`dsc`/`zmd`/`gml`), and a one-line preview.
  Active item: teal left-border + tint.
- Footer/last item: "+ new web_ui channel" dashed button.
- **Note:** current engine endpoints hard-code `channel="web_ui"` and
  `user_identifier="portal"`. True multi-channel + channel creation requires a
  small API change — see `API_CONTRACTS.md` § "Channel scoping (proposed)". For
  S1, a single `web_ui` channel is acceptable; the grouped list is the target.

#### Region: Conversation (flexible)
- **Header (52px):** persona avatar + title (`assistant · scratch`), chips for
  `model_name`, `memory_mode` (violet), `tool_policy.mode`, and the
  **RENDERED ⇄ CONTEXT ↦ LLM segmented toggle** (right-aligned).
- **Budget bar:** a stacked horizontal bar of the context budget (system / LTM /
  history / reply-reserve) with a legend and `used / max ctx` total. Hidden in
  CONTEXT view (where per-row token counts are shown instead). The token numbers
  should come from `POST /api/extra/tokencount` and/or the `/assemble` payload —
  do **not** estimate client-side in production (the prototype estimates for
  demo only).
- **Transcript** (scrolling): see "Message row anatomy" and "Response-type
  treatments" below.
- **Composer:** an `LTM recall` toggle chip (violet when on; drives whether
  `ltm_block` is fetched/injected), a `/ dev-command` hint, keyboard hints, a
  growing textarea, and a SEND button. `Enter` sends, `Shift+Enter` newline.
  A leading `/` routes to `POST /persona/{name}/dev_command` instead of chat.

#### Region: Inspector (360px, collapsible) — three tabs
- **Persona** — read/edit the persona. **Critical split:**
  - **`▣ base params`** section ("sent to every provider"): `model_name`,
    `memory_mode`, `temperature` (slider), `max_tokens`, `history_messages`,
    `max_context_tokens`, `thinking_level`, `chat_template`, `tool_policy.mode`.
  - **`⚠ kobold-only`** section ("passthrough route · `provider_extra`"),
    **collapsible**: `top_p`, `top_k`, `rep_pen`, `rep_pen_range`, `min_p`, `tfs`,
    `mirostat`/`tau`/`eta`, `instruct_tags`, `sampler_order`.
  - **Distinguish base vs kobold-only by which params have a devoted persona
    property** vs. which live in `kobold_extras` / `provider_extra("kobold", …)`.
    See `API_CONTRACTS.md` for the exact field origins.
  - A **security-blocked banner** shows when `persona.security_blocked` is true,
    listing `security_block_reasons`.
  - Saves go to `PATCH /api/v1/persona/{name}`, which returns `rejected_fields`
    and `unknown_fields` — **surface rejections** (don't pretend a save was clean).
- **Tools** — list from `GET /api/v1/tools/catalog`. Each tool: name, description,
  capability badges (write/read, sensitivity high/med/low, locality local/remote,
  produces_untrusted), and an enable toggle reflecting `persona.enabled_tools`.
- **Raw req** — the **parity inspector** (the priority feature). See its own
  section below.

---

## Message row anatomy (hi-fi — recreate faithfully)

Every assistant/user chunk renders as a row: `[avatar] [body]`. The body contains,
in order, only the parts present on that chunk:

1. **Meta line:** author label (`portal` for user, `assistant` for assistant),
   timestamp, **id tag** (`#<interaction_id>`, or `ephemeral · <chunk_id>` for
   parked), and — **only when `has_versions` is true** — **version chevrons**
   `‹ k/n ›`. Do **not** show chevrons on chunks without versions.
2. **Reasoning fold** — if `chunk.reasoning` is present (the server also folds it
   into `content` as `<think>…</think>`; split it back out): a collapsible block
   labeled `⟁ reasoning` with a token count, collapsed by default.
3. **Tool call card(s)** — for each entry in `chunk.tool_context`: a collapsible
   card showing `→ toolname(args)` with capability badges, `call_id`/`group_id`,
   full args, and the result (or `awaiting approval — tool not yet run` when the
   result is null on a parked write). Tool cards are **embedded within the
   assistant turn** (explicit product decision — they are not separate rows).
4. **Body text** — the chunk's prose content (with `<think>` stripped).
5. **Row hover actions** — assistant: `⟲ regen`, `✎ edit`, `✕ del`; user:
   `✎ edit`, `✕ del`. (Wired in S3.)

**Editing model:** `✎ edit` → `PATCH /interaction/{id}` with new `content`.
`✕ del` → `DELETE /interaction/{id}` (soft-suppress, idempotent; reply chains
preserved; suppressed rows are filtered server-side). `⟲ regen` → a retry chat
turn (`derpr_retry: true`) that archives the prior assistant row and creates a new
version; chevrons then appear (`has_versions`). Version navigation:
`GET /interaction/{id}/versions` + `POST /interaction/{id}/select_version/{k}`.

---

## Response-type treatments (see `design/Response Types.html`)

`DoneEvent.response_type` distinguishes terminal-turn kinds. Each must read
differently so an aborted/errored turn never looks like a clean reply. Map these
descriptive names to the real `ResponseType` enum:

| Treatment | Trigger | Rendering |
|---|---|---|
| **normal** | prose reply | standard assistant row, `assistant_id` set |
| **tool-only** | tools ran, empty content | assistant row with only tool cards + a "tool-only turn" chip |
| **parked** | CONFIRM write awaiting approval | ephemeral row (amber left-border), pending tool card, approve/deny bar (POSTs to `/confirm` with the resume token), `assistant_id: null` + `ephemeral_chunk_id` |
| **aborted** | user stopped generation | row with partial text + a dashed "aborted · partial flushed · regen to continue" marker |
| **error** | engine/provider failure | red row, no assistant row committed, dismiss/retry actions |
| **security-blocked** | `persona.security_blocked` | red system row, submit refused before assembly, "review block reasons" action |
| **dev-command** | `/` prefixed input routed to `dev_command` endpoint | thin collapsible dev row (see below) — `--ink-faint` left-border, collapsed by default, ephemeral (not persisted) |

---

## Dev message rows (client-side ephemeral)

Dev commands (`/set temp 0.4`, `/detail`, etc.) short-circuit in the engine's
`preprocess_message` and are **never persisted to the DB**. They are not part of
the DP-130 transcript contract and do not appear in `GET /transcript` or the LLM
context. The UI renders them as **thin, collapsible, visually subdued rows**
interleaved chronologically with transcript chunks.

**Routing:** a leading `/` in the composer routes to
`POST /api/v1/persona/{name}/dev_command` instead of `/v1/chat/completions`.

**State:** dev messages live in a client-side `devMessages[]` array:
```jsonc
{ id: 3, command: "/set temp 0.4", response: "Temperature set to 0.40",
  mutated: true, timestamp: "2026-06-05T14:01:12Z", afterChunkId: 1041 }
```

**Rendering (collapsed, one-liner):**
```
▸ DEV  /set temp 0.4  →  Temperature set to 0.40  MUTATED  14:01
```

**Rendering (expanded on click):**
- Full response text (pre-wrapped)
- "ephemeral · not persisted · vanishes on refresh" note

**Design decisions:**
- `--ink-faint` left-border (grey, not a semantic color — dev messages are meta)
- `--accent-dim` for the command text (teal, links it to engine interaction)
- `--write` amber `MUTATED` badge when `mutated: true` (persona config changed)
- Excluded from CONTEXT view (they are not part of the LLM prompt)
- No edit/delete/regen actions (ephemeral, non-addressable)
- Collapsed by default to stay unobtrusive

---

## Raw request parity inspector (THE priority feature)

Goal: **see the exact request the engine will send, and guarantee it cannot
diverge from what `chat_system.stream_response` assembles internally.**

The inspector (Raw req tab) renders, top to bottom:
1. **Parity banner.** Green "✓ parity verified — dry-run of
   `chat_system.stream_response`, same code path as a live submit, not
   reconstructed client-side." If the dry-run source is unavailable and the UI
   falls back to client reconstruction, the banner turns **red** ("⚠ client
   fallback — may drift"). The UI must visibly distinguish the two.
2. **Routing:** `route` and `model_name`.
3. **`local_inference_config`:** the resolved sampling params actually forwarded
   (key left, value right, one per row). Includes base params plus any kobold
   sampler extras the engine pulls into the config for the route.
4. **`messages[]`:** the assembled array, each line tagged with its source row
   (`persona.prompt`, `ltm_block`, `#1041`, `#1042 · v3 canonical`, …) so a wire
   line maps back to a transcript row. History is **rebuilt from DB**; the client
   array is discarded.
5. **Footer:** "Edit any line by editing its source row — never a free-text blob.
   The next submit re-runs this exact assembly."

**This requires a new backend endpoint** — a dry-run assembler. Spec in
`API_CONTRACTS.md` § "`/assemble` (proposed)". Without it, the inspector can only
*approximate* (client fallback), which defeats the purpose. Building `/assemble`
is the heart of Sprint S5.

---

## Interactions & Behavior (summary)

- **Collapse toggles** animate grid columns to 0 (`0.22s` ease).
- **RENDERED ⇄ CONTEXT** swaps the transcript renderer; CONTEXT hides the budget
  bar and shows per-row token counts.
- **Reasoning folds / tool cards** toggle open with a caret rotation; `max-height`
  transition `0.2s`.
- **LTM toggle** controls whether `ltm_block` is fetched and injected.
- **Inspector tabs** switch panes; **kobold-only** section collapses independently.
- **Chat send** opens an SSE stream to `/v1/chat/completions`; render token deltas,
  `derpr-tool-start` / `derpr-tool-result` frames, then the `derpr` id-frame
  (hydrate the new chunk's id/version state), then `[DONE]`.
- **CONFIRM approve/deny** compose a confirmation turn (next `/chat/completions`).
- **Persona save** → `PATCH /persona`; surface `rejected_fields`/`unknown_fields`.

Full SSE event grammar, retry semantics, and the `derpr` id-frame fields are in
`API_CONTRACTS.md`.

---

## State Management

- **`activePersona`** — drives `PUT /api/v1/model`; the conversation, persona
  inspector, tools, and budget all key off it.
- **`activeChannel`** — currently `web_ui` only; future-proof the shape.
- **`chunks[]`** — the transcript, keyed by `interaction_id` / `ephemeral_chunk_id`.
  Hydrated from `GET /transcript`; mutated by stream id-frames and by
  edit/delete/version actions. **Never index by position.**
- **`viewMode`** — `'rendered' | 'context'`.
- **`ltmOn`** — boolean; whether to fetch/inject `ltm_block`.
- **`persona`** — full persona object (`GET /persona/{name}`); editable buffer for
  the inspector with dirty-tracking and rejection surfacing.
- **`tools[]`** — `GET /tools/catalog`; enabled state from `persona.enabled_tools`.
- **`stream`** — in-flight SSE state: partial text, partial reasoning, live tool
  calls, abort handle.
- **`pendingConfirmation`** — the trailing ephemeral chunk, if any.
- **`assembled`** — the `/assemble` payload for the Raw req tab + parity flag.

Re-sync rule: after any terminal turn or mutation, **re-fetch `GET /transcript`**
as the authoritative state; the SSE id-frame is an optimization, not the source of
truth. The transcript endpoint is the single re-sync source (it even surfaces the
live parked confirmation as a trailing ephemeral chunk).

---

## Design Tokens

From `design/portal.css` `:root`. Adopt as-is or map to house style, but **keep
the semantic color roles.**

**Colors**
| Token | Hex | Role |
|---|---|---|
| `--bg` | `#0d0f14` | app background |
| `--panel` | `#12151d` | panels / rails |
| `--panel-2` | `#161a24` | raised surfaces |
| `--panel-3` | `#1c2130` | controls |
| `--raise` | `#222838` | hover / tooltip |
| `--line` | `rgba(150,170,205,0.13)` | hairline borders |
| `--line-2` | `rgba(165,185,220,0.24)` | stronger borders |
| `--ink` | `#d6dde9` | primary text |
| `--ink-2` | `#aab4c4` | secondary text |
| `--ink-dim` | `#7a8597` | tertiary text |
| `--ink-faint` | `#515b6c` | faint labels |
| `--accent` (teal) | `#5ad6cf` | **engine / interactive** |
| `--accent-2` | `#8be9e3` | bright teal accents |
| `--accent-dim` | `#2c6f6c` | teal borders |
| `--write` (amber) | `#e7ad62` | **write / mutation / kobold-only** |
| `--danger` (red) | `#e57272` | **danger / error / deny** |
| `--ok` (green) | `#84cf8f` | **read / approve** |
| `--mem` (violet) | `#a394f2` | **memory / LTM** |
| source: `--dsc` `#8aa0e0` · `--zmd` `#e7ad62` · `--gml` `#e57272` | | channel source badges |

Each color has a matching `-bg` tint (e.g. `--accent-bg: rgba(90,214,207,0.10)`).

**Typography**
- Chrome / labels / code: **JetBrains Mono** (400/500/700).
- Message body & descriptions: **IBM Plex Sans** (400/500/600).
- Base size 13px; message body 12.5px; meta/labels 9–11px; caps labels use
  `letter-spacing: 1–1.4px; text-transform: uppercase`.

**Radii:** `--r: 7px`, `--r2: 10px`. **Layout widths:** rail 60, channels 250,
inspector 360 (responsive shrink at ≤1100px).

---

## Assets

None binary. All iconography is **Unicode glyphs** (`▤ ◈ ⊞ ▰ ◔ ❏ ⚙ ⟁ → ‹ ›` etc.).
The implementer may swap these for the repo's icon set, but the glyph approach
keeps the tool dependency-free and is fine to keep. Fonts load from Google Fonts
in the prototype; vendor them locally in production.

---

## Files in this bundle

- `README.md` — this file (self-sufficient overview + component spec + tokens).
- `API_CONTRACTS.md` — **every endpoint** (existing + proposed), the transcript
  chunk shape, persona/tool shapes, the full SSE grammar, retry & version
  semantics, and the proposed `/assemble` and channel-scoping changes. **This is
  the authoritative contract reference — read it with `portal-data.js`.**
- `SPRINTS.md` — the encapsulated sprint plan (S1–S6) with explicit acceptance
  criteria, so each can ship independently.
- `design/DERPR Portal — Control Room.html` — the primary working prototype.
- `design/portal.css` — full theme + component styles (token source).
- `design/portal-app.js` — render + interaction logic (read for DOM structure
  and the contract→view mapping).
- `design/portal-data.js` — **canonical data shapes**, annotated with the
  endpoint each payload comes from. Mirror these exactly.
- `design/Response Types.html` — the six response-type row treatments.
- `design/DERPR Portal Wireframes.html` + `design/wireframe.css` — the original
  two-direction low-fi exploration (Control Room vs. Focus Workspace), included
  for context on why the Control Room layout was chosen.

## Source files in the DERPR repo this design targets

- `src/interfaces/kobold_engine_adapter.py` — the **engine adapter** (port 5003).
  Most endpoints already exist here; the new UI talks to this, not the
  passthrough adapter.
- `src/interfaces/kobold_export.py` — `build_transcript()` (the chunk projection)
  and `build_kobold_savefile()`.
- `src/interfaces/_persona_patch.py` — base vs kobold param key sets
  (`_KNOWN_PATCH_KEYS_ENGINE`, `_apply_kobold_sampler_extras`,
  `get_kobold_extras_for_get`) — the authoritative base-vs-kobold split.
- `src/interfaces/portal_render.py` — how the old portal HTML is served (replace).
- `src/chat_system.py` — `stream_response`, event types (`TokenEvent`,
  `ToolCallStartEvent`, `ToolCallResultEvent`, `DoneEvent`, `ErrorEvent`),
  `_pending_confirmations`. The `/assemble` dry-run must reuse this assembler.
