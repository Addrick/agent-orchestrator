# Sprint Plan — DERPR Portal

Encapsulated, independently-shippable sprints to reach **parity** with the current
Kobold-Lite portal (with the new row-as-object / DB-aligned model), then a couple
of small extensions. Each sprint has a hard **Done =** line so it can't sprawl.
The goal of this structure: settle every state in the design (already done in the
prototype) so implementation is hitting a fixed target, not discovering edge cases
mid-build.

Order is dependency-driven. S1→S5 = parity. S6 = first real extension.

Reference: `README.md` (component spec), `API_CONTRACTS.md` (endpoints),
`design/portal-data.js` (data shapes), `design/portal-app.js` (contract→view).

---

## S0 — Front-end scaffold & theme (foundation)

**Build:** choose the front-end stack (see README "About the Design Files");
set up the app shell, the four-region collapsible grid, top bar, nav rail (with
the 5 reserved expansion docks rendered as disabled "soon" items), and the design
tokens from `portal.css`. Wire the three collapse toggles. Replace
`portal_render.py`'s served HTML entry point.

**Done =** the empty Control Room shell renders, all three side panels collapse/
expand smoothly, tokens/fonts match, nav rail shows CHAT active + 5 disabled
expansion slots. No data yet.

---

## S1 — Read-only transcript render (the parity keystone)

**Build:**
- Fetch `GET /api/v1/persona/{name}` (system prompt + header chips) and
  `GET /api/v1/session/{persona}/transcript` (chunks).
- Render the conversation in **RENDERED** view: pinned system-prompt row, LTM
  injected row (from `GET /ltm_block` when LTM toggle on), then chunks.
- Full message-row anatomy **read-only**: meta line + id tag, collapsible
  reasoning fold (split `<think>` out of content), embedded tool-call cards with
  capability badges, body text. Version chevrons appear **only** when
  `has_versions` (read-only display of count for now).
- Channel list grouped by source (single `web_ui` entry acceptable).
- Budget bar from `POST /api/extra/tokencount` or static for now.

**Must respect:** address chunks by `interaction_id`/`ephemeral_chunk_id`, never
by array index (invariants C1/C2). System & LTM are stitched, not chunks.

**Done =** loading the portal shows the persona's conversation identical to the
DB, with reasoning/tool/version affordances rendering from real chunk fields, and
no drift between a page reload and the DB state.

---

## S2 — Submit & stream a turn

**Build:**
- Composer → `POST /v1/chat/completions` (SSE) with `derpr_user_text` (+ resolved
  params). Handle the event grammar: token deltas → live assistant bubble;
  `derpr-tool-start`/`derpr-tool-result` → live tool cards; `derpr` id-frame →
  hydrate the new chunk's ids/`response_type`; `[DONE]` → finalize.
- After `[DONE]`, **re-fetch `GET /transcript`** as the authoritative re-sync.
- Abort button → `POST /api/v1/abort`; render the **aborted/partial** treatment.
- Error frame → **error** treatment (no committed assistant row).
- `Enter` send / `Shift+Enter` newline; `/`-prefixed input →
  `POST /persona/{name}/dev_command` and render its response (not a chat turn).

**Done =** a user can send a message, watch tokens/tools/reasoning stream in, and
the resulting turn persists as exactly one user row + one assistant row that match
the DB on reload. Abort and error states render correctly.

---

## S3 — Row mutations (edit / delete / regenerate / versions)

**Build:**
- `✎ edit` → inline edit → `PATCH /interaction/{id}` → re-sync.
- `✕ del` → `DELETE /interaction/{id}` (idempotent soft-suppress); row disappears,
  reply chain intact.
- `⟲ regen` → chat turn with `derpr_retry: true`; afterwards the chunk gains
  versions.
- Version chevrons `‹ k/n ›` → `GET /interaction/{id}/versions` +
  `POST /interaction/{id}/select_version/{k}`; re-sync chevrons from the response.

**Done =** every mutation writes back to a discrete DB row (never a blob), the UI
re-syncs from the server, and regenerate produces a navigable version stack. This
sprint is the explicit kill of the Kobold "single textbox" footgun.

---

## S4 — Persona inspector, tools & CONTEXT view

**Build:**
- **Persona tab:** read `GET /persona`; render the **base-params** section and the
  collapsible **kobold-only** section per the README split (authoritative origins
  in `_persona_patch.py`). Edits → `PATCH /persona`; **surface
  `rejected_fields`/`unknown_fields`.** Security-blocked banner when applicable.
- **Tools tab:** `GET /tools/catalog` list with capability badges + enable
  toggles reflecting `persona.enabled_tools`.
- **CONTEXT ↦ LLM** transcript view: the assembled prompt as role-tagged rows with
  per-row token counts (budget bar hidden). May use client assembly *temporarily*
  here, but the authoritative version is S5.

**Done =** the persona can be inspected and edited with base/kobold params clearly
separated, rejections surfaced, tools listed with correct badges, and the CONTEXT
toggle shows a role-tagged assembled view.

---

## S5 — Raw request parity inspector + `/assemble` endpoint (the priority)

**Build (backend + frontend):**
- **Backend:** factor prompt assembly out of `stream_response` into a single pure
  builder; add `GET /api/v1/session/{persona}/assemble` (dry-run, inference off)
  returning `{parity, route, model_name, params, messages[]}` per
  `API_CONTRACTS.md` §9. Same code path as a live submit ⇒ no drift by
  construction.
- **Frontend:** the **Raw req** inspector tab — parity banner (green verified /
  red client-fallback), routing, `local_inference_config`, and `messages[]` with
  each line tagged to its source row. Reads from `/assemble`.

**Done =** opening Raw req shows the exact request the engine will send, proven to
match the live assembler (green parity banner sourced from the shared builder),
with every wire line traceable to its transcript row. **This is the headline
capability: "what I see == what gets sent," guaranteed.**

---

## S6 — (First extension) Multi-channel + CONFIRM resolve

Two smaller pieces; split if needed.

**6a — CONFIRM resolve:**
- Render the parked ephemeral chunk (amber, pending tool card, approve/deny bar)
  from the `derpr-confirm` frame (structured calls + resume token) and the trailing
  transcript ephemeral chunk. Approve/deny **POST to
  `POST /api/v1/persona/{name}/confirm`** (`{approved, token}`), which streams the
  continuation back over SSE (handle a chained `derpr-confirm`). Re-sync from
  `/transcript`. (See `API_CONTRACTS.md` §7 — the dedicated endpoint, not a
  free-text chat turn.)
- **Done =** a CONFIRM-mode write parks visibly and resolves correctly via
  `/confirm`; the ephemeral chunk becomes a persisted row on approval or is
  cancelled on denial.

**6b — Channel scoping (needs the API change in `API_CONTRACTS.md` §10):**
- Add `channel` (and `user_identifier`/`server_id` as needed) params to
  `/transcript`, `/ltm_block`, `/v1/chat/completions`. Channel list switches the
  active channel; "+ new channel" submits the first turn with a fresh tag.
- **Done =** the UI can read existing channels across sources and create new
  `web_ui` channels. Confirm `memory_mode` isolation semantics with the engine
  team before shipping.

---

## Out of scope (reserved expansion — DO NOT build now)

Leave permanent nav-rail homes only: **Memory inspector**, **Agent monitor**,
**Budget visualizer**, **Analytics/cost**, **Persona library** (compare/fork/A-B).
These are future sprints; the layout already reserves space.

---

## Suggested cut line for a first usable PoC

**S0–S3 = a working, DB-aligned chat PoC** (read, send/stream, mutate). **S4–S5**
deliver the inspector + the parity guarantee that motivated the rework. Ship
S0–S5 before touching S6 or any expansion area.
