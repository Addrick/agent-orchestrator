# API Contracts — DERPR Portal

Authoritative reference for the bespoke UI. Read alongside `design/portal-data.js`,
whose mock objects mirror these payloads exactly. The UI talks to the **engine
adapter** (`src/interfaces/kobold_engine_adapter.py`), default port **5003**.

Legend: ✅ **exists today** · 🟡 **proposed / needs backend work**

---

## 1. The transcript chunk contract (DP-130) — the render source ✅

`GET /api/v1/session/{persona}/transcript?max_turns={n}` → `{ "chunks": [ ... ] }`

Produced by `build_transcript()` in `kobold_export.py`. Each chunk:

```jsonc
{
  "interaction_id": 1042,        // int | null  (null only when ephemeral)
  "role": "assistant",           // "user" | "assistant"  (system/tool-only rows are NOT chunks)
  "content": "<think>…</think>\nbody text",  // reasoning folded in as <think>…</think>
  "ephemeral": false,            // true => not yet persisted (parked confirmation)
  "reasoning": "…",              // string | null  (the raw reasoning, also folded into content)
  "tool_context": [ /* see §5 */ ] ,  // array | null
  "has_versions": true,          // edit/regen archives exist → show version chevrons
  "ephemeral_chunk_id": "pend_7f3c"   // present ONLY on the ephemeral parked chunk
}
```

**Invariants the UI must respect (server guarantees these):**
- **C1:** every chunk has exactly one `interaction_id` OR `ephemeral: true` —
  never both, never neither.
- **C2 (gametext alignment):** chunks are ordered; one slot per visible story
  turn. `interaction_id` may be `null` for an unaddressable renderable row —
  guard with `if (interaction_id)` before calling id-scoped endpoints.
- **C3:** the chat stream emits a `derpr` id-frame on **every** terminal turn
  (normal, parked, tool-only, abort) carrying the ids — but `GET /transcript` is
  the **authoritative re-sync source**; prefer re-fetching it after a turn.
- **C5:** suppressed (deleted) rows are filtered server-side; the UI never sees
  them.

**Display stitching:** `transcript` does NOT include the system prompt or LTM
block. Fetch those separately (§3, §4) and render them as pinned/injected rows in
RENDERED view.

**Parked confirmation:** when a write is awaiting approval, `GET /transcript`
appends a trailing chunk with `ephemeral: true`, `interaction_id: null`,
`ephemeral_chunk_id` set, and the pending tool in `tool_context` (result `null`).

---

## 2. Persona ✅

### `GET /api/v1/persona/{name}`
```jsonc
{
  "name": "assistant",
  "display_name": "Assistant",
  "prompt": "You are a terse internal IT-support assistant…",
  // ---- BASE PARAMS (devoted persona properties; sent to every provider) ----
  "model_name": "gpt-4o-mini",
  "temperature": 0.4,
  "max_tokens": 1024,            // from get_response_token_limit()
  "history_messages": 24,        // get_base_history_messages()
  "thinking_level": "medium",
  "memory_mode": "GLOBAL",       // CHANNEL_ISOLATED|SERVER_WIDE|PERSONAL|GLOBAL|TICKET_ISOLATED
  "max_context_tokens": 16384,
  "chat_template": "chatml",     // engine adapter only (not in passthrough adapter)
  "tool_policy": { "mode": "CONFIRM", /* + .to_dict() fields */ },
  "enabled_tools": ["search_tickets", "reset_vpn_cert", "email_user", "lookup_user"],
  // ---- KOBOLD-ONLY (provider_extra("kobold", …); only used on kcpp route) ----
  "top_p": 0.92,                 // persona props but kobold-route samplers
  "top_k": 40,
  "instruct_tags": { /* or null */ },
  "kobold_extras": { "rep_pen": 1.07, "rep_pen_range": 320, "min_p": 0.05,
                     "tfs": 1.0, "mirostat": 0, "mirostat_tau": 5.0,
                     "mirostat_eta": 0.1, "sampler_order": [6,0,1,3,4,2,5] },
  // ---- SECURITY ----
  "security_blocked": false,
  "security_block_reasons": []
}
```

**Base vs kobold-only split (authoritative source):** see `_persona_patch.py`.
`_KNOWN_PATCH_KEYS_ENGINE` is the accepted PATCH key set; `kobold_extras` come from
`get_kobold_extras_for_get(p)` and are applied by `_apply_kobold_sampler_extras`.
**Rule for the inspector:** a field that maps to a devoted persona getter/setter
(`temperature`, `model_name`, `memory_mode`, `max_tokens`, `history_messages`,
`max_context_tokens`, `thinking_level`, `chat_template`, `tool_policy`) → **base
params**. A field that lives in `kobold_extras` / `provider_extra("kobold", …)`
(`rep_pen*`, `min_p`, `tfs`, `mirostat*`, `sampler_order`, `instruct_tags`) →
**kobold-only**. `top_p`/`top_k` are persona props but are kobold-route samplers —
the prototype groups them under kobold-only; confirm with the engine team which
side they belong on.

### `PATCH /api/v1/persona/{name}`
Body: any subset of the above keys. Returns:
```jsonc
{ "result": "success",
  "rejected_fields": ["temperature"],   // coerced/invalid values — SURFACE THESE
  "unknown_fields": [] }
```
Numeric setters silently coerce bad input; the UI **must** show `rejected_fields`
to the user rather than implying a clean save. On save failure: HTTP 500 with
`{ "error": "save_failed", "detail", "rejected_fields", "unknown_fields" }`.

### Other persona routes ✅
- `GET /api/v1/model` → `{ "result": "<active persona>" }`
- `PUT /api/v1/model` body `{ "model": "<persona>" }` → sets active persona.
- `GET /v1/models` → OpenAI-style list of personas.
- `GET /api/v1/models/list` → `{ "models": ["…"] }` (for the model dropdown).
- `POST /api/v1/persona/{name}/reset` → starts a new conversation.
- `POST /api/v1/persona/{name}/dev_command` body `{ "command": "set temp 0.4" }`
  → `{ "response": "...", "mutated": bool }`. Route composer input starting with
  `/` here. 400 `{ "response": "Not a dev command" }` if not recognized.

---

## 3. LTM memory block ✅

`GET /api/v1/session/{persona}/ltm_block?query={text}`
→ `{ "block": "<memory>…</memory>" }` or `{ "block": null }`.

Called before each submit when LTM is on. The block is injected at the engine's
author's-note position. In RENDERED view, show it as a violet "LTM recalled"
injected row; in CONTEXT/Raw view it appears as the author's-note message.

---

## 4. Tools catalog ✅

`GET /api/v1/tools/catalog` →
```jsonc
{ "tools": [
  { "name": "reset_vpn_cert",
    "description": "Reissue a user's VPN client certificate.",
    "is_write": true,
    "capabilities": { "locality": "local",    // "local" | "remote"
                      "sensitivity": "high",   // "low" | "medium" | "high"
                      "produces_untrusted": false } }
] }
```
Drives the Tools inspector tab and the capability badges on tool cards. Enabled
state per persona comes from `persona.enabled_tools`.

---

## 5. Chat stream — submit, tokens, tools, ids ✅

`POST /v1/chat/completions` (also `/chat/completions`). **SSE.** The engine
adapter is a thin transcoder over `chat_system.stream_response` and **discards the
client `messages` array** — history is rebuilt from the DB.

**Request body (the parts the UI sets):**
```jsonc
{
  "stream": true,
  "derpr_user_text": "Email Jane the confirmation too.",  // the authoritative new user turn
  "derpr_retry": false,            // true = regenerate: archive prior assistant row, make a new version
  "model": "assistant",            // persona selector (stripped before upstream)
  // sampling params the engine extracts into local_inference_config:
  "temperature": 0.4, "top_p": 0.92, "top_k": 40, "max_tokens": 1024, "stop": null,
  "rep_pen": 1.07, "min_p": 0.05, "tfs": 1.0  // kobold extras pulled through for the route
  // "messages": [...]  // IGNORED by the engine adapter — do not rely on it
}
```

**SSE event grammar (in order):**
```
data: {"object":"chat.completion.chunk","choices":[{"delta":{"content":"…"}}]}      // token deltas (repeat)
event: derpr-tool-start
data: {"tool_name":"reset_vpn_cert","arguments":{…},"call_id":"c_91a2","group_id":"g_1"}
event: derpr-tool-result
data: {"call_id":"c_91a2","tool_name":"reset_vpn_cert","result":"…","error":null,"group_id":"g_1"}
event: derpr-confirm
data: {"text":"…","persona":"assistant","token":"pend_7f3c","calls":[{"name":"reset_vpn_cert","arguments":{…},"id":"c_91a2"}],"audit_info":{…}}   // ONLY on a CONFIRM-mode park; see §7
event: derpr
data: {"user_id":1045,"assistant_id":1047,"response_type":"NORMAL","ephemeral_chunk_id":null}   // id-frame, EVERY terminal turn
data: [DONE]
```

The **`derpr-confirm`** frame is emitted (before the terminal `derpr` id-frame)
**only when a write parks** under CONFIRM mode. It carries the structured pending
write(s) + a resume `token`; the portal renders an approve/deny surface from it and
resolves via the dedicated `/confirm` endpoint (§7). On a parked turn the terminal
`derpr` frame has `assistant_id: null` and a set `ephemeral_chunk_id`.

**The `derpr` id-frame** (`DoneEvent`) carries:
- `user_id` — interaction_id of the user turn just logged (null on retry/parked).
- `assistant_id` — interaction_id of the assistant row (**null when parked**).
- `response_type` — map to `ResponseType` enum → drives row treatment (§ README
  response-type table).
- `ephemeral_chunk_id` — set when the turn parked for confirmation; the stable id
  of the unpersisted confirmation chunk.

**Error:** `data: {"error":{"message":"…"}}` then `data: [DONE]` → render an
error row, no assistant row committed.

**Abort:** `POST /api/v1/abort` (or `/api/extra/abort`). The engine flushes the
partial assistant row; render the "aborted · partial" treatment.

**Non-streaming** variant returns a normal OpenAI completion object plus
`derpr_assistant_id`.

---

## 6. Row mutations — edit / delete / versions ✅

- `PATCH /api/v1/interaction/{id}` body `{ "content": "…" }` → edit a row's text.
- `DELETE /api/v1/interaction/{id}` → soft-suppress. **Idempotent:** returns
  `{ "result":"success", "already_suppressed": bool }`. Reply chains preserved;
  suppressed rows vanish from transcript/history/export.
- `GET /api/v1/interaction/{id}/versions` →
  `{ "interaction_id", "versions": [ { …, "content", reasoning folded } ] }`,
  **canonical last**. Drives the chevron stack.
- `POST /api/v1/interaction/{id}/select_version/{k}` → swap archive position `k`
  with canonical (0-indexed pre-swap); returns new canonical + refreshed
  `versions` so the UI re-syncs chevrons in one round-trip.

**Regenerate:** send a chat turn with `derpr_retry: true`. The engine archives the
current assistant row and updates canonical on completion; afterwards that chunk's
`has_versions` becomes true.

---

## 7. CONFIRM (tool approval) — dedicated `/confirm` endpoint ✅

> **Updated 2026-06-04 to match the engine.** Earlier drafts of this handoff said
> there was no approval endpoint and that approve/deny had to be composed as a
> free-text chat turn (the old Discord-style flow). That is **no longer true.** The
> engine adapter now exposes a structured `/confirm` endpoint and emits a
> `derpr-confirm` SSE frame (§5). Build to the endpoint below, not to a free-text
> affirmation. (Reworked in engine commit `8506d26` "drive CONFIRM-mode resume
> through ToolLoop".)

When `persona.tool_policy.mode == "CONFIRM"`, a write tool **parks**:
- During the chat stream the engine emits a **`derpr-confirm`** frame (§5) with the
  structured pending `calls[]`, the persona, a resume `token`, and `audit_info`;
  the terminal `derpr` id-frame then carries `assistant_id: null` +
  `ephemeral_chunk_id`. `GET /transcript` also surfaces the park as a trailing
  ephemeral chunk on a fresh load.
- **Resolution is a dedicated endpoint:**

  `POST /api/v1/persona/{name}/confirm` body `{ "approved": bool, "token": "<from the derpr-confirm frame>" }`

  Returns an **SSE stream** using the same wire protocol as `/v1/chat/completions`
  (token deltas → tool frames → the model's follow-up turn → terminal `derpr`
  id-frame → `[DONE]`). A chained write re-surfaces as another `derpr-confirm`
  frame, so the approve/deny surface must handle recursion.
- The `token` guards against resuming a **stale park** (the model proposed
  different writes since the frame was shown); send the token from the frame. Omit
  or send empty to skip the staleness check.
- The portal user is always `"portal"`, matching the park key `(user, persona)` in
  `_pending_confirmations`.

So the approve/deny buttons **POST to `/confirm`** (one click, structured) — they
do **not** compose a free-text chat turn.

> **Edit-args** (adjust a proposed call before approving) has **no structured
> endpoint** — that path would still require composing a corrective chat turn, or a
> new endpoint. Confirm the desired UX with the engine team before building it.

---

## 8. Token counting & misc ✅

- `POST /api/extra/tokencount` body `{ "prompt": "…" }` → forwarded to KCPP; use
  for the budget bar / per-row counts. **Do not estimate client-side** in
  production (the prototype estimates only for demo).
- `GET /api/extra/version`, `/api/v1/info/version`, `/api/v1/config/*` — capability
  probes; forwarded with fallbacks. Generally not needed by the new UI.
- `GET /api/v1/session/{persona}/kobold_export` — legacy Kobold savefile; not used
  by the new UI.

---

## 9. 🟡 `/assemble` — dry-run request assembler (PROPOSED; build this)

**The parity primitive.** The Raw req inspector must show the *exact* request the
engine will send, sourced from the engine's own assembler — not reconstructed in
the browser.

Proposed:
```
GET /api/v1/session/{persona}/assemble?message={text}&channel={chan}&ltm={bool}&retry={bool}
```
Runs the **same path as `stream_response`** (history rebuild from DB + LTM +
prompt assembly + param resolution) **with inference disabled**, and returns what
would be sent:
```jsonc
{
  "parity": { "source": "engine.dry_run",            // vs "client_fallback"
              "builder": "chat_system.stream_response",
              "matches_live": true },
  "route": "engine · POST /v1/chat/completions",
  "model_name": "gpt-4o-mini",
  "params": { "temperature":0.4, "top_p":0.92, "top_k":40, "max_tokens":1024,
              "stop":null, "rep_pen":1.07, "min_p":0.05, "tfs":1.0 },
  "messages": [
    { "role":"system",    "content":"…",  "src":"persona.prompt" },
    { "role":"system",    "content":"[author's-note]…", "src":"ltm_block" },
    { "role":"user",      "content":"…",  "src":"#1041" },
    { "role":"assistant", "content":"…",  "src":"#1042 · v3 canonical" },
    { "role":"user",      "content":"…",  "src":"#1045" }
  ]
}
```
**Implementation note:** factor the assembly out of `stream_response` into a pure
builder that both the live path and `/assemble` call, so there is *one* code path
and drift is impossible by construction. The `src` tags map each wire line back to
a transcript row (or to `persona.prompt` / `ltm_block`). `mock` shape lives in
`design/portal-data.js` → `ASSEMBLED_REQUEST`.

If `/assemble` is unavailable, the UI may client-reconstruct as a fallback **but
must flip the parity banner to red** (`source: "client_fallback"`).

---

## 10. 🟡 Channel scoping (PROPOSED; needed for multi-channel)

Today the engine adapter hard-codes `channel="web_ui"` and
`user_identifier="portal"` in every call (`_log_interaction`, `stream_response`,
`ltm_block`, retry). `channel` on the engine side is already a **source-agnostic
tag**, so no new concept is required — but the HTTP surface needs to **accept a
`channel` (and optionally `user_identifier`/`server_id`) parameter** on:
- `GET /transcript`, `GET /ltm_block`, and `POST /v1/chat/completions`.

Creating a "new channel" then just means submitting the first turn with a fresh
`channel` string. **For S1–S5 a single `web_ui` channel is fine**; channel
scoping is its own sprint (S6) and should be confirmed with the engine team since
it touches `memory_mode` isolation semantics (CHANNEL_ISOLATED etc.).
