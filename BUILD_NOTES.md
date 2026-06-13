# DERPR Portal (DP-132) — Build Notes

Bespoke React/Vite/TypeScript web UI replacing the Kobold-Lite PoC. Premise:
**every message is a discrete, id-addressable object** matching the DB 1:1.
Built on branch `feature/DP-132-bespoke-ui-portal` (not merged).

## Sprints completed

| Sprint | Commit | Status |
|--------|--------|--------|
| **S0** scaffold, theme & collapsible Control Room shell | `🏗️ DP-132` | ✅ done |
| **S1** read-only transcript render (parity keystone)    | `✨ DP-132` | ✅ done |
| **S2** submit & stream a turn over SSE                  | `✨ DP-133` | ✅ done |
| **S3** id-addressed row mutations (edit/del/regen/versions) | `✨ DP-134` | ✅ done |

S4 (persona edit / tools toggle / CONTEXT-authoritative), S5 (`/assemble`
parity inspector), and S6 (CONFIRM resolve / multi-channel) are **out of scope**
and were not built. The inspector Persona/Tools panes render **read-only**; the
Raw-req tab shows an explicit "S5" placeholder rather than a client
reconstruction (so it never implies a parity guarantee it can't make).

## How to run

Project root: `src/interfaces/web_assets/derpr_ui/` (inside this worktree).
Node v24 / npm 11.

```bash
cd src/interfaces/web_assets/derpr_ui
npm install          # already done in this worktree
npm run dev          # dev server (Vite) — proxies /api + /v1 to :5003
npm run build        # tsc -b && vite build → dist/  (base = /derpr/)
npm run lint         # eslint — clean
```

Acceptance bar met: `npm run build` (type-checks via `tsc -b`) and `npm run lint`
both pass clean. **Note:** I could not start the dev server in this build
environment — listening sockets are blocked (EACCES on every port, sandboxed and
unsandboxed). The build/typecheck/lint are the verifiable signals here; the dev
server should run normally on a real machine.

### Serving from the engine (production path)

The engine adapter (`kobold_engine_adapter.py`, port **5003**) was given **one
additive change** (no other backend edits):

- `GET /derpr` → returns the built `dist/index.html`.
- A `StaticFiles` mount at `/derpr` serves the hashed JS/CSS assets.
- The Vite `base` is `/derpr/`, so emitted asset URLs resolve under the mount.
- If `dist/` is absent the route returns a 503 with a "run `npm run build`" hint,
  so the engine still boots without the front-end artifacts.

The existing `/portal` route and the Kobold-Lite PoC are untouched. After
`npm run build`, open `http://<host>:5003/derpr`.

`dist/` is gitignored (standard Vite). Build it on deploy (or wire a build step
into the engine's deploy if you want `/derpr` live without a manual `npm run build`).

## Mocked vs live

The API client (`src/api/client.ts`) is the **only** module that talks to the
engine. It is **live-first with mock fallback**: each call hits the real endpoint;
if the engine is unreachable it falls back to typed fixtures in `src/api/mock.ts`
(mirrors `design/portal-data.js` exactly) so the UI renders without a backend.
`usingMock()` drives the "mock · offline" indicator in the top-bar connection chip.

Live endpoints consumed (all on the adapter):
- `GET /api/v1/model`, `GET /api/v1/models/list`, `PUT /api/v1/model`
- `GET /api/v1/persona/{name}`, `PATCH /api/v1/persona/{name}`
- `GET /api/v1/tools/catalog`
- `GET /api/v1/session/{persona}/transcript`, `…/ltm_block`
- `POST /v1/chat/completions` (SSE), `POST /api/v1/abort`
- `POST /api/v1/persona/{name}/dev_command`
- `PATCH` / `DELETE /api/v1/interaction/{id}`,
  `GET /api/v1/interaction/{id}/versions`,
  `POST /api/v1/interaction/{id}/select_version/{k}`

Single-user, single-channel: `user_identifier="portal"`, one `web_ui` channel.
There is no channel-listing endpoint yet (S6); the channel rail shows the single
live channel against the engine, or the grouped demo list when offline.

## Key design decisions / deviations

- **State model:** chunks are keyed by `interaction_id` / `ephemeral_chunk_id`,
  never by array position (invariants C1/C2). React keys use the id, falling back
  to a stable slot key only for the unaddressable case. After every terminal turn
  or mutation the store re-fetches `GET /transcript` as the authoritative re-sync;
  the SSE `derpr` id-frame is treated as an optimization, not source of truth.
- **Response-type treatments:** the engine's `ResponseType` enum is coarse
  (`LLM_GENERATION` / `DEV_COMMAND` / `PENDING_CONFIRMATION`), so the README's six
  descriptive treatments (normal / tool-only / parked / aborted / error /
  security-blocked) are *derived* in `state/util.ts → deriveTreatment()` from the
  wire `response_type` plus stream signals (parked, aborted, errored, tools-with-
  empty-content). The parked ephemeral chunk renders (amber border, pending tool
  card) but its approve/deny **resolve flow is intentionally disabled (S6)**.
- **SSE transport:** uses `fetch` + a `ReadableStream` reader, not `EventSource`
  (which is GET-only and can't carry the POST body). `src/api/stream.ts` parses
  the SSE block grammar in order and exposes a `streamConfirm` helper ready for S6.
- **Token counts:** the budget bar and CONTEXT view estimate tokens client-side
  (`estimateTokens`, marked DEMO-only). Production counts must come from
  `POST /api/extra/tokencount` or the `/assemble` payload — wired in S4/S5. The
  budget total is prefixed `~` and the CONTEXT note flags the approximation.
- **CSS minify disabled:** Vite 8's default `lightningcss` minifier mis-parses the
  ported `portal.css` theme (a false-positive "dangling combinator"), and esbuild
  isn't bundled in this Vite build. `build.cssMinify: false` in `vite.config.ts`.
  Unminified CSS (33 kB → 6 kB gzip) is fine for an internal tool. Worth revisiting.
- **Persona editing read-only:** the inspector shows the base-vs-kobold param split
  faithfully but does not PATCH (that, with `rejected_fields`/`unknown_fields`
  surfacing, is S4). `patchPersona` is implemented in the client and ready to wire.
- **Pre-commit hook:** the repo's pre-commit hook writes `.claude/.memory_update_
  pending`; the worktree had no `.claude/` dir, so I created an (untracked) one so
  commits succeed.

## Left incomplete / next steps (out of scope, by design)

- S4: persona PATCH + rejection surfacing, tool enable toggles, authoritative
  CONTEXT view.
- S5: backend `GET /api/v1/session/{persona}/assemble` dry-run builder + the Raw-req
  parity inspector (the headline feature — currently a placeholder).
- S6: CONFIRM approve/deny via `POST /api/v1/persona/{name}/confirm` (the SSE client
  already has `streamConfirm`); multi-channel scoping.
