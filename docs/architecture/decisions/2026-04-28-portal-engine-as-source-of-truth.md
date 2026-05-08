---
name: Portal as thin shell over engine — backend owns history, params pass through
description: Phase D direction — portal adapter is a transcoder; engine rebuilds context from DB; kobold-specific params flow via provider_extras, not mirrored in DERPR UI; jinja explains why OAI route is preferred over kobold-native
type: project
---

**Decision (2026-04-28):** Phase D of the portal engine reintegration commits to a single design intent that resolves several Phase D ambiguities at once.

## Intent

The portal is one of several interfaces (Discord, Gmail, agents, portal). All interfaces are **thin shells over the engine**. The engine is the single source of truth for:

- History assembly (from DB, not from client payload)
- LTM retrieval and injection placement
- Tool orchestration / agent dispatch
- User-turn logging, retry archive, assistant commit

Multiple paths for any of these = duplication = inhibits future test simplicity. The portal's bespoke passthrough was scaffolding that solved a specific Stage-1 problem (verbatim kobold contract). It is not a feature to preserve.

## Concrete consequences for Phase D

1. **OAI route (`/v1/chat/completions`) → goes through `_orchestrate`.** Engine rebuilds messages from DB. Client `data["messages"]` is **discarded**; only `derpr_user_text` sidecar is consumed as the user turn. Retry/edit/delete already round-trip to DB via existing PATCH/DELETE/version routes — DB == UI state by construction.
2. **Native route (`/api/extra/generate/stream`) → adapter shim stays untouched** in Phase D. The kobold-rendered prompt blob cannot be rebuilt from DB without re-introducing the template-tag drift hazard that `2026-04-19-portal-phase2-approach.md` rejected. No sibling `_orchestrate_prompt` kernel — the redundancy isn't worth the duplication.
3. **Native route deprecation is queued as future work, not Phase D.** See "Why OAI is preferred" below.

## Param strategy

- DERPR's persona panel UI stays minimal. Do **not** mirror kobold-lite's parameter sprawl (mirostat, dry, xtc, sampler order, etc.) into DERPR's UI.
- Kobold-specific params flow through unchanged via `GenerationParams.provider_extras["kobold"]` (Phase A surface). Engine passes them to `StreamEngine` which forwards to KCPP.
- BotLogic's planned dotted-path setter (`set kobold.<key> <value>`, Phase E) is the occasional CLI escape hatch when experimenting.
- Trade-off accepted: a new prompt-handling feature in upstream kobold-lite (jinja was the historical example) requires manual mirror in DERPR. This is rare and copy-update is cheap.

## Why OAI route is preferred over kobold native

Historical: kobold-native (`/api/extra/generate/stream`) was the original wire format. When upstream kobold-lite shipped jinja chat templates, the implementation hung off OAI-shape endpoints (`/v1/chat/completions`). DERPR followed and OAI became the de-facto path. Native still ships because kobold-lite's `useoaichatcompl` toggle remains user-facing.

**Open investigation (queued, not in Phase D):** does every kobold-lite feature now reach feature parity over OAI? If yes, the native route + adapter shim can be removed entirely (forced toggle or stripped UI control). If something (e.g. specific sampler sliders, abort semantics) only fires on native, those need OAI parity first or the toggle stays. Until that audit lands, treat the native route as legacy-but-supported.

## How to apply

- Phase D plan (`plans/portal_engine_reintegration.md`) updated to reflect: OAI-only migration; kernel-via-DB; native route untouched; sibling kernel rejected; helper extraction + cancel-flush + retry trailing-pop fix as the engine-side prerequisites.
- When evaluating future portal features: if it requires duplicating engine logic (history rebuild, retry semantics, LTM placement, tool orchestration), **stop and route through the engine instead**. The portal layer should only do HTTP/SSE transcoding.
- When evaluating future kobold params: add to `provider_extras["kobold"]` plumbing, not to DERPR UI.

## References

- `plans/portal_engine_reintegration.md` — Phase D scope and ordering
- `decisions/2026-04-19-portal-phase2-approach.md` — kobold owns templating *for the prompt path* (template-tag drift hazard); does not apply to OAI route
- `decisions/2026-04-19-kobold-portal-passthrough.md` — Stage 1 verbatim passthrough, scaffolding this decision walks back
