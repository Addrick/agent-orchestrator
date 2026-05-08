---
name: Portal ↔ persona settings sync
description: Map of kobold-lite localsettings fields to Persona getters/setters and provider_extras for bidirectional sync; identifies what's already wired and what's missing
type: project
---

## Goal

Make kobold-lite's settings popup edits round-trip into the active persona so the kobold UI and persona DB are a single source of truth. Today sync is one-way (persona → kobold sliders on switch via `applyPersonaToUI`); reverse direction is missing.

## Current state

### Already synced both ways (persona popup ↔ persona DB)
The persona settings popup in `portal.html` (`onPersonaChange` / `savePersonaFromUI`) already PATCHes these:

| Persona field | UI input id | Persona accessor |
|---|---|---|
| `prompt` | `systemPromptInput` | `get_prompt` / `set_prompt` |
| `model_name` | `modelTargetSelect` | `get_model_name` / `set_model_name` |
| `temperature` | `tempInput` | `get_temperature` / `set_temperature` |
| `top_p` | `topPInput` | `get_top_p` / `set_top_p` |
| `top_k` | `topKInput` | `get_top_k` / `set_top_k` |
| `max_tokens` | `maxTokensInput` | `get_response_token_limit` / `set_response_token_limit` |
| `history_messages` | `historySlider` | `get_base_history_messages` / `set_history_messages` |
| `max_context_tokens` | `maxContextTokensInput` | `get_max_context_tokens` / `set_max_context_tokens` |
| `memory_mode` | `memoryModeSelect` | `get_memory_mode` / `set_memory_mode` |
| `instruct_tags` | (kobold tag fields) | `get_provider_extra("kobold", "instruct_tags")` / `set_provider_extra(...)` |

### Synced one-way (persona → kobold sliders)
`applyPersonaToUI` pushes persona values into `localsettings` + slider DOM on persona load/switch:

| `localsettings` key | Slider DOM id | Persona source |
|---|---|---|
| `temperature` | `temperature` | `get_temperature` |
| `top_p` | `top_p` | `get_top_p` |
| `top_k` | `top_k` | `get_top_k` |
| `max_length` | `max_length` | `get_response_token_limit` |
| `max_context_length` | `max_context_length` | `get_max_context_tokens` |
| `instruct_sysprompt` | `instruct_sysprompt` | `get_prompt` |
| `chatopponent` | (none) | persona role |

**Missing reverse direction:** edits in kobold-lite's `confirm_settings()` popup never PATCH the persona. User-tuned sliders evaporate on persona reload.

### Not synced at all (kobold-only today, no persona equivalent)
These live in `localsettings` and ride OAI `provider_extras["kobold"]` per request, but have no persona getter/setter — every persona reuses kobold's session-level defaults:

| `localsettings` key | Notes |
|---|---|
| `rep_pen` | Already plumbed through `provider_extras["kobold"]` per request |
| `rep_pen_range` | Same |
| `rep_pen_slope` | Same |
| `min_p` | Same |
| `typical` (typ_s) | Same |
| `tfs` (tfs_s) | Same |
| `presence_penalty` | Per-request only |
| `sampler_order` | Per-request only |
| `top_a` | Per-request only |
| `dynatemp_range` / `dynatemp_exponent` | Per-request only |
| `mirostat` / `mirostat_tau` / `mirostat_eta` | Per-request only |
| `smoothing_factor` | Per-request only |

## Adapter duality note

Two adapters run in parallel: `kobold_adapter.py` (legacy passthrough on `KOBOLD_PORT`) and `kobold_engine_adapter.py` (Phase E engine-orchestrated, on `KOBOLD_PORT+1`). Phase 2 PATCH/GET work must land in **both** for parity until the legacy adapter is retired. Today only the engine adapter handles `instruct_tags` in PATCH (`kobold_engine_adapter.py:330`); the legacy adapter's PATCH stops at `max_context_tokens`.

## Plan

### Phase 1 — bidirectional sync for fields that already have persona equivalents — SHIPPED

`portal.html:16669` `persist_sampler_settings_to_persona` exists and is hooked at the end of `confirm_settings` (line 17010). PATCHes:

```
localsettings.temperature        → temperature
localsettings.top_p              → top_p
localsettings.top_k              → top_k
localsettings.max_length         → max_tokens
localsettings.max_context_length → max_context_tokens
```

**Known gap (addressed in baseline-guard followup):** unconditional PATCH on every save. A fresh tab with default kobold sliders that opens the settings popup and clicks Save will overwrite a tuned persona with defaults. Fix: cache the persona's loaded sampler values in a baseline (set in `applyPersonaToUI`); skip PATCH when no tracked field differs from baseline.

### Phase 2 — promote ungoverned kobold extras to persona fields — SHIPPED 2026-05-01

Option **B** locked: kobold sampler extras (`rep_pen`, `rep_pen_range`, `rep_pen_slope`, `min_p`, `typical`, `tfs`) now persona-stored via `provider_extras["kobold"]`. Niche knobs (`mirostat`, `dynatemp`, `smoothing_factor`, `sampler_order`, `top_a`, `presence_penalty`) intentionally left kobold-only.

**As-shipped:**

| # | Item | As shipped |
|---|------|------------|
| 1 | Shared helper module | New `src/interfaces/_persona_patch.py` — exports `_KNOWN_PATCH_KEYS_LEGACY` / `_KNOWN_PATCH_KEYS_ENGINE`, `_apply_kobold_sampler_extras(persona, data, rejected)`, `get_kobold_extras_for_get(persona)`. Both adapters import from here so PATCH/GET stays in lockstep. |
| 2 | PATCH (both adapters) | Each of the six sampler keys → `Persona.set_provider_extra("kobold", k, coerced)`. `None` / `"clear"` / `""` → `clear_provider_extra`. Coercion failure (`ValueError`/`TypeError`) appends the key to `rejected_fields` and leaves prior value intact. Legacy adapter also gained `instruct_tags` PATCH (was engine-only). |
| 3 | GET (both adapters) | New `kobold_extras` block — only includes keys actually set on the persona; absent keys omitted so the portal can distinguish unset from set. Legacy adapter also added `instruct_tags` to GET (was engine-only). |
| 4 | Portal `persist_sampler_settings_to_persona` | Body extended with the six kobold keys read from `localsettings.*`. |
| 5 | Portal `applyPersonaToUI` | Reads `window._derpr_active_persona.kobold_extras` (stashed by `onPersonaChange`) and pushes each set key into `localsettings.*` + slider DOM via `setBoth`. |

### Phase 3 — drift guards — SHIPPED 2026-05-01 (server only)

**As-shipped:**

| # | Item | As shipped |
|---|------|------------|
| 1 | Unknown-field tracking | Both PATCH routes compute `unknown = sorted(set(data.keys()) - _KNOWN_PATCH_KEYS)`, log a warning, and return them in `unknown_fields` on the JSON response (alongside `rejected_fields`). |
| 2 | Baseline guard | `persist_sampler_settings_to_persona` skips PATCH when no tracked field differs from the baseline cached by `applyPersonaToUI` in `window._derpr_persona_sampler_baseline`. Baseline refreshes after a successful PATCH. Prevents fresh-tab default-slider overwrite. |
| 3 | Client-side mismatch warning | **Deferred.** Plan called for a console.warn when `localsettings.*` differs from a freshly-loaded persona value, but in practice this fires legitimately on every persona switch where the user had tuned sliders for the *previous* persona. Server-side `unknown_fields` already catches the regression class this was meant for (frontend sending keys the backend doesn't know). |

## Tests (as-shipped)

`tests/interfaces/test_kobold_adapter.py` — 5 new (uses `KoboldEngineAdapter`):
- `test_patch_persona_writes_kobold_sampler_extras` — all six keys land in `provider_extras["kobold"]`.
- `test_patch_persona_clear_kobold_extra_via_none` — `None` / `"clear"` clears the key.
- `test_patch_persona_kobold_extra_bad_input_rejected` — non-coercible input → `rejected_fields`, prior value retained.
- `test_patch_persona_unknown_field_returned` — unknown keys round-trip into `unknown_fields`.
- `test_get_persona_includes_kobold_extras` — GET surfaces only set keys.

699/699 unit+integration pass; mypy delta 0.

## Notes

- `instruct_tags` (and `instruct_gentag` via `assistant_gen` mapping) sync via `confirm_chat_and_instruct_tags` → `persist_instruct_tags_to_persona`. Legacy adapter's PATCH/GET now also handles `instruct_tags` (was engine-only).
- `max_context_length`: persona is authoritative — kobold-lite's OAI payload (line ~20990) doesn't include it. Engine falls back to `persona.max_context_tokens`.
- Niche knobs deferred: `mirostat`, `dynatemp`, `smoothing_factor`, `sampler_order`, `top_a`, `presence_penalty`. Promote later if usage justifies it.
