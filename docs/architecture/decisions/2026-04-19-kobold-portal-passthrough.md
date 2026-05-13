---
name: Kobold portal Stage 1 passthrough
description: Web UI Stage 1 scope — verbatim kobold-lite passthrough to local KoboldCPP, with Override mode deferred and tag-schema work flagged for later
type: project
---

**Decision (2026-04-19):** `kobold_adapter.py` Stage 1 implements pure verbatim passthrough of kobold-lite prompts to local KoboldCPP (`LOCAL_LLM_URL`). No extraction, no re-rendering, no ChatSystem involvement on the default path.

**Why:** Prior implementation ran every request through `ChatSystem.stream_response` → persona `chat_template`, which rewrapped kobold-lite's already-rendered prompt in a different template (chatml/gemma/llama3). Kobold-lite's native thinking-tag markers (e.g. `<|channel>thought\n`) belonged to kobold's template, not DERPR's — gluing them on via `assistant_prefill` produced corrupted output. User only cares about `local` model for now; cross-template reasoning support is deferred.

**How to apply:**
- Default adapter request path = verbatim HTTP forward of `{prompt, params, ...}` to `{LOCAL_LLM_URL_base}/api/extra/generate/stream` (or `/api/v1/generate` for sync). Relay SSE back unchanged.
- Persona sampling params are pushed from backend into kobold-lite UI sliders via the frontend `applyPersonaToUI()`; the UI's outgoing params are then forwarded unmodified. No server-side merging.
- Persona system prompt likewise baked into kobold-lite's `memorytext` frontend-side. Server does not inject system content in passthrough.
- Abort: adapter must close its upstream `httpx` stream AND POST to KoboldCPP's `/api/extra/abort` to stop generation server-side.

---

## Deferred work / known limitations

**1. `history_override` toggle stubbed in Stage 1.**
- Frontend toggle was removed because the name is ambiguous and the feature only works partially. A future toggle should be split into explicit options: "Use kobold-lite text buffer as prompt" vs "Rebuild prompt from DERPR user_db (L0 + LTM)".
- The Override path (extract last user turn → `ChatSystem` rebuild) was known-broken for local models because the rewrap lost kobold's thinking tags. Re-introducing it needs the tag-schema work below.

**2. Tag-schema system for local adapter is a separate design job.**
- Local KoboldCPP requires specific system-formatting tags to function correctly (chat template markers, thinking triggers). These vary per model.
- Future: a "local" adapter with schema selection (chatml / gemma / llama3 / deepseek / gpt-oss / etc.), letting DERPR-rebuilt prompts target the correct template. This is prerequisite for a functional History DB Override mode on local models.
- Other providers (OpenAI, Anthropic, Gemini) have their own reasoning mechanisms and don't need this.

**3. Proxy vs Override spec terminology diverged from implementation.**
- Spec (`docs/plans/...`) uses `raw_passthrough`. Implementation settled on `history_override` because it's more self-documenting in the UI context. When tag-schema work resumes, prefer explicit names reflecting the *source* of prompt text (kobold_buffer vs derpr_db) over passthrough/override terminology.

**4. `assistant_prefill` mechanism removed.**
- Was an overengineered bridge to carry kobold's reasoning-mode triggers through ChatSystem's rewrap. Solved the wrong problem. If local-model reasoning support is revisited, build it into the tag-schema adapter directly, not as a generic prefill field on `inference_config`.
