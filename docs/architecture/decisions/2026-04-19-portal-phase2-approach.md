---
name: Portal Phase 2 architectural approach
description: DB-as-source via kobold savefile + author's note for LTM; rationale for rejecting server-side prompt rebuild and prompt-start LTM placement
type: project
---

**Decision (2026-04-19):** Phase 2 of the kobold-lite portal will use two mechanisms to integrate DERPR's memory system:

1. **DB-as-source via kobold's public savefile JSON.** When the user selects "DERPR Database" history mode, the server exports DERPR `message_history` as a kobold-lite savefile and the portal ingests it via kobold-lite's native `load_file` path. Passthrough then continues unchanged.
2. **LTM injection via kobold's `authornote` field** (not prompt-start, not server-side prompt splicing). When LTM generation is enabled, the server intercepts the outgoing request, runs DERPR's LTM retrieval over the latest user turn, and writes retrieved summaries into the `authornote` field of the forwarded payload. The original user `authornote` is backed up and restored across toggle transitions.

**Why:**

- **Rejecting server-side prompt rebuild (the Stage 1 deferred path).** The original plan was to extract the latest user turn, rebuild the full prompt through DERPR's ChatSystem using its own chat template, and forward to KoboldCPP. This was known-broken for local models because DERPR's rewrap lost kobold-lite's native thinking-tag markers (`<|channel>thought\n`, etc.) — they belonged to kobold's template, not DERPR's. The prerequisite tag-schema adapter work is substantial and fragile. The DB-as-source approach **sidesteps the problem entirely**: kobold-lite renders the prompt using its own template, DERPR never rewraps, and future kobold-lite updates remain compatible as a simple extension of existing passthrough behavior.

- **Rejecting prompt-start placement for LTM.** KoboldCPP exposes only llama.cpp's prefix cache — matching is left-to-right on identical tokens, with full recompute from the first divergence. There is no mid-prompt "reserved slot" or cache-hole API. Placing the LTM summary block at prompt start means every retrieval change invalidates the entire KV cache and forces a full recompute of the conversation on each turn. This is slow on local hardware. Author's note placement puts the mutable region near the end of the prompt, so only the last few hundred tokens recompute per turn — nearly free by comparison. Trade-off: recency bias in how the model weighs injected memory. Accepted as a reasonable cost, and matches a user-visible kobold feature the user already understands.

- **Rejecting direct DOM / internal-state injection.** Using kobold-lite's public savefile JSON format keeps DERPR coupled only to kobold's stable save/load contract, not to its in-memory data structures. Kobold-lite can change how it represents turns internally without breaking the exporter.

- **Rejecting DB round-trip for UI edits in this phase.** Load-once-on-toggle is simple and testable. Round-tripping UI edits back to the DB requires stable message-id tagging in the exported savefile and surviving kobold's own save/load cycle. Flagged as backlog; not worth the complexity until 2.1/2.2 are working.

**How to apply:**

- Adapter grows a `GET /api/v1/session/{persona}/kobold_export` route that builds a kobold savefile JSON from `message_history`. `max_turns` defaults to the persona's existing history-limit setting — do not introduce a new config key for this.
- Tool-call and tool-result rows are skipped during export (logged for audit). Tool support is backlog.
- The portal's existing disabled History Override checkbox is replaced by a two-state toggle labeled "Kobold Native" / "DERPR Database" with a sub-checkbox "LTM Generation" that is only enabled when the toggle is on the DERPR side.
- When LTM injection is active, the author's note region is marked as DERPR-managed in the UI. The user's prior authornote is preserved and restored on toggle-off.
- Split shipping: Phase 2.1 lands DB-as-source (LTM checkbox inert), Phase 2.2 lands LTM injection. Two PRs, two sessions.

---

## Deferred / flagged for future work

- **Per-response memory provenance UI** ("show thoughts" for memories). Requires `response_id` threaded through SSE and a dedicated endpoint. Not needed for 2.1/2.2 correctness.
- **Edit/delete sync between kobold UI and DERPR DB.** Currently one-way at load. Backlog.
- **LTM block placement experiment.** If upstream llama.cpp lands chunked prefill / semantic cache, prompt-start placement becomes cheap and may yield better quality than author's note.
- **Tag-schema adapter.** Was the prerequisite for the rejected server-side-rebuild path. No longer on critical path; monitor if future requirements reopen the question.
