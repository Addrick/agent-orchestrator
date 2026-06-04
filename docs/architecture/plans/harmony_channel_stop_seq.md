# Plan: Harmony / Qwen3 channel-aware stop sequences

> **SHIPPED (reconciled 2026-06-04).** Option A landed in `src/stream_engine.py:31-38` (strips bare `<|im_end|>`), with an in-file comment citing this plan.

DP-ID: TBD

## Why

Local Qwen3 instruct models emit harmony channels inside a single
assistant turn:

```
<|im_start|>assistant
<|channel|>thinking<|message|>...<|im_end|>
<|channel|>tool_call<|message|><tool_call>{...}</tool_call><|im_end|>
<|channel|>final<|message|>...<|im_end|>
<|im_start|>user
```

The current ChatML template (`src/stream_engine.py:25-32`) sets
`stop: ["<|im_end|>", "<|im_start|>"]`. kcpp halts on the FIRST
`<|im_end|>` it sees — which is the boundary between the first channel
(usually `thinking` or a chatty lead-in) and the next one. The model
intended to emit `<tool_call>...</tool_call>` immediately after but never
got there.

Observed 2026-05-17 (`logs/sse_dumps/derpr_sse_20260517T201954_*.log`):
model output 107 chars ending with `"Now let's try **recall_memory**"`
then a clean stream close. No error, no cancel. The next channel boundary
ate the rest of the turn.

This compounds with Phase E (channel-marker leak): even when the parser
correctly strips `<|channel|>` tokens server-side, the stop sequence
prevents the model from generating the channels we want to keep.

## Goal

Let the model emit multi-channel responses without prematurely halting,
while still stopping cleanly at end-of-turn (`<|im_start|>user`) and on
EOS.

## Non-goals

- Generic harmony-format parsing on the engine side. The existing
  `_ToolCallStreamParser` plus the Phase E `<|...|>` stripper already
  cover what the portal needs to see.
- Per-model template auto-detection. We do that crudely already; this
  plan reuses `template_name` from kobold-lite's `instruct_*` block.
- Streaming-mid-channel UI rendering (e.g., show thinking channel as
  reflective-process). Out of scope; the tool_trace plan handles
  tool channels via SSE events instead.

## Design

### Option A — drop bare `<|im_end|>` from ChatML stop list

```python
"chatml": {
    ...
    "stop": ["<|im_start|>user", "<|im_start|>system"],
}
```

Rely on the model's EOS token (configured at the kcpp side) for normal
end-of-turn termination. `<|im_start|>user` catches the case where the
model continues into a new role.

Pros: minimal change, fixes Qwen3 immediately, no template fork.
Cons: a model that fails to emit EOS and just stops at `<|im_end|>` will
now hang until `max_tokens` — but kcpp's `bypass_eos: false` keeps EOS
working. Non-harmony ChatML models still work because their `<|im_end|>`
coincides with an EOS token anyway.

### Option B — fork a `chatml_harmony` template

```python
"chatml_harmony": {
    "system":    "<|im_start|>system\n{content}<|im_end|>\n",
    "user":      "<|im_start|>user\n{content}<|im_end|>\n",
    "assistant": "<|im_start|>assistant\n{content}<|im_end|>\n",
    "assistant_start": "<|im_start|>assistant\n",
    "stop": ["<|im_start|>user", "<|im_start|>system", "<|im_start|>tool"],
},
```

Auto-select based on the model name (`"qwen"`, `"qwq"`, `"harmony"` in
the `/api/v1/model` response) OR via a persona-level
`template_override: "chatml_harmony"` field. Default ChatML behavior
unchanged for models that legitimately use `<|im_end|>` as a hard stop.

Pros: surgical, opt-in, preserves behavior for non-harmony models.
Cons: more code, requires detection logic.

### Recommendation

**Option A.** The non-harmony case where a model emits `<|im_end|>`
without an EOS token is unusual; kcpp's EOS handling is already the
canonical stop signal. Bare `<|im_end|>` as a stop string is a
belt-and-suspenders relic from when EOS pass-through was unreliable.

Validate by:

1. Toggling the stop list change behind an env var
   (`DERPR_CHATML_STRICT_STOP=1` to keep old behavior) for a short bake.
2. Sweeping the recent `logs/sse_dumps/*` for turns that ended at exactly
   `<|im_end|>`-equivalent length and confirm they were Qwen3 cutoffs,
   not legitimate completions.
3. Running the existing engine tests; add one that asserts the stop
   list no longer contains bare `<|im_end|>`.

If the bake surfaces a non-harmony model hanging, escalate to Option B.

## Phases

| Phase | Scope | Files |
|---|---|---|
| **A** | Remove bare `<|im_end|>` from `CHAT_TEMPLATES["chatml"]["stop"]`. Add `<|im_start|>user` / `<|im_start|>system` instead. Add unit test asserting the new stop set. | `src/stream_engine.py`, `tests/test_stream_engine.py` |
| **B** | Capture a fresh SSE dump from the same prompt that previously truncated. Confirm the model now generates through the tool_call channel cleanly. | (manual) |
| **C** | If Option A causes regressions, add Option B's `chatml_harmony` template + detection on model name. | `src/stream_engine.py` |

## References

- `logs/sse_dumps/derpr_sse_20260517T201954_*.log` — concrete cutoff
  trace. 107-char final_text ending at `**recall_memory**`.
- `docs/architecture/plans/portal_tool_trace_ui.md` Phase E — the
  sibling channel-marker-strip work; this plan is its inverse on the
  output side.
- Qwen3 harmony format reference (external; cite the model card when
  filing the DP ticket).
