---
name: Google free-tier rate limits
description: Confirmed RPM/RPD limits per model family and grounding quota — encoded in global_config.py
type: reference
---

Confirmed free-tier rate limits (2026-03):
- Gemini 2.5 (Flash, Flash-Lite): 5 RPM, 20 RPD
- Gemini 3.1 Flash: 15 RPM
- Gemma 3 27b-it: 30 RPM, 14,400 RPD

Google Search Grounding quota:
- Gemini 2.5: ~500 (functionally unlimited)
- Gemini 3.1: 0 (not available)
- Gemma: not supported

Encoded in `config/global_config.py` as `RATE_LIMIT_GEMINI_25_RPM`, etc. Split limiters by model family to avoid unnecessary throttling.

## gemini-embedding-001 (free tier)

- 100 items/minute (each item in a `batchEmbedContents` call counts as 1 RPM request)
- 30,000 input tokens/minute
- Per-input cap: 2048 tokens (enforced by `EmbeddingService._truncate_texts`)

## "Invisible" 429s — per-call limit rejections

A single `batchEmbedContents` request whose total input exceeds the 30k tokens/minute budget is rejected at the gate. The API returns a 429 whose `details` payload contains only a `Help` link — no `RetryInfo`, no `QuotaFailure.violations` — and the rejected request does **not** show up in the AI Studio usage pane (it was never admitted to the quota counter).

This was the root cause of the long-standing memory-agent stall: the agent was sending one large batch per cycle and the entire batch was being rejected, not throttled. The fix shipped in commit `ac78ecd` (2026-04-06) was token-aware chunking in `MemoryAgent._chunk_messages` that caps each outbound batch at ~25k estimated tokens (chars/4).

When diagnosing a 429 with this signature, look at the *total token size of the rejected call*, not RPM history. Adam has confirmed he has not hit burst-pattern / edge-protection 429s on this project.
