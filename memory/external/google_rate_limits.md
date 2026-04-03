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
