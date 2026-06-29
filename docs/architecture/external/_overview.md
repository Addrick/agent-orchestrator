---
name: External references overview (L1)
description: Pointers to external systems and resources — repo URL, API limits, etc.
type: reference
---

- **GitHub repo:** Addrick/agent-orchestrator (renamed from derpr-python). URL: https://github.com/Addrick/agent-orchestrator.git. GHCR package `orchestratr` (was derpr-python, DP-259).
- **Google rate limits:** Free-tier RPM/RPD per model family documented in `google_rate_limits.md`, encoded in global_config.py
- **Supply chain attacks:** TeamPCP campaign (March 2026) compromised litellm PyPI package via Trivy GitHub Actions. OpenViking supports litellm as optional provider (not a dependency). See `supply_chain_attacks.md` for full kill chain.
- **Hindsight upstream API quirks:** Field names (`retain_mission` vs deprecated `mission`), retain bundle shape, observations tier, stats endpoint. See `hindsight_upstream_api.md`.

For detail: read individual files in this directory.
