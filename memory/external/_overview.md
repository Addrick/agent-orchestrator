---
name: External references overview (L1)
description: Pointers to external systems and resources — repo URL, API limits, etc.
type: reference
---

- **GitHub repo:** Addrick/llm-orchestrator (renamed from derpr-python). URL: https://github.com/Addrick/llm-orchestrator.git
- **Google rate limits:** Free-tier RPM/RPD per model family documented in `google_rate_limits.md`, encoded in global_config.py
- **Supply chain attacks:** TeamPCP campaign (March 2026) compromised litellm PyPI package via Trivy GitHub Actions. OpenViking supports litellm as optional provider (not a dependency). See `supply_chain_attacks.md` for full kill chain.

- **Claude Code config:** User-level settings in `~/.claude/settings.json`. Autocompact at 90% via `env.CLAUDE_AUTOCOMPACT_PCT_OVERRIDE`. See `claude_code_config.md` for full config and scope precedence.

For detail: read individual files in this directory.
