---
name: Supply chain attack — TeamPCP / litellm / Trivy (March 2026)
description: Major supply chain attack compromised litellm PyPI package via Trivy GitHub Actions — affects dependency decisions
type: reference
---

**Date:** March 2026

**Kill chain:** hackerbot-claw (AI-powered bot) exploited Trivy's GitHub Actions workflows -> stole org credentials -> TeamPCP poisoned 76/77 Trivy action tags -> compromised Trivy action in litellm's CI stole PYPI_PUBLISH token -> two backdoored litellm versions (1.82.7, 1.82.8) published to PyPI for ~5.5 hours on March 24, 2026.

**Malware impact:** Credential harvesting (SSH, cloud, K8s, DB, crypto wallets), lateral movement via Kubernetes, persistent backdoor polling every 50 minutes.

**Scope:** litellm had ~3.4M daily downloads. Google ADK was affected (unpinned litellm dependency). Netflix and other large companies reportedly impacted.

**Current status (2026-03-27):** Compromised versions removed from PyPI. v1.82.6 is safe. New releases paused pending pipeline review. Google Mandiant engaged.

**Relevance to this project:**
- OpenViking supports litellm as an optional provider (not a dependency). Minor risk that OpenViking devs use litellm internally and may have had credentials exposed, but no direct supply chain risk to installing OpenViking itself.
- General lesson: always pin dependencies to specific versions, never `>=X` without upper bound
- The broader TeamPCP campaign hit 5 ecosystems in one month (Trivy, Checkmarx, LiteLLM, KICS, Telnyx)
