---
name: Test suite reorganization
description: Completed 2026-03-18 — split into 4 tiers (unit, integration, zammad-live, llm-live)
type: project
---

Test suite reorganized into 4 tiers (2026-03-18):
1. **Unit** (no marker) — 170+ tests, everything mocked
2. **Integration** (`@pytest.mark.integration`) — multi-component with mocked externals
3. **Zammad Live** (`@pytest.mark.zammad_live`) — requires live Zammad
4. **LLM Live** (`@pytest.mark.llm_live`) — requires API keys

Infrastructure:
- `tests/conftest.py` loads `.env.test` (override=True) and auto-skips live tests
- `.env.test` (gitignored) stores test Zammad credentials at `http://10.0.0.70:8081`
- Pre-commit hook runs `pytest -m "not zammad_live and not llm_live"`
- All config in `pytest.ini`
