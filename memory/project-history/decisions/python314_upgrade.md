---
name: Python 3.14 upgrade
description: Completed 2026-03-20 — jumped from 3.11/3.12 to 3.14.3 stable
type: project
---

Completed 2026-03-20. Jumped from 3.11/3.12 straight to 3.14.3 (stable).

- System PATH updated to Python 3.14
- Dockerfile bumped to `python:3.14-slim`
- CI workflow bumped to `python-version: ["3.14"]`
- Requirements recompiled with `pip-compile --upgrade`
- All 235 tests pass

**Upstream deprecation warnings (not blocking):**
- `google-genai`: uses `_UnionGenericAlias`, deprecated in 3.14, removal in 3.17
- `discord.py`: uses `asyncio.iscoroutinefunction`, deprecated in 3.14, removal in 3.16
