---
name: Persona config deployment model
description: Which persona config files are tracked vs gitignored, and how they seed production
type: project
---

- `config/default_personas.json` — tracked in git, seeds prod deployments (AWS)
- `config/system_personas.json` — tracked in git, seeds prod deployments (AWS)
- `data/personas.json` — gitignored, local dev only, holds runtime state; overrides defaults on startup

Edit `default_personas.json` or `system_personas.json` for production changes, not `personas.json`.
