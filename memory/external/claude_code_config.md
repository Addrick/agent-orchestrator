---
name: Claude Code configuration
description: Claude Code CLI settings and environment variable overrides — autocompact threshold, model, effort level
type: reference
---

## User-level settings (`~/.claude/settings.json`)

- `model`: "opus"
- `effortLevel`: "high"
- `autoUpdatesChannel`: "latest"
- `env.CLAUDE_AUTOCOMPACT_PCT_OVERRIDE`: "90" — triggers context compaction at 90% instead of default ~95%

## How `env` works

The `env` key in settings.json **merges on top of** the inherited shell environment — it does not replace it. Only the listed variables are added/overridden; PATH, CUDA, etc. remain intact.

## Scope precedence

Managed > Command-line > Local (`.claude/settings.local.json`) > Project (`.claude/settings.json`) > User (`~/.claude/settings.json`)
