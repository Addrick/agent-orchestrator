---
name: Claude Code configuration
description: Claude Code CLI settings and environment variable overrides — autocompact threshold, model, effort level, venv setup
type: reference
---

## User-level settings (`~/.claude/settings.json`)

- `model`: "opus"
- `effortLevel`: "high"
- `autoUpdatesChannel`: "latest"
- `env.CLAUDE_AUTOCOMPACT_PCT_OVERRIDE`: "90" — triggers context compaction at 90% instead of default ~95%

## How `env` works

The `env` key adds/overrides specific variables in the inherited shell environment. Variables not listed remain intact. **Do not set `PATH` here** — it replaces the inherited PATH entirely and breaks things. Use `BASH_ENV` instead for venv activation.

## Scope precedence

Managed > Command-line > Local (`.claude/settings.local.json`) > Project (`.claude/settings.json`) > User (`~/.claude/settings.json`)

## Project-local settings (`.claude/settings.local.json`)

- `env.BASH_ENV`: `/c/Users/adama/PycharmProjects/derpr-python/.venv/Scripts/activate` — auto-activates the project venv before every bash command. This is the correct way to expose venv tools (pytest, flake8, mypy) to Claude without touching PATH.

## Windows machine setup (non-desktop)

On the laptop, Python entries must be manually ordered above the Windows Store entries in the user PATH (via Environment Variables in System Properties). Windows Store installs a `python.exe` stub that shadows the real Python if it appears first. Once reordered, `python` resolves to 3.14 in PowerShell and Claude inherits it correctly.
