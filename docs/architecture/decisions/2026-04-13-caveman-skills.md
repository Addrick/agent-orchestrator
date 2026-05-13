---
name: Global Caveman Skill Installation
date: 2026-04-13
description: Decision to install the Caveman skill suite globally and enable "Always-on" mode to optimize token usage.
type: decision
---

# Decision: Global Caveman Skill Installation

The user (Adam) values token efficiency and technically accurate but terse communication. We decided to install the [Caveman](https://github.com/JuliusBrussee/caveman) skill suite globally to assist with this across all projects.

## Logic / Rationale

- **Efficiency**: Standard LLM outputs contain significant filler ("Sure!", articles, pleasantries). Caveman style reduces output tokens by ~75%.
- **Global Scope**: Preference for terseness applies across all of Adam's projects, regardless of specific codebase.
- **Convenience**: Manual activation every session is friction. Automatic activation ensures consistent behavior.

## Installation Details

- **Path**: `~/.gemini/antigravity/skills/`
- **Skills**:
  - `caveman`: Terseness logic (lite/full/ultra/wenyan).
  - `caveman-commit`: Terse, conventional-commit messages.
  - `caveman-review`: Actionable, one-line PR feedback.
  - `caveman-help`: Quick-reference map.

## Configuration (The "Always-on" Choice)

Instead of using environment variables (`CAVEMAN_DEFAULT_MODE`) or a separate config file, we modified the `caveman/SKILL.md`'s metadata description.

- **Choice**: Set description to "Always-active... Auto-activates on session start."
- **Reason**: Antigravity agents scan skill metadata at startup. Explicitly stating "Always-active" in the metadata ensures the agent loads the skill instructions automatically, avoiding the need for external OS-level configuration or "overkill" environment variables.

## Verification

- Verified directory structure and file existence.
- Verified semantic loading by confirming the agent understands its always-on mandate.
