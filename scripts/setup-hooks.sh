#!/bin/sh
# One-time per-clone dev bootstrap: activate the tracked git hooks.
#
# Git will not honor a hooks directory committed to the repo unless each clone
# opts in (a security measure), so this must be run once after cloning. It is
# idempotent and cross-platform (POSIX sh — runs in git-bash on Windows and in
# the default shell on macOS/Linux):
#
#     sh scripts/setup-hooks.sh
#
# What the tracked hooks do (.githooks/):
#   - pre-commit:    writes .claude/.memory_update_pending (memory-update nudge)
#   - post-checkout: seeds gitignored .env/.env.test + builds the per-worktree
#                    .venv (uv) on new `git worktree add`
#   - pre-push:      runs the local CI gate (scripts/ci_check.py, ~90s); skip
#                    with `git push --no-verify` (DP-250)
#
# core.hooksPath is shared across worktrees, so setting it in the main clone is
# enough for every worktree of that clone.

set -e
cd "$(git rev-parse --show-toplevel)"
git config core.hooksPath .githooks
echo "core.hooksPath = $(git config --get core.hooksPath)  (tracked hooks active)"
