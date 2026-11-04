# AGENTS.md: how you work in this workspace

Written by DERPR (DP-488) and overwritten on its next start, like `.opencode/`. Edits made
here are lost. To change these rules, ask Adam.

You work offline and alone. DERPR's task queue gives you one ticket at a time, runs a
hidden check on your work when you end your turn and, if it passes, merges it. Each rule
below exists because skipping it once cost real time.

## The workspace

```
AGENTS.md              these rules
memory/                your notes, a git repo of its own
projects/<project>/    one git repo per project; its main branch holds finished work
worktrees/<ticket>/    the ticket's worktree, branch ticket/<ticket>, cut from main
```

## Where you work

- **Work only in your ticket's worktree**, `worktrees/<ticket>/`. Never edit a project's
  main checkout in `projects/<project>/` or another ticket's worktree: DERPR merges your
  branch into main when the check passes, and a change made anywhere else is not part of
  the merge.
- **Commit in the worktree, staging files by path** (`git add path/to/file`). Never
  `git add -A` or `git add .`: they sweep in scratch files, databases and caches. Check
  `git status` before you commit; it should list only what you meant to change.
- **Ignore what builds and tests leave behind.** When the check passes, DERPR commits
  everything left in your worktree before it merges, except what `.gitignore` covers. A
  new project's `.gitignore` already covers Python and Node caches, virtualenvs and
  `*.db`; when your work makes some other output (a build folder, a data file), add it
  to `.gitignore` in the same commit.
- **Run the tests inside the worktree**, from its top folder, the way the check will.
  Tests run from anywhere else test different files.
- There is no remote: no fetch, no push, no pull requests. The box has no internet, so
  nothing can be installed. If a package you need is missing, say so with `notify_adam`
  instead of working around it.
- If the ticket says the previous attempt's merge conflicted, merge the project's main
  branch into your branch in the worktree, resolve the conflicts, run the tests and commit.

## How you work

- **Read your notes before you start**: `memory/MEMORY.md`, then the `_overview.md` of
  any directory that looks relevant, then `memory/tickets/<ticket>.md`. Earlier tickets and
  other projects are in `memory/` and `projects/`; look there when you need them.
- **Fix the root cause, not the symptom**, even when a workaround is faster. Every fix
  ends with the test that stops it coming back, or a note saying why there can't be one.
- **Before building something, look for what already does that job.** Search by what the
  code does (what else stores, retries, parses, schedules?), not by a name you expect;
  code that does the same job often has a different name.
- **Stop and ask.** After 3 failed fixes for one problem, or if you can't say in one
  sentence what is broken and why, stop. Send Adam a concrete question with
  `notify_adam`: what you tried, what you saw, what you need to know. "I'm stuck" is not a
  question.
- **Never report something as verified unless you ran it** and saw the result. Say what
  you ran. "Should work" is not "works".
- **Update docs in the same commit as the code** they describe (README, usage notes,
  comments that state behaviour).

## Your notes: tiered memory

`memory/` is your long-term memory. Read it top-down, write it bottom-up.

- `memory/MEMORY.md` (L0): an index only, one line per directory. Never put notes in it.
- `memory/<dir>/_overview.md` (L1): a short summary of the directory, usually enough to
  decide whether to read further.
- Detail files (L2), such as `memory/tickets/<ticket>.md` and
  `memory/projects/<project>.md`: read only when L1 isn't enough.

When you learn something worth keeping (why a decision was made, what a bug's root cause
was, how a project is laid out and tested, what failed and why), write the detail file
first, then update its directory's `_overview.md`, then `MEMORY.md` only if a directory's
summary changed. **Commit the notes in the same turn you write them**, staging by path
(`git -C memory add <path>` then `git -C memory commit`): a note that is not committed is
not memory.

- `memory/tickets/<ticket>.md` is your record of the ticket. DERPR creates it from the
  ticket text and adds one line after each attempt with its outcome. Add what you tried,
  what happened and what you decided.
- **A status line in a note is a claim, not evidence.** Before acting on "X is done" or
  "Y is broken", check it against the code, `git log` or a test run. When a note and the
  code disagree, the code wins: fix the note.
