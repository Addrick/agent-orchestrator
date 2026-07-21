---
name: remote-windows-ssh
description: Drive a remote Windows box's git-bash or PowerShell over SSH without fighting the quoting stack. Use when the shell host is one Windows box (e.g. dt21) and you must run commands on ANOTHER Windows box over ssh — omen (10.0.0.67), or any `ssh <host> "..."` where the far side is Windows + git-bash. Encodes the PS→ssh→cmd→bash quoting traps that silently mangle inline commands into garbage, and the script-file transport that sidesteps all of them.
---

# Driving a remote Windows box over SSH

You are on a Windows shell (PowerShell primary, Bash tool = git-bash) and need to run
commands on **another** Windows box over `ssh`. Every command crosses four shells, each
with its own quoting rules:

```
PowerShell  →  ssh  →  cmd.exe (remote default shell)  →  git-bash (bash.exe)
```

Inline one-liners get mangled at one of these boundaries almost every time. **The fix is
not better quoting — it is to stop inlining.**

## ⚠️ The one rule: ship a script file

For anything past a trivial single token, **write a `.sh`, `scp` it, run it by path.**
Zero inline quoting survives to break.

```powershell
# 1. Write locally (use the Write tool), then:
scp C:\path\local.sh omen:"C:/Users/adama/x.sh"
# 2. Run by PATH — no inner -c, no quotes to mangle:
ssh omen "C:\PROGRA~1\Git\bin\bash.exe /c/Users/adama/x.sh"
```

`C:\PROGRA~1\Git\bin\bash.exe` = the 8.3 short path for `C:\Program Files\Git\bin\bash.exe`,
dodging the space. Inside the `.sh`, quoting is normal bash — you control it fully.

This pattern produced correct output every time this session; inline `-c "..."` failed
repeatedly. When you catch yourself building `ssh host "bash -c \"...\""`, stop and write a file.

## The traps (why inline fails)

| Trap | Symptom | Fix |
|------|---------|-----|
| PS expands `$VAR` in **double**-quoted outer string | `$HOME` → the *local* box's path sent to remote (`C:UsersAdam...`) | Single-quote the PS outer string when it contains `$HOME`/`$x` meant for the remote |
| cmd.exe mangles inner `-c "quoted"` | infinite `$'\260e': command not found` spew, or empty output | Script-file transport (above). Never `bash -c "quoted"` over ssh→cmd |
| PS 5.1 has no `&&` / `\|\|` | `The token '&&' is not a valid statement separator` | Put the `&&` chain **inside** the `.sh`, or use `;`/`if ($?)` in PS |
| `bash` on PATH = **WSL**, not git-bash | `$HOME`=`/home/..`, scripts can't find files | Invoke git-bash by explicit path `C:\PROGRA~1\Git\bin\bash.exe`; never bare `bash` |
| `bash -lc` (login shell) over ssh+pipe | garbage/`$'\260e'` spew | Use non-login: `bash /path/x.sh`. Login profile is usually fine; the spew is the quoting layer, not a corrupt profile |
| PS blocks `< file` redirection | `The '<' operator is reserved for future use` | Feed stdin from inside the `.sh`, or `Get-Content x \| ssh ...` |
| Background task hang on interactive prompt | `ssh` sits forever (password / host-key prompt) | Add `-o BatchMode=yes -o StrictHostKeyChecking=accept-new`; kill with TaskStop if it backgrounds |

## Verified-working recipes

**Run a remote script, capture output:**
```powershell
scp C:\tmp\job.sh omen:"C:/Users/adama/job.sh" 2>&1 | Out-Null
ssh omen "C:\PROGRA~1\Git\bin\bash.exe /c/Users/adama/job.sh"
```

**One trivial remote command (no inner quotes needed):** cmd builtins are safe:
```powershell
ssh omen "dir C:\Users\adama\.claude /b"
ssh omen "del /q C:\Users\adama\tmp-*.sh"
```

**PS outer string with a remote `$HOME` — single-quote it:**
```powershell
ssh omen 'C:\PROGRA~1\Git\bin\bash.exe -c "echo $HOME"'   # $HOME NOT expanded by PS
```

**Read back / validate a pushed file:** `scp` it down and inspect locally rather than
crafting a remote `type`/`findstr` with nested quotes.

## Git / gh over SSH on Windows = 401 (separate trap)

Key-based SSH → a Windows **network** logon token → cannot unlock **DPAPI** → Credential
Manager / gh token read as "invalid". So `git clone/pull/push` and `gh` **fail over ssh
regardless of reauth at the keyboard.** Options:
- Run the git/gh command **at the remote machine's keyboard** (interactive logon), or
- `scp` the file(s) in from the shell host (what we do for one-off file updates), or
- a token in an env var (`GH_TOKEN`) bypasses DPAPI — but that is a broad plaintext
  secret; **omen's owner declined it**, so default to scp/keyboard.

See memory `infrastructure/devices/omen.md` for omen's specifics (user `adama`, ssh alias
`omen`, python3 shim, framework cutover).
