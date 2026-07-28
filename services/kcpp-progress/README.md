# kcpp-progress

Serves KoboldCPP's **live prompt-ingestion progress** over HTTP, for the portal
statusline (DP-311).

## Why this exists

`GET /api/extra/perf` cannot report ingestion progress. Every `last_*` field it
returns is frozen for the duration of a generation and only updates on
completion — measured, not assumed. The per-batch counter exists **only** on
KoboldCPP's stdout:

```
Processing Prompt [BATCH] (2048 / 24310 tokens)
Generating (17 / 400 tokens)
```

The engine runs on a different host than the model, so it cannot read that log
directly, and giving the engine SSH into the model host crosses the project's
tool-security line (open egress + credential inheritance). Hence a small
read-only sidecar next to KoboldCPP.

## ⚠️ KoboldCPP must be run unbuffered

Progress records end in **CR, not LF**, and KCPP's stdout is redirected to a file
(no tty), so C stdio buffers them fully: without this fix the updates arrive in
20-30 second clumps, which is useless for a live bar. Wrap `ExecStart` in
`stdbuf -o0`:

```ini
# /etc/systemd/system/koboldcpp-<unit>.service.d/unbuffered.conf
[Service]
ExecStart=
ExecStart=/usr/bin/stdbuf -o0 -e0 /opt/koboldcpp/koboldcpp --model … (full original args)
```

Byte volume is unchanged — the same characters are written, just flushed per
write instead of per buffer. Measured on CT101 with a 24,310-token prefill:

| | updates | arrival |
|---|---|---|
| buffered (before) | 12, in **2 clumps** | 20-30s late |
| `stdbuf -o0` (after) | 12, one per 2048 tokens | **~3.5s apart, live** |

Verify the preload took: `tr '\0' '\n' < /proc/$(pgrep -f koboldcpp)/environ | grep -i preload`
→ `LD_PRELOAD=/usr/libexec/coreutils/libstdbuf.so`.

## Install (on the KoboldCPP host, e.g. CT101)

```bash
install -D -m 755 kcpp_progress.py /opt/kcpp-progress/kcpp_progress.py
install -D -m 644 kcpp-progress.service /etc/systemd/system/kcpp-progress.service
systemctl daemon-reload && systemctl enable --now kcpp-progress
curl -s localhost:5011/progress
```

Then point the engine at it:

```
KOBOLD_PROGRESS_URL=http://10.0.0.72:5011
```

Unset, the engine's `/api/extra/prefill` reports `{"available": false}` and the
statusline silently falls back to last-completed counters. Nothing breaks where
the sidecar is not deployed (e.g. the dt21 kobold farm).

## API

`GET /progress`

```json
{"phase": "prefill", "processed": 8192, "total": 24310,
 "generated": 0, "generate_total": 0, "age_s": 0.4, "run": 3, "source": "log"}
```

`phase` is `prefill` | `generate` | `idle`. A run whose output goes silent for
30s reverts to `idle` — the sidecar sees only output, so a crashed or aborted run
would otherwise pin the bar forever. `run` increments on each new prefill, so a
consumer can distinguish "same run, further along" from "a new run".

`GET /healthz` → `{"ok": true}`.

## Security

**Integers only.** The service never stores or returns raw log lines — the log
carries generated text, and prompts can appear in error paths. Records are
matched with anchored regexes and discarded; unmatched input is dropped. The
systemd unit runs it read-only against `/var/log/kobold` with `ProtectSystem=strict`
and `NoNewPrivileges`.

**Do not add a "recent log lines" debug endpoint.** That is the one change that
would turn this from a counter into a content leak.

The port is unauthenticated on the LAN, matching KoboldCPP's own posture on the
same host. It exposes token *counts*, never content. Do not expose it beyond the
LAN.
