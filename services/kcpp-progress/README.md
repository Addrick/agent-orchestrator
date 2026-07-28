# kcpp-progress

Serves KoboldCPP's live prompt-ingestion progress over HTTP, for the portal statusline (DP-311).

No API has these counters — `/api/extra/perf`'s `last_*` fields are frozen for the duration of a run.
They exist only on KCPP's stdout (`Processing Prompt [BATCH] (2048 / 24310 tokens)`). The engine is
on a different host, and engine→model SSH crosses the project's tool-security line, so: a read-only
sidecar next to the model.

## ⚠️ KoboldCPP must run unbuffered

Progress records end in **CR, not LF**, and kcpp's stdout is a file, so stdio buffers them into
20-30s clumps — useless for a live bar. Add a drop-in re-declaring ExecStart under `stdbuf -o0 -e0`:

```ini
# /etc/systemd/system/koboldcpp-<unit>.service.d/unbuffered.conf
[Service]
ExecStart=
ExecStart=/usr/bin/stdbuf -o0 -e0 /opt/koboldcpp/koboldcpp --model … (repeat the FULL original argv)
```

Updates then land every ~3.5s, same byte volume. `systemctl show <unit> -p ExecStart` must list
exactly one — and changing a kcpp flag means editing both the unit and this drop-in.
Verify: `tr '\0' '\n' < /proc/$(pgrep -f koboldcpp)/environ | grep -i preload` → `…/libstdbuf.so`.

## Install (on the KoboldCPP host)

```bash
install -D -m 755 kcpp_progress.py /opt/kcpp-progress/kcpp_progress.py
install -D -m 644 kcpp-progress.service /etc/systemd/system/kcpp-progress.service
systemctl daemon-reload && systemctl enable --now kcpp-progress
curl -s localhost:5011/progress
```

Engine side: `KOBOLD_PROGRESS_URL=http://10.0.0.72:5011`. Unset, `/api/extra/prefill` reports
`available: false` and the statusline falls back to last-completed counters — nothing breaks where
the sidecar is absent.

## API

`GET /progress` → `{"phase": "prefill", "processed": 8192, "total": 24310, "generated": 0,
"generate_total": 0, "age_s": 0.4, "run": 3, "source": "log"}` · `GET /healthz` → `{"ok": true}`

`phase` = `prefill` | `generate` | `idle`; a run silent for 30s reverts to `idle` (the sidecar sees
output, not state, so a crashed run would otherwise pin the bar). `run` increments per prefill, so a
counter that jumps backwards is distinguishable from a new run.

## Security

**Integers only — never raw log lines.** The log carries generated text, and prompts appear on error
paths. Unmatched input is dropped; the unit runs `ProtectSystem=strict`, `NoNewPrivileges`, read-only
on `/var/log/kobold`. Unauthenticated on the LAN, matching kcpp's own posture on that host; do not
expose it further.

**Do not add a "recent log lines" endpoint** — that single change turns a counter into a content leak.
