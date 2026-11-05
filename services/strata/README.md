# `services/strata` — CT102's Strata, the parts kept in the repo (DP-405B)

Strata itself lives on CT102 at `/opt/strata/Strata` (not in this repo). Its unit is
`/etc/systemd/system/strata.service` plus the per-model drop-in (`dau-abl-tb.conf` today).
This directory holds what DP-405B adds on top.

## Keep the live conversation across a restart

Strata keeps its conversation cache in RAM, so every restart (update, model swap, node reboot) made
the next turn of the conversation in progress re-read its whole prompt: about 50 s at 130k tokens on
the R9700. With `slot_save_path` set (CT102: `/opt/strata/slots`, since 2026-11-04), Strata's
llama-server-style `POST /slots/0?action=save|restore {"filename"}` writes the live conversation to disk
and reads it back.

| File | On CT102 at | What it does |
|---|---|---|
| `strata-slot-persist` | `/opt/strata/strata-slot-persist` (mode 755) | `save`: saves the live conversation as `shutdown-live-<model>.bin`, writing under `.part` and renaming once Strata answers 200. `restore`: waits for the model to load, restores the file and renames it `.restored`. Never fails, never blocks. |
| `slot-persist.conf` | `/etc/systemd/system/strata.service.d/slot-persist.conf` | `ExecStop=… save` (runs while the server is still up) and `TimeoutStopSec=100`. Also `Wants=strata-restore.service`. |
| `strata-restore.service` | `/etc/systemd/system/strata-restore.service` | oneshot `… restore`, `After=`/`PartOf=strata.service`. It is a separate unit because an `ExecStartPost` waiting out a 2-5 min load would exceed strata's 90 s `TimeoutStartSec`. |

Measured on CT102 (engine 0.1.52, 2026-11-04):
- Save: 12k tokens in 0.45 s (419 MB), 52k in 0.79 s (1.03 GB), 136k in 1.59 s (2.31 GB).
- Restore: 0.42 s, 1.09 s and 2.48 s for the same three sizes.
- After a restore, the same prompt reused all but the last 7 tokens.

**Limits:**
- Only the live conversation is saved. Conversations parked in the RAM cache are still lost on restart.
- A crash saves nothing, because `ExecStop` finds no server. The previous file is then the stale one, and it was already renamed `.restored`, so it is never restored twice.
- The model id is in the file name, so a restart on another model ignores the file.

### Install (by hand; CI does not deploy CT102)

From the node (`ssh -i ~/.ssh/derpr-host root@10.0.0.71`), with the three files copied to the node's `/tmp`:

```bash
pct push 102 /tmp/strata-slot-persist /opt/strata/strata-slot-persist --perms 755
pct push 102 /tmp/slot-persist.conf /etc/systemd/system/strata.service.d/slot-persist.conf
pct push 102 /tmp/strata-restore.service /etc/systemd/system/strata-restore.service
pct exec 102 -- systemctl daemon-reload
```

The next stop saves and the next start restores; nothing restarts now. To check: `systemctl stop strata`
should leave `/opt/strata/slots/shutdown-live-<model>.bin`, and after `systemctl start strata` the
journal of `strata-restore` should say `restored`. Do that check only when a restart is wanted anyway, since
it is an outage.

**Rollback:** `rm` the drop-in and `strata-restore.service`, then `systemctl daemon-reload`.

## Not built: saving a preempted Hindsight job

See `services/inference-router/README.md` → "Background work". A job sent with
`strata_checkpoint: false` cannot be saved (`400 no complete session to save`), and dropping the flag
brings back the chat evictions it exists to prevent. The trade-off is waiting on a decision.
