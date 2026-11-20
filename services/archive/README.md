# `services/archive` — long-term copies on the r9700 HDD (DP-527)

CT100's root disk (65 G, shared by every prod container) holds only recent
state. Anything worth keeping longer goes to the 1 TB archive HDD in **r9700**
(`/srv/archive`, NTFS, Adam's old data drive: never reformat), over a second NFS
export beside the vzdump one from DP-521. The jobs run on **host `derpr`**
(`10.0.0.74`) and reach into CT100 with `pct exec` / `pct pull`, so CT100 needs no
mount, no config change and no restart (a CT100 restart re-locks payroll).

| File | Installed at | What it does |
|---|---|---|
| `daemon.json` | CT100 `/etc/docker/daemon.json` | Default log rotation for every container: `json-file`, 4 × 25 MB, rotated files gzipped. Applies to a container at **create** time only. |
| `archive-lib.sh` | derpr `/usr/local/lib/derpr-archive/` | Shared guard: mount the archive if it isn't, refuse to run unless it is mounted with 5 GB free. An unmounted `/mnt/hdd-archive` is a directory on derpr's NVMe. |
| `derpr-log-archive` + `.service`/`.timer` | derpr `/usr/local/sbin/`, `/etc/systemd/system/` | Every 6 h (`00/6:05`): each CT100 container's lines since the last good run → `docker-logs/ct100/YYYY/MM/DD/<name>-<HHMM>Z.log.zst`. Keeps 365 days. |
| `hindsight-backup` + units | same | Daily 02:45: Hindsight `pg_dump -Fc`, proved by a TOC scan and a schema-only restore before it counts; keeps 14. Moved from r9700, where it had failed since CT100 left that host. |
| `zammad-backup-sync` + units | same | Daily 04:15: copies new files from prod Zammad's own backup volume (`HOLD_DAYS=10` there); keeps 60 days. |
| `derpr-archive.exports` | r9700 `/etc/exports.d/` | `/srv/archive/derpr-archive` → `10.0.0.74` only, `fsid=75`. |

All three jobs share one rule: **a run with any failure neither advances nor
prunes.** The log job retries the same window (possible duplicate lines, never a
gap); the dump jobs keep every old copy.

## Archive layout (`r9700:/srv/archive/derpr-archive` = `derpr:/mnt/hdd-archive`)

```
docker-logs/ct100/YYYY/MM/DD/*.log.zst   archive.log
docker-logs/ct100/backfill-pre-2026-11-20T1732Z/   one-off: everything before rotation
hindsight-backups/hindsight-*.dump       backup.log
zammad-backups/prod/*_zammad_*           sync.log
manual/                                  hand-made dumps (e.g. pre-upgrade)
```

Read a log: `zstd -dc <file> | less`. Every line carries docker's RFC 3339 timestamp.

## Deploy (by hand; CI does not cover it)

```bash
# r9700
mkdir -p /etc/exports.d /srv/archive/derpr-archive
scp derpr-archive.exports root@10.0.0.71:/etc/exports.d/ && ssh root@10.0.0.71 exportfs -ra
# derpr
mkdir -p /usr/local/lib/derpr-archive /mnt/hdd-archive /var/lib/derpr-archive
echo '10.0.0.71:/srv/archive/derpr-archive /mnt/hdd-archive nfs4 vers=4.2,soft,timeo=600,nofail,_netdev 0 0' >> /etc/fstab
scp archive-lib.sh root@10.0.0.74:/usr/local/lib/derpr-archive/
scp derpr-log-archive hindsight-backup zammad-backup-sync root@10.0.0.74:/usr/local/sbin/
scp *.service *.timer root@10.0.0.74:/etc/systemd/system/
ssh root@10.0.0.74 'systemctl daemon-reload && mount /mnt/hdd-archive && systemctl enable --now derpr-log-archive.timer hindsight-backup.timer zammad-backup-sync.timer'
# CT100 rotation: install daemon.json, restart dockerd, then recreate every container
```

⚠️ **Recreating a container deletes its log.** Run `systemctl start
derpr-log-archive` on derpr immediately before any recreate you do by hand.
Changing `daemon.json` changes nothing for existing containers until they are
recreated: compose projects with `up -d --force-recreate`, hand-run containers
(`r9700-proxy`, `inference-router`: `services/inference-router/README.md`) with
`docker rm -f` + their `docker run`.

## Not covered

- `zammad-cloudflared` still has unrotated logs: a hand-run container whose tunnel
  token is on its command line. Recreate it from its source when it next changes.
- Payroll (`/opt/payroll`, CNMI-Tax repo) is out of scope: it has its own encrypted
  backup path. Its containers get rotation the next time they are recreated.
- Same box ≠ offsite: this survives a derpr or CT100 loss, not an r9700/house loss.
