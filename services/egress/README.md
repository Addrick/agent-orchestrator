# `services/egress` — each sandbox box's WAN (DP-456, DP-465)

The sandbox boxes (`codr-vm` VM200 `10.0.0.75`, `stagr-vm` VM201 `10.0.0.76`,
both on host `derpr`) have no DNS and no direct internet. A box's only way out
is **its own tinyproxy instance** in the egress container (CT110 on `r9700`,
`10.0.0.77`), which admits only that box and only allowlisted hosts. The
instance *is* the box's WAN switch: started = on, stopped = off. User-facing
behaviour: `docs/user_guide.md`, "Preparing codr's work". Why it is built this
way: decision `sandbox-net-switch-mechanics` (notes repo).

| File | In CT110 at | What it is |
|---|---|---|
| `stagr.conf` / `codr.conf` | `/etc/egress/<box>.conf` | One instance each: stagr `10.0.0.77:8888`, codr `:8889`; `Allow` only that box; default-deny host filter at `/run/egress/<box>.filter`; CONNECT to 443 only; log `/var/log/tinyproxy/<box>.log`. |
| `tinyproxy@.service` | `/etc/systemd/system/` | The instance unit. **Never enabled**: a container restart leaves every WAN off. `ExecStartPre` writes the filter, `ExecStopPost` deletes the approved hosts. |
| `egress-ctl` | `/usr/local/sbin/egress-ctl` | `egress-ctl <box> on\|off\|status\|allow <host>\|list\|conns\|log` (`filter` is the unit's). `allow` needs WAN on and reloads (SIGUSR1: rereads the filter, keeps tunnels). |
| `default.list` | `/etc/egress/default.list` | Always-allowed hosts: a line admits the domain and its subdomains, `=host` only that host. |
| `logrotate` | `/etc/logrotate.d/egress` | Weekly, `copytruncate` (egress-ctl reads the live log). |

derpr reaches this only through the node verb `services/pve/derpr-egress`
(`pct exec 110 -- egress-ctl …`) behind the same SSH key and wrapper as the
other node tools. Nothing in the container listens for derpr.

**Web only.** tinyproxy carries plain HTTP and HTTPS (CONNECT to 443), for
clients that honour `HTTP(S)_PROXY`. Not: `git@` clones (ssh), other TLS ports,
UDP/QUIC/ICMP, tools that ignore the proxy env.

## Firewall (Proxmox, not in this repo)

- `r9700:/etc/pve/firewall/110.fw`: in only `10.0.0.76 → 8888` and
  `10.0.0.75 → 8889`; out DNS to the router, then RFC1918/link-local dropped,
  then 80/443. **All IPv6 is dropped**: the home LAN has a global v6 prefix, so
  v6 cannot be split into LAN and internet by address. `/etc/gai.conf` in the
  container prefers IPv4. The datacenter firewall on `r9700` (`cluster.fw`)
  must be enabled or no guest rule applies.
- `derpr:/etc/pve/firewall/{200,201}.fw`: out to the box's own proxy port,
  always on (nothing listens there while WAN is off). The box's proxy env is in
  `/etc/environment` and apt's `80proxy`. LAN: `services/sandbox-net/`.

## Deploying

```bash
scp services/egress/{egress-ctl,stagr.conf,codr.conf,default.list,tinyproxy@.service,logrotate} root@10.0.0.71:/tmp/
ssh root@10.0.0.71 'cd /tmp && pct exec 110 -- mkdir -p /etc/egress \
  && pct push 110 egress-ctl /usr/local/sbin/egress-ctl --perms 0755 \
  && pct push 110 stagr.conf /etc/egress/stagr.conf --perms 0644 \
  && pct push 110 codr.conf /etc/egress/codr.conf --perms 0644 \
  && pct push 110 default.list /etc/egress/default.list --perms 0644 \
  && pct push 110 tinyproxy@.service /etc/systemd/system/tinyproxy@.service --perms 0644 \
  && pct push 110 logrotate /etc/logrotate.d/egress --perms 0644 \
  && pct exec 110 -- systemctl daemon-reload'
```

First time from the DP-456 layout: `systemctl disable --now tinyproxy
egress-refresh.timer`, remove `egress-refresh.{service,timer}`,
`/etc/tinyproxy/filter` and `/var/lib/egress/granted`.

Refusals: `pct exec 110 -- egress-ctl stagr log`, or grep `refused` in
`/var/log/tinyproxy/<box>.log`.
