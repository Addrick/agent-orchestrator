# `services/egress` — the stagr setup VM's way out (DP-456)

The setup VM (`stagr`, VM201 on host `derpr`, `10.0.0.76`) has no LAN and no
direct internet. Its only route out is tinyproxy in the egress container (CT110 on
`r9700`, `10.0.0.77:8888`), which admits only allowlisted domains. User-facing
behaviour: `docs/user_guide.md`, "Preparing codr's work".

| File | In CT110 at | What it is |
|---|---|---|
| `tinyproxy.conf` | `/etc/tinyproxy/tinyproxy.conf` | Listens on `10.0.0.77:8888`, admits only `10.0.0.76`, default-deny host filter, CONNECT to 443 only. |
| `egress-ctl` | `/usr/local/sbin/egress-ctl` | Writes the filter from the defaults plus live grants: `allow <domain> <minutes>`, `list`, `refresh`. |
| `default.list` | `/etc/egress/default.list` | Always-allowed domains (each admits its subdomains). |
| `egress-refresh.service` / `.timer` | `/etc/systemd/system/` | Drop expired grants every minute. |

derpr changes the allowlist only through the node verb `services/pve/derpr-egress`
(`pct exec 110 -- egress-ctl …`) behind the same SSH key and wrapper as the other
node tools. Nothing in the container listens for derpr.

## Firewall (Proxmox, not in this repo)

- `r9700:/etc/pve/firewall/110.fw`: in only `10.0.0.76 → 8888`; out DNS to the
  router, then RFC1918/link-local dropped, then 80/443. **All IPv6 is dropped**: the
  home LAN has a global v6 prefix, so v6 cannot be split into LAN and internet by
  address. `/etc/gai.conf` in the container prefers IPv4 so lookups don't stall on
  AAAA records. The datacenter firewall on `r9700` (`cluster.fw`) must be enabled or
  no guest rule applies.
- `derpr:/etc/pve/firewall/201.fw` (stagr): in `dt21 → 22, 4096` and `CT100 → 4096`;
  out only `10.0.0.77:8888`. No DNS: tools must use the proxy (`/etc/environment`
  and apt's `80proxy` set it).

## Deploying

```bash
scp services/egress/{egress-ctl,tinyproxy.conf,default.list,egress-refresh.service,egress-refresh.timer} root@10.0.0.71:/tmp/
ssh root@10.0.0.71 'cd /tmp && pct push 110 egress-ctl /usr/local/sbin/egress-ctl --perms 0755 \
  && pct push 110 tinyproxy.conf /etc/tinyproxy/tinyproxy.conf --perms 0644 \
  && pct push 110 default.list /etc/egress/default.list --perms 0644 \
  && pct push 110 egress-refresh.service /etc/systemd/system/egress-refresh.service --perms 0644 \
  && pct push 110 egress-refresh.timer /etc/systemd/system/egress-refresh.timer --perms 0644 \
  && pct exec 110 -- sh -c "systemctl daemon-reload && systemctl enable --now egress-refresh.timer && /usr/local/sbin/egress-ctl refresh && systemctl restart tinyproxy"'
```

Refusals: `pct exec 110 -- grep refused /var/log/tinyproxy/tinyproxy.log`.
