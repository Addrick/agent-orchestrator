# `services/sandbox-net` — each sandbox box's LAN switch (DP-465)

These live on **host `derpr`** (`10.0.0.74`), which holds `codr-vm` (VM200,
`10.0.0.75`) and `stagr-vm` (VM201, `10.0.0.76`) and their firewall files.
derpr reaches them over its **second** node key; the first key and
`services/pve/` are r9700's. User-facing behaviour: `docs/user_guide.md`,
"Preparing codr's work". WAN is not here: `services/egress/`.

| File | On host derpr at | What it is |
|---|---|---|
| `agent-net` | `/usr/local/sbin/agent-net` | `agent-net <codr\|stagr> lan on\|off\|status\|conns`. Flips `GROUP agent-lan # TOGGLE lan` in the box's `.fw` (a leading `\|` = off), recompiles, and on `off` deletes the box's LAN conntrack flows outside its always-on rules. Also the boxes' hookscript: `agent-net <vmid> pre-start` turns LAN off. |
| `derpr-net-wrapper` | `/usr/local/bin/derpr-net-wrapper` | Forced command for the second key: only the `agent-net` verbs above and the drift-check hash of these two files. |

Like `derpr-pve-wrapper`, the wrapper is the second half of every tool that
uses it: change it in the same commit as `src/stagr/net.py`, redeploy, and
verify with the container's key. `tests/services/test_net_wrapper.py` feeds
every argv derpr emits through it.

## Host state this relies on (Proxmox, not in this repo)

- `/etc/pve/firewall/cluster.fw`: `[group agent-lan]` = `OUT ACCEPT -dest
  10.0.0.0/24`. IPv4 only, keep it that way: link-local v6 reaches the LAN.
- Each box's `.fw` has `|GROUP agent-lan # TOGGLE lan` (off), plus its
  always-on rules: in from dt21/CT100 to opencode, out to the inference servers
  (codr) and to its own egress proxy port on `10.0.0.77` (both). The old
  `agent-internet` group (direct internet) is gone: WAN is the proxy.
- `nf_conntrack_tcp_loose=0` in `/etc/sysctl.d/90-dp465-conntrack.conf`. With `1`, a
  deleted flow is re-created by its next packet and `off` cuts nothing.

## Deploying

```bash
scp services/sandbox-net/{agent-net,derpr-net-wrapper} root@10.0.0.74:/tmp/
ssh root@10.0.0.74 'install -m 0755 /tmp/agent-net /usr/local/sbin/agent-net \
  && install -m 0755 /tmp/derpr-net-wrapper /usr/local/bin/derpr-net-wrapper \
  && mkdir -p /var/lib/vz/snippets && ln -sf /usr/local/sbin/agent-net /var/lib/vz/snippets/agent-net-hook \
  && qm set 200 --hookscript local:snippets/agent-net-hook \
  && qm set 201 --hookscript local:snippets/agent-net-hook'
```

The key: generate a pair for derpr's container, put the public half in
`/root/.ssh/authorized_keys` on host derpr as

```
command="/usr/local/bin/derpr-net-wrapper",no-pty,no-agent-forwarding,no-port-forwarding,no-X11-forwarding ssh-ed25519 AAAA… derpr-net
```

and mount the private half into the container at `PVE_NET_SSH_KEY`
(`/run/secrets/pve_derpr_net`). Check from the container:
`ssh -i /run/secrets/pve_derpr_net root@10.0.0.74 /usr/local/sbin/agent-net stagr lan status`
prints `lan: off`, and `… id` prints `derpr-net: not allowed`.
