# `services/pve` — the node half of derpr's Proxmox tooling

Three artifacts that live on the **Proxmox node**, not in the derpr container:

| File | Node path | What it is |
|---|---|---|
| `derpr-pve-wrapper` | `/usr/local/bin/derpr-pve-wrapper` | The forced-command allowlist for derpr's SSH key (DP-267). |
| `derpr-model-install` | `/usr/local/sbin/derpr-model-install` | The one verb behind `install_model` (DP-265). |
| `derpr-model-tier` | `/usr/local/sbin/derpr-model-tier` | Hot/cold gguf tiering: `list`, `pin`, `unpin`, `promote` (DP-340). |
| `koboldcpp-model.service.in` | `/usr/local/share/derpr/koboldcpp-model.service.in` | Unit template the installer fills in. |
| `gguf_header.py` | `/usr/local/share/derpr/gguf_header.py` | Reads `n_layer` / `n_kv_head` / `head_dim` / `ssm_layers` out of a downloaded gguf. Deploy it **with** `derpr-model-install`: since DP-360 the installer gates on this script's exit status rather than on a pattern over its output, so an older reader still works (it also exits 0) but an older *installer* silently drops any fragment whose first key is not `n_layer`. |

They are versioned here because the node copies are deployment artifacts of
them — and because the alternative has already cost us a silent production
outage (below).

---

## ⚠️ The wrapper is the second half of every proxmox tool

`derpr-pve-wrapper` is what sshd runs instead of whatever derpr asked for. If
derpr emits a command shape the wrapper does not admit, the tool fails **in
production and only in production**:

- A developer's own node key is unrestricted, so anything run from a workstation
  — including a "live smoke test against the real node" — bypasses the wrapper
  entirely and proves nothing about the deployed path.
- The container's key *is* wrapped, so the same call from the deployed bot comes
  back `derpr-pve: not allowed`, exit 1.

**This is not hypothetical.** DP-332 shipped `list_models` and `gpu_status` with
three new command shapes (`systemctl list-unit-files`, `ls /sys/class/drm`, `cat
…/mem_info_vram_*`) and no wrapper change. Unit tests passed, mypy passed, and a
live read-only smoke test passed — from a workstation. On the deployed container
both tools were dead. The wrapper in this directory restores that parity and adds
DP-265's verb.

**Rule:** when you change an argv in `src/proxmox/handler.py` or
`src/huggingface/handler.py`, change this wrapper in the same commit, redeploy
it, and verify with **the container's key**.

### Verifying against the real gate

⚠️ **The gate check runs the command it admits.** `allow()` ends in `exec`, so a
probe is not a dry run: probing `derpr-model-tier pin foo.gguf` *pins* it, and
probing `promote` *starts a promotion job*. Redirecting stdout hides the output,
not the side effect. Probe read-only shapes only, and reason about the mutating
verbs from the `case` block.

From the node, with no key involved at all:

```bash
for c in \
  "pct list" \
  "pct exec 101 -- systemctl list-unit-files --type=service --no-legend --no-pager" \
  "pct exec 101 -- ls /sys/class/drm" \
  "pct exec 101 -- cat /sys/class/drm/card1/device/mem_info_vram_total /sys/class/drm/card1/device/mem_info_vram_used" \
  "/usr/local/sbin/derpr-model-tier list" \
  "/usr/local/sbin/derpr-model-tier run-promote x.gguf j1" \
  "id"
do
  printf '%s -> ' "$c"
  SSH_ORIGINAL_COMMAND="$c" /usr/local/bin/derpr-pve-wrapper >/dev/null 2>&1 \
    && echo ALLOW || echo DENY
done
```

The last two must print `DENY` — `run-promote` is systemd's verb and sshd must
never reach it. `journalctl -t derpr-pve` carries the audit trail.

---

## Deploying

All commands run as root on the Proxmox node (`10.0.0.71` on this deployment).

```bash
# 1. wrapper — scp it, never pipe it. A heredoc over ssh mangles the
#    metacharacter `case` block (this ate the block once during DP-267).
scp services/pve/derpr-pve-wrapper root@<node>:/usr/local/bin/derpr-pve-wrapper
ssh root@<node> 'chmod 755 /usr/local/bin/derpr-pve-wrapper && bash -n /usr/local/bin/derpr-pve-wrapper'

# 2. installer + its data files
scp services/pve/derpr-model-install root@<node>:/usr/local/sbin/derpr-model-install
ssh root@<node> 'chmod 755 /usr/local/sbin/derpr-model-install && bash -n /usr/local/sbin/derpr-model-install'
ssh root@<node> 'mkdir -p /usr/local/share/derpr'
# 2b. tiering (DP-340). Same rule: scp, then syntax-check on the node.
scp services/pve/derpr-model-tier root@<node>:/usr/local/sbin/derpr-model-tier
ssh root@<node> 'chmod 755 /usr/local/sbin/derpr-model-tier && bash -n /usr/local/sbin/derpr-model-tier'

scp services/pve/koboldcpp-model.service.in root@<node>:/usr/local/share/derpr/
scp services/pve/gguf_header.py root@<node>:/usr/local/share/derpr/
```

The `authorized_keys` line the wrapper hangs off (already present since DP-267):

```
command="/usr/local/bin/derpr-pve-wrapper",no-pty,no-port-forwarding,no-X11-forwarding,no-agent-forwarding ssh-ed25519 AAAA… derpr-container
```

⚠️ **Never edit `authorized_keys` in place.** Write a temp file and `mv` it.
`grep -v … file > file` truncates the file that gates your own access; that is
how the node got locked out during DP-267, recovered only via the PVE web
console.

### Tiering prerequisites (DP-340)

`derpr-model-tier` assumes the archive disk is mounted and will refuse to invent
it. Before first use:

```bash
# the archive disk, with nofail so it can never block the node's boot
mkdir -p /srv/archive
# /etc/fstab:
# UUID=<uuid> /srv/archive ntfs3 rw,noatime,uid=0,gid=0,umask=022,nofail,x-systemd.device-timeout=10s 0 0
mount /srv/archive && mkdir -p /srv/archive/models /srv/archive/.jobs /srv/archive/.tier
```

⚠️ **Move the existing ggufs to the archive before relying on eviction.** The hot
tier only evicts a model that has a verified archive copy, so until each gguf
exists in both places a promotion will refuse with `unarchived_victim` rather
than delete anything. That refusal is the invariant working, not a bug.

Optional, in `/etc/default/derpr-model-tier`:

```sh
HOT_CAPACITY_BYTES=128849018880   # 120 GiB — cap the hot tier below the volume
MARGIN_BYTES=5368709120           # 5 GiB kept free beyond the incoming model
```

`HOT_CAPACITY_BYTES` matters before DP-341 shrinks the models LV: without it the
volume is far bigger than the tier should be and eviction never fires.

### Site settings

`derpr-model-install` reads `/etc/default/derpr-model-install` if present:

```sh
MODELS_DIR=/srv/models           # where downloads land (DP-416: the dedicated
                                 # models NVMe; never a thin LV, df must be true)
JOBS_DIR=/srv/archive/.jobs      # job records, on a different disk from the
                                 # download so a full models disk can still
                                 # record the failure
CT_VMID=101                      # GPU container
CT_MODELS_DIR=/opt/koboldcpp/models
KCPP_DIR=/opt/koboldcpp
KCPP_PORT=5001
MIN_MARGIN_BYTES=2147483648      # free space kept beyond the download
LOCK_WAIT=60                     # seconds to wait for the install lock (DP-349)
SMARTCACHE_SLOTS=4               # dense models: --smartcache N (DP-364)
SMARTCACHEGRID_MB=32768          # hybrid models: --smartcachegrid MB
```

The two cache settings are **box constants**. Which one a unit gets is not a
setting at all: the installer reads it off the downloaded file (below).

Concurrent installs are admitted under one lock (DP-349). The precheck and the
`systemd-run` that commits to it are a single critical section, and each running
job writes a reservation under `$JOBS_DIR/.reservations` recording the bytes it
has promised, so a second job's precheck subtracts space that is in flight and
not only space already written. Reservations are released when a job finishes
and pruned when its `modelinstall-<job>` unit is gone, which also deletes the
`.part` a crashed job left behind. Two jobs installing the same name are
refused; `LOCK_WAIT` bounds how long a precheck waits before giving up.

Downloads land as `<MODELS_DIR>/<name>.gguf` — named for the **unit name**, not
for the repo's file name, because two unrelated repos publishing
`model-Q4_K_M.gguf` is ordinary and the second would otherwise be uninstallable.

Hub downloads are **anonymous**: there is no HuggingFace token, here or on the
derpr side (DP-347). Gated and private repos are out of scope — the public gguf
repos this exists to install from need no auth. Do not re-add a token file; the
old one put a bearer on `curl`'s argv, readable through `/proc/<pid>/cmdline`
for the length of a multi-GB download.

### Completion ping (DP-343)

Both scripts POST the **job id** to derpr when a job reaches `done` or `failed`,
so derpr can wake a persona instead of waiting to be asked. Off unless a URL is
set. Add to **both** `/etc/default/derpr-model-install` and
`/etc/default/derpr-model-tier`:

```sh
# On this deployment: CT100 (10.0.0.70), host port 5004, which docker-compose
# publishes straight to the container's 5003. Deliberately NOT host 5003 —
# that is Caddy with `tls internal`, whose self-signed cert `curl -f` refuses
# and which would need the node to carry Caddy's root CA for no gain on a LAN
# hop between two guests of the same node.
DERPR_CALLBACK_URL=http://10.0.0.70:5004/api/v1/model_job/complete
DERPR_CALLBACK_TIMEOUT=10
```

**One setting, and no credential (DP-355).** The route is unauthenticated by
design: the ping carries no facts, so there is nothing for a credential to
protect. DP-343 shipped a shared bearer token here and it was deleted — the
threat it was argued against (a forged "install finished") is already dead
because derpr re-reads the job over SSH, and the only property it really bought
was stopping an unauthenticated LAN host from costing derpr one SSH round-trip
per POST. That is a denial of service anyone already on this LAN has cheaper
ways to cause, and it did not justify a secret generated, `chmod`-ed and kept in
sync across two hosts forever.

⚠️ **LAN-only accepted risk**, indexed with the rest of them in
`pre-public-exposure-checklist`. If derpr is ever exposed beyond the LAN this
route needs both a credential and TLS, and neither is a one-line change. Do not
re-add a token on its own and call it hardened.

⚠️ **Never point `DERPR_CALLBACK_URL` at a control-plane route, and never give
the node `DERPR_CONTROL_TOKEN`.** The reason this route can be open is that it
opens exactly one door, which accepts a job id and nothing else. The operator
token opens persona edits and park approval — a node holding it would be an
operator able to approve its own parks.

The POST body is `{"job_id": "..."}`; derpr answers it by re-reading the job over
its own SSH connection, so the node is not trusted to report the outcome. The
ping fires after the job file is renamed into place, every failure path is a log
line (`journalctl -t derpr-model-install -t derpr-model-tier`), and an
unreachable derpr costs the announcement and nothing else — the job stays `done`.

### derpr side

```
HF_TOOLS_ENABLED=true
```

plus the existing `PVE_*` settings — `install_model` rides the same key and host
as the proxmox tools. Give the persona
`service_bindings: ["proxmox", "huggingface"]`.

For the DP-343 ping, also:

```
MODEL_JOB_ALERT_CHANNEL_ID=<discord channel id to post the report into>
```

**One setting, and it does not name a conversation.** The ping resumes the
turn that *started* the job: `install_model` and the cold-tier promotion park a
`node_job` deferral under the job id, and that row already carries the persona,
the channel and the user. So the report lands where you asked for the install,
a `CHANNEL_ISOLATED` persona still sees the instruction you gave it earlier, and
any `set_active_model` it parks appears as an approval card you can answer —
with nothing configured.

`MODEL_JOB_ALERT_CHANNEL_ID` is not a coordinate either: the resume has no
listener holding a stream open, so this is simply where the reply is posted.
Unset it and the turn still runs (and can still park) — nothing is announced.

---

## What `derpr-model-install` does, and what it refuses

```
derpr-model-install install <repo> <file> <name> <ctx> <size> <sha256> <job_id> <kv>
derpr-model-install status  <job_id>
derpr-model-install run     <repo> <file> <name> <ctx> <size> <sha256> <job_id> <kv>
```

`run` is what `systemd-run` executes and is **not** in the wrapper's allowlist —
it is reachable locally only.

`<kv>` (DP-364) is the KV precision, proposed by the persona and shown on the
approval card. The cache mode is derived, from `gguf_header.py --ssm-layers`:

| | from | written into the unit as |
|---|---|---|
| KV precision | `<kv>`: `f16` · `q8` · `q4` | `--quantkv 0` · `1` · `2` |
| cache mode, hybrid | `ssm_layers > 0` | `--smartcachegrid $SMARTCACHEGRID_MB` |
| cache mode, dense | `ssm_layers = 0` | `--smartcache $SMARTCACHE_SLOTS` |

A hybrid's recurrent state can't be rewound, so only a checkpoint lets an edit
skip reprocessing; a dense cache already survives edits and wants swap slots for
switching conversations. Never both: the fork arms the grid only on a
recurrent/hybrid model, and `koboldcpp.py` drops `--smartcache` whenever
`--smartcachegrid` is set — grid on a dense model would leave it with neither.

Every other flag in `koboldcpp-model.service.in` is a box constant, including
`--multiuser 1`. **The unit file is the whole configuration** — nothing rewrites
its argv at exec. (CT101 used to run a policy wrapper over the koboldcpp binary
that did, because the template hardcoded one `--quantkv` / `--smartcache` for
every model; it is removed as part of DP-364's rollout.)

⚠️ **Deploy the node artifacts and the container together.** The install arity
changed (9 → 10 words at the wrapper), and there is no compatibility shape: a
new container against an old wrapper is refused, and an old container against a
new wrapper is refused. Installs are rare and human-approved, so a short window
where `install_model` fails loudly is the chosen cost — a default precision for
the old shape would bring back exactly the one-size unit this replaced.
**`gguf_header.py` is now required too**: without it the cache mode can't be
read and every install fails with `cache_mode_unknown`.

Size and sha256 are **arguments**, read from the Hub by derpr and displayed on
the approval card. The node never asks HuggingFace what the file *should* be, so
what a human approved is what gets enforced — and the node needs no JSON parser,
which matters because a stock PVE node has no `jq`.

It refuses, before any bytes move:

- a unit named `koboldcpp-<name>.service` that already exists on the container —
  overwriting one silently repoints a name `list_models` already publishes;
- a destination file that exists with a **different** sha256 (an identical one is
  reused, so a retry is cheap);
- insufficient free space on the archive disk — the larger of 2 GiB or 5% of the
  download is kept free, less what running jobs have already reserved. Downloads
  no longer touch `/srv/models` (DP-340): that is a thin LV, where `df` cannot see
  the pool running out;
- a KV precision outside `f16` / `q8` / `q4`.

And after downloading:

- a sha256 mismatch **deletes** the partial file and fails the job. Size matching
  is not proof and has fooled this project before;
- a file whose SSM block count `gguf_header.py --ssm-layers` cannot read fails
  with `cache_mode_unknown` and writes no unit — either guess would be silently
  wrong for one architecture. The verified file is **kept**, so a retry reuses it.

The unit it writes is **disabled and not started**. Putting a model on `:5001` is
`set_active_model`'s job and gets its own approval.

Job state lives in `<JOBS_DIR>/<job_id>.json` (`/srv/archive/.jobs` by default —
on the archive disk, so a read-only `/srv/models` cannot stop a failure from
being recorded), written to a temp file and
renamed, so a poll never reads a half-written document. Every value in it is
either regex-gated input or a fixed-vocabulary token — no HTTP body, no `curl`
message. That is what lets `install_status` claim `produces_untrusted: False` on
derpr's side.
