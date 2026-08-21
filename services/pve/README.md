# `services/pve` — the node half of derpr's Proxmox tooling

Three artifacts that live on the **Proxmox node**, not in the derpr container:

| File | Node path | What it is |
|---|---|---|
| `derpr-pve-wrapper` | `/usr/local/bin/derpr-pve-wrapper` | The forced-command allowlist for derpr's SSH key (DP-267). |
| `derpr-model-install` | `/usr/local/sbin/derpr-model-install` | The one verb behind `install_model` (DP-265). |
| `koboldcpp-model.service.in` | `/usr/local/share/derpr/koboldcpp-model.service.in` | Unit template the installer fills in. |
| `gguf_header.py` | `/usr/local/share/derpr/gguf_header.py` | Reads `n_layer` / `n_kv_head` / `head_dim` out of a downloaded gguf. |

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

From the node, with no key involved at all:

```bash
for c in \
  "pct list" \
  "pct exec 101 -- systemctl list-unit-files --type=service --no-legend --no-pager" \
  "pct exec 101 -- ls /sys/class/drm" \
  "pct exec 101 -- cat /sys/class/drm/card1/device/mem_info_vram_total /sys/class/drm/card1/device/mem_info_vram_used" \
  "id"
do
  printf '%s -> ' "$c"
  SSH_ORIGINAL_COMMAND="$c" /usr/local/bin/derpr-pve-wrapper >/dev/null 2>&1 \
    && echo ALLOW || echo DENY
done
```

The last one must print `DENY`. `journalctl -t derpr-pve` carries the audit trail.

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

### Site settings

`derpr-model-install` reads `/etc/default/derpr-model-install` if present:

```sh
MODELS_DIR=/srv/models          # node-side models dir (bind-mounted RO into the CT)
CT_VMID=101                     # GPU container
CT_MODELS_DIR=/opt/koboldcpp/models
KCPP_DIR=/opt/koboldcpp
KCPP_PORT=5001
MIN_MARGIN_BYTES=2147483648     # free space kept beyond the download
```

For gated repos, put a HuggingFace read token in
`/etc/derpr-model-install.token` (`chmod 600`). Absent, public repos still work.

### derpr side

```
HF_TOOLS_ENABLED=true
```

plus the existing `PVE_*` settings — `install_model` rides the same key and host
as the proxmox tools. Give the persona
`service_bindings: ["proxmox", "huggingface"]`.

---

## What `derpr-model-install` does, and what it refuses

```
derpr-model-install install <repo> <file> <name> <ctx> <size> <sha256> <job_id>
derpr-model-install status  <job_id>
derpr-model-install run     <repo> <file> <name> <ctx> <size> <sha256> <job_id>
```

`run` is what `systemd-run` executes and is **not** in the wrapper's allowlist —
it is reachable locally only.

Size and sha256 are **arguments**, read from the Hub by derpr and displayed on
the approval card. The node never asks HuggingFace what the file *should* be, so
what a human approved is what gets enforced — and the node needs no JSON parser,
which matters because a stock PVE node has no `jq`.

It refuses, before any bytes move:

- a unit named `koboldcpp-<name>.service` that already exists on the container —
  overwriting one silently repoints a name `list_models` already publishes;
- a destination file that exists with a **different** sha256 (an identical one is
  reused, so a retry is cheap);
- insufficient free space on the models dir — the larger of 2 GiB or 5% of the
  download is kept free. `/srv/models` is a thin LV: filling it takes `:5001` and
  every other guest's models with it, so this refuses rather than truncating.

And after downloading:

- a sha256 mismatch **deletes** the partial file and fails the job. Size matching
  is not proof and has fooled this project before.

The unit it writes is **disabled and not started**. Putting a model on `:5001` is
`set_active_model`'s job and gets its own approval.

Job state lives in `<MODELS_DIR>/.jobs/<job_id>.json`, written to a temp file and
renamed, so a poll never reads a half-written document. Every value in it is
either regex-gated input or a fixed-vocabulary token — no HTTP body, no `curl`
message. That is what lets `install_status` claim `produces_untrusted: False` on
derpr's side.
