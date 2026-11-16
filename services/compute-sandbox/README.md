# `services/compute-sandbox` — the box side of `run_python` (DP-522)

Lives on **stagr-vm** (VM201, `10.0.0.76`, on host `derpr`). derpr reaches it the way it
reaches every stagr command: `OpencodeClient.run_shell` on stagr's `opencode serve`, as user
`agent` (passwordless sudo). User-facing behaviour: `docs/user_guide.md`, "`run_python`".

| File | On stagr-vm at | What it is |
|---|---|---|
| `derpr-compute` | `/usr/local/bin/derpr-compute` (root, 0755) | `derpr-compute SECONDS`, script on stdin. Runs it as `compute` in a transient systemd unit with no network, no `/home`, a read-only system, a private `/tmp`, and memory/CPU/task/time caps. |

The guards are this file's, not derpr's: `src/stagr/compute.py` only sends
`sudo -n /usr/local/bin/derpr-compute <seconds>` with the script on stdin. Change both in the
same commit and redeploy this one.

## Host state this relies on

- User `compute`: system user, no home, shell `/usr/sbin/nologin`, not in `sudo`, no SSH key.
- systemd ≥ 247 (`PrivateIPC`, `ProtectProc`); Debian 13 ships 257.

## Deploying

```bash
scp -i ~/.ssh/derpr-host services/compute-sandbox/derpr-compute agent@10.0.0.76:/tmp/
ssh -i ~/.ssh/derpr-host agent@10.0.0.76 'sudo useradd --system --no-create-home --home-dir /nonexistent --shell /usr/sbin/nologin compute 2>/dev/null; sudo install -o root -g root -m 0755 /tmp/derpr-compute /usr/local/bin/derpr-compute && rm /tmp/derpr-compute'
```

## Verifying (run after every deploy)

Each must print what its comment says:

```bash
ssh -i ~/.ssh/derpr-host agent@10.0.0.76 bash -s <<'EOF'
echo 'print(2**100)' | sudo -n derpr-compute 10                                   # 1267650600228229401496703205376
echo 'import socket; socket.create_connection(("1.1.1.1",80),3)' | sudo -n derpr-compute 10; echo "exit $?"   # error, exit 1
echo 'import os; print(os.listdir("/home"))' | sudo -n derpr-compute 10           # PermissionError
echo 'import socket; socket.getaddrinfo("pypi.org",443)' | sudo -n derpr-compute 10; echo "exit $?"   # gaierror at once, exit 1 (a hang = it reached a resolver)
echo 'import os; print(os.listdir("/run"))' | sudo -n derpr-compute 10; echo "exit $?"   # PermissionError, exit 1
echo 'open("/usr/x","w")' | sudo -n derpr-compute 10; echo "exit $?"              # Read-only file system, exit 1
echo 'import time; time.sleep(30)' | sudo -n derpr-compute 3; echo "exit $?"      # exit 124
echo 'x=bytearray(1<<30)' | sudo -n derpr-compute 10; echo "exit $?"              # MemoryError or killed, nonzero
EOF
```
