# `services/inference-router` — the local inference front door (DP-483)

Everything that wants "the local model" talks to `10.0.0.70:5001` on CT100: derpr, codr, codr's
subagents. User-facing behaviour: `docs/user_guide.md`, "More than one server behind `:5001`".

```
:5001 inference-router ─┬─ r9700: 127.0.0.1:5006 r9700-proxy (nginx) ─ CT101 .72, CT102 .73 backup
                        └─ dt21:  10.0.0.69:5001 Strata, on demand
```

| File | On CT100 at | What it is |
|---|---|---|
| `inference_router.py` | `/opt/configs/derpr/inference_router.py` | The router, standard library only. |
| `r9700-lb.conf` | `/opt/configs/derpr/r9700-lb.conf` | CT101/CT102 failover for the one R9700. Unchanged job, now behind the router on loopback. |

## Why not nginx

Strata runs one request at a time and keeps a conversation's cache on the box that served it.
`hash` sticks to the cache but sends two conversations to one box half the time, with the other box
idle. `least_conn` spreads them, and hops every turn, re-reading a 200k-token prompt (2+ minutes)
each time. The router needs both facts at once, and so does its anti-thrash rule, which nginx
open source cannot express. It remains the mechanism for the CT101/CT102 failover beneath it.

## Rules

Conversation identity is the request's own leading messages: `prefix_chain` hashes them cumulatively
and each box remembers the prefixes it has served, so clients send no session header. Rules are in
`choose()`; its tests read as the spec. Restarting the router forgets the prefixes, so each
conversation may re-read once.

- `ROUTER_BACKENDS` — `name=host:port,…`, primary first (default `r9700=127.0.0.1:5006,dt21=10.0.0.69:5001`).
- `ROUTER_MOVE_CHARS` (40000) — a conversation with less than this cached is cheap to move.
- `ROUTER_QUIET_S` (120) — a box idle this long has no conversation between steps on it.
- `ROUTER_HOST` / `ROUTER_PORT` (`0.0.0.0` / `5001`).

A request is held in the router until its chosen box is free, oldest first, because once sent it is
stuck in that box's queue even if the other frees up (seen live: 208 s behind the R9700 with dt21 idle).
A client that leaves while held never reaches a box. **A held request gets no headers until its box
takes it**, and opencode gives up on headers after 5 min by default and resends at the back of the
queue, so opencode clients set the provider's `headerTimeout` to 30 min (codr's `workspace_files()`,
VM200's `~/.config/opencode/opencode.json`; DP-490). Once a box answers, Strata sends `: keep-alive`
every ~3 s during prefill, so opencode's 5 min `chunkTimeout` is not at risk. Only generation paths (`GENERATION_PATHS`) are held
or counted; other calls go straight to the first healthy box.

Each box is polled every 3 s: `/v1/models` (up, model, context, loaded) and `/metrics` (busy with work
that did not come through the router, which is how Adam's direct use of dt21 counts). A box is used
only if it is **loaded and serves the primary's model**, so a kcpp swap on the R9700 takes dt21 out of
rotation by itself, and a lazy Strata is never made to load.

The `Host` header passes through untouched, so every Strata needs the callers' hosts in
`allowed_hosts` (it answers 403 otherwise): `10.0.0.70`, `host.docker.internal`.

## Deploy (by hand; CI does not cover it)

Survives derpr deploys on purpose: it is not in the compose file, like the nginx proxies before it.
Replacing the proxy that holds `:5001` cuts generations in flight. Check `/metrics` `live.state` first.

```bash
docker rm -f r9700-proxy
docker run -d --name r9700-proxy --restart unless-stopped -p 127.0.0.1:5006:5001 \
  -v /opt/configs/derpr/r9700-lb.conf:/etc/nginx/nginx.conf:ro nginx:1.27-alpine
docker run -d --name inference-router --restart unless-stopped --network host \
  -v /opt/configs/derpr/inference_router.py:/app/inference_router.py:ro \
  python:3.13-alpine python -u /app/inference_router.py
```

Update the router: copy the file, `docker restart inference-router`.

Verify: `curl -s 10.0.0.70:5001/router/status`. Logs: `docker logs inference-router`, one line per
request, box and match sizes only; no message content is logged or stored.

**Rollback:** `docker rm -f inference-router r9700-proxy`, then the `r9700-proxy` line above with
`-p 5001:5001`.

## dt21

Start Menu "Strata Flash-Next 5001" runs `healer_strata-flash-next-5001.ps1` (copy: notes repo
`infrastructure/scripts/dt21/`): visible window, no `--lazy`, closing it stops serving, and it joins the router within
seconds of loading. Test a new config on another port (`--port 5107`), which the router never polls. Strata 0.2.7 lives in
`F:\Machine Learning\Strata-0.2.7`, separate from the 0.1.44.1 install that the `:5105` Hindsight slot and the A/B
baselines depend on. Cannot run beside `:5105`: both want the card.

**Leave it in full-RAM mode: no `--resident-budget-gib`, no `--resident-experts`.** Windows pages out the cold experts when
another program wants RAM: with a program holding 32 GiB, decode stayed 85-88 tok/s. Either flag costs a quarter or more of
the speed even when nothing is read from the SSD (measured 2026-11-02; table in the notes repo, `benchmarks/strata-dt21.md`).
`--vram-reserve-mib 1024` stays: with 176 MiB of VRAM free, the first run hung the GPU.
