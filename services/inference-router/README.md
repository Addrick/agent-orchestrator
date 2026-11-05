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
- `ROUTER_BACKGROUND_PORT` (unset = off) — the listener Hindsight's pool reaches (DP-405B, below).
- `ROUTER_BACKGROUND_BACKENDS` (the primary) — boxes that may take background work. dt21's `:5001` is Adam's own.
- `ROUTER_BACKGROUND_HOLDOFF_S` (`ROUTER_QUIET_S`) — seconds since the last chat before a box takes a background job.
- `ROUTER_PUBLIC_HOST` (`10.0.0.70`) — the `Host` a loopback backend is addressed as (polls, background jobs).

## Background work: Hindsight behind chat (DP-405B)

Hindsight's nginx pool (`hindsight-kobold-proxy`, CT100 `/opt/configs/hindsight/kobold-lb.conf`) lists
`10.0.0.70:<ROUTER_BACKGROUND_PORT>` as one more worker. On that port:

- A job runs only on a background box that is **running Strata** (its `/metrics` has an `engine` block; kcpp's
  cancel-on-disconnect is unverified), has no chat in flight or queued, runs no other job, and has had no chat for
  the hold-off. Otherwise: **503 at once, never held.** nginx then gives the job to the next worker, and Hindsight's
  startup LLM probe (fail-open, capped at 10 s in 1.2.0) is never slowed by the router.
- **Chat preempts.** A chat request routed to a box that is running a job closes that job's upstream (Strata
  stops within its 0.5 s disconnect watch, verified on CT102 2026-11-04), the job gets a 503, and the chat goes in.
- The job's reply is read **whole** before anything is sent. A reply that fails its framing (`Content-Length`,
  chunked terminator, a stream's last event (`data: [DONE]`, `message_stop`, `response.completed`), or unframed JSON that does not parse) becomes a 502. A box's own 5xx
  (Strata's 502 for a schema failure) is passed on and does **not** mark the box down.
- Each job is sent with `"strata_checkpoint": false` and `Host: ROUTER_PUBLIC_HOST`. Without the flag Strata
  parks every job as a conversation, and five of them evicted a parked 9.6k-token chat (next turn: 0 reused, 4.9 s).
  With the flag the turn reused 9,630 of 9,650 tokens (0.15 s). Injected here rather than through Hindsight's
  `LLM_EXTRA_BODY`, because only the router knows that this request is going to Strata.
- **Not built: keeping a preempted job's prompt on disk.** On CT102 (2026-11-04), a job cancelled mid-generation
  *without* the flag could be saved: `/slots/0?action=save` stored the prompt plus the partial output, and after a
  restore the same prompt reused all but 7 tokens. That held at 12k, 52k and 136k tokens, with saves of 0.45 / 0.79 /
  1.59 s and restores of 0.42 / 1.09 / 2.48 s. **With the flag the save is refused** (`400 no complete session to
  save`, after a cancel and after a completed reply alike). In every run the RAM park alone also gave full reuse on
  the retry, so the disk copy only matters once the park is evicted, and evicting chats' parks is exactly what
  the flag prevents. Saving jobs means sending them without the flag, which is a trade-off awaiting a decision.
- Jobs do not move `last_active`, so neither the hold-off nor the quiet rule sees them as chat.
  `/router/status` → `background` holds counters for admitted, refused, preempted and incomplete jobs.

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

**A box is marked down only when it cannot answer**: a connect error, timeout or reset, or three failed polls.
An HTTP error from the engine is passed to the client with its status and body unchanged, and the box stays
up. Example: Strata's 502 `structured_output_failed` for a reply that failed its JSON schema. Until DP-405B,
that 502 marked the R9700 down until the next poll and re-sent the chat to dt21. A 502/503/504 fails over
only if the proxy in front generated it: `r9700-lb.conf` tags its own errors `X-Proxy-Error`, and an HTML
error page counts too. The status is kept at 502 rather than remapped to 422/500 because it is Strata's
answer. No proxy in this chain acts on an engine's status any more: `r9700-proxy` only fails over on
`error timeout`. Hindsight's pool re-sends a 502 to another worker on purpose, and Hindsight retries any 5xx.

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
  -e ROUTER_BACKGROUND_PORT=5007 \
  -v /opt/configs/derpr/inference_router.py:/app/inference_router.py:ro \
  python:3.13-alpine python -u /app/inference_router.py
```

Deploy `r9700-lb.conf` before the router. Its `X-Proxy-Error` tag is how the router tells a stopped CT from an
engine's error, though an HTML error page from an older conf still counts as a stopped CT. CT102's
save-on-stop / restore-on-start units are a separate install: `services/strata/README.md`.

Background work also needs, in `kobold-lb.conf`'s `kobold_pool`, a
`server 10.0.0.70:5007 max_fails=2 fail_timeout=60s;` line, and in its `location /` the line
`proxy_next_upstream error timeout http_502 http_503 non_idempotent;`. That way a refused or preempted
job (503) or a schema failure (502) goes to the next worker. Without `non_idempotent`, nginx never re-sends
a POST. Then run `docker restart hindsight-kobold-proxy`.

Update the router: copy the file, `docker restart inference-router`.

Verify: `curl -s 10.0.0.70:5001/router/status`. Since DP-499 it also lists each box's running `jobs` and the
`held` queue, each with its `requester` (the `X-Inference-Requester` header, else `hindsight` on the background port,
else the caller's IP), and each box's last `/metrics` (Strata) or `/api/extra/perf` (kcpp) sample, which the portal's
per-server statusline rows read. Logs: `docker logs inference-router`, one line per
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
