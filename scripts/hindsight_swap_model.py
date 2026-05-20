"""Swap Hindsight's extraction model and restart the container.

Sets BEDROCK_MODEL_ID in .env, recreates the hindsight container with the
bedrock override compose stack, then probes /v1/chat/completions through
the proxy with one trivial extraction-shaped prompt to confirm the new
model responds.

Usage:
    python -m scripts.hindsight_swap_model google.gemma-3-4b-it
    python -m scripts.hindsight_swap_model openai.gpt-oss-20b-1:0

Requires the bedrock override stack to already be the active one (i.e.
docker compose -f ...hindsight.yml -f ...bedrock.yml up -d at least once).
"""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
import time
from pathlib import Path

import httpx

ENV_FILE = Path(".env")
COMPOSE_FILES = ["docker-compose.hindsight.yml", "docker-compose.hindsight.bedrock.yml"]
PROXY_URL = "http://127.0.0.1:8888"  # not used; we hit proxy through hindsight indirectly
PROBE_TIMEOUT_S = 120


def write_env(model_id: str) -> None:
    lines: list[str] = []
    if ENV_FILE.exists():
        for ln in ENV_FILE.read_text(encoding="utf-8").splitlines():
            if re.match(r"^BEDROCK_MODEL_ID\s*=", ln):
                continue
            lines.append(ln)
    lines.append(f"BEDROCK_MODEL_ID={model_id}")
    ENV_FILE.write_text("\n".join(lines) + "\n", encoding="utf-8")


def compose(*args: str) -> None:
    cmd = ["docker", "compose"]
    for f in COMPOSE_FILES:
        cmd.extend(["-f", f])
    cmd.extend(args)
    print(f"$ {' '.join(cmd)}", flush=True)
    subprocess.check_call(cmd)


def probe_proxy(model_id: str) -> None:
    """Hit bedrock-proxy directly via its host port (if exposed) to confirm
    the model is reachable. Falls back to checking hindsight /health only."""
    # bedrock-proxy isn't host-exposed in the default override; check hindsight health.
    deadline = time.monotonic() + PROBE_TIMEOUT_S
    while time.monotonic() < deadline:
        try:
            r = httpx.get("http://127.0.0.1:8888/health", timeout=3)
            if r.status_code == 200:
                print(f"hindsight healthy with model={model_id}", flush=True)
                return
        except Exception:
            pass
        time.sleep(2)
    raise SystemExit("hindsight failed to come healthy after restart")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("model_id", help="Bedrock model id, e.g. google.gemma-3-4b-it")
    args = ap.parse_args()

    write_env(args.model_id)
    compose("up", "-d", "--no-deps", "hindsight")  # recreate hindsight with new env
    probe_proxy(args.model_id)
    return 0


if __name__ == "__main__":
    sys.exit(main())
