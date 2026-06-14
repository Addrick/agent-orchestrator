#!/usr/bin/env python
"""Local CI gate runner — mirrors .github/workflows/deploy.yml.

Runs the same checks GitHub does (flake8 hard subset, missing-deps,
mypy, pytest) and prints one pass/fail line per stage plus a summary.
Exits non-zero if any stage fails, so PyCharm/CI surface red on failure.

Usage:
    python scripts/ci_check.py            # full gate (pytest -m "not integration")
    python scripts/ci_check.py --fast     # unit-only, parallel, skip live/integration
    python scripts/ci_check.py --no-tests # lint + mypy only (quick structural check)

Run it from the repo root. Uses the interpreter it was launched with
(sys.executable), so point your PyCharm run config at the project venv.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
PY = sys.executable


def run(name: str, cmd: list[str]) -> tuple[str, bool, float]:
    """Run one stage; return (name, ok, seconds). Streams output live."""
    print(f"\n{'=' * 70}\n> {name}\n  $ {' '.join(cmd)}\n{'=' * 70}", flush=True)
    start = time.monotonic()
    result = subprocess.run(cmd, cwd=REPO_ROOT)
    elapsed = time.monotonic() - start
    ok = result.returncode == 0
    print(f"  {'PASS' if ok else 'FAIL'} ({elapsed:.1f}s, exit {result.returncode})")
    return name, ok, elapsed


def main() -> int:
    parser = argparse.ArgumentParser(description="Local CI gate runner.")
    parser.add_argument("--fast", action="store_true",
                        help="unit-only pytest, parallel; skip integration + live tiers")
    parser.add_argument("--no-tests", action="store_true",
                        help="lint + mypy only, no pytest")
    args = parser.parse_args()

    stages: list[tuple[str, list[str]]] = [
        # CI's blocking flake8 pass: syntax errors / undefined names / bad imports.
        ("flake8 (errors)", [PY, "-m", "flake8", "src/", "--count",
                             "--select=E9,F63,F7,F82", "--show-source", "--statistics"]),
        ("missing-deps", [PY, "scripts/check_missing_deps.py"]),
        ("mypy", [PY, "-m", "mypy", "src/", "--config-file", "mypy.ini"]),
    ]

    if not args.no_tests:
        if args.fast:
            pytest_cmd = [PY, "-m", "pytest", "-n", "auto",
                          "-m", "not integration and not zammad_live "
                                "and not llm_live and not discord_live"]
        else:
            # Exactly what CI runs. Live tiers auto-skip without credentials.
            pytest_cmd = [PY, "-m", "pytest", "-m", "not integration"]
        stages.append(("pytest", pytest_cmd))

    results = [run(name, cmd) for name, cmd in stages]

    print(f"\n{'=' * 70}\nSUMMARY\n{'=' * 70}")
    width = max(len(n) for n, _, _ in results)
    for name, ok, elapsed in results:
        mark = "PASS" if ok else "FAIL"
        print(f"  {name.ljust(width)}  {mark}  ({elapsed:.1f}s)")

    failed = [n for n, ok, _ in results if not ok]
    if failed:
        print(f"\n[X] {len(failed)} stage(s) failed: {', '.join(failed)}")
        return 1
    print("\n[OK] All stages passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
