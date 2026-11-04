"""DP-491: security floors for the committed lockfiles.

Asserts on the *committed* locks rather than the installed environment: CI installs only
``requirements-dev.txt``, but production runs ``requirements-voice.txt``, so an
installed-metadata check would never see the prod lock.

Each floor is the lowest version that fixes every advisory pip-audit reported against the
lock at the time. A regen that drops a pin back below its floor fails here, not in prod.
Raise a floor when a new advisory lands; delete one only when the package leaves every lock.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest
from packaging.version import Version

REPO_ROOT = Path(__file__).resolve().parent.parent
LOCKS = ("requirements.txt", "requirements-dev.txt", "requirements-voice.txt")

# package -> minimum fixed version (DP-491, pip-audit 2026-11-04)
SECURITY_FLOORS: dict[str, str] = {
    "aiohttp": "3.14.4",
    "anyio": "4.14.2",
    "brotli": "1.2.1",
    "certifi": "2026.10.18",
    "filelock": "4.2.1",
    "h2": "4.4.1",
    "hpack": "4.2.0",
    "mcp": "1.30.0",
    "msgpack": "1.2.3",
    "multidict": "6.9.1",
    "oauthlib": "4.0.0",
    "opentelemetry-api": "1.45.1",
    "pip": "26.2",
    "pyasn1": "0.6.5",
    "pyjwt": "2.15.0",
    "setuptools": "84.1.0",
    "urllib3": "2.8.0",
}

# package -> exclusive upper bound: majors deliberately deferred to their own ticket.
MAJOR_CAPS: dict[str, str] = {
    "mcp": "2",  # mcp 2.x is a breaking client API migration (DP-491 §Deferred)
}

_PIN = re.compile(r"^([A-Za-z0-9][A-Za-z0-9._-]*)==([^\s;]+)", re.MULTILINE)


def _canon(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def _pins(lock: str) -> dict[str, Version]:
    text = (REPO_ROOT / lock).read_text(encoding="utf-8")
    return {_canon(n): Version(v) for n, v in _PIN.findall(text)}


def test_lock_parser_reads_real_pins() -> None:
    # Guards against a parser regression silently making every floor test vacuous.
    pins = _pins("requirements.txt")
    assert len(pins) > 50
    assert "aiohttp" in pins


@pytest.mark.parametrize("lock", LOCKS)
@pytest.mark.parametrize("package,floor", sorted(SECURITY_FLOORS.items()))
def test_lock_meets_security_floor(lock: str, package: str, floor: str) -> None:
    pinned = _pins(lock).get(_canon(package))
    if pinned is None:
        pytest.skip(f"{package} not in {lock}")
    assert pinned >= Version(floor), f"{lock}: {package}=={pinned} is below security floor {floor}"


@pytest.mark.parametrize("lock", LOCKS)
@pytest.mark.parametrize("package,cap", sorted(MAJOR_CAPS.items()))
def test_lock_respects_major_cap(lock: str, package: str, cap: str) -> None:
    pinned = _pins(lock).get(_canon(package))
    if pinned is None:
        pytest.skip(f"{package} not in {lock}")
    assert pinned < Version(cap), f"{lock}: {package}=={pinned} crosses deferred major {cap}"


def test_every_floor_is_present_in_some_lock() -> None:
    # A floor for a package no lock contains anymore is dead weight; prune it.
    present = set().union(*(_pins(lock) for lock in LOCKS))
    stale = sorted(p for p in SECURITY_FLOORS if _canon(p) not in present)
    assert not stale, f"floors for packages in no lock: {stale}"
