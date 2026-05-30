"""Dispatcher config for the agy subagent (DP-127).

Loads an optional JSON config that controls the three external knobs the agy
CLI actually exposes, as established by the 2026-05-29 doc review (see
``memory/project/decisions/2026-05-29-agy-sdk-oauth-finding.md``):

- ``model`` — passed to ``agy -p`` as ``--model``. NOTE: in print mode on the
  OAuth tier this is currently a **no-op** (the tier serves Gemini 3.5 Flash
  regardless; an invalid id doesn't even error). We still thread it through for
  forward-compatibility — if Google opens more print-tier models it just works.

- ``agents_md`` / ``agents_md_path`` — persona / task contract. agy auto-reads
  ``AGENTS.md`` from the workspace, so we stage one into the worktree before
  spawn (and restore the worktree afterwards so a project's own ``AGENTS.md`` is
  never clobbered). This is the clean way to inject the subagent contract and
  also tightens the CLI's weaker context isolation — no ``--system`` flag exists.

- ``mcp_servers`` — custom tools. agy auto-loads MCP servers from the **global**
  ``~/.gemini/config/mcp_config.json`` (the OAuth creds live in the data dir, so
  we cannot isolate a per-run data dir without breaking auth). We merge declared
  servers in idempotently and atomically, preserving any already present. Schema
  is the standard gemini-cli ``mcpServers`` map (stdio ``command``/``args``/``env``
  or HTTP/SSE ``url``).

Config JSON shape (all keys optional)::

    {
      "model": "gemini-3-pro",
      "agents_md": "You are a coding subagent. ...",
      "agents_md_path": "/abs/path/to/AGENTS.md",
      "mcp_servers": {
        "my_tool": {"command": "python3", "args": ["server.py"], "env": {"K": "V"}}
      }
    }
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

# agy reads MCP servers from this global file (gemini-cli lineage). Overridable
# in tests via the ``config_path`` argument of ``ensure_mcp_servers``.
DEFAULT_MCP_CONFIG_PATH = Path.home() / ".gemini" / "config" / "mcp_config.json"

# Name of the file agy auto-reads from the workspace for persona/contract.
AGENTS_FILENAME = "AGENTS.md"


class DispatchConfigError(ValueError):
    """Raised when a dispatcher config file is present but malformed."""


@dataclass
class DispatchConfig:
    """Parsed dispatcher config. All fields optional / default-empty."""

    model: str | None = None
    agents_md: str | None = None
    agents_md_path: Path | None = None
    mcp_servers: dict[str, Any] = field(default_factory=dict)


def _opt_str(data: dict[str, Any], key: str) -> str | None:
    value = data.get(key)
    if value is not None and not isinstance(value, str):
        raise DispatchConfigError(f"config '{key}' must be a string")
    return value


def _validate_mcp_servers(data: dict[str, Any]) -> dict[str, Any]:
    mcp_servers = data.get("mcp_servers", {})
    if not isinstance(mcp_servers, dict):
        raise DispatchConfigError("config 'mcp_servers' must be an object")
    for name, spec in mcp_servers.items():
        if not isinstance(spec, dict):
            raise DispatchConfigError(f"config 'mcp_servers.{name}' must be an object")
    return mcp_servers


def load_dispatch_config(path: Path) -> DispatchConfig:
    """Load and validate a dispatcher config JSON file.

    Raises :class:`DispatchConfigError` if the file is unreadable, not an
    object, or has a field of the wrong type.
    """
    try:
        raw_text = path.read_text(encoding="utf-8")
    except OSError as e:
        raise DispatchConfigError(f"cannot read config {path}: {e}") from e
    try:
        data = json.loads(raw_text)
    except json.JSONDecodeError as e:
        raise DispatchConfigError(f"config {path} is not valid JSON: {e}") from e
    if not isinstance(data, dict):
        raise DispatchConfigError(f"config {path} must be a JSON object")

    agents_md_path_raw = _opt_str(data, "agents_md_path")
    return DispatchConfig(
        model=_opt_str(data, "model"),
        agents_md=_opt_str(data, "agents_md"),
        agents_md_path=Path(agents_md_path_raw) if agents_md_path_raw else None,
        mcp_servers=_validate_mcp_servers(data),
    )


def _resolve_agents_md_text(config: DispatchConfig) -> str | None:
    """Return the AGENTS.md content to stage, or None if none configured.

    Inline ``agents_md`` takes precedence over ``agents_md_path``.
    """
    if config.agents_md is not None:
        return config.agents_md
    if config.agents_md_path is not None:
        try:
            return config.agents_md_path.read_text(encoding="utf-8")
        except OSError as e:
            raise DispatchConfigError(
                f"cannot read agents_md_path {config.agents_md_path}: {e}"
            ) from e
    return None


def stage_agents_md(worktree: Path, config: DispatchConfig) -> Callable[[], None]:
    """Stage ``AGENTS.md`` into the worktree; return a restore callable.

    If the worktree already has an ``AGENTS.md`` (a project's own), its bytes are
    captured and our content is appended below a delimiter so project rules are
    preserved, not clobbered. The returned callable restores the original state
    (rewrites the saved bytes, or removes the file we created) and must be called
    in the caller's ``finally``. A no-op callable is returned when nothing is
    staged.
    """
    text = _resolve_agents_md_text(config)
    if text is None:
        return lambda: None

    target = worktree / AGENTS_FILENAME
    existed = target.exists()
    original = target.read_text(encoding="utf-8") if existed else None

    if original is not None:
        merged = (
            original.rstrip("\n")
            + "\n\n<!-- agy_dispatch: injected subagent contract -->\n"
            + text
        )
    else:
        merged = text
    target.write_text(merged, encoding="utf-8")

    def restore() -> None:
        try:
            if original is not None:
                target.write_text(original, encoding="utf-8")
            elif target.exists():
                target.unlink()
        except OSError:
            pass

    return restore


def ensure_mcp_servers(
    servers: dict[str, Any],
    config_path: Path | None = None,
) -> list[str]:
    """Idempotently merge ``servers`` into agy's global ``mcp_config.json``.

    Preserves any servers already present (only the declared names are added /
    overwritten). Writes atomically (temp file + ``os.replace``) so a concurrent
    agy run never sees a half-written file. Returns the sorted list of names we
    ensured. No-op (returns ``[]``) when ``servers`` is empty.

    ``config_path`` is resolved at call time (not import time) so tests can
    monkeypatch :data:`DEFAULT_MCP_CONFIG_PATH`.
    """
    if not servers:
        return []
    if config_path is None:
        config_path = DEFAULT_MCP_CONFIG_PATH

    existing: dict[str, Any] = {}
    if config_path.exists():
        try:
            loaded = json.loads(config_path.read_text(encoding="utf-8") or "{}")
            if isinstance(loaded, dict) and isinstance(loaded.get("mcpServers"), dict):
                existing = dict(loaded["mcpServers"])
        except (OSError, json.JSONDecodeError):
            # A corrupt/empty global file shouldn't abort the dispatch; start
            # from an empty server map and overwrite cleanly.
            existing = {}

    existing.update(servers)
    payload = {"mcpServers": existing}

    config_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = config_path.with_suffix(config_path.suffix + ".dispatch-tmp")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(tmp, config_path)

    return sorted(servers.keys())
