"""Dispatcher config for the agy subagent (DP-127).

Loads an optional JSON config that controls the three external knobs the agy
CLI actually exposes, as established by the 2026-05-29 doc review (see
``memory/project/decisions/2026-05-29-agy-sdk-oauth-finding.md``):

- ``model`` — the agy CLI model, written to ``antigravity-cli/settings.json``'s
  ``model`` key (its **human-readable display name**, e.g. ``Gemini 3.5 Flash
  (High)`` — exactly what ``agy models`` prints). This is the real, externally
  manageable lever: the ``--model`` CLI flag is a no-op in print mode, and the
  ``last_selected_agent_model`` placeholder id in ``antigravity_state.pbtxt`` is
  ignored when written externally (verified 2026-06-03). We set this key before
  the run and restore the prior value afterward. Defaults to
  :data:`DEFAULT_AGY_MODEL` when unset.

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

# agy auto-registers every ``--add-dir`` folder as a "project" (workspace) here,
# one JSON per folder, and never removes them — so each dispatch leaves a
# one-off worktree workspace cluttering the Antigravity UI. We prune the entry
# for our disposable worktree after the run. Overridable in tests.
DEFAULT_PROJECTS_DIR = Path.home() / ".gemini" / "config" / "projects"

# The agy CLI's own settings file. Its ``model`` key (a human-readable display
# name like "Gemini 3.5 Flash (High)") is the one lever that actually selects the
# CLI model — confirmed by the interactive header reflecting the written value.
# Overridable in tests.
DEFAULT_AGY_CLI_SETTINGS_PATH = Path.home() / ".gemini" / "antigravity-cli" / "settings.json"

# Default model for dispatched subagents when a config doesn't specify one.
# "High" = Gemini 3.5 Flash with high thinking effort.
DEFAULT_AGY_MODEL = "Gemini 3.5 Flash (High)"

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


def _read_json_object(path: Path) -> dict[str, Any]:
    """Best-effort load of a JSON object file; ``{}`` if missing/corrupt."""
    try:
        loaded = json.loads(path.read_text(encoding="utf-8") or "{}")
    except (OSError, json.JSONDecodeError):
        return {}
    return loaded if isinstance(loaded, dict) else {}


def _restore_settings_model(
    settings_path: Path, had_model: bool, original_model: Any
) -> None:
    """Put the ``model`` key back to its prior value (or remove it)."""
    current = _read_json_object(settings_path)
    if had_model:
        current["model"] = original_model
    else:
        current.pop("model", None)
    try:
        settings_path.write_text(json.dumps(current, indent=2), encoding="utf-8")
    except OSError:
        pass


def set_cli_model(
    config: DispatchConfig,
    settings_path: Path | None = None,
) -> Callable[[], None]:
    """Set the agy CLI ``model`` for this run; return a restore callable.

    Writes ``config.model`` (or :data:`DEFAULT_AGY_MODEL`) into the agy CLI
    settings file's ``model`` key, preserving all other keys, and returns a
    callable that restores the prior value (or removes the key if it was absent).
    Must be called in the caller's ``finally``. Best-effort and non-fatal: a
    missing/corrupt settings file is replaced with a minimal one, and a
    write/restore IO error never aborts the dispatch.
    """
    if settings_path is None:
        settings_path = DEFAULT_AGY_CLI_SETTINGS_PATH
    target_model = config.model or DEFAULT_AGY_MODEL

    settings = _read_json_object(settings_path)
    had_model = "model" in settings
    original_model = settings.get("model")
    if original_model == target_model:
        return lambda: None  # already desired — nothing to change or restore

    settings["model"] = target_model
    try:
        settings_path.parent.mkdir(parents=True, exist_ok=True)
        settings_path.write_text(json.dumps(settings, indent=2), encoding="utf-8")
    except OSError:
        return lambda: None

    return lambda: _restore_settings_model(settings_path, had_model, original_model)


def _project_json_targets_worktree(path: Path, worktree_resolved: str) -> bool:
    """True if an Antigravity project JSON registers ``worktree_resolved``.

    Matches either the top-level ``name`` (agy stores the folder path there for
    an ``--add-dir`` workspace) or any ``folderUri``/``gitFolder.folderUri`` in
    ``projectResources``. Unreadable/foreign files are treated as non-matching.
    """
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    if not isinstance(data, dict):
        return False
    if data.get("name") == worktree_resolved:
        return True
    want_uri = Path(worktree_resolved).as_uri()
    resources = data.get("projectResources", {})
    for res in (resources.get("resources", []) if isinstance(resources, dict) else []):
        if not isinstance(res, dict):
            continue
        if res.get("folderUri") == want_uri:
            return True
        git_folder = res.get("gitFolder")
        if isinstance(git_folder, dict) and git_folder.get("folderUri") == want_uri:
            return True
    return False


def _prune_new_workspace_files(
    projects_dir: Path, before: set[str], worktree_resolved: str
) -> None:
    """Delete project JSONs created since ``before`` that register the worktree."""
    try:
        current = list(projects_dir.glob("*.json"))
    except OSError:
        return
    for path in current:
        if path.name in before:
            continue
        if _project_json_targets_worktree(path, worktree_resolved):
            try:
                path.unlink()
            except OSError:
                pass


def prune_workspace_on_exit(
    worktree: Path,
    projects_dir: Path | None = None,
) -> Callable[[], None]:
    """Snapshot the projects dir now; return a callable that prunes ours later.

    Call BEFORE spawning agy (snapshots the existing project JSONs); the returned
    callable, run in the caller's ``finally``, deletes only project files that
    (a) did not exist at snapshot time AND (b) register ``worktree``. This removes
    the disposable worktree's one-off Antigravity workspace without touching the
    user's real projects. Best-effort — IO errors never abort teardown.
    """
    if projects_dir is None:
        projects_dir = DEFAULT_PROJECTS_DIR
    try:
        before = {p.name for p in projects_dir.glob("*.json")}
    except OSError:
        before = set()
    worktree_resolved = str(worktree.resolve())

    return lambda: _prune_new_workspace_files(projects_dir, before, worktree_resolved)


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
