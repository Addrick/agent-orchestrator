"""Tests for scripts/agy_config.py — the agy dispatcher config layer.

Covers config loading/validation, AGENTS.md staging (with no-clobber + restore),
and idempotent atomic MCP-server merging into the global mcp_config.json.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.agy_config import (
    DEFAULT_AGY_MODEL,
    DispatchConfig,
    DispatchConfigError,
    ensure_mcp_servers,
    load_dispatch_config,
    prune_workspace_on_exit,
    set_cli_model,
    stage_agents_md,
)


# ---------- load_dispatch_config ----------


def _write(path: Path, obj) -> Path:
    path.write_text(json.dumps(obj), encoding="utf-8")
    return path


def test_load_happy_all_fields(tmp_path: Path) -> None:
    cfg = _write(
        tmp_path / "c.json",
        {
            "model": "gemini-3-pro",
            "agents_md": "be good",
            "agents_md_path": "/x/AGENTS.md",
            "mcp_servers": {"t": {"command": "python3", "args": ["s.py"]}},
        },
    )
    out = load_dispatch_config(cfg)
    assert out.model == "gemini-3-pro"
    assert out.agents_md == "be good"
    assert out.agents_md_path == Path("/x/AGENTS.md")
    assert out.mcp_servers == {"t": {"command": "python3", "args": ["s.py"]}}


def test_load_empty_object_yields_defaults(tmp_path: Path) -> None:
    out = load_dispatch_config(_write(tmp_path / "c.json", {}))
    assert out == DispatchConfig()


def test_load_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(DispatchConfigError):
        load_dispatch_config(tmp_path / "nope.json")


def test_load_non_json_raises(tmp_path: Path) -> None:
    p = tmp_path / "c.json"
    p.write_text("not json{", encoding="utf-8")
    with pytest.raises(DispatchConfigError):
        load_dispatch_config(p)


def test_load_non_object_raises(tmp_path: Path) -> None:
    with pytest.raises(DispatchConfigError):
        load_dispatch_config(_write(tmp_path / "c.json", [1, 2, 3]))


def test_load_wrong_type_fields_raise(tmp_path: Path) -> None:
    for bad in (
        {"model": 5},
        {"agents_md": 5},
        {"agents_md_path": 5},
        {"mcp_servers": []},
        {"mcp_servers": {"t": "notdict"}},
    ):
        with pytest.raises(DispatchConfigError):
            load_dispatch_config(_write(tmp_path / "c.json", bad))


# ---------- stage_agents_md ----------


def test_stage_inline_writes_and_restore_removes(tmp_path: Path) -> None:
    cfg = DispatchConfig(agents_md="you are a subagent")
    restore = stage_agents_md(tmp_path, cfg)
    staged = tmp_path / "AGENTS.md"
    assert staged.read_text() == "you are a subagent"
    restore()
    assert not staged.exists()


def test_stage_from_path(tmp_path: Path) -> None:
    src = tmp_path / "persona.md"
    src.write_text("persona from file", encoding="utf-8")
    cfg = DispatchConfig(agents_md_path=src)
    restore = stage_agents_md(tmp_path / "wt", DispatchConfig())  # nothing
    restore()  # no-op safe
    wt = tmp_path / "wt"
    wt.mkdir()
    restore = stage_agents_md(wt, cfg)
    assert (wt / "AGENTS.md").read_text() == "persona from file"
    restore()


def test_stage_inline_precedence_over_path(tmp_path: Path) -> None:
    src = tmp_path / "persona.md"
    src.write_text("from file", encoding="utf-8")
    cfg = DispatchConfig(agents_md="inline wins", agents_md_path=src)
    stage_agents_md(tmp_path, cfg)
    assert (tmp_path / "AGENTS.md").read_text() == "inline wins"


def test_stage_no_clobber_appends_and_restores_original(tmp_path: Path) -> None:
    existing = tmp_path / "AGENTS.md"
    existing.write_text("PROJECT RULES\n", encoding="utf-8")
    cfg = DispatchConfig(agents_md="subagent contract")
    restore = stage_agents_md(tmp_path, cfg)
    merged = existing.read_text()
    assert "PROJECT RULES" in merged
    assert "subagent contract" in merged
    # Restore must put the original byte-for-byte back, not leave our block.
    restore()
    assert existing.read_text() == "PROJECT RULES\n"


def test_stage_nothing_configured_is_noop(tmp_path: Path) -> None:
    restore = stage_agents_md(tmp_path, DispatchConfig())
    assert not (tmp_path / "AGENTS.md").exists()
    restore()  # must not raise


def test_stage_missing_path_raises(tmp_path: Path) -> None:
    cfg = DispatchConfig(agents_md_path=tmp_path / "does-not-exist.md")
    with pytest.raises(DispatchConfigError):
        stage_agents_md(tmp_path, cfg)


# ---------- ensure_mcp_servers ----------


def test_ensure_empty_is_noop(tmp_path: Path) -> None:
    cfg_path = tmp_path / "mcp_config.json"
    assert ensure_mcp_servers({}, config_path=cfg_path) == []
    assert not cfg_path.exists()


def test_ensure_writes_fresh_config(tmp_path: Path) -> None:
    cfg_path = tmp_path / "config" / "mcp_config.json"
    servers = {"t": {"command": "python3", "args": ["s.py"]}}
    ensured = ensure_mcp_servers(servers, config_path=cfg_path)
    assert ensured == ["t"]
    data = json.loads(cfg_path.read_text())
    assert data == {"mcpServers": {"t": {"command": "python3", "args": ["s.py"]}}}


def test_ensure_merges_preserving_existing(tmp_path: Path) -> None:
    cfg_path = tmp_path / "mcp_config.json"
    cfg_path.write_text(
        json.dumps({"mcpServers": {"keep": {"command": "node"}}}),
        encoding="utf-8",
    )
    ensure_mcp_servers({"new": {"command": "python3"}}, config_path=cfg_path)
    data = json.loads(cfg_path.read_text())
    assert set(data["mcpServers"]) == {"keep", "new"}
    assert data["mcpServers"]["keep"] == {"command": "node"}


def test_ensure_is_idempotent(tmp_path: Path) -> None:
    cfg_path = tmp_path / "mcp_config.json"
    servers = {"t": {"command": "python3"}}
    ensure_mcp_servers(servers, config_path=cfg_path)
    first = cfg_path.read_text()
    ensure_mcp_servers(servers, config_path=cfg_path)
    assert cfg_path.read_text() == first


def test_ensure_recovers_from_corrupt_global_file(tmp_path: Path) -> None:
    cfg_path = tmp_path / "mcp_config.json"
    cfg_path.write_text("}{ corrupt", encoding="utf-8")
    ensure_mcp_servers({"t": {"command": "python3"}}, config_path=cfg_path)
    data = json.loads(cfg_path.read_text())
    assert data == {"mcpServers": {"t": {"command": "python3"}}}


def test_ensure_no_tmp_file_left_behind(tmp_path: Path) -> None:
    cfg_path = tmp_path / "mcp_config.json"
    ensure_mcp_servers({"t": {"command": "python3"}}, config_path=cfg_path)
    leftovers = list(tmp_path.glob("*.dispatch-tmp"))
    assert leftovers == []


# ---------- prune_workspace_on_exit ----------


def _project_json(pid: str, folder: Path) -> dict:
    """Shape of an agy --add-dir workspace record."""
    return {
        "id": pid,
        "name": str(folder),
        "projectResources": {"resources": [{"folderUri": folder.as_uri()}]},
    }


def test_prune_removes_only_new_workspace_for_worktree(tmp_path: Path) -> None:
    projects = tmp_path / "projects"
    projects.mkdir()
    worktree = tmp_path / "wt"
    worktree.mkdir()
    # A pre-existing, unrelated real project that must survive.
    other = projects / "keep.json"
    _write(other, _project_json("keep", tmp_path / "some-other-repo"))

    prune = prune_workspace_on_exit(worktree, projects_dir=projects)
    # agy now registers the worktree as a one-off workspace mid-run.
    ours = projects / "ours.json"
    _write(ours, _project_json("ours", worktree))

    prune()
    assert not ours.exists()          # our one-off workspace pruned
    assert other.exists()             # unrelated project untouched


def test_prune_keeps_preexisting_workspace_for_same_worktree(tmp_path: Path) -> None:
    projects = tmp_path / "projects"
    projects.mkdir()
    worktree = tmp_path / "wt"
    worktree.mkdir()
    # A workspace for this same path that already existed at snapshot time is
    # the user's own — only entries created during the run are pruned.
    pre = projects / "pre.json"
    _write(pre, _project_json("pre", worktree))

    prune = prune_workspace_on_exit(worktree, projects_dir=projects)
    prune()
    assert pre.exists()


def test_prune_matches_git_folder_uri(tmp_path: Path) -> None:
    projects = tmp_path / "projects"
    projects.mkdir()
    worktree = tmp_path / "wt"
    worktree.mkdir()
    prune = prune_workspace_on_exit(worktree, projects_dir=projects)
    nested = projects / "git.json"
    _write(nested, {
        "id": "g",
        "name": "label",
        "projectResources": {
            "resources": [{"gitFolder": {"folderUri": worktree.as_uri()}}]
        },
    })
    prune()
    assert not nested.exists()


def test_prune_ignores_foreign_and_corrupt_files(tmp_path: Path) -> None:
    projects = tmp_path / "projects"
    projects.mkdir()
    worktree = tmp_path / "wt"
    worktree.mkdir()
    prune = prune_workspace_on_exit(worktree, projects_dir=projects)
    corrupt = projects / "corrupt.json"
    corrupt.write_text("}{ not json", encoding="utf-8")
    foreign = projects / "foreign.json"
    _write(foreign, _project_json("f", tmp_path / "elsewhere"))
    prune()
    assert corrupt.exists() and foreign.exists()


def test_prune_missing_projects_dir_is_noop(tmp_path: Path) -> None:
    # No projects dir at all (fresh machine) must not raise.
    prune = prune_workspace_on_exit(tmp_path / "wt", projects_dir=tmp_path / "nope")
    prune()


# ---------- set_cli_model ----------


def test_set_cli_model_writes_and_restores_prior(tmp_path: Path) -> None:
    settings = tmp_path / "settings.json"
    settings.write_text(json.dumps({
        "colorScheme": "tokyo night", "model": "Gemini 3.5 Flash (Low)",
    }), encoding="utf-8")
    restore = set_cli_model(DispatchConfig(model="Gemini 3.5 Flash (High)"),
                            settings_path=settings)
    data = json.loads(settings.read_text())
    assert data["model"] == "Gemini 3.5 Flash (High)"
    assert data["colorScheme"] == "tokyo night"   # other keys preserved
    restore()
    assert json.loads(settings.read_text())["model"] == "Gemini 3.5 Flash (Low)"


def test_set_cli_model_defaults_when_config_unset(tmp_path: Path) -> None:
    settings = tmp_path / "settings.json"
    settings.write_text(json.dumps({"model": "Gemini 3.1 Pro (High)"}), encoding="utf-8")
    set_cli_model(DispatchConfig(), settings_path=settings)
    assert json.loads(settings.read_text())["model"] == DEFAULT_AGY_MODEL


def test_set_cli_model_restore_removes_key_when_absent(tmp_path: Path) -> None:
    settings = tmp_path / "settings.json"
    settings.write_text(json.dumps({"colorScheme": "x"}), encoding="utf-8")  # no model
    restore = set_cli_model(DispatchConfig(model="Gemini 3.5 Flash (High)"),
                            settings_path=settings)
    assert json.loads(settings.read_text())["model"] == "Gemini 3.5 Flash (High)"
    restore()
    data = json.loads(settings.read_text())
    assert "model" not in data           # restored to absent
    assert data["colorScheme"] == "x"


def test_set_cli_model_noop_when_already_target(tmp_path: Path) -> None:
    settings = tmp_path / "settings.json"
    settings.write_text(json.dumps({"model": DEFAULT_AGY_MODEL}), encoding="utf-8")
    before = settings.read_text()
    restore = set_cli_model(DispatchConfig(), settings_path=settings)
    assert settings.read_text() == before   # untouched
    restore()
    assert settings.read_text() == before


def test_set_cli_model_creates_missing_settings_file(tmp_path: Path) -> None:
    settings = tmp_path / "nested" / "settings.json"
    restore = set_cli_model(DispatchConfig(model="Gemini 3.5 Flash (High)"),
                            settings_path=settings)
    assert json.loads(settings.read_text())["model"] == "Gemini 3.5 Flash (High)"
    restore()
    # Was absent before -> key removed on restore (file may remain, key gone).
    assert "model" not in json.loads(settings.read_text())


def test_set_cli_model_recovers_from_corrupt_settings(tmp_path: Path) -> None:
    settings = tmp_path / "settings.json"
    settings.write_text("}{ not json", encoding="utf-8")
    set_cli_model(DispatchConfig(model="Gemini 3.5 Flash (High)"), settings_path=settings)
    assert json.loads(settings.read_text())["model"] == "Gemini 3.5 Flash (High)"
