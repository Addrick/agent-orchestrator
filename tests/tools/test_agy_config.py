"""Tests for scripts/agy_config.py — the agy dispatcher config layer.

Covers config loading/validation, AGENTS.md staging (with no-clobber + restore),
and idempotent atomic MCP-server merging into the global mcp_config.json.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.agy_config import (
    DispatchConfig,
    DispatchConfigError,
    ensure_mcp_servers,
    load_dispatch_config,
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
