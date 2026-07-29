# tests/utils/test_cc_sandbox.py
"""Unit tests for the unified `claude --settings` sandbox builder (DP-314).

Before DP-314 the cc-* engine route and the fixr dispatcher each built this
block independently; `docs/capability_map.md` recorded the pair as a
security-relevant divergence risk. The parity tests at the bottom are the
regression guard: they fail if the two call sites drift apart again.
"""

from typing import Any, Dict, Optional

import pytest

from config import global_config
from src.utils.cc_sandbox import build_sandbox_settings


@pytest.fixture
def sandboxed(monkeypatch):
    """CC_SANDBOX on, no domains, no weaker-nested — the neutral baseline."""
    monkeypatch.setattr(global_config, "CC_SANDBOX", True)
    monkeypatch.setattr(global_config, "CC_SANDBOX_WEAKER_NESTED", False)
    monkeypatch.setattr(global_config, "CC_SANDBOX_ALLOWED_DOMAINS", [])


def _sandbox(settings: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    assert settings is not None
    return settings["sandbox"]


def test_returns_none_when_sandbox_disabled(monkeypatch):
    """None, not an empty block — the caller omits `--settings` entirely."""
    monkeypatch.setattr(global_config, "CC_SANDBOX", False)
    assert build_sandbox_settings() is None
    assert build_sandbox_settings(allow_write=["/srv/notes"]) is None


def test_baseline_auto_allows_sandboxed_bash(sandboxed):
    sandbox = _sandbox(build_sandbox_settings())
    assert sandbox["enabled"] is True
    # A headless run cannot answer an approval prompt; the OS boundary is what
    # bounds it instead.
    assert sandbox["autoAllowBashIfSandboxed"] is True
    assert "network" not in sandbox
    assert "filesystem" not in sandbox


def test_weaker_nested_flag_propagates(sandboxed, monkeypatch):
    monkeypatch.setattr(global_config, "CC_SANDBOX_WEAKER_NESTED", True)
    assert _sandbox(build_sandbox_settings())["enableWeakerNestedSandbox"] is True


def test_extra_domains_append_without_duplicating(sandboxed, monkeypatch):
    monkeypatch.setattr(global_config, "CC_SANDBOX_ALLOWED_DOMAINS", ["github.com"])
    sandbox = _sandbox(build_sandbox_settings(extra_domains=["bridge.test", "github.com"]))
    assert sandbox["network"]["allowedDomains"] == ["github.com", "bridge.test"]


def test_allow_write_only_appears_when_paths_given(sandboxed):
    """An empty `filesystem` key would widen nothing but would still be a lie in
    the payload; omit it."""
    assert "filesystem" not in _sandbox(build_sandbox_settings(allow_write=[]))
    sandbox = _sandbox(build_sandbox_settings(allow_write=["/srv/notes"]))
    assert sandbox["filesystem"]["allowWrite"] == ["/srv/notes"]


def test_allow_write_deduplicates_and_drops_empties(sandboxed):
    sandbox = _sandbox(
        build_sandbox_settings(allow_write=["/srv/notes", "", "/srv/notes", "/srv/x"])
    )
    assert sandbox["filesystem"]["allowWrite"] == ["/srv/notes", "/srv/x"]


# --- parity: the two call sites must not drift again -------------------------


def test_engine_and_dispatcher_agree_on_common_policy(sandboxed, monkeypatch):
    """`providers/cc.build_cc_sandbox_settings` and `dispatcher._sandbox_settings`
    were independent copies until DP-314. With no notes clone and a non-capable
    dispatch, they must now produce byte-identical settings."""
    import src.engine.providers.cc as cc
    from src.self_edit.dispatcher import Dispatcher

    monkeypatch.setattr(global_config, "CC_SANDBOX_ALLOWED_DOMAINS", ["github.com"])
    monkeypatch.setattr(cc, "notes_allow_write_paths", lambda: [])
    monkeypatch.setattr(
        "src.self_edit.dispatcher.notes_allow_write_paths", lambda: []
    )

    assert cc.build_cc_sandbox_settings() == Dispatcher._sandbox_settings()


def test_both_call_sites_grant_write_to_the_notes_clone(sandboxed, monkeypatch):
    """The link into the workspace points OUT of it, so without this entry the
    agent's memory writes die with an EACCES it cannot interpret."""
    import src.engine.providers.cc as cc
    from src.self_edit.dispatcher import Dispatcher

    monkeypatch.setattr(cc, "notes_allow_write_paths", lambda: ["/srv/notes"])
    monkeypatch.setattr(
        "src.self_edit.dispatcher.notes_allow_write_paths", lambda: ["/srv/notes"]
    )

    for settings in (cc.build_cc_sandbox_settings(), Dispatcher._sandbox_settings()):
        assert _sandbox(settings)["filesystem"]["allowWrite"] == ["/srv/notes"]
