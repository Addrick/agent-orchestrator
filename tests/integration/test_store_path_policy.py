"""Persistence-store path policy (DP-303).

The rule this file enforces: **a store may own its own file, but not its own
path convention.** Every persistent store must resolve its path through
`config.global_config` and default under `DATA_DIR`, because `DATA_DIR` is the
only directory docker-compose mounts as a volume.

Why it is a test and not a paragraph in a doc: `_TrustOverrideStore` and
`_DocScopeStore` derived their paths from `Path(__file__).parent.parent`, which
placed them in `/app/src/memory/` — inside the image layer. The deploy job does
`docker compose pull && up -d`, so every merge to master destroyed them. That
took every operator "this memory unit is untrusted" flip and the audit trail
proving it existed with it, silently re-trusting quarantined units.

Nothing about that failure was visible in code review: the existing tests all
pass explicit `tmp_path` values, so the defaults were never exercised. The
policy is only enforceable if the *defaults* are what gets asserted.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from config import global_config


def _paths_under(root: Path, candidate: str) -> bool:
    try:
        Path(candidate).resolve().relative_to(root)
    except ValueError:
        return False
    return True


# Every persistent store's configured path. Adding a store means adding a row
# here; that is the point of the test.
STORE_PATH_SETTINGS = [
    "MEMORY_DATABASE_FILE",
    "CC_FIXR_REGISTRY_DB",
    "HINDSIGHT_OVERRIDE_DB",
    "HINDSIGHT_DOC_SCOPE_DB",
    "MCP_SERVERS_FILE",
    "PERSONA_SAVE_FILE",
]


@pytest.mark.parametrize("setting", STORE_PATH_SETTINGS)
def test_store_path_defaults_live_under_data_dir(setting: str) -> None:
    """Persistent state must land on the mounted volume, not in the image."""
    value = getattr(global_config, setting)
    data_dir = Path(global_config.DATA_DIR).resolve()
    assert _paths_under(data_dir, str(value)), (
        f"{setting} resolves to {value!r}, which is outside DATA_DIR "
        f"({data_dir}). Only DATA_DIR is a mounted volume — state stored "
        f"elsewhere is destroyed on every image rebuild (DP-303)."
    )


def test_hindsight_backend_defaults_to_configured_paths(monkeypatch, tmp_path) -> None:
    """The backend must read its store paths from config, not the source tree.

    Constructed with no path arguments — the production call site in
    `MemoryManager` passes none — the two sibling stores must land wherever
    config says. Repointing config and seeing both stores follow is what
    distinguishes "reads config" from "happens to agree with config today".
    """
    from src.memory.backend import hindsight as hindsight_module
    from src.memory.backend.hindsight import HindsightBackend

    override = tmp_path / "ov.db"
    doc_scope = tmp_path / "ds.db"
    monkeypatch.setattr(global_config, "HINDSIGHT_OVERRIDE_DB", str(override))
    monkeypatch.setattr(global_config, "HINDSIGHT_DOC_SCOPE_DB", str(doc_scope))

    backend = HindsightBackend(url="http://localhost:8888")

    assert Path(backend._overrides.db_path) == override
    assert Path(backend._doc_scope.db_path) == doc_scope
    # And the source tree is not a fallback for either.
    src_root = Path(hindsight_module.__file__).resolve().parents[3]
    for path in (backend._overrides.db_path, backend._doc_scope.db_path):
        assert not _paths_under(src_root / "src", path), (
            f"{path} sits inside the source tree; it will be wiped by the next "
            f"image rebuild (DP-303)."
        )
