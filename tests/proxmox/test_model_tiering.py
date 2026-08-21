"""Hot/cold gguf tiering (DP-340).

The behaviour under test exists because of a real outage: on 2026-08-21 a 22 GB
model download filled the node's thin pool and put CT100 *and* CT101 into ext4
`emergency_ro`, taking zammad, the chatbot and hindsight's database down
together. Downloads now land on the archive disk and only reach the SSD via an
explicit promotion.

Two properties matter more than the happy path and are asserted directly:

1. **A node without the tier script must keep working.** The node artifacts in
   `services/pve/` and the container image are two independent deploys, so this
   code will meet nodes that have never heard of `derpr-model-tier`.
2. **Promoting must not disturb :5001.** A promotion is minutes long; if it
   disabled the running model first, every cold selection would take the
   service down for the length of a copy.
"""

from __future__ import annotations

import pytest

from config import global_config
from src.proxmox.handler import ProxmoxToolHandler
from tests.proxmox.test_proxmox_tools import FakeRunner

UNIT_FILES = (
    "koboldcpp-hotmodel.service  disabled  enabled\n"
    "koboldcpp-coldmodel.service  disabled  enabled"
)

# FakeRunner synthesises `--model /models/<unit>.gguf` from the unit name, so
# the basenames the handler will look up are these.
HOT_FILE = "koboldcpp-hotmodel.service.gguf"
COLD_FILE = "koboldcpp-coldmodel.service.gguf"

TIERS = {
    HOT_FILE: {
        "file": HOT_FILE, "tier": "hot", "size_bytes": 24_000_000_000,
        "pinned": True, "last_served": 1_700_000_000, "archived": True,
    },
    COLD_FILE: {
        "file": COLD_FILE, "tier": "cold", "size_bytes": 24_000_000_000,
        "pinned": False, "last_served": 0, "archived": True,
    },
}


@pytest.fixture
def enabled(monkeypatch):
    monkeypatch.setattr(global_config, "PVE_TOOLS_ENABLED", True)
    monkeypatch.setattr(global_config, "PVE_MODEL_HOST_VMID", "101")


def _handler(**kw) -> tuple[ProxmoxToolHandler, FakeRunner]:
    runner = FakeRunner(unit_files=UNIT_FILES, **kw)
    return ProxmoxToolHandler(runner), runner  # type: ignore[arg-type]


def _joined(runner: FakeRunner) -> list[str]:
    return [" ".join(c) for c in runner.calls]


# -- list_models -------------------------------------------------------------

@pytest.mark.asyncio
async def test_list_models_reports_the_tier(enabled):
    h, _ = _handler(tiers=TIERS)
    out = await h._list_models()
    assert out["status"] == "ok"
    tiers = {m["name"]: m["tier"] for m in out["models"]}
    assert tiers == {"hotmodel": "hot", "coldmodel": "cold"}


@pytest.mark.asyncio
async def test_a_cold_model_is_listed_not_hidden(enabled):
    """Before DP-340 an absent gguf meant "hide the unit" (DP-264, since it
    would take :5001 down). A cold model is a different thing: it is installed
    and selectable, just not instantly. Hiding it would make it look lost."""
    h, _ = _handler(tiers=TIERS)
    out = await h._list_models()
    assert "coldmodel" in [m["name"] for m in out["models"]]


@pytest.mark.asyncio
async def test_a_pin_is_surfaced(enabled):
    h, _ = _handler(tiers=TIERS)
    out = await h._list_models()
    rows = {m["name"]: m for m in out["models"]}
    assert rows["hotmodel"].get("pinned") is True
    assert "pinned" not in rows["coldmodel"]


@pytest.mark.asyncio
async def test_the_result_warns_that_a_cold_pick_is_slow(enabled):
    """The cost of picking a cold model is only knowable from what this call
    just read, so it belongs in the result rather than in a persona prompt."""
    h, _ = _handler(tiers=TIERS)
    out = await h._list_models()
    note = out.get("note", "")
    assert "coldmodel" in note
    assert "promot" in note.lower()
    assert "install_status" in note


@pytest.mark.asyncio
async def test_no_note_when_everything_is_hot(enabled):
    tiers = {HOT_FILE: dict(TIERS[HOT_FILE]),
             COLD_FILE: dict(TIERS[COLD_FILE], tier="hot")}
    h, _ = _handler(tiers=tiers)
    out = await h._list_models()
    assert "note" not in out


# -- set_active_model --------------------------------------------------------

@pytest.mark.asyncio
async def test_a_hot_model_still_swaps_directly(enabled):
    h, runner = _handler(tiers=TIERS)
    out = await h._set_active_model("hotmodel")
    assert out["status"] == "ok"
    assert out.get("state") != "promoting"
    assert out["active_model"] == "hotmodel"
    assert any("systemctl enable --now" in c for c in _joined(runner))


@pytest.mark.asyncio
async def test_a_cold_model_promotes_instead_of_swapping(enabled):
    h, runner = _handler(tiers=TIERS)
    out = await h._set_active_model("coldmodel")
    assert out["status"] == "ok"
    assert out["state"] == "promoting"
    assert out["job_id"].startswith("coldmodel-")
    assert out["file"] == COLD_FILE
    assert any("derpr-model-tier promote" in c for c in _joined(runner))


@pytest.mark.asyncio
async def test_promoting_never_disables_the_running_model(enabled):
    """THE safety property. A promotion is minutes long; disabling first would
    mean every cold selection takes :5001 down for the whole copy, and a failed
    copy would leave it down with nothing serving."""
    h, runner = _handler(tiers=TIERS)
    await h._set_active_model("coldmodel")
    calls = _joined(runner)
    assert not any("systemctl disable" in c for c in calls), calls
    assert not any("systemctl enable" in c for c in calls), calls


@pytest.mark.asyncio
async def test_the_promotion_note_says_a_second_call_is_needed(enabled):
    """A model told only "promoting" will assume :5001 changed. It did not."""
    h, _ = _handler(tiers=TIERS)
    out = await h._set_active_model("coldmodel")
    note = out["note"]
    assert "did NOT change" in note
    assert "install_status" in note
    assert "again" in note


@pytest.mark.asyncio
async def test_the_promoted_filename_comes_from_the_node_not_the_caller(enabled):
    """`exfil_capable: False` depends on nothing the model typed reaching argv.

    The promote argv must carry the node's own basename from `tier list`, never
    the `name` string the caller passed.
    """
    h, runner = _handler(tiers=TIERS)
    await h._set_active_model("coldmodel")
    promote = next(c for c in runner.calls if c[:2][-1:] == ["promote"])
    assert promote[2] == COLD_FILE


@pytest.mark.asyncio
async def test_a_gguf_in_neither_tier_is_refused(enabled):
    """Still the DP-264 refusal: never disable a running model to start a unit
    whose weights do not exist anywhere."""
    h, runner = _handler(tiers={HOT_FILE: TIERS[HOT_FILE]})
    out = await h._set_active_model("coldmodel")
    assert out["status"] == "error"
    assert "either tier" in out["message"]
    assert not any("systemctl disable" in c for c in _joined(runner))


# -- graceful degradation onto a node that has no tier script ----------------

@pytest.mark.asyncio
async def test_a_node_without_the_tier_script_still_lists_models(enabled):
    """The node artifacts and the container image are separate deploys, and
    DP-332 shipped a handler against a six-week-old node. A missing tier script
    must degrade to the pre-DP-340 behaviour, not break every model tool."""
    h, _ = _handler(tiers=None)
    out = await h._list_models()
    assert out["status"] == "ok"
    assert {m["name"] for m in out["models"]} == {"hotmodel", "coldmodel"}
    assert all(m["tier"] == "hot" for m in out["models"])
    assert "note" not in out


@pytest.mark.asyncio
async def test_a_node_without_the_tier_script_still_swaps(enabled):
    h, runner = _handler(tiers=None)
    out = await h._set_active_model("hotmodel")
    assert out["status"] == "ok"
    assert out["active_model"] == "hotmodel"
    assert any("systemctl enable --now" in c for c in _joined(runner))


@pytest.mark.asyncio
async def test_without_the_tier_script_a_missing_gguf_is_still_refused(enabled):
    """The old presence probe has to keep working, or the fallback path quietly
    loses the protection it existed for."""
    h, runner = _handler(tiers=None, present_units={"koboldcpp-hotmodel.service"})
    out = await h._set_active_model("coldmodel")
    assert out["status"] == "error"
    assert "not on disk" in out["message"]
    assert not any("systemctl disable" in c for c in _joined(runner))


@pytest.mark.asyncio
async def test_unparseable_tier_output_degrades_rather_than_raising(enabled):
    """A tier script that answers with garbage is treated as absent. It is the
    one silent failure here, and it is deliberate — the alternative is every
    model tool dying because one node-side script regressed its output."""
    h, runner = _handler(tiers="this is not json at all")
    out = await h._list_models()
    assert out["status"] == "ok"
    # Proof it took the garbage branch and not the missing-script one: the
    # script WAS called and DID exit 0, and the handler still fell back.
    assert any("derpr-model-tier list" in " ".join(c) for c in runner.calls)
    assert all(m["tier"] == "hot" for m in out["models"])
    assert "note" not in out


@pytest.mark.asyncio
async def test_tier_output_with_an_error_status_degrades_too(enabled):
    """`{"status": "error"}` is a well-formed answer meaning "I could not tell
    you", which must not be read as "there are no models"."""
    h, _ = _handler(tiers='{"status": "error", "message": "archive not mounted"}')
    out = await h._list_models()
    assert out["status"] == "ok"
    assert {m["name"] for m in out["models"]} == {"hotmodel", "coldmodel"}
