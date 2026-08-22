"""DP-343 — the node's completion ping, turned into a persona turn.

The property under test throughout is that the PING IS A DOORBELL: it carries a
job id, and everything the persona is told comes from the SSH status read. The
node half (the `curl` in `derpr-model-install` / `derpr-model-tier`) is covered
in tests/services/test_job_completion_ping.py; this file starts at the job id.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from config import global_config
from src.huggingface.completion import JobCompletionBridge


PERSONA = "hypr"
CHANNEL = "infra"
USER = "1234567890"
ALERT_CHANNEL = "999888777"


@pytest.fixture(autouse=True)
def wake_configured(monkeypatch):
    monkeypatch.setattr(global_config, "MODEL_JOB_WAKE_PERSONA", PERSONA, raising=False)
    monkeypatch.setattr(global_config, "MODEL_JOB_WAKE_CHANNEL", CHANNEL, raising=False)
    monkeypatch.setattr(global_config, "MODEL_JOB_WAKE_USER", USER, raising=False)
    monkeypatch.setattr(global_config, "MODEL_JOB_ALERT_CHANNEL_ID", ALERT_CHANNEL,
                        raising=False)


def _job(**over):
    job = {
        "job_id": "newmodel-abc123",
        "state": "done",
        "step": "installed",
        "reason": "",
        "repo": "bartowski/Qwen3-32B-GGUF",
        "file": "Qwen3-32B-Q4_K_M.gguf",
        "name": "newmodel",
        "unit": "koboldcpp-newmodel.service",
        "size_bytes": 20_000_000_000,
        "contextsize": 8192,
    }
    job.update(over)
    return job


def _bridge(status=None, reply="all done", raises=None):
    """A bridge with all three collaborators faked."""
    handler = SimpleNamespace(job_status=AsyncMock(
        return_value=status if status is not None
        else {"status": "ok", "job": _job(), "note": "KV cache for this model: 1024 bytes per token"}
    ))
    chat = SimpleNamespace(generate_response=AsyncMock(
        side_effect=raises if raises else None,
        return_value=(reply, "chat", 1, 2),
    ))
    notifier = SimpleNamespace(send=AsyncMock(return_value=True))
    return JobCompletionBridge(handler, chat, notifier), handler, chat, notifier


# ---------------------------------------------------------------------------
# the happy path
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_finished_install_wakes_persona_and_announces():
    bridge, handler, chat, notifier = _bridge()

    res = await bridge.handle("newmodel-abc123")

    assert res == {"status": "ok", "woke": True, "announced": True,
                   "job_id": "newmodel-abc123", "state": "done"}
    handler.job_status.assert_awaited_once_with("newmodel-abc123")
    kwargs = chat.generate_response.await_args.kwargs
    assert kwargs["persona_name"] == PERSONA
    # The three that make a park approvable: the operator's own identifier and
    # the channel whose history holds the instruction that may say "activate it".
    assert kwargs["user_identifier"] == USER
    assert kwargs["channel"] == CHANNEL
    notifier.send.assert_awaited_once()
    assert notifier.send.await_args.kwargs["recipient"] == ALERT_CHANNEL
    assert notifier.send.await_args.kwargs["body"] == "all done"


@pytest.mark.asyncio
async def test_wake_message_carries_the_verified_facts():
    bridge, _h, chat, _n = _bridge()

    await bridge.handle("newmodel-abc123")

    msg = chat.generate_response.await_args.kwargs["message"]
    assert "newmodel" in msg
    assert "koboldcpp-newmodel.service" in msg
    assert "bartowski/Qwen3-32B-GGUF" in msg
    # The KV arithmetic install_status already computed rides along, so the
    # model sizing a contextsize does not have to ask for it in a second turn.
    assert "KV cache for this model" in msg
    # An install lands DISABLED; the turn must be told not to swap on its own.
    assert "set_active_model" in msg


@pytest.mark.asyncio
async def test_promotion_gets_the_promotion_instruction():
    """A promotion is the step BEFORE serving, and says so.

    `set_active_model` on a cold model starts the copy and returns without
    touching :5001 — so the finished promotion is exactly the moment to call it
    again. An install-shaped instruction here would tell the persona the model
    is installed, which it was, hours ago.
    """
    bridge, _h, chat, _n = _bridge(status={
        "status": "ok",
        "job": _job(kind="promote", step="promoted", repo="", unit="",
                    name="Qwen3-32B-Q4_K_M"),
    })

    await bridge.handle("newmodel-abc123")

    msg = chat.generate_response.await_args.kwargs["message"]
    assert "promotion" in msg
    assert "did NOT change what :5001 is serving" in msg


@pytest.mark.asyncio
async def test_failed_job_reports_step_and_reason():
    bridge, _h, chat, _n = _bridge(status={
        "status": "ok",
        "job": _job(state="failed", step="verify", reason="sha256_mismatch"),
    })

    res = await bridge.handle("newmodel-abc123")

    assert res["state"] == "failed"
    msg = chat.generate_response.await_args.kwargs["message"]
    assert "sha256_mismatch" in msg and "verify" in msg
    assert "Do not retry it on your own" in msg


# ---------------------------------------------------------------------------
# what the ping is NOT allowed to do
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_unreadable_job_does_not_wake_anyone():
    """The node says a job ended; its own status verb cannot produce the record.

    Waking a persona with "something finished, I can't say what" is worse than
    the silence this feature replaced — and it is also the shape a forged ping
    for a nonexistent job takes.
    """
    bridge, _h, chat, notifier = _bridge(status={
        "status": "error", "message": "node script exited 1",
    })

    res = await bridge.handle("no-such-job")

    assert res["status"] == "error"
    chat.generate_response.assert_not_awaited()
    notifier.send.assert_not_awaited()


@pytest.mark.asyncio
async def test_non_terminal_job_is_ignored():
    bridge, _h, chat, _n = _bridge(status={
        "status": "ok", "job": _job(state="running", step="download"),
    })

    res = await bridge.handle("newmodel-abc123")

    assert res["status"] == "ignored"
    chat.generate_response.assert_not_awaited()


@pytest.mark.asyncio
async def test_repeated_ping_wakes_once():
    """The node retries its POST twice.

    A reply lost on the way back is indistinguishable from one that never came,
    so without this a retry re-runs a whole persona turn — and a turn that parks
    `set_active_model` would park it twice.
    """
    bridge, _h, chat, notifier = _bridge()

    first = await bridge.handle("newmodel-abc123")
    second = await bridge.handle("newmodel-abc123")

    assert first["woke"] is True
    assert second == {"status": "ignored", "reason": "already handled",
                      "job_id": "newmodel-abc123"}
    assert chat.generate_response.await_count == 1
    assert notifier.send.await_count == 1


@pytest.mark.asyncio
async def test_empty_job_id_rejected_without_a_status_read():
    bridge, handler, chat, _n = _bridge()

    res = await bridge.handle("   ")

    assert res["status"] == "error"
    handler.job_status.assert_not_awaited()
    chat.generate_response.assert_not_awaited()


# ---------------------------------------------------------------------------
# degraded configurations — every one of these is a live deployment state
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
@pytest.mark.parametrize("missing", ["MODEL_JOB_WAKE_PERSONA",
                                     "MODEL_JOB_WAKE_CHANNEL",
                                     "MODEL_JOB_WAKE_USER"])
async def test_no_wake_when_wake_config_is_incomplete(monkeypatch, missing):
    """Two of the three are what make a parked write approvable.

    A wake filed under a synthetic user, or in a channel nobody talks in, raises
    approval cards `_post_pending_proposals` will never render — so an
    incomplete config must produce no turn at all rather than an unanswerable
    one.
    """
    monkeypatch.setattr(global_config, missing, "", raising=False)
    bridge, _h, chat, notifier = _bridge()

    res = await bridge.handle("newmodel-abc123")

    assert res == {"status": "ok", "woke": False,
                   "job_id": "newmodel-abc123", "state": "done"}
    chat.generate_response.assert_not_awaited()
    notifier.send.assert_not_awaited()


@pytest.mark.asyncio
async def test_no_alert_channel_still_runs_the_turn(monkeypatch):
    """Announcement off, wake on: the turn can still park an activation."""
    monkeypatch.setattr(global_config, "MODEL_JOB_ALERT_CHANNEL_ID", "", raising=False)
    bridge, _h, chat, notifier = _bridge()

    res = await bridge.handle("newmodel-abc123")

    assert res["woke"] is True and res["announced"] is False
    chat.generate_response.assert_awaited_once()
    notifier.send.assert_not_awaited()


@pytest.mark.asyncio
async def test_no_notification_router_does_not_crash():
    handler = SimpleNamespace(job_status=AsyncMock(
        return_value={"status": "ok", "job": _job()}))
    chat = SimpleNamespace(generate_response=AsyncMock(return_value=("hi", "chat", 1, 2)))
    bridge = JobCompletionBridge(handler, chat, None)

    res = await bridge.handle("newmodel-abc123")

    assert res["woke"] is True and res["announced"] is False


@pytest.mark.asyncio
async def test_failed_turn_is_reported_not_raised():
    """The only consumer of the HTTP status is a `curl` in a bash script that
    has already finished the job it is reporting. A 500 there teaches nobody
    anything and the retry re-runs the same failure."""
    bridge, _h, _c, notifier = _bridge(raises=RuntimeError("engine down"))

    res = await bridge.handle("newmodel-abc123")

    assert res["status"] == "error"
    notifier.send.assert_not_awaited()


@pytest.mark.asyncio
async def test_failed_announcement_does_not_fail_the_wake():
    bridge, _h, _c, notifier = _bridge()
    notifier.send = AsyncMock(side_effect=RuntimeError("discord down"))

    res = await bridge.handle("newmodel-abc123")

    assert res["woke"] is True and res["announced"] is False
