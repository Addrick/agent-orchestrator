"""AgentManager config-application tests.

Covers the agent schedule wiring, including DP-221's removal of the legacy
`poll_interval` config key (the old format `agents.json` used before the nested
`schedule` block). After the cleanup, `schedule` is the only accepted form and a
bare `poll_interval` is ignored.
"""

import asyncio
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock
from zoneinfo import ZoneInfo

import pytest

from config import global_config
from src.agents.agent_manager import AgentManager
from src.agents.base import Agent


class _NoopAgent(Agent):
    """Minimal agent whose deploy loop idles until stopped."""

    def __init__(self, chat_system):
        super().__init__(chat_system, inject_personas=False)

    async def deploy(self) -> None:  # pragma: no cover - never iterates in tests
        return None


def _make_manager() -> AgentManager:
    chat_system = MagicMock()
    # Point the manager at a non-existent config path so no file config leaks in.
    mgr = AgentManager(
        chat_system=chat_system,
        memory_manager=chat_system.memory_manager,
        config_path=__import__("pathlib").Path("/nonexistent/agents.json"),
    )
    mgr.register("noop", _NoopAgent)
    return mgr


@pytest.mark.asyncio
async def test_schedule_block_is_applied():
    mgr = _make_manager()
    await mgr.start_agent("noop", config_overrides={"schedule": {"interval": 5}})
    try:
        instance = mgr._running["noop"].instance
        assert instance.schedule == {"interval": 5}
    finally:
        await mgr.stop_agent("noop")
        await asyncio.gather(mgr._running["noop"].task, return_exceptions=True)


@pytest.mark.asyncio
async def test_legacy_poll_interval_is_ignored():
    """DP-221: the retired `poll_interval` key no longer maps to a schedule."""
    mgr = _make_manager()
    await mgr.start_agent("noop", config_overrides={"poll_interval": 30})
    try:
        instance = mgr._running["noop"].instance
        # schedule is untouched (class default), not derived from poll_interval.
        assert instance.schedule == {}
        assert "interval" not in instance.schedule
    finally:
        await mgr.stop_agent("noop")
        await asyncio.gather(mgr._running["noop"].task, return_exceptions=True)


# --- daily_at is wall-clock in LOCAL_TZ, not the host clock (DP-412) ---

def test_local_now_follows_local_tz_not_host_clock(monkeypatch):
    monkeypatch.setattr(global_config, "LOCAL_TZ", "Asia/Tokyo")
    assert Agent._local_now().utcoffset() == timedelta(hours=9)


def test_next_daily_run_is_wall_clock_in_local_zone():
    """`daily_at: "17:00"` is 5 PM Eastern on a UTC host, across DST."""
    ny = ZoneInfo("America/New_York")
    utc = timezone.utc
    # 16:30 EDT -> today's 17:00 EDT (21:00 UTC)
    before = datetime(2026, 10, 1, 20, 30, tzinfo=utc).astimezone(ny)
    assert Agent._next_daily_run("17:00", before) == datetime(2026, 10, 1, 21, 0, tzinfo=utc)
    # 17:30 EDT -> already past, so tomorrow's
    after = datetime(2026, 10, 1, 21, 30, tzinfo=utc).astimezone(ny)
    assert Agent._next_daily_run("17:00", after) == datetime(2026, 10, 2, 21, 0, tzinfo=utc)
    # EST: the same wall-clock time is 22:00 UTC
    winter = datetime(2026, 12, 1, 12, 0, tzinfo=ny)
    assert Agent._next_daily_run("17:00", winter) == datetime(2026, 12, 1, 22, 0, tzinfo=utc)


@pytest.mark.asyncio
async def test_signal_stop_all_signals_without_awaiting():
    """DP-304: AppManager needs to signal agents *before* it drains long-running
    tasks — agents and MemoryConsolidator contend on MemoryManager._lock, so an
    agent left deploying for the whole drain window races the work item the
    drain exists to let finish. shutdown_all() awaits, so it can't be that call.
    """
    mgr = _make_manager()
    await mgr.start_agent("noop")
    try:
        instance = mgr._running["noop"].instance
        assert instance._shutdown_event.is_set() is False

        mgr.signal_stop_all()

        # Signalled synchronously; nothing was awaited to make that happen.
        assert instance._shutdown_event.is_set() is True
    finally:
        mgr._running["noop"].instance.stop()
        await asyncio.gather(mgr._running["noop"].task, return_exceptions=True)


@pytest.mark.asyncio
async def test_signal_stop_all_survives_a_failing_agent():
    from src.agents.agent_manager import RunningAgent

    mgr = _make_manager()
    await mgr.start_agent("noop")
    try:
        boom = MagicMock()
        boom.stop.side_effect = RuntimeError("stop exploded")
        mgr._running["boom"] = RunningAgent(
            instance=boom, task=mgr._running["noop"].task
        )

        mgr.signal_stop_all()  # must not propagate

        boom.stop.assert_called_once()
        assert mgr._running["noop"].instance._shutdown_event.is_set() is True
    finally:
        mgr._running.pop("boom", None)
        mgr._running["noop"].instance.stop()
        await asyncio.gather(mgr._running["noop"].task, return_exceptions=True)


# ---------- single-shot inference agents (DP-292) ----------

class _InferenceAgent:
    """Minimal single-shot agent: takes chat_system, optionally a router."""

    def __init__(self, chat_system, notification_router=None):
        self.chat_system = chat_system
        self.notification_router = notification_router

    async def tag(self, body):
        return "ok"


class _RouterlessAgent:
    def __init__(self, chat_system):
        self.chat_system = chat_system


def test_register_and_get_inference_agent_builds_and_caches():
    mgr = _make_manager()
    mgr.register_inference_agent("infer", _RouterlessAgent)
    a = mgr.get_inference_agent("infer")
    assert isinstance(a, _RouterlessAgent)
    assert a.chat_system is mgr._chat_system
    # Cached: same instance on second lookup.
    assert mgr.get_inference_agent("infer") is a


def test_get_unregistered_inference_agent_returns_none():
    mgr = _make_manager()
    assert mgr.get_inference_agent("nope") is None


def test_inference_agent_receives_convention_di():
    """A single-shot agent whose __init__ wants notification_router gets it —
    the same DI path scheduled agents use (this is what unblocks the DM feature)."""
    mgr = _make_manager()
    router = MagicMock()
    mgr.notification_router = router
    mgr.register_inference_agent("infer", _InferenceAgent)
    a = mgr.get_inference_agent("infer")
    assert a.notification_router is router


def test_inference_agent_not_started_as_task():
    mgr = _make_manager()
    mgr.register_inference_agent("infer", _RouterlessAgent)
    mgr.get_inference_agent("infer")
    # Registering/building an inference agent must not launch a loop task.
    assert "infer" not in mgr._running


# ---------- DP-293: agent_config injection must not mutate shared config ----------

class _ConfigAgent(Agent):
    """Agent whose __init__ requests agent_config (triggers the DI merge)."""

    def __init__(self, chat_system, agent_config=None):
        super().__init__(chat_system, inject_personas=False)
        self.agent_config = agent_config

    async def deploy(self) -> None:  # pragma: no cover
        return None


def test_agent_config_injection_does_not_mutate_shared_config():
    """`_recipients` must be added to a COPY, not the live `self._config`
    block — otherwise the recipients map leaks into every later read of that
    agent's config and re-accumulates on each rebuild."""
    mgr = _make_manager()
    mgr.register("cfg", _ConfigAgent)
    mgr._config = {
        "agents": {"cfg": {"persona": "p"}},
        "recipients": {"adrich": {"discord_user_id": "123"}},
    }

    inst = mgr._build_agent_instance("cfg", _ConfigAgent, {})
    # The injected config sees the recipients...
    assert inst.agent_config["_recipients"] == {"adrich": {"discord_user_id": "123"}}
    # ...but the shared source block is untouched.
    assert "_recipients" not in mgr._config["agents"]["cfg"]
    assert mgr._config["agents"]["cfg"] == {"persona": "p"}


# ---------- DP-294: inference-agent injection by matching param name ----------

class _ConsumerAgent(Agent):
    """A scheduled agent whose __init__ names a registered inference agent —
    DP-294's ZammadBot(content_classifier=…) pattern in miniature."""

    def __init__(self, chat_system, infer=None):
        super().__init__(chat_system, inject_personas=False)
        self.infer = infer

    async def deploy(self) -> None:  # pragma: no cover - never iterates
        return None


def test_scheduled_agent_gets_registered_inference_agent_by_di():
    """A scheduled agent whose __init__ names a registered inference agent
    receives the shared, cached instance via convention-DI (DP-294) — no ad-hoc
    construction, and the same object get_inference_agent hands out."""
    mgr = _make_manager()
    mgr.register_inference_agent("infer", _RouterlessAgent)
    kwargs = mgr._di_kwargs("consumer", _ConsumerAgent, {})
    assert isinstance(kwargs["infer"], _RouterlessAgent)
    assert kwargs["infer"] is mgr.get_inference_agent("infer")


def test_unregistered_inference_param_is_not_injected():
    """A param that isn't a registered inference agent is left alone (the
    consumer's own default applies) — the injection is name-scoped to the
    inference registry, not any param."""
    mgr = _make_manager()
    kwargs = mgr._di_kwargs("consumer", _ConsumerAgent, {})
    assert "infer" not in kwargs


# --- DP-451: agents.json is per-instance, seeded once from the shipped copy ---

def _point_config_at(monkeypatch, tmp_path, shipped: dict):
    default = tmp_path / "config" / "agents.json"
    default.parent.mkdir()
    default.write_text(__import__("json").dumps(shipped))
    save = tmp_path / "data" / "agents.json"
    monkeypatch.setattr(global_config, "DEFAULT_AGENTS_FILE", default)
    monkeypatch.setattr(global_config, "AGENTS_SAVE_FILE", save)
    return default, save


def test_missing_instance_config_is_seeded_from_shipped(monkeypatch, tmp_path):
    shipped = {"agents": {"reminder": {"auto_start": True}}}
    _, save = _point_config_at(monkeypatch, tmp_path, shipped)

    mgr = AgentManager(chat_system=MagicMock(), memory_manager=MagicMock())

    assert save.exists()
    assert mgr.config == shipped


def test_instance_config_wins_over_shipped(monkeypatch, tmp_path):
    """A dev box that turned an agent off keeps it off when the shipped copy changes."""
    import json
    _, save = _point_config_at(monkeypatch, tmp_path, {"agents": {"reminder": {"auto_start": True}}})
    save.parent.mkdir()
    local = {"agents": {"reminder": {"auto_start": False}}}
    save.write_text(json.dumps(local))

    mgr = AgentManager(chat_system=MagicMock(), memory_manager=MagicMock())

    assert mgr.config == local
    assert json.loads(save.read_text()) == local


def test_explicit_config_path_is_never_seeded(monkeypatch, tmp_path):
    _point_config_at(monkeypatch, tmp_path, {"agents": {"reminder": {"auto_start": True}}})
    explicit = tmp_path / "elsewhere" / "agents.json"

    mgr = AgentManager(chat_system=MagicMock(), memory_manager=MagicMock(), config_path=explicit)

    assert not explicit.exists()
    assert mgr.config == {}


# --- codr agent (DP-449) --------------------------------------------------------


def test_codr_agent_built_from_the_shipped_config():
    """The shipped `codr` block reaches the agent through DI: target, schedule
    and every tuning key, with nothing else needed."""
    from src.agents.codr_agent import CodrAgent

    mgr = AgentManager(chat_system=MagicMock(), memory_manager=MagicMock(),
                       notification_router=MagicMock(), config_path=global_config.DEFAULT_AGENTS_FILE)
    mgr.register("codr", CodrAgent)
    shipped = mgr.config["agents"]["codr"]
    assert shipped["auto_start"] is False
    agent = mgr._build_agent_instance("codr", CodrAgent, shipped)
    assert agent.cfg["notification_target"] == {"channel": "discord_channel", "recipient": "1562589289332609075"}
    assert agent.cfg["pool_url"] == "http://10.0.0.70:5002/v1"
    assert agent.persona_name == "codr" and agent.directory.endswith("/codr")
    assert "_recipients" not in agent.cfg
    assert {t.id for t in agent._tasks()} >= {"01-csvstats", "02-seeded-bug", "03-notes-api"}


def test_codr_agent_with_an_empty_config_block_uses_the_defaults(tmp_path):
    """An instance agents.json seeded before DP-449 has no `codr` block."""
    from src.agents.codr_agent import DEFAULTS, CodrAgent

    mgr = AgentManager(chat_system=MagicMock(), memory_manager=MagicMock(),
                       notification_router=MagicMock(), config_path=tmp_path / "agents.json")
    mgr.register("codr", CodrAgent)
    agent = mgr._build_agent_instance("codr", CodrAgent, {})
    assert {k: agent.cfg[k] for k in DEFAULTS} == DEFAULTS
    assert "notification_target" not in agent.cfg      # posts are logged, not sent


def test_codr_agent_registered_at_startup():
    from src.agents.codr_agent import CodrAgent
    from src.main import _register_agents

    for zammad in (MagicMock(), None):
        manager = MagicMock()
        _register_agents(manager, zammad_client=zammad)
        registered = {c.args[0]: c.args[1] for c in manager.register.call_args_list}
        assert registered.get("codr") is CodrAgent
