"""AgentManager config-application tests.

Covers the agent schedule wiring, including DP-221's removal of the legacy
`poll_interval` config key (the old format `agents.json` used before the nested
`schedule` block). After the cleanup, `schedule` is the only accepted form and a
bare `poll_interval` is ignored.
"""

import asyncio
from unittest.mock import MagicMock

import pytest

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
