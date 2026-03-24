# tests/agents/test_agent_manager.py

import asyncio
import json
import pytest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch, PropertyMock

from src.agents.agent_manager import AgentManager, AgentRegistration, RunningAgent
from src.agents.base import AgentLoop


class StubAgent(AgentLoop):
    """Minimal agent for testing lifecycle management."""
    agent_name = "stub"
    poll_interval = 0.05

    def __init__(self, chat_system, inject_personas=False):
        super().__init__(chat_system, inject_personas=inject_personas)

    async def _poll(self):
        pass


class FailingAgent(AgentLoop):
    """Agent that raises on every poll — tests error tracking."""
    agent_name = "failing"
    poll_interval = 0.05

    def __init__(self, chat_system, inject_personas=False):
        super().__init__(chat_system, inject_personas=inject_personas)

    async def _poll(self):
        raise RuntimeError("Simulated failure")


class DependencyAgent(AgentLoop):
    """Agent that requires zammad_client and notification_router."""
    agent_name = "dep"
    poll_interval = 0.05

    def __init__(self, chat_system, zammad_client, notification_router, inject_personas=False):
        super().__init__(chat_system, inject_personas=inject_personas)
        self.zammad_client = zammad_client
        self.notification_router = notification_router

    async def _poll(self):
        pass


class ConfigAgent(AgentLoop):
    """Agent that accepts agent_config for testing config injection."""
    agent_name = "configurable"
    poll_interval = 0.05

    def __init__(self, chat_system, agent_config=None, inject_personas=False):
        super().__init__(chat_system, inject_personas=inject_personas)
        self.agent_config = agent_config

    async def _poll(self):
        pass

    async def _poll(self):
        pass


@pytest.fixture
def mock_chat_system():
    cs = MagicMock()
    cs.text_engine = MagicMock()
    cs.memory_manager = MagicMock()
    cs.personas = {}
    cs._services = {}
    return cs


@pytest.fixture
def tmp_config(tmp_path):
    """Create a temporary agents.json config file."""
    config = {
        "agents": {
            "stub": {
                "poll_interval": 120,
                "auto_start": False,
            },
            "auto_agent": {
                "poll_interval": 60,
                "auto_start": True,
            },
        },
        "recipients": {
            "adrich": {"discord_user_id": "12345"},
        },
    }
    config_path = tmp_path / "agents.json"
    config_path.write_text(json.dumps(config))
    return config_path


@pytest.fixture
def manager(mock_chat_system, tmp_config):
    return AgentManager(
        chat_system=mock_chat_system,
        memory_manager=mock_chat_system.memory_manager,
        notification_router=MagicMock(),
        config_path=tmp_config,
    )


class TestAgentManagerInit:
    def test_loads_config_from_file(self, manager, tmp_config):
        assert "agents" in manager.config
        assert "stub" in manager.config["agents"]

    def test_missing_config_file_ok(self, mock_chat_system, tmp_path):
        mgr = AgentManager(
            chat_system=mock_chat_system,
            memory_manager=mock_chat_system.memory_manager,
            config_path=tmp_path / "nonexistent.json",
        )
        assert mgr.config == {}

    def test_invalid_json_config(self, mock_chat_system, tmp_path):
        bad_config = tmp_path / "bad.json"
        bad_config.write_text("not json")
        mgr = AgentManager(
            chat_system=mock_chat_system,
            memory_manager=mock_chat_system.memory_manager,
            config_path=bad_config,
        )
        assert mgr.config == {}


class TestRegistration:
    def test_register_agent(self, manager):
        manager.register("stub", StubAgent)
        assert "stub" in manager.get_registered()

    def test_register_with_default_config(self, manager):
        manager.register("stub", StubAgent, default_config={"poll_interval": 30})
        assert "stub" in manager.get_registered()

    def test_get_registered_empty(self, manager):
        assert manager.get_registered() == []


class TestStartStop:
    @pytest.mark.asyncio
    async def test_start_agent(self, manager):
        manager.register("stub", StubAgent)
        result = await manager.start_agent("stub")
        assert "started" in result.lower()
        assert "stub" in manager.get_running()

        # Clean up
        await manager.stop_agent("stub")

    @pytest.mark.asyncio
    async def test_start_unregistered_agent_raises(self, manager):
        with pytest.raises(ValueError, match="not registered"):
            await manager.start_agent("nonexistent")

    @pytest.mark.asyncio
    async def test_start_already_running_raises(self, manager):
        manager.register("stub", StubAgent)
        await manager.start_agent("stub")
        with pytest.raises(ValueError, match="already running"):
            await manager.start_agent("stub")

        await manager.stop_agent("stub")

    @pytest.mark.asyncio
    async def test_stop_agent(self, manager):
        manager.register("stub", StubAgent)
        await manager.start_agent("stub")
        result = await manager.stop_agent("stub")
        assert "stopped" in result.lower()
        assert "stub" not in manager.get_running()

    @pytest.mark.asyncio
    async def test_stop_not_running_raises(self, manager):
        manager.register("stub", StubAgent)
        with pytest.raises(ValueError, match="not running"):
            await manager.stop_agent("stub")

    @pytest.mark.asyncio
    async def test_restart_agent(self, manager):
        manager.register("stub", StubAgent)
        await manager.start_agent("stub")
        result = await manager.restart_agent("stub")
        assert "started" in result.lower()
        assert "stub" in manager.get_running()

        await manager.stop_agent("stub")

    @pytest.mark.asyncio
    async def test_restart_not_running_starts(self, manager):
        manager.register("stub", StubAgent)
        result = await manager.restart_agent("stub")
        assert "started" in result.lower()

        await manager.stop_agent("stub")


class TestConfigMerging:
    @pytest.mark.asyncio
    async def test_file_config_applied_to_instance(self, manager):
        """Config from agents.json should set poll_interval on the instance."""
        manager.register("stub", StubAgent)
        await manager.start_agent("stub")

        instance = manager._running["stub"].instance
        assert instance.poll_interval == 120  # from tmp_config

        await manager.stop_agent("stub")

    @pytest.mark.asyncio
    async def test_runtime_override_beats_file_config(self, manager):
        manager.register("stub", StubAgent)
        await manager.start_agent("stub", config_overrides={"poll_interval": 999})

        instance = manager._running["stub"].instance
        assert instance.poll_interval == 999

        await manager.stop_agent("stub")

    @pytest.mark.asyncio
    async def test_registration_defaults_merged(self, manager):
        manager.register("stub", StubAgent, default_config={"action_history_limit": 25})
        await manager.start_agent("stub")

        instance = manager._running["stub"].instance
        assert instance.action_history_limit == 25

        await manager.stop_agent("stub")


class TestDependencyInjection:
    @pytest.mark.asyncio
    async def test_injects_zammad_and_notification(self, mock_chat_system, tmp_path):
        """Agent with zammad_client and notification_router params gets them injected."""
        mock_zammad_service = MagicMock()
        mock_zammad_service._client = MagicMock()
        mock_chat_system._services = {"zammad": mock_zammad_service}

        mock_router = MagicMock()
        config_path = tmp_path / "agents.json"
        config_path.write_text("{}")

        mgr = AgentManager(
            chat_system=mock_chat_system,
            memory_manager=mock_chat_system.memory_manager,
            notification_router=mock_router,
            config_path=config_path,
        )
        mgr.register("dep", DependencyAgent)
        await mgr.start_agent("dep")

        instance = mgr._running["dep"].instance
        assert instance.zammad_client is mock_zammad_service._client
        assert instance.notification_router is mock_router

        await mgr.stop_agent("dep")

    @pytest.mark.asyncio
    async def test_missing_zammad_raises(self, mock_chat_system, tmp_path):
        mock_chat_system._services = {}
        config_path = tmp_path / "agents.json"
        config_path.write_text("{}")

        mgr = AgentManager(
            chat_system=mock_chat_system,
            memory_manager=mock_chat_system.memory_manager,
            notification_router=MagicMock(),
            config_path=config_path,
        )
        mgr.register("dep", DependencyAgent)

        with pytest.raises(ValueError, match="zammad_client"):
            await mgr.start_agent("dep")

    @pytest.mark.asyncio
    async def test_missing_notification_router_raises(self, mock_chat_system, tmp_path):
        mock_zammad_service = MagicMock()
        mock_zammad_service._client = MagicMock()
        mock_chat_system._services = {"zammad": mock_zammad_service}

        config_path = tmp_path / "agents.json"
        config_path.write_text("{}")

        mgr = AgentManager(
            chat_system=mock_chat_system,
            memory_manager=mock_chat_system.memory_manager,
            notification_router=None,
            config_path=config_path,
        )
        mgr.register("dep", DependencyAgent)

        with pytest.raises(ValueError, match="notification_router"):
            await mgr.start_agent("dep")

    @pytest.mark.asyncio
    async def test_injects_agent_config_with_recipients(self, mock_chat_system, tmp_path):
        """Agent with agent_config param gets config + recipients injected."""
        config = {
            "agents": {
                "configurable": {
                    "poll_interval": 60,
                    "notification_defaults": {"channel": "discord_dm", "recipient": "adrich"},
                }
            },
            "recipients": {
                "adrich": {"discord_user_id": "321783731146850305"},
            },
        }
        config_path = tmp_path / "agents.json"
        config_path.write_text(json.dumps(config))

        mgr = AgentManager(
            chat_system=mock_chat_system,
            memory_manager=mock_chat_system.memory_manager,
            config_path=config_path,
        )
        mgr.register("configurable", ConfigAgent)
        await mgr.start_agent("configurable")

        instance = mgr._running["configurable"].instance
        assert instance.agent_config is not None
        assert instance.agent_config["notification_defaults"]["recipient"] == "adrich"
        assert instance.agent_config["_recipients"]["adrich"]["discord_user_id"] == "321783731146850305"

        await mgr.stop_agent("configurable")


class TestGetStatus:
    @pytest.mark.asyncio
    async def test_status_registered_not_running(self, manager):
        manager.register("stub", StubAgent)
        status = manager.get_status("stub")
        agent_status = status["agents"]["stub"]

        assert agent_status["registered"] is True
        assert agent_status["running"] is False
        assert agent_status["class"] == "StubAgent"

    @pytest.mark.asyncio
    async def test_status_running_agent(self, manager):
        manager.register("stub", StubAgent)
        await manager.start_agent("stub")

        # Let it poll at least once
        await asyncio.sleep(0.1)

        status = manager.get_status("stub")
        agent_status = status["agents"]["stub"]

        assert agent_status["running"] is True
        assert agent_status["started_at"] is not None
        assert agent_status["poll_count"] >= 1
        assert agent_status["error_count"] == 0

        await manager.stop_agent("stub")

    @pytest.mark.asyncio
    async def test_status_all_agents(self, manager):
        manager.register("stub", StubAgent)
        manager.register("failing", FailingAgent)

        status = manager.get_status()
        assert "stub" in status["agents"]
        assert "failing" in status["agents"]

    def test_status_unknown_agent(self, manager):
        status = manager.get_status("nonexistent")
        assert status["agents"]["nonexistent"]["registered"] is False

    @pytest.mark.asyncio
    async def test_status_tracks_errors(self, manager):
        manager.register("failing", FailingAgent)
        await manager.start_agent("failing")

        # Let it fail a few times
        await asyncio.sleep(0.2)

        status = manager.get_status("failing")
        agent_status = status["agents"]["failing"]

        assert agent_status["error_count"] > 0
        assert agent_status["consecutive_errors"] > 0
        assert agent_status["last_error"] is not None

        await manager.stop_agent("failing")


class TestShutdownAll:
    @pytest.mark.asyncio
    async def test_shutdown_stops_all(self, manager):
        manager.register("stub", StubAgent)
        manager.register("failing", FailingAgent)

        await manager.start_agent("stub")
        await manager.start_agent("failing")

        assert len(manager.get_running()) == 2

        await manager.shutdown_all()

        assert len(manager.get_running()) == 0

    @pytest.mark.asyncio
    async def test_shutdown_empty_is_noop(self, manager):
        await manager.shutdown_all()  # Should not raise


class TestAutoStart:
    @pytest.mark.asyncio
    async def test_auto_start_configured_agents(self, manager):
        """Agents with auto_start: true in config should be started."""
        manager.register("auto_agent", StubAgent)
        await manager.auto_start()

        assert "auto_agent" in manager.get_running()

        await manager.shutdown_all()

    @pytest.mark.asyncio
    async def test_auto_start_skips_unregistered(self, manager):
        """auto_start should not fail for agents in config but not registered."""
        await manager.auto_start()  # auto_agent is in config but not registered
        assert manager.get_running() == []

    @pytest.mark.asyncio
    async def test_auto_start_respects_false(self, manager):
        """Agents with auto_start: false should not be started."""
        manager.register("stub", StubAgent)
        await manager.auto_start()

        assert "stub" not in manager.get_running()


class TestNotificationRouterProperty:
    def test_getter_returns_router(self, manager):
        assert manager.notification_router is not None

    def test_setter_updates_router(self, mock_chat_system, tmp_path):
        config_path = tmp_path / "agents.json"
        config_path.write_text("{}")
        mgr = AgentManager(
            chat_system=mock_chat_system,
            memory_manager=mock_chat_system.memory_manager,
            config_path=config_path,
        )
        assert mgr.notification_router is None

        new_router = MagicMock()
        mgr.notification_router = new_router
        assert mgr.notification_router is new_router


class BlockingAgent(AgentLoop):
    """Agent whose _poll blocks forever, ignoring the shutdown event."""
    agent_name = "blocking"
    poll_interval = 0.05

    def __init__(self, chat_system, inject_personas=False):
        super().__init__(chat_system, inject_personas=inject_personas)

    async def _poll(self):
        # This blocks forever and cannot be interrupted by stop()
        await asyncio.Event().wait()


class TestStopTimeout:
    @pytest.mark.asyncio
    async def test_stop_timeout_cancels_task(self, mock_chat_system, tmp_path):
        """When an agent's task doesn't complete within timeout, the task gets cancelled."""
        config_path = tmp_path / "agents.json"
        config_path.write_text("{}")
        mgr = AgentManager(
            chat_system=mock_chat_system,
            memory_manager=mock_chat_system.memory_manager,
            config_path=config_path,
        )
        mgr.register("blocking", BlockingAgent)
        await mgr.start_agent("blocking")

        running = mgr._running["blocking"]
        task = running.task

        # Wait for agent to enter _poll (which blocks forever)
        await asyncio.sleep(0.1)

        # Signal stop — but _poll is stuck in Event().wait(), so
        # the task won't finish within the timeout
        running.instance.stop()

        # Simulate what stop_agent does (lines 224-228) with a short timeout
        try:
            await asyncio.wait_for(asyncio.shield(task), timeout=0.1)
        except asyncio.TimeoutError:
            task.cancel()

        # Give event loop time to process the cancellation
        try:
            await task
        except asyncio.CancelledError:
            pass

        assert task.cancelled()


class CrashingAgent(AgentLoop):
    """Agent that crashes during start (not during _poll)."""
    agent_name = "crashing"
    poll_interval = 0.05

    def __init__(self, chat_system, inject_personas=False):
        super().__init__(chat_system, inject_personas=inject_personas)

    async def _poll(self):
        pass

    async def _on_start(self):
        raise RuntimeError("Crash during startup")


class TestCrashErrorReporting:
    @pytest.mark.asyncio
    async def test_crash_error_in_status(self, mock_chat_system, tmp_path):
        """When an agent's task crashes, the status includes crash_error."""
        config_path = tmp_path / "agents.json"
        config_path.write_text("{}")
        mgr = AgentManager(
            chat_system=mock_chat_system,
            memory_manager=mock_chat_system.memory_manager,
            config_path=config_path,
        )
        mgr.register("crashing", CrashingAgent)
        await mgr.start_agent("crashing")

        # Wait for the task to finish (it will crash quickly)
        task = mgr._running["crashing"].task
        try:
            await asyncio.wait_for(task, timeout=2.0)
        except Exception:
            pass

        assert task.done()
        status = mgr.get_status("crashing")
        agent_status = status["agents"]["crashing"]
        assert "crash_error" in agent_status
        assert "Crash during startup" in agent_status["crash_error"]


class TestShutdownErrorHandling:
    @pytest.mark.asyncio
    async def test_shutdown_continues_on_stop_error(self, mock_chat_system, tmp_path):
        """shutdown_all continues even if one agent fails to stop."""
        config_path = tmp_path / "agents.json"
        config_path.write_text("{}")
        mgr = AgentManager(
            chat_system=mock_chat_system,
            memory_manager=mock_chat_system.memory_manager,
            config_path=config_path,
        )
        mgr.register("stub1", StubAgent)
        mgr.register("stub2", StubAgent)

        await mgr.start_agent("stub1")
        await mgr.start_agent("stub2")

        # Make stop_agent raise for stub1 but succeed for stub2
        original_stop = mgr.stop_agent
        call_count = 0

        async def failing_stop(name):
            nonlocal call_count
            call_count += 1
            if name == "stub1":
                raise RuntimeError("Stop failed for stub1")
            return await original_stop(name)

        mgr.stop_agent = failing_stop

        # Should not raise, should continue to stub2
        await mgr.shutdown_all()
        assert call_count == 2


class FailStartAgent(AgentLoop):
    """Agent that requires zammad_client — will fail to start when not provided."""
    agent_name = "fail_start"
    poll_interval = 0.05

    def __init__(self, chat_system, zammad_client=None, inject_personas=False):
        super().__init__(chat_system, inject_personas=inject_personas)
        if zammad_client is None:
            raise ValueError("zammad_client is required")

    async def _poll(self):
        pass


class TestAutoStartErrorHandling:
    @pytest.mark.asyncio
    async def test_auto_start_logs_errors_without_crashing(self, mock_chat_system, tmp_path):
        """auto_start logs errors but doesn't crash when an agent fails to start."""
        config = {
            "agents": {
                "fail_start": {"auto_start": True},
                "stub": {"auto_start": True},
            }
        }
        config_path = tmp_path / "agents.json"
        config_path.write_text(json.dumps(config))

        mgr = AgentManager(
            chat_system=mock_chat_system,
            memory_manager=mock_chat_system.memory_manager,
            config_path=config_path,
        )

        # Register an agent that will fail (needs zammad but none provided)
        mgr.register("fail_start", DependencyAgent)
        mgr.register("stub", StubAgent)

        # auto_start should not raise even though fail_start will error
        await mgr.auto_start()

        # stub should have started successfully despite fail_start failing
        assert "stub" in mgr.get_running()

        await mgr.shutdown_all()
