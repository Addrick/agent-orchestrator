# tests/agents/test_agent_service.py

import pytest
from unittest.mock import MagicMock, patch

from src.agents.agent_service import AgentServiceIntegration
from src.tools.tool_manager import ToolManager


class TestAgentServiceIntegrationInit:
    def test_stores_agent_manager_reference(self):
        mock_agent_manager = MagicMock()
        mock_memory_manager = MagicMock()
        service = AgentServiceIntegration(mock_agent_manager, mock_memory_manager)
        assert service._agent_manager is mock_agent_manager

    def test_stores_memory_manager_reference(self):
        mock_agent_manager = MagicMock()
        mock_memory_manager = MagicMock()
        service = AgentServiceIntegration(mock_agent_manager, mock_memory_manager)
        assert service._memory_manager is mock_memory_manager


class TestAgentServiceName:
    def test_name_returns_agents(self):
        service = AgentServiceIntegration(MagicMock(), MagicMock())
        assert service.name == "agents"


class TestAgentServiceRegisterTools:
    def test_registers_all_tools(self):
        mock_agent_manager = MagicMock()
        mock_memory_manager = MagicMock()
        service = AgentServiceIntegration(mock_agent_manager, mock_memory_manager)

        mock_tool_manager = MagicMock()
        service.register_tools(mock_tool_manager)

        assert mock_tool_manager.register.call_count == 4

        registered_names = {call.args[0] for call in mock_tool_manager.register.call_args_list}
        assert registered_names == {
            "get_agent_status", "get_agent_history",
            "lookup_agent_history", "manage_agent",
        }

    def test_passes_correct_dependencies_to_handler(self):
        mock_agent_manager = MagicMock()
        mock_memory_manager = MagicMock()
        service = AgentServiceIntegration(mock_agent_manager, mock_memory_manager)

        mock_tool_manager = MagicMock()

        with patch('src.tools.agent_tool_handler.AgentToolHandler') as MockHandler:
            mock_handler_instance = MagicMock()
            MockHandler.return_value = mock_handler_instance

            service.register_tools(mock_tool_manager)

            MockHandler.assert_called_once_with(mock_agent_manager, mock_memory_manager)
            mock_handler_instance.register.assert_called_once_with(mock_tool_manager)


@pytest.mark.integration
class TestAgentToolRegistrationChain:
    """Integration: real ToolManager wired through AgentServiceIntegration."""

    def test_agent_tools_appear_in_definitions_after_registration(self):
        """After register_tools, ToolManager.get_tool_definitions includes the 3 agent tools."""
        manager = ToolManager()
        service = AgentServiceIntegration(MagicMock(), MagicMock())
        service.register_tools(manager)

        tool_names = {t['function']['name'] for t in manager.get_tool_definitions()
                      if t.get('type') == 'function'}
        assert {'get_agent_status', 'get_agent_history', 'manage_agent'} <= tool_names

    def test_agent_tools_have_correct_service_binding(self):
        """All agent tools declare service_binding='agents'."""
        manager = ToolManager()
        service = AgentServiceIntegration(MagicMock(), MagicMock())
        service.register_tools(manager)

        agent_tools = [t for t in manager.get_tool_definitions()
                       if t.get('service_binding') == 'agents']
        assert len(agent_tools) == 4

    def test_unregistered_tools_excluded_from_definitions(self):
        """ToolManager with no registered handlers returns no callable tools."""
        manager = ToolManager()
        callable_tools = [t for t in manager.get_tool_definitions()
                          if t.get('type') == 'function']
        assert len(callable_tools) == 0
