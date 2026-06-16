# tests/test_zammad_approval_enrichment.py

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
import json
from src.tools.tool_loop import ToolLoop, _LoopFinishedEvent
from src.tools.tool_manager import ToolManager, ZammadToolHandler
from src.persona import Persona, ExecutionMode
from src.generation_events import TokenEvent, ToolCallStartEvent, ToolCallResultEvent, ResponseType

@pytest.mark.asyncio
async def test_zammad_approval_enrichment():
    # 1. Setup Mocks
    mock_engine = MagicMock()
    # Mock stream_messages to yield a tool call event and then a done event
    async def mock_stream(*args, **kwargs):
        yield {"type": "tool_calls", "calls": [{"id": "call_1", "name": "update_ticket", "arguments": {"ticket_id": 123, "state": "closed"}}]}
        yield {"type": "done", "full_text": "I'm closing the ticket."}
    
    mock_engine.stream_messages.side_effect = mock_stream
    
    mock_zammad = MagicMock()
    mock_zammad.get_ticket.return_value = {"id": 123, "number": "202605130001", "title": "Broken Printer"}
    mock_zammad.get_tags.return_value = []
    
    tool_manager = ToolManager()
    zammad_handler = ZammadToolHandler(mock_zammad)
    zammad_handler.register(tool_manager)
    
    mock_persona = MagicMock(spec=Persona)
    mock_persona.get_prompt.return_value = "You are a helpful assistant."
    mock_persona.get_config_for_engine.return_value = {}
    mock_persona.get_execution_mode.return_value = ExecutionMode.CONFIRM
    mock_persona.get_self_edit.return_value = False
    
    # 2. Run ToolLoop
    tool_loop = ToolLoop(mock_engine, tool_manager)
    
    events = []
    async for ev in tool_loop.run(
        persona=mock_persona,
        conversation_history=[],
        params=MagicMock(),
        tools=tool_manager.get_tool_definitions()
    ):
        events.append(ev)
    
    # 3. Assertions
    # We expect: 
    # 1. TokenEvent (not yielded because tool_calls came first in this mock)
    # 2. _LoopFinishedEvent with ResponseType.PENDING_CONFIRMATION
    
    finished_event = next(ev for ev in events if isinstance(ev, _LoopFinishedEvent))
    assert finished_event.response_type == ResponseType.PENDING_CONFIRMATION
    
    # Check the final text for the enrichment and expanded tags
    text = finished_event.final_text
    print(f"\nGenerated confirmation text:\n{text}")
    
    assert "[ZAMMAD, INTERNAL]" in text
    assert "#202605130001 (Broken Printer)" in text
    assert "update_ticket" in text
    assert '{"ticket_id": 123, "state": "closed"}' in text

@pytest.mark.asyncio
async def test_zammad_merge_enrichment():
    # 1. Setup Mocks
    mock_engine = MagicMock()
    async def mock_stream(*args, **kwargs):
        yield {"type": "tool_calls", "calls": [{"id": "call_2", "name": "merge_tickets", "arguments": {"source_ticket_id": 123, "target_ticket_id": 456}}]}
        yield {"type": "done", "full_text": "Merging tickets."}
    
    mock_engine.stream_messages.side_effect = mock_stream
    
    mock_zammad = MagicMock()
    def get_ticket_side_effect(ticket_id):
        if ticket_id == 123:
            return {"id": 123, "number": "202605130001", "title": "Old Ticket"}
        if ticket_id == 456:
            return {"id": 456, "number": "202605130002", "title": "New Ticket"}
        return None
        
    mock_zammad.get_ticket.side_effect = get_ticket_side_effect
    
    tool_manager = ToolManager()
    zammad_handler = ZammadToolHandler(mock_zammad)
    zammad_handler.register(tool_manager)
    
    mock_persona = MagicMock(spec=Persona)
    mock_persona.get_prompt.return_value = "You are a helpful assistant."
    mock_persona.get_config_for_engine.return_value = {}
    mock_persona.get_execution_mode.return_value = ExecutionMode.CONFIRM
    mock_persona.get_self_edit.return_value = False
    
    # 2. Run ToolLoop
    tool_loop = ToolLoop(mock_engine, tool_manager)
    
    events = []
    async for ev in tool_loop.run(
        persona=mock_persona,
        conversation_history=[],
        params=MagicMock(),
        tools=tool_manager.get_tool_definitions()
    ):
        events.append(ev)
    
    # 3. Assertions
    finished_event = next(ev for ev in events if isinstance(ev, _LoopFinishedEvent))
    text = finished_event.final_text
    print(f"\nGenerated merge confirmation text:\n{text}")
    
    assert "[ZAMMAD, INTERNAL, IRREVERSIBLE, HIGH-IMPACT]" in text
    assert "Merge #202605130001 into #202605130002 ('New Ticket')" in text
