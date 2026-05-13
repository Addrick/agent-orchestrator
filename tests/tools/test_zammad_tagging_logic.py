# tests/tools/test_zammad_tagging_logic.py

import pytest
from unittest.mock import MagicMock, patch
from src.tools.tool_manager import ZammadToolHandler

@pytest.fixture
def mock_zammad_client():
    client = MagicMock()
    # Mock get_tags to return some existing tags
    client.get_tags.return_value = ["existing_tag"]
    # Mock update_ticket to return a success response
    client.update_ticket.return_value = {"id": 123, "tags": "existing_tag"}
    return client

@pytest.fixture
def handler(mock_zammad_client):
    return ZammadToolHandler(mock_zammad_client)

@pytest.mark.asyncio
async def test_update_ticket_uses_tags_api(handler, mock_zammad_client):
    """
    Test that update_ticket uses the dedicated Tags API instead of the generic Ticket API.
    This test is expected to FAIL with the current implementation because it sends tags 
    via the generic PUT /tickets/{id} payload.
    """
    ticket_id = 123
    new_tags = ["phishing"]
    
    # Execute the tool
    await handler._update_ticket(ticket_id=ticket_id, tags=new_tags)
    
    # EXPECTED BEHAVIOR (The Fix):
    # 1. Should fetch current tags
    mock_zammad_client.get_tags.assert_called_with(ticket_id=ticket_id)
    
    # 2. Should add the new tag 'phishing'
    mock_zammad_client.add_tag.assert_called_with(ticket_id=ticket_id, tag="phishing")
    
    # 3. Should remove the old tag 'existing_tag' (since it's an overwrite)
    # This requires ZammadClient.remove_tag to exist and be called.
    mock_zammad_client.remove_tag.assert_called_with(ticket_id=ticket_id, tag="existing_tag")
    
    # 4. Should NOT include 'tags' in the generic update_ticket payload
    # If ONLY tags were updated, update_ticket shouldn't even be called (it uses get_ticket instead)
    if mock_zammad_client.update_ticket.called:
        last_update_call = mock_zammad_client.update_ticket.call_args
        assert "tags" not in last_update_call.kwargs["payload"]
    else:
        # If not called, it's also correct (it means we avoided the bug)
        mock_zammad_client.get_ticket.assert_called_with(ticket_id=ticket_id)

@pytest.mark.asyncio
async def test_update_ticket_mixed_fields(handler, mock_zammad_client):
    """
    Test that update_ticket correctly handles both tags and other fields.
    """
    ticket_id = 123
    new_tags = ["phishing"]
    
    # Execute the tool with tags AND state
    await handler._update_ticket(ticket_id=ticket_id, tags=new_tags, state="closed")
    
    # 1. Tags API should still be called
    mock_zammad_client.add_tag.assert_called_with(ticket_id=ticket_id, tag="phishing")
    
    # 2. Ticket API should be called for the state update
    # But it should NOT contain 'tags'
    mock_zammad_client.update_ticket.assert_called_once()
    last_update_call = mock_zammad_client.update_ticket.call_args
    assert last_update_call.kwargs["payload"] == {"state": "closed"}
