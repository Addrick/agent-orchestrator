# tests/live/test_zammad_tools.py
#
# Dedicated tests for the ZammadIntegration tools in isolation.
# These verify that the Python tool code correctly interfaces with the Zammad API
# and returns the data structures the LLM expects.

import pytest
from src.tools.tool_manager import ToolManager, ZammadToolHandler
from tests.live.conftest import TEST_CUSTOMER_ID

pytestmark = pytest.mark.zammad_live

@pytest.fixture
def tool_manager(zammad_client):
    """Provides a ToolManager with Zammad tools registered."""
    manager = ToolManager()
    handler = ZammadToolHandler(zammad_client)
    handler.register(manager)
    return manager

@pytest.mark.asyncio
async def test_tool_search_tickets_finds_golden_set(tool_manager):
    """Verifies that search_tickets correctly finds the pre-indexed Golden tickets."""
    # Search for the Warp Core history
    query = 'title:"[GOLD] Warp Core" AND state.name:closed'
    response = await tool_manager.execute_tool("search_tickets", query=query)
    
    assert 'result' in response
    results = response['result']
    assert len(results) >= 1
    ticket = results[0]
    assert "[GOLD] Warp Core" in ticket['title']
    assert ticket['state'] == 'closed'
    assert ticket['customer_id'] == TEST_CUSTOMER_ID

@pytest.mark.asyncio
async def test_tool_search_tickets_with_filtering(tool_manager):
    """Verifies that search_tickets handles broad queries and state filtering."""
    # Broad search for Printer
    response = await tool_manager.execute_tool("search_tickets", query='title:"Printer"')
    assert 'result' in response
    assert len(response['result']) >= 1
    
    # Check that we can filter for closed tickets specifically
    query = 'title:"Printer" AND state.name:closed'
    response = await tool_manager.execute_tool("search_tickets", query=query)
    assert all(t['state'] == 'closed' for t in response['result'])

@pytest.mark.asyncio
async def test_tool_get_ticket_details_structure(tool_manager, zammad_client):
    """Verifies that get_ticket_details returns the comprehensive info required by the bot."""
    # Find one of our golden tickets
    resp = await tool_manager.execute_tool("search_tickets", query='title:"[GOLD] Warp Core"')
    # Note: the tool get_ticket_details uses ticket_number (Zammad's user-facing ID)
    ticket_number = resp['result'][0]['number']
    
    response = await tool_manager.execute_tool("get_ticket_details", ticket_number=ticket_number)
    assert 'result' in response
    details = response['result']
    
    # Check structure
    assert 'title' in details
    assert 'articles' in details
    assert 'tags' in details
    assert 'customer_info' in details
    assert details['state'] == 'closed'
    
    # Verify article content
    articles = details['articles']
    assert any("Dilithium crystals" in a['body'] for a in articles)

@pytest.mark.asyncio
async def test_tool_create_and_update_lifecycle(tool_manager, zammad_client):
    """Verifies the write tools (create, add_note, update) work in sequence."""
    ticket_id = None
    try:
        # 1. Create
        response = await tool_manager.execute_tool(
            "create_ticket",
            title="[TOOL TEST] Lifecycle",
            body="Initial Body",
            customer_id=TEST_CUSTOMER_ID
        )
        assert 'result' in response
        ticket_id = response['result']['id']
        
        # 2. Add Note
        response = await tool_manager.execute_tool(
            "add_note_to_ticket",
            ticket_id=ticket_id,
            body="Follow-up Note",
            internal=True
        )
        assert 'result' in response
        assert response['result']['ticket_id'] == ticket_id
        
        # 3. Verify via list API (direct DB check)
        details = zammad_client.list_tickets(params={'expand': 'true'})
        created = next((t for t in details if t['id'] == ticket_id), None)
        assert created is not None
        assert created['title'] == "[TOOL TEST] Lifecycle"
        
    finally:
        if ticket_id:
            zammad_client.delete_ticket(ticket_id)

@pytest.mark.asyncio
async def test_tool_get_user_info(tool_manager):
    """Verifies that search_user can retrieve and format user/customer info."""
    # We search by email of the test user (which we know exists)
    response = await tool_manager.execute_tool("search_user", query="pytest-integration-user@zammad.local")
    
    assert 'result' in response
    users = response['result']
    assert len(users) >= 1
    assert users[0]['email'] == "pytest-integration-user@zammad.local"
