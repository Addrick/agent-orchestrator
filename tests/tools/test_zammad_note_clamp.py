# tests/tools/test_zammad_note_clamp.py

"""add_note_to_ticket writes internal notes only (customer-visible replies
will be a separate tool, classified as egress)."""

import pytest
from unittest.mock import MagicMock

from src.tools.tool_manager import ZammadToolHandler


@pytest.mark.asyncio
async def test_add_note_is_internal():
    client = MagicMock()
    client.add_article_to_ticket.return_value = {"id": 1, "ticket_id": 42}
    await ZammadToolHandler(client)._add_note_to_ticket(ticket_id=42, body="note")
    client.add_article_to_ticket.assert_called_once_with(
        ticket_id=42, body="note", internal=True
    )
