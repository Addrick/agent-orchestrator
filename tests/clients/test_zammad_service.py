# tests/clients/test_zammad_service.py

import pytest
from unittest.mock import MagicMock

from src.clients.zammad_service import ZammadIntegration
from src.clients.zammad_client import ZammadClient


@pytest.fixture
def zammad_integration():
    """Provides a ZammadIntegration with a mocked ZammadClient."""
    mock_client = MagicMock(spec=ZammadClient)
    mock_client.api_url = "http://zammad.local"
    return ZammadIntegration(mock_client), mock_client


def test_name(zammad_integration):
    integration, _ = zammad_integration
    assert integration.name == "zammad"
