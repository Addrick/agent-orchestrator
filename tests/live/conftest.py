# tests/live/conftest.py

import os
import time
import asyncio
import random
from typing import Callable, List, Any

import pytest
import requests
from unittest.mock import patch

from src.clients.zammad_client import ZammadClient
from src.clients.zammad_service import ZammadIntegration
from memory.memory_manager import MemoryManager
from src.engine import TextEngine
from src.chat_system import ChatSystem
from src.persona import Persona, MemoryMode
from config.global_config import (
    TEST_MEMORY_DATABASE_FILE,
    ZAMMAD_BOT_EMAIL,
    ZAMMAD_BOT_FIRSTNAME,
    ZAMMAD_BOT_LASTNAME,
)


# ---------------------------------------------------------------------------
# Elasticsearch helpers
# ---------------------------------------------------------------------------

ES_INDEX_PREFIX = "zammad_production"


def refresh_es_index():
    """Forces Elasticsearch to flush pending writes so they become searchable immediately."""
    es_url = os.environ.get("ZAMMAD_ES_URL")
    if not es_url:
        return
    try:
        requests.post(f"{es_url}/{ES_INDEX_PREFIX}_*/_refresh", timeout=5)
    except requests.exceptions.RequestException:
        pass  # Non-fatal — tests fall back to polling


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

async def wait_for_search(search_func: Callable[..., List[Any]], assertion_func: Callable[[List[Any]], bool],
                          timeout: int = 15, interval: float = 0.2):
    """Polls search with forced ES refresh until assertion passes or timeout.
    The bottleneck is Zammad's scheduler pushing data to ES, not ES itself.
    Tight polling with refresh catches it the moment Zammad flushes."""
    start_time = time.time()
    last_results = []
    while time.time() - start_time < timeout:
        refresh_es_index()
        try:
            results = await asyncio.to_thread(search_func)
            last_results = results
            if assertion_func(results):
                return
        except Exception as e:
            print(f"Search failed with error: {e}")
        await asyncio.sleep(interval)

    print(f"DEBUG: Timeout reached. Last search results ({len(last_results)}): {last_results}")
    pytest.fail(f"Search assertion did not pass within {timeout} seconds.")


async def wait_for_tag(zammad_client, ticket_id, tag, timeout=10):
    """Polls the ticket tags until the specified tag appears."""
    start = time.time()
    current_tags = []
    while time.time() - start < timeout:
        try:
            current_tags = await asyncio.to_thread(zammad_client.get_tags, ticket_id)
            if tag in current_tags:
                return
        except Exception as e:
            print(f"Error fetching ticket tags: {e}")
        await asyncio.sleep(1)
    pytest.fail(f"Tag '{tag}' not found on ticket {ticket_id} after {timeout}s. Current tags: {current_tags}")


# ---------------------------------------------------------------------------
# Zammad fixtures (shared by test_zammad_live.py and test_full_system_zammad.py)
# ---------------------------------------------------------------------------

PERSISTENT_TEST_USER_EMAIL = "pytest-integration-user@zammad.local"


@pytest.fixture(scope="module")
def zammad_client():
    """Provides a live ZammadClient, skipping if connection fails."""
    try:
        client = ZammadClient()
        client.get_self()
        return client
    except (ValueError, requests.exceptions.RequestException) as e:
        pytest.skip(f"Zammad unavailable: {e}")


@pytest.fixture(scope="module")
def bot_identity(zammad_client):
    """Ensures the Zammad Bot user exists for the tests."""
    users = zammad_client.search_user(f"email:{ZAMMAD_BOT_EMAIL}")
    if not users:
        print(f"\n[SETUP] Creating Bot User: {ZAMMAD_BOT_EMAIL}")
        zammad_client.create_user(
            email=ZAMMAD_BOT_EMAIL,
            firstname=ZAMMAD_BOT_FIRSTNAME,
            lastname=ZAMMAD_BOT_LASTNAME,
            roles=["Agent"]
        )
    else:
        print(f"\n[SETUP] Bot User found: {ZAMMAD_BOT_EMAIL}")
    return ZAMMAD_BOT_EMAIL


@pytest.fixture(scope="function")
def live_chat_system():
    """Sets up a fully integrated ChatSystem with a live Zammad connection for each test."""
    db_path = f"{TEST_MEMORY_DATABASE_FILE}.{random.randint(1000, 9999)}"
    if os.path.exists(db_path):
        os.remove(db_path)

    memory_manager = MemoryManager(db_path=db_path)
    memory_manager.create_schema()

    try:
        zammad_client = ZammadClient()
        zammad_client.get_self()
    except Exception as e:
        pytest.skip(f"Skipping live tests: Zammad client setup failed. Error: {e}")

    text_engine = TextEngine()

    test_personas = {
        "test_persona": Persona(
            persona_name="test_persona", model_name="mock_model", prompt="You are a test persona.",
            enabled_tools=['*'], memory_mode=MemoryMode.CHANNEL_ISOLATED, context_length=10
        ),
        "capped_persona": Persona(
            persona_name='capped_persona', model_name='mock', prompt='talk', context_length=100
        )
    }

    with patch('src.chat_system.load_personas_from_file', return_value=test_personas):
        chat_system = ChatSystem(
            memory_manager=memory_manager, text_engine=text_engine,
        )
        chat_system.register_service(ZammadIntegration(zammad_client))

    try:
        yield chat_system, memory_manager, zammad_client
    finally:
        memory_manager.close()
        time.sleep(0.1)
        if os.path.exists(db_path):
            try:
                os.remove(db_path)
            except PermissionError as e:
                print(f"\n[TEARDOWN WARNING] Could not remove test database file: {e}")


@pytest.fixture(scope="function")
def managed_zammad_user(live_chat_system):
    """Finds or creates a single persistent user for Zammad-related tests."""
    _, _, zammad_client = live_chat_system
    users = zammad_client.search_user(query=PERSISTENT_TEST_USER_EMAIL)
    if users:
        user_id = users[0]['id']
    else:
        user_data = zammad_client.create_user(email=PERSISTENT_TEST_USER_EMAIL, firstname="Pytest", lastname="User")
        user_id = user_data['id']

    tickets = zammad_client.search_tickets(query=f"customer_id:{user_id}")
    for ticket in tickets:
        zammad_client.delete_ticket(ticket['id'])

    yield {"id": user_id, "identifier": f"Pytest User <{PERSISTENT_TEST_USER_EMAIL}>"}


# ---------------------------------------------------------------------------
# LLM live test cost controls
# ---------------------------------------------------------------------------

LLM_LIVE_MAX_TOKENS = 100
LLM_LIVE_MODEL = "gemma-3-27b-it"  # multimodal, generous free-tier rate limits
LLM_LIVE_MAX_TESTS = 10
