# tests/live/conftest.py

import os
import time
import asyncio
import random
from typing import Callable, List, Any

import pytest
import requests
import logging
from unittest.mock import patch

logger = logging.getLogger(__name__)

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
TEST_CUSTOMER_ID = 100


def refresh_es_index():
    """Forces Elasticsearch to flush pending writes so they become searchable immediately."""
    es_url = os.environ.get("ZAMMAD_ES_URL")
    if not es_url:
        es_url = "http://10.0.0.70:9200"
        os.environ["ZAMMAD_ES_URL"] = es_url
        print(f"DEBUG: Using HARDCODED ES_URL: {es_url}")

    if not es_url:
        return

    try:
        r = requests.post(f"{es_url}/{ES_INDEX_PREFIX}_*/_refresh", timeout=5)
        if r.status_code != 200:
            logger.warning(f"ES Refresh returned {r.status_code}: {r.text}")
    except Exception as e:
        logger.debug(f"ES Refresh skipped or failed for {es_url}: {e}")


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

async def wait_for_search(search_func: Callable[..., List[Any]], assertion_func: Callable[[List[Any]], bool],
                          timeout: int = 15, interval: float = 0.5):
    """Polls search with forced ES refresh until assertion passes or timeout."""
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
            logger.debug(f"Search attempt failed: {e}")
        await asyncio.sleep(interval)

    pytest.fail(f"Search assertion did not pass within {timeout} seconds. Last results: {last_results}")


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
    """Initializes the ZammadClient and ensures a 'Golden Set' of history exists."""
    client = ZammadClient()
    try:
        client.get_self()
    except Exception as e:
        pytest.skip(f"Zammad unavailable: {e}")

    user_id = TEST_CUSTOMER_ID

    # 1. Cleanup: Delete all tickets for the test user EXCEPT the Golden Set
    # We use a direct list (non-ES) to be fast and reliable.
    try:
        tickets = client.list_tickets(params={'expand': 'true'})
        for t in tickets:
            if t.get('customer_id') == user_id:
                tags = client.get_tags(t['id'])
                if "gold-history" not in tags:
                    client.delete_ticket(t['id'])
    except Exception as e:
        print(f"[SETUP] Cleanup failed: {e}")

    # 2. Golden Set Setup: Ensure standard history exists for search tests
    golden_tickets = [
        {
            "title": "[GOLD] Warp Core Phase Variance",
            "body": "Problem: Dilithium crystals are out of alignment in the main warp core.",
            "solution": "Solution: Initiated a manual phase realignment of the core.",
            "tags": ["gold-history", "warp-core"]
        },
        {
            "title": "[GOLD] Printer Paper Jam",
            "body": "Paper jam in tray 2 of the LaserJet.",
            "solution": "Cleared the jam and reset the rollers.",
            "tags": ["gold-history", "printer"]
        }
    ]

    for spec in golden_tickets:
        # Check if already exists via direct list (no ES dependency)
        try:
            tickets = client.list_tickets(params={'expand': 'true'})
            existing = [t for t in tickets if t['title'] == spec["title"] and t['customer_id'] == user_id]
            if not existing:
                print(f"[SETUP] Creating Golden Ticket: {spec['title']}")
                t = client.create_ticket(
                    title=spec["title"],
                    group="Users",
                    customer_id=user_id,
                    article_body=spec["body"]
                )
                client.add_article_to_ticket(t['id'], body=spec["solution"], internal=False)
                client.add_tag(t['id'], "gold-history")
                for tag in spec["tags"]:
                    client.add_tag(t['id'], tag)
                client.update_ticket(t['id'], {'state': 'closed'})
                # We created a ticket, so we'll definitely need a refresh
                needs_refresh = True
        except Exception as e:
            print(f"[SETUP] Golden Ticket check/creation failed: {e}")

    # 3. Synchronize: Wait for Golden tickets to be searchable (one-time setup cost)
    print("[SETUP] Waiting for Golden tickets to index...")
    start_sync = time.time()
    while time.time() - start_sync < 30:
        refresh_es_index()
        try:
            # Search specifically for one of our golden tickets
            found = client.search_tickets(query=f'title:"[GOLD] Warp Core" AND customer_id:{user_id}')
            if found:
                print(f"[SETUP] Golden tickets indexed successfully in {int(time.time() - start_sync)}s")
                break
        except Exception:
            pass
        time.sleep(2)
    else:
        print("[WARNING] Golden tickets not searchable after 30s. Tests may fail.")

    return client


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
