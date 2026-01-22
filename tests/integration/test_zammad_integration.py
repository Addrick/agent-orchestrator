# tests/integration/test_zammad_integration.py

import pytest
import asyncio
import time
import os
from typing import Callable, List, Any
from datetime import datetime
import requests
from unittest.mock import MagicMock, AsyncMock, patch
from src.clients.zammad_client import ZammadClient
from src.interfaces.zammad_bot import ZammadBot
from src.chat_system import ChatSystem
from src.engine import TextEngine
from src.persona import Persona
from config.global_config import TRIAGE_PERSONA_NAME, LOCAL_LLM_URL, ZAMMAD_TRIAGE_TAG

# Mark all tests in this file as 'integration'.
pytestmark = pytest.mark.integration

TEST_USER_EMAIL = "pytest-lifecycle-user@zammad.local"

# Constants for the End-to-End Test (Fantastical Theme)
REAL_HISTORY_TITLE = "[Test] Warp Core Phase Variance"
REAL_NEW_TITLE = "[Test] Warp Core is humming loudly"

# Constants for New Tests
CLEAN_SLATE_TITLE = "[Test] Unique Issue 999"
COMPRESSION_HISTORY_TITLE = "[Test] Long History Log"
COMPRESSION_NEW_TITLE = "[Test] Long New Request"
IDEMPOTENCY_TITLE = "[Test] Idempotency Check"


async def _wait_for_search(search_func: Callable[..., List[Any]], assertion_func: Callable[[List[Any]], bool],
                           timeout: int = 30, interval: float = 1.0):
    """
    Repeatedly calls a search function and checks an assertion until it passes or times out.
    Includes debug printing to help diagnose indexing issues.
    """
    start_time = time.time()
    last_results = []
    while time.time() - start_time < timeout:
        try:
            results = await asyncio.to_thread(search_func)
            last_results = results
            if assertion_func(results):
                return  # Success
        except Exception as e:
            print(f"Search failed with error: {e}")

        await asyncio.sleep(interval)

    # Debug output on failure
    print(f"DEBUG: Timeout reached. Last search results ({len(last_results)}): {last_results}")
    pytest.fail(f"Search assertion did not pass within {timeout} seconds.")


async def wait_for_tag(zammad_client, ticket_id, tag, timeout=10):
    """
    Polls the ticket tags using the dedicated tags endpoint until the specified tag appears.
    """
    start = time.time()
    current_tags = []
    while time.time() - start < timeout:
        try:
            current_tags = await asyncio.to_thread(zammad_client.get_tags, ticket_id)
            if tag in current_tags:
                return  # Success
        except Exception as e:
            print(f"Error fetching ticket tags: {e}")

        await asyncio.sleep(1)

    pytest.fail(f"Tag '{tag}' not found on ticket {ticket_id} after {timeout}s. Current tags: {current_tags}")


def check_local_llm_health():
    """
    Checks if the Local LLM (KoboldCPP) is reachable and has a model loaded.
    Returns True if healthy, False otherwise.
    """
    try:
        url = f"{LOCAL_LLM_URL}/models"
        response = requests.get(url, timeout=2)
        if response.status_code == 200:
            return True

        url = f"{LOCAL_LLM_URL.replace('/v1', '')}/api/v1/model"
        response = requests.get(url, timeout=2)
        if response.status_code == 200:
            return True

    except Exception as e:
        print(f"Local LLM Health Check Failed: {e}")
    return False


@pytest.fixture(scope="module")
def zammad_client():
    try:
        client = ZammadClient()
        client.get_self()
        return client
    except (ValueError, requests.exceptions.RequestException) as e:
        pytest.skip(f"Skipping Zammad integration tests: Cannot connect. Error: {e}")


@pytest.fixture(scope="function")
def managed_test_user(zammad_client: ZammadClient) -> int:
    """
    Provides a persistent test user.
    Performs PRE-RUN cleanup by deleting old test tickets to avoid DB clutter.
    Leaves new tickets in place for inspection.
    """
    users = zammad_client.search_user(query=TEST_USER_EMAIL)
    if users:
        user_id = users[0]['id']
    else:
        user_data = zammad_client.create_user(
            email=TEST_USER_EMAIL,
            firstname="Pytest",
            lastname="LifecycleUser",
            note="Persistent integration test user."
        )
        user_id = user_data['id']

    titles_to_clean = [
        REAL_HISTORY_TITLE, REAL_NEW_TITLE,
        CLEAN_SLATE_TITLE,
        COMPRESSION_HISTORY_TITLE, COMPRESSION_NEW_TITLE,
        IDEMPOTENCY_TITLE
    ]

    print(f"\n[CLEANUP] Checking for old test tickets for user {user_id}...")
    for title in titles_to_clean:
        query = f'customer_id:{user_id} AND title:"{title}"'
        old_tickets = zammad_client.search_tickets(query=query)
        for t in old_tickets:
            print(f"[CLEANUP] Deleting old ticket #{t['id']} ('{t['title']}')...")
            zammad_client.delete_ticket(t['id'])

    yield user_id


@pytest.mark.asyncio
async def test_zammad_bot_end_to_end_real_flow(zammad_client: ZammadClient, managed_test_user: int):
    """
    TRUE INTEGRATION TEST: Zammad <-> Bot <-> Local LLM (KoboldCPP).
    Verifies the Happy Path with History.
    """
    if not check_local_llm_health():
        pytest.skip(f"Local LLM at {LOCAL_LLM_URL} is not reachable. Skipping real integration test.")

    solved_ticket_id = None
    new_ticket_id = None

    try:
        # 1. Create History Ticket (Fantastical Theme)
        solved_ticket = zammad_client.create_ticket(
            title=REAL_HISTORY_TITLE,
            group="Users",
            customer_id=managed_test_user,
            article_body="Problem: Dilithium crystals are out of alignment in the main warp core."
        )
        solved_ticket_id = solved_ticket['id']
        zammad_client.add_article_to_ticket(solved_ticket_id,
                                            body="Solution: Initiated a manual phase realignment of the core.",
                                            internal=False)
        zammad_client.update_ticket(solved_ticket_id, {'state': 'closed'})

        # 2. Wait for Indexing
        await _wait_for_search(
            search_func=lambda: zammad_client.search_tickets(
                query=f'title:"Warp" AND title:"Core" AND state.name:closed'),
            assertion_func=lambda results: any(t['id'] == solved_ticket_id for t in results),
            timeout=30
        )

        # 3. Create New Ticket
        new_ticket = zammad_client.create_ticket(
            title=REAL_NEW_TITLE,
            group="Users",
            customer_id=managed_test_user,
            article_body="The engines are making a loud humming sound and the warp core seems unstable."
        )
        new_ticket_id = new_ticket['id']
        print(f"\n[INFO] Created Real LLM Test Ticket ID: {new_ticket_id} (Warp Core)")

        # 4. Setup Bot
        memory_manager = MagicMock()
        text_engine = TextEngine()
        chat_system = ChatSystem(memory_manager, text_engine, zammad_client)

        chat_system.personas[TRIAGE_PERSONA_NAME] = Persona(
            persona_name=TRIAGE_PERSONA_NAME,
            model_name="local",
            prompt="You are a support triage assistant.",
            token_limit=300
        )
        bot = ZammadBot(chat_system)

        # 5. Spy and Run
        captured_interactions = []
        real_generate = text_engine.generate_response

        async def spy_generate_response(*args, **kwargs):
            result = await real_generate(*args, **kwargs)
            captured_interactions.append({"args": args, "kwargs": kwargs, "result": result})
            return result

        with patch.object(text_engine, 'generate_response', side_effect=spy_generate_response):
            await bot._process_ticket(new_ticket_id)

            if len(captured_interactions) >= 2:
                last_interaction = captured_interactions[-1]
                history_list = last_interaction['kwargs']['context_object']['history']
                raw_prompt = history_list[-1]['content'] if history_list else "NO PROMPT FOUND"
                raw_response_obj = last_interaction['result'][0]
                raw_response_content = raw_response_obj.get('content', 'NO CONTENT')

                dump_body = (
                    f"[TEST APPARATUS - AI TRIAGE REQUEST DUMP]\n\n"
                    f"--- PROMPT SENT TO LLM ---\n{raw_prompt}\n\n"
                    f"--- RAW RESPONSE FROM LLM ---\n{raw_response_content}"
                )

                zammad_client.add_article_to_ticket(
                    ticket_id=new_ticket_id,
                    body=dump_body,
                    internal=True
                )

        # 6. Verify
        articles = zammad_client.get_ticket_articles(new_ticket_id)
        ai_note = next((a for a in articles if a['internal'] is True and "[ AI TRIAGE CONTEXT DUMP ]" in a['body']),
                       None)
        assert ai_note is not None, "Bot failed to post the triage note."
        print(f"\n[VISUAL INSPECTION] Real LLM Note:\n{ai_note['body'][:300]}...")

    finally:
        print("\n[CLEANUP] Closing test tickets...")
        if solved_ticket_id: zammad_client.update_ticket(solved_ticket_id, {'state': 'closed'})
        if new_ticket_id: zammad_client.update_ticket(new_ticket_id, {'state': 'closed'})


@pytest.mark.asyncio
async def test_zammad_bot_clean_slate(zammad_client: ZammadClient, managed_test_user: int):
    """
    Verifies behavior when NO history exists.
    """
    if not check_local_llm_health():
        pytest.skip("Local LLM unreachable.")

    new_ticket_id = None
    try:
        new_ticket = zammad_client.create_ticket(
            title=CLEAN_SLATE_TITLE,
            group="Users",
            customer_id=managed_test_user,
            article_body="This is a unique issue with no precedent."
        )
        new_ticket_id = new_ticket['id']

        memory_manager = MagicMock()
        text_engine = TextEngine()
        chat_system = ChatSystem(memory_manager, text_engine, zammad_client)
        chat_system.personas[TRIAGE_PERSONA_NAME] = Persona(
            persona_name=TRIAGE_PERSONA_NAME, model_name="local", prompt="Triage", token_limit=300
        )
        bot = ZammadBot(chat_system)

        await bot._process_ticket(new_ticket_id)

        articles = zammad_client.get_ticket_articles(new_ticket_id)
        ai_note = next((a for a in articles if a['internal'] is True and "[ AI TRIAGE CONTEXT DUMP ]" in a['body']),
                       None)
        assert ai_note is not None

        body = ai_note['body']
        assert "GLOBAL MATCHES FOUND:\nNone" in body
        assert "USER HISTORY FOUND:\nNone" in body

    finally:
        if new_ticket_id: zammad_client.update_ticket(new_ticket_id, {'state': 'closed'})


@pytest.mark.asyncio
async def test_zammad_bot_adaptive_compression(zammad_client: ZammadClient, managed_test_user: int):
    """
    Verifies the Adaptive Compression logic.
    """
    if not check_local_llm_health():
        pytest.skip("Local LLM unreachable.")

    solved_ticket_id = None
    new_ticket_id = None

    try:
        long_text = "Lorem ipsum dolor sit amet. " * 100
        solved_ticket = zammad_client.create_ticket(
            title=COMPRESSION_HISTORY_TITLE,
            group="Users",
            customer_id=managed_test_user,
            article_body=f"Problem: {long_text}"
        )
        solved_ticket_id = solved_ticket['id']
        zammad_client.update_ticket(solved_ticket_id, {'state': 'closed'})

        await _wait_for_search(
            search_func=lambda: zammad_client.search_tickets(
                query=f'title:"Long" AND title:"History" AND state.name:closed'),
            assertion_func=lambda results: any(t['id'] == solved_ticket_id for t in results),
            timeout=30
        )

        new_ticket = zammad_client.create_ticket(
            title=COMPRESSION_NEW_TITLE,
            group="Users",
            customer_id=managed_test_user,
            article_body=f"New Issue: {long_text}"
        )
        new_ticket_id = new_ticket['id']

        memory_manager = MagicMock()
        text_engine = TextEngine()
        chat_system = ChatSystem(memory_manager, text_engine, zammad_client)
        chat_system.personas[TRIAGE_PERSONA_NAME] = Persona(
            persona_name=TRIAGE_PERSONA_NAME, model_name="local", prompt="Triage", token_limit=300
        )
        bot = ZammadBot(chat_system)

        with patch('src.interfaces.zammad_bot.TRIAGE_MAX_CONTEXT_CHARS', 1000):
            await bot._process_ticket(new_ticket_id)

        articles = zammad_client.get_ticket_articles(new_ticket_id)
        ai_note = next((a for a in articles if a['internal'] is True and "[ AI TRIAGE CONTEXT DUMP ]" in a['body']),
                       None)
        assert ai_note is not None

        print(f"\n[COMPRESSION TEST] Note Body:\n{ai_note['body'][:500]}...")

    finally:
        if solved_ticket_id: zammad_client.update_ticket(solved_ticket_id, {'state': 'closed'})
        if new_ticket_id: zammad_client.update_ticket(new_ticket_id, {'state': 'closed'})


@pytest.mark.asyncio
async def test_zammad_bot_idempotency(zammad_client: ZammadClient, managed_test_user: int):
    """
    Verifies that tickets are tagged and not re-processed.
    """
    if not check_local_llm_health():
        pytest.skip("Local LLM unreachable.")

    new_ticket_id = None
    try:
        new_ticket = zammad_client.create_ticket(
            title=IDEMPOTENCY_TITLE,
            group="Users",
            customer_id=managed_test_user,
            article_body="Test for idempotency."
        )
        new_ticket_id = new_ticket['id']

        memory_manager = MagicMock()
        text_engine = TextEngine()
        chat_system = ChatSystem(memory_manager, text_engine, zammad_client)
        chat_system.personas[TRIAGE_PERSONA_NAME] = Persona(
            persona_name=TRIAGE_PERSONA_NAME, model_name="local", prompt="Triage", token_limit=300
        )
        bot = ZammadBot(chat_system)

        await bot._process_ticket(new_ticket_id)

        await wait_for_tag(zammad_client, new_ticket_id, ZAMMAD_TRIAGE_TAG)

        query = f"state.name:new AND NOT tags:{ZAMMAD_TRIAGE_TAG}"
        await _wait_for_search(
            search_func=lambda: zammad_client.search_tickets(query),
            assertion_func=lambda results: not any(t['id'] == new_ticket_id for t in results),
            timeout=20
        )

    finally:
        if new_ticket_id: zammad_client.update_ticket(new_ticket_id, {'state': 'closed'})
