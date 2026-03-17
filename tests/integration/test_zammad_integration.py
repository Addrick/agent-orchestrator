# tests/integration/test_zammad_integration.py

import pytest
import asyncio
import time
from typing import Callable, List, Any
import requests
from unittest.mock import MagicMock, patch
from src.clients.zammad_client import ZammadClient
from src.interfaces.zammad_bot import ZammadBot
from src.chat_system import ChatSystem
from src.engine import TextEngine
from config.global_config import (
    TRIAGE_SCOUT_NAME,
    TRIAGE_SUMMARIZER_NAME,
    TRIAGE_ANALYST_NAME,
    TRIAGE_FILTER_NAME,
    ZAMMAD_TRIAGE_TAG,
    ZAMMAD_BOT_EMAIL,
    ZAMMAD_BOT_FIRSTNAME,
    ZAMMAD_BOT_LASTNAME
)

# Mark all tests in this file as 'integration'.
pytestmark = pytest.mark.integration

TEST_USER_EMAIL = "pytest-lifecycle-user@zammad.local"

# Constants for the End-to-End Test
REAL_HISTORY_TITLE = "[Test] Warp Core Phase Variance"
REAL_NEW_TITLE = "[Test] Warp Core is humming loudly"

# Constants for New Tests
CLEAN_SLATE_TITLE = "[Test] Unique Issue 999"
COMPRESSION_HISTORY_TITLE = "[Test] Long History Log"
COMPRESSION_NEW_TITLE = "[Test] Long New Request"
IDEMPOTENCY_TITLE = "[Test] Idempotency Check"
FALLBACK_TITLE = "[Test] Fallback Check"

# Constants for Filtering Test
FILTER_RELEVANT_TITLE = "[Test] Printer Paper Jam"
FILTER_IRRELEVANT_TITLE = "[Test] Printer 3D Model Request"
FILTER_NEW_TITLE = "[Test] Printer is stuck"


async def _mock_llm_generate(persona_config, context_object, tools=None):
    """
    Deterministic stand-in for TextEngine.generate_response used across all
    integration tests.  Dispatches on the persona_prompt so each pipeline
    stage gets a plausible, structurally-valid response without touching the
    real API.
    """
    sys_prompt = context_object.get('persona_prompt', '')

    if 'keyword extraction' in sys_prompt:
        return {"type": "text", "content": "network error connectivity"}, {}

    if 'summarization' in sys_prompt:
        return {"type": "text", "content": "Mock summary: ticket describes a technical issue requiring investigation."}, {}

    if 'relevance classifier' in sys_prompt:
        return {"type": "text", "content": "RELEVANT"}, {}

    # Analyst (and any unrecognised persona)
    return {"type": "text", "content": "Mock triage: Issue noted. No similar history found. Recommend standard investigation procedure."}, {}


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


@pytest.fixture(scope="module")
def zammad_client():
    try:
        client = ZammadClient()
        client.get_self()
        return client
    except (ValueError, requests.exceptions.RequestException) as e:
        pytest.skip(f"Skipping Zammad integration tests: Cannot connect. Error: {e}")


@pytest.fixture(scope="module")
def bot_identity(zammad_client):
    """
    Ensures the Zammad Bot user exists for the tests.
    """
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


@pytest.fixture(scope="module")
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
        IDEMPOTENCY_TITLE, FALLBACK_TITLE,
        FILTER_RELEVANT_TITLE, FILTER_IRRELEVANT_TITLE, FILTER_NEW_TITLE
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
async def test_zammad_bot_end_to_end_real_flow(zammad_client: ZammadClient, managed_test_user: int, bot_identity):
    """
    Verifies the Happy Path with History: bot finds a related closed ticket,
    runs the full triage pipeline, and posts a note on the new ticket.
    """
    solved_ticket_id = None
    new_ticket_id = None

    try:
        # 1. Create History Ticket
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
        print(f"\n[INFO] Created Test Ticket ID: {new_ticket_id}")

        # 4. Setup Bot and Run
        memory_manager = MagicMock()
        text_engine = TextEngine()
        chat_system = ChatSystem(memory_manager, text_engine, zammad_client)
        bot = ZammadBot(chat_system)

        # Scout returns keywords that match the history ticket's title/body so it
        # is actually retrieved and included in the triage note.
        async def mock_e2e_generate(persona_config, context_object, tools=None):
            if 'keyword extraction' in context_object.get('persona_prompt', ''):
                return {"type": "text", "content": "warp core dilithium"}, {}
            return await _mock_llm_generate(persona_config, context_object, tools)

        with patch.object(text_engine, 'generate_response', side_effect=mock_e2e_generate):
            await bot._process_ticket(new_ticket_id)

        # 5. Verify note was posted and the history ticket is referenced in it
        articles = zammad_client.get_ticket_articles(new_ticket_id)
        ai_note = next((a for a in articles if a['internal'] is True and "[ AI TRIAGE CONTEXT DUMP ]" in a['body']),
                       None)
        assert ai_note is not None, "Bot failed to post the triage note."
        assert f"{REAL_HISTORY_TITLE} (Ticket #{solved_ticket_id})" in ai_note['body'], \
            "History ticket should have been found and listed in the triage note."
        print(f"\n[VERIFY] Triage note posted. Excerpt:\n{ai_note['body'][:300]}...")

    finally:
        print("\n[CLEANUP] Closing test tickets...")
        if solved_ticket_id: zammad_client.update_ticket(solved_ticket_id, {'state': 'closed'})
        if new_ticket_id: zammad_client.update_ticket(new_ticket_id, {'state': 'closed'})


@pytest.mark.asyncio
async def test_zammad_bot_clean_slate(zammad_client: ZammadClient, managed_test_user: int, bot_identity):
    """
    Verifies behavior when NO history exists.
    """
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
        bot = ZammadBot(chat_system)

        with patch.object(text_engine, 'generate_response', side_effect=_mock_llm_generate):
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
async def test_zammad_bot_adaptive_compression(zammad_client: ZammadClient, managed_test_user: int, bot_identity):
    """
    Verifies the Adaptive Compression logic.
    """
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
        bot = ZammadBot(chat_system)

        with patch.object(text_engine, 'generate_response', side_effect=_mock_llm_generate):
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
async def test_zammad_bot_idempotency(zammad_client: ZammadClient, managed_test_user: int, bot_identity):
    """
    Verifies that tickets are tagged and not re-processed.
    """
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
        bot = ZammadBot(chat_system)

        with patch.object(text_engine, 'generate_response', side_effect=_mock_llm_generate):
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


@pytest.mark.asyncio
async def test_zammad_bot_impersonation_fallback(zammad_client: ZammadClient, managed_test_user: int, bot_identity):
    """
    Verifies that if impersonation fails (e.g. invalid user), the bot falls back
    to posting as the API token owner.
    """
    ticket_id = None
    try:
        # 1. Create Ticket
        ticket = zammad_client.create_ticket(
            title=FALLBACK_TITLE,
            group="Users",
            customer_id=managed_test_user,
            article_body="Testing permission fallback."
        )
        ticket_id = ticket['id']

        # 2. Setup Bot
        memory_manager = MagicMock()
        text_engine = TextEngine()
        chat_system = ChatSystem(memory_manager, text_engine, zammad_client)
        bot = ZammadBot(chat_system)

        # 3. Run with Invalid Email to force impersonation failure
        with patch.object(text_engine, 'generate_response', side_effect=_mock_llm_generate):
            with patch("src.interfaces.zammad_bot.ZAMMAD_BOT_EMAIL", "nonexistent_ghost@example.com"):
                await bot._process_ticket(ticket_id)

        # 4. Verify Tag (Success implies fallback worked)
        await wait_for_tag(zammad_client, ticket_id, ZAMMAD_TRIAGE_TAG)

        # 5. Verify Note Exists
        articles = zammad_client.get_ticket_articles(ticket_id)
        ai_note = next((a for a in articles if a['internal'] is True and "[ AI TRIAGE CONTEXT DUMP ]" in a['body']),
                       None)
        assert ai_note is not None, "Fallback failed to post note."

        # 6. Verify Author is API Token Owner (Self)
        myself = zammad_client.get_self()
        assert ai_note['created_by_id'] == myself['id']
        print(f"\n[FALLBACK TEST] Note posted successfully by ID {ai_note['created_by_id']} (API Owner).")

    finally:
        if ticket_id: zammad_client.update_ticket(ticket_id, {'state': 'closed'})


@pytest.mark.asyncio
async def test_zammad_bot_filtering_logic(zammad_client: ZammadClient, managed_test_user: int, bot_identity):
    """
    Verifies that the 'triage_filter' persona correctly filters out irrelevant tickets.
    The filter mock returns content-aware RELEVANT/IRRELEVANT decisions; all other
    pipeline stages use the shared mock.
    """
    relevant_id = None
    irrelevant_id = None
    new_ticket_id = None

    try:
        # 1. Create Relevant History
        t1 = zammad_client.create_ticket(
            title=FILTER_RELEVANT_TITLE,
            group="Users",
            customer_id=managed_test_user,
            article_body="Paper jam in tray 2."
        )
        relevant_id = t1['id']
        zammad_client.update_ticket(relevant_id, {'state': 'closed'})

        # 2. Create Irrelevant History (Shares keyword 'Printer')
        t2 = zammad_client.create_ticket(
            title=FILTER_IRRELEVANT_TITLE,
            group="Users",
            customer_id=managed_test_user,
            article_body="I need a 3D model of a printer."
        )
        irrelevant_id = t2['id']
        zammad_client.update_ticket(irrelevant_id, {'state': 'closed'})

        # Wait for Indexing
        await _wait_for_search(
            search_func=lambda: zammad_client.search_tickets(query=f'title:"Printer" AND state.name:closed'),
            assertion_func=lambda results: any(t['id'] == relevant_id for t in results) and any(
                t['id'] == irrelevant_id for t in results),
            timeout=30
        )

        # 3. Create New Ticket
        new_ticket = zammad_client.create_ticket(
            title=FILTER_NEW_TITLE,
            group="Users",
            customer_id=managed_test_user,
            article_body="My printer is stuck and won't print."
        )
        new_ticket_id = new_ticket['id']

        # 4. Setup Bot
        memory_manager = MagicMock()
        text_engine = TextEngine()
        chat_system = ChatSystem(memory_manager, text_engine, zammad_client)
        bot = ZammadBot(chat_system)

        async def mock_filter_generate(persona_config, context_object, tools=None):
            sys_prompt = context_object.get('persona_prompt', '')
            if 'keyword extraction' in sys_prompt:
                return {"type": "text", "content": "Printer"}, {}
            if 'relevance classifier' in sys_prompt:
                prompt_lower = context_object['history'][0]['content'].lower()
                if 'paper jam' in prompt_lower:
                    return {"type": "text", "content": "RELEVANT"}, {}
                if '3d model' in prompt_lower:
                    return {"type": "text", "content": "IRRELEVANT"}, {}
            return await _mock_llm_generate(persona_config, context_object, tools)

        with patch.object(text_engine, 'generate_response', side_effect=mock_filter_generate):
            await bot._process_ticket(new_ticket_id)

        # 5. Verify
        articles = zammad_client.get_ticket_articles(new_ticket_id)
        ai_note = next((a for a in articles if a['internal'] is True and "[ AI TRIAGE CONTEXT DUMP ]" in a['body']),
                       None)
        assert ai_note is not None

        body = ai_note['body']

        # Relevant ticket should be listed normally
        assert f"{FILTER_RELEVANT_TITLE} (Ticket #{relevant_id})" in body

        # Irrelevant ticket should be absent from global matches or marked irrelevant in user history
        assert f"{FILTER_IRRELEVANT_TITLE} (Ticket #{irrelevant_id}) (Irrelevant)" in body or \
               f"{FILTER_IRRELEVANT_TITLE} (Ticket #{irrelevant_id})" not in body

    finally:
        if relevant_id: zammad_client.update_ticket(relevant_id, {'state': 'closed'})
        if irrelevant_id: zammad_client.update_ticket(irrelevant_id, {'state': 'closed'})
        if new_ticket_id: zammad_client.update_ticket(new_ticket_id, {'state': 'closed'})
