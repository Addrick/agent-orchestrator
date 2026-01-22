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
from config.global_config import TRIAGE_PERSONA_NAME, LOCAL_LLM_URL

# Mark all tests in this file as 'integration'.
pytestmark = pytest.mark.integration

TEST_USER_EMAIL = "pytest-lifecycle-user@zammad.local"

# Constants for "Prompt Dump" Test
FLUX_HISTORY_TITLE = "[Test] Flux Capacitor Malfunction"
FLUX_NEW_TITLE = "[Test] My Flux Capacitor is Broken"

# Constants for "Realistic" Test
WARP_HISTORY_TITLE = "[Test] WarpDrive Containment Breach"
WARP_NEW_TITLE = "[Test] WarpDrive fluctuating"

# Constants for "Real Local LLM" Test
REAL_HISTORY_TITLE = "[Test] Network Outage Sector 7"
REAL_NEW_TITLE = "[Test] Sector 7 is offline"


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


def check_local_llm_health():
    """
    Checks if the Local LLM (KoboldCPP) is reachable and has a model loaded.
    Returns True if healthy, False otherwise.
    """
    try:
        # Check version/info endpoint
        # KoboldCPP usually has /api/v1/model or /api/extra/version
        # We'll try the OpenAI-compatible models endpoint first as it's standard
        url = f"{LOCAL_LLM_URL}/models"
        response = requests.get(url, timeout=2)
        if response.status_code == 200:
            return True

        # Fallback check for Kobold specific
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
    # 1. Find or Create User
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

    # 2. Pre-Run Cleanup: Delete specific test tickets from previous runs
    # We clean up ALL known test titles here
    titles_to_clean = [FLUX_HISTORY_TITLE, FLUX_NEW_TITLE, WARP_HISTORY_TITLE, WARP_NEW_TITLE, REAL_HISTORY_TITLE,
                       REAL_NEW_TITLE]

    print(f"\n[CLEANUP] Checking for old test tickets for user {user_id}...")
    for title in titles_to_clean:
        # Quote the title to handle spaces/brackets correctly in search
        query = f'customer_id:{user_id} AND title:"{title}"'
        old_tickets = zammad_client.search_tickets(query=query)
        for t in old_tickets:
            print(f"[CLEANUP] Deleting old ticket #{t['id']} ('{t['title']}')...")
            zammad_client.delete_ticket(t['id'])

    yield user_id


@pytest.mark.asyncio
async def test_zammad_bot_real_local_llm_flow(zammad_client: ZammadClient, managed_test_user: int):
    """
    TRUE INTEGRATION TEST: Zammad <-> Bot <-> Local LLM (KoboldCPP).
    1. Checks if Local LLM is up.
    2. Creates History/New tickets.
    3. Runs Bot with REAL TextEngine (no mocks for generation).
    4. Spies on the generation call to capture the prompt.
    5. Posts the prompt dump.
    """
    if not check_local_llm_health():
        pytest.skip(f"Local LLM at {LOCAL_LLM_URL} is not reachable. Skipping real integration test.")

    solved_ticket_id = None
    new_ticket_id = None

    try:
        # 1. Create History Ticket
        solved_ticket = zammad_client.create_ticket(
            title=REAL_HISTORY_TITLE,
            group="Users",
            customer_id=managed_test_user,
            article_body="Problem: Network switch in Sector 7 is unresponsive."
        )
        solved_ticket_id = solved_ticket['id']
        zammad_client.add_article_to_ticket(solved_ticket_id, body="Solution: Power cycled the switch via PDU.",
                                            internal=False)
        zammad_client.update_ticket(solved_ticket_id, {'state': 'closed'})

        # 2. Wait for Indexing
        await _wait_for_search(
            search_func=lambda: zammad_client.search_tickets(query=f'title:"Sector 7" AND state.name:closed'),
            assertion_func=lambda results: any(t['id'] == solved_ticket_id for t in results),
            timeout=30
        )

        # 3. Create New Ticket
        new_ticket = zammad_client.create_ticket(
            title=REAL_NEW_TITLE,
            group="Users",
            customer_id=managed_test_user,
            article_body="I cannot connect to the internet in Sector 7."
        )
        new_ticket_id = new_ticket['id']
        print(f"\n[INFO] Created Real LLM Test Ticket ID: {new_ticket_id}")

        # 4. Setup Bot with REAL Components
        memory_manager = MagicMock()  # Not used for triage but required for init
        text_engine = TextEngine()  # Real Engine

        chat_system = ChatSystem(memory_manager, text_engine, zammad_client)

        # Inject Triage Persona forcing LOCAL model
        chat_system.personas[TRIAGE_PERSONA_NAME] = Persona(
            persona_name=TRIAGE_PERSONA_NAME,
            model_name="local",  # Force local model
            prompt="You are a support triage assistant. Analyze the ticket, user history, and similar past solutions.",
            token_limit=300
        )

        bot = ZammadBot(chat_system)

        # 5. Spy on generate_response to capture the prompt
        # We wrap the real method so it still executes!
        with patch.object(text_engine, 'generate_response', wraps=text_engine.generate_response) as spy_generate:

            # 6. Run Bot
            await bot._process_ticket(new_ticket_id)

            # 7. Capture Prompt
            # The bot calls generate_response twice:
            # 1. Keyword Extraction (Local)
            # 2. Final Analysis (Local, because we forced the persona)

            # We want the last call (Analysis)
            if spy_generate.call_count >= 2:
                last_call = spy_generate.call_args_list[-1]
                _, kwargs = last_call
                raw_prompt = kwargs['context_object']['current_message']['text']

                zammad_client.add_article_to_ticket(
                    ticket_id=new_ticket_id,
                    body=f"RAW PROMPT DUMP (REAL LLM TEST):\n\n{raw_prompt}",
                    internal=True
                )
            else:
                print("[WARNING] generate_response was not called twice as expected. Prompt dump might be missing.")

        # 8. Verify
        articles = zammad_client.get_ticket_articles(new_ticket_id)
        ai_note = next((a for a in articles if a['internal'] is True and "ANALYSIS" in a['body']),
                       None)  # "ANALYSIS" might vary depending on LLM output, but usually prompt guides it

        # Since it's a real LLM, we can't assert exact text, but we check if *an* internal note was posted
        # We check for the Context Dump header which is hardcoded in the bot
        context_dump_note = next(
            (a for a in articles if a['internal'] is True and "[ AI TRIAGE CONTEXT DUMP ]" in a['body']), None)

        assert context_dump_note is not None, "Bot failed to post the triage note."
        print(f"\n[VISUAL INSPECTION] Real LLM Note:\n{context_dump_note['body'][:300]}...")

    finally:
        if solved_ticket_id: zammad_client.update_ticket(solved_ticket_id, {'state': 'closed'})
        if new_ticket_id: zammad_client.update_ticket(new_ticket_id, {'state': 'closed'})


@pytest.mark.asyncio
async def test_zammad_bot_realistic_triage(zammad_client: ZammadClient, managed_test_user: int):
    """
    Test 1: Realistic Triage (Mocked).
    Uses 'WarpDrive' scenario.
    Mocks a full, professional LLM response.
    Goal: Visual inspection of the final Zammad note formatting.
    """
    solved_ticket_id = None
    new_ticket_id = None

    try:
        # 1. Create History Ticket
        solved_ticket = zammad_client.create_ticket(
            title=WARP_HISTORY_TITLE,
            group="Users",
            customer_id=managed_test_user,
            article_body="Problem: Plasma leak detected."
        )
        solved_ticket_id = solved_ticket['id']
        zammad_client.add_article_to_ticket(solved_ticket_id, body="Solution: Realign dilithium matrix.",
                                            internal=False)
        zammad_client.update_ticket(solved_ticket_id, {'state': 'closed'})

        # 2. Wait for Indexing
        await _wait_for_search(
            search_func=lambda: zammad_client.search_tickets(query=f'title:"WarpDrive" AND state.name:closed'),
            assertion_func=lambda results: any(t['id'] == solved_ticket_id for t in results),
            timeout=30
        )

        # 3. Create New Ticket
        new_ticket = zammad_client.create_ticket(
            title=WARP_NEW_TITLE,
            group="Users",
            customer_id=managed_test_user,
            article_body="Engines are unstable."
        )
        new_ticket_id = new_ticket['id']
        print(f"\n[INFO] Created Realistic Test Ticket ID: {new_ticket_id} (WarpDrive)")

        # 4. Setup Bot with Realistic Mock
        mock_chat_system = MagicMock(spec=ChatSystem)
        mock_chat_system.zammad_client = zammad_client
        mock_chat_system.personas = {}
        mock_chat_system.text_engine = MagicMock(spec=TextEngine)

        async def mock_generate_response(persona_config, context_object, tools=None):
            model = persona_config.get('model_name')
            if model == 'local':
                return {"type": "text", "content": "WarpDrive containment"}, {}
            else:
                # Realistic Analysis
                analysis = (
                    f"ANALYSIS: The user is reporting instability in the WarpDrive engines. "
                    f"Global history indicates a similar 'Containment Breach' (Ticket #{solved_ticket_id}) was resolved by realigning the dilithium matrix. "
                    f"Recommendation: Advise the user to check matrix alignment immediately."
                )
                return {"type": "text", "content": analysis}, {}

        mock_chat_system.text_engine.generate_response = AsyncMock(side_effect=mock_generate_response)
        bot = ZammadBot(mock_chat_system)

        # 5. Run
        await bot._process_ticket(new_ticket_id)

        # 6. Verify
        articles = zammad_client.get_ticket_articles(new_ticket_id)
        triage_note = next((a for a in articles if a['internal'] is True and "ANALYSIS:" in a['body']), None)
        assert triage_note is not None
        print(f"\n[VISUAL INSPECTION] Created Note Body for Ticket {new_ticket_id}:\n{triage_note['body']}")

    finally:
        # 7. Close tickets
        if solved_ticket_id: zammad_client.update_ticket(solved_ticket_id, {'state': 'closed'})
        if new_ticket_id: zammad_client.update_ticket(new_ticket_id, {'state': 'closed'})


@pytest.mark.asyncio
async def test_zammad_bot_triage_with_prompt_dump(zammad_client: ZammadClient, managed_test_user: int):
    """
    Test 2: Triage with Prompt Dump (Mocked).
    Uses 'Flux Capacitor' scenario.
    Captures the raw prompt sent to the LLM and posts it as a second note.
    Goal: Debugging the exact context sent to the AI.
    """
    solved_ticket_id = None
    new_ticket_id = None

    try:
        # 1. Create History Ticket
        solved_ticket = zammad_client.create_ticket(
            title=FLUX_HISTORY_TITLE,
            group="Users",
            customer_id=managed_test_user,
            article_body="Problem: My device is not charging."
        )
        solved_ticket_id = solved_ticket['id']
        zammad_client.add_article_to_ticket(solved_ticket_id, body="Solution: Charge to 1.21 Gigawatts.",
                                            internal=False)
        zammad_client.update_ticket(solved_ticket_id, {'state': 'closed'})

        # 2. Wait for Indexing
        await _wait_for_search(
            search_func=lambda: zammad_client.search_tickets(
                query=f'title:"Flux" AND title:"Capacitor" AND state.name:closed'),
            assertion_func=lambda results: any(t['id'] == solved_ticket_id for t in results),
            timeout=30
        )

        # 3. Create New Ticket
        new_ticket = zammad_client.create_ticket(
            title=FLUX_NEW_TITLE,
            group="Users",
            customer_id=managed_test_user,
            article_body="It won't turn on."
        )
        new_ticket_id = new_ticket['id']
        print(f"\n[INFO] Created Prompt Dump Test Ticket ID: {new_ticket_id} (Flux Capacitor)")

        # 4. Setup Bot with Mock
        mock_chat_system = MagicMock(spec=ChatSystem)
        mock_chat_system.zammad_client = zammad_client
        mock_chat_system.personas = {}
        mock_chat_system.text_engine = MagicMock(spec=TextEngine)

        async def mock_generate_response(persona_config, context_object, tools=None):
            model = persona_config.get('model_name')
            if model == 'local':
                # Return keywords with space
                return {"type": "text", "content": "Flux Capacitor"}, {}
            else:
                # Realistic Analysis
                analysis = (
                    f"ANALYSIS: The user is reporting a Flux Capacitor malfunction. "
                    f"Global history confirms a matching solution in Ticket #{solved_ticket_id}: 'Charge to 1.21 Gigawatts'. "
                    f"Recommendation: Provide these instructions to the user."
                )
                return {"type": "text", "content": analysis}, {}

        mock_chat_system.text_engine.generate_response = AsyncMock(side_effect=mock_generate_response)
        bot = ZammadBot(mock_chat_system)

        # 5. Run Bot
        await bot._process_ticket(new_ticket_id)

        # 6. Capture and Post Raw Prompt
        main_llm_call = mock_chat_system.text_engine.generate_response.call_args_list[-1]
        _, kwargs = main_llm_call
        raw_prompt = kwargs['context_object']['current_message']['text']

        zammad_client.add_article_to_ticket(
            ticket_id=new_ticket_id,
            body=f"RAW PROMPT DUMP (TEST APPARATUS):\n\n{raw_prompt}",
            internal=True
        )

        # 7. Verify
        articles = zammad_client.get_ticket_articles(new_ticket_id)

        # Check for AI Note
        ai_note = next((a for a in articles if a['internal'] is True and "ANALYSIS:" in a['body']), None)
        assert ai_note is not None, "AI Triage note missing"

        # Check for Prompt Dump Note
        dump_note = next((a for a in articles if a['internal'] is True and "RAW PROMPT DUMP" in a['body']), None)
        assert dump_note is not None, "Test Apparatus Prompt Dump missing"

        print(f"\n[VISUAL INSPECTION] AI Note:\n{ai_note['body'][:200]}...")
        print(f"\n[VISUAL INSPECTION] Prompt Dump:\n{dump_note['body'][:200]}...")

    finally:
        # 8. Close tickets
        if solved_ticket_id: zammad_client.update_ticket(solved_ticket_id, {'state': 'closed'})
        if new_ticket_id: zammad_client.update_ticket(new_ticket_id, {'state': 'closed'})
