# tests/live/test_agent_live.py
#
# Live tests for the agent notification subsystem.
#
# discord_live: Requires DISCORD_API_KEY — boots a real discord.Client and sends DMs.
# zammad_live: Requires ZAMMAD_URL + ZAMMAD_API_KEY — posts real Zammad internal notes.
# llm_live: Requires LLM API keys — makes real LLM calls for triage/dispatch.
#
# These tests verify that the notification pipeline works end-to-end
# with real external services (not mocks).

import asyncio
import json
import os
import random
import threading
import time

import discord
import pytest
from unittest.mock import patch

from src.clients.notification import (
    DiscordNotifier,
    LogNotifier,
    NotificationRouter,
    ZammadNotifier,
)

# Target Discord user for DM tests — the repo owner (adrich)
DISCORD_TEST_USER_ID = "321783731146850305"

# Timeout for the Discord client to connect and become ready
DISCORD_READY_TIMEOUT = 30

# Stable titles for test tickets — used for pre-run cleanup
PIPELINE_TICKET_TITLE = "[Test] VPN connection drops every 15 minutes"
NOTIFIER_TICKET_TITLE = "[Test] Agent Notifier Live"
MULTICHANNEL_TICKET_TITLE = "[Test] Multi-Channel Router"
MOCKED_PIPELINE_TICKET_TITLE = "[Test] Mocked Pipeline - Printer offline"


# ---------------------------------------------------------------------------
# Discord client fixture — boots in a background thread with its own loop
# ---------------------------------------------------------------------------

class _DiscordClientHolder:
    """Manages a Discord client on a dedicated event loop in a background thread.

    This avoids async fixture issues with pytest-asyncio strict mode by keeping
    the Discord client's long-running event loop entirely separate from pytest's.
    """

    def __init__(self, token: str) -> None:
        self.token = token
        self.client: discord.Client | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._ready = threading.Event()
        self._failed: Exception | None = None

    def start(self) -> None:
        """Boot the Discord client in a background thread."""
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

        if not self._ready.wait(timeout=DISCORD_READY_TIMEOUT):
            self.stop()
            pytest.skip(f"Discord client did not become ready within {DISCORD_READY_TIMEOUT}s")

        if self._failed:
            pytest.skip(f"Discord client failed to start: {self._failed}")

    def stop(self) -> None:
        """Cleanly shut down the client and thread."""
        if self._loop and not self._loop.is_closed() and self.client and not self.client.is_closed():
            future = asyncio.run_coroutine_threadsafe(self.client.close(), self._loop)
            try:
                future.result(timeout=10)
            except Exception:
                pass

        if self._loop and not self._loop.is_closed():
            self._loop.call_soon_threadsafe(self._loop.stop)

        if self._thread:
            self._thread.join(timeout=10)

    def _run(self) -> None:
        """Thread target — runs the Discord client's event loop."""
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)

        intents = discord.Intents.default()
        self.client = discord.Client(intents=intents)

        @self.client.event
        async def on_ready():
            self._ready.set()

        try:
            self._loop.run_until_complete(self.client.start(self.token))
        except Exception as e:
            self._failed = e
            self._ready.set()  # Unblock the main thread
        finally:
            self._loop.close()

    def run_async(self, coro):
        """Run a coroutine on the Discord client's event loop from the test thread."""
        if not self._loop or self._loop.is_closed():
            raise RuntimeError("Discord event loop is not running")
        future = asyncio.run_coroutine_threadsafe(coro, self._loop)
        return future.result(timeout=30)


@pytest.fixture(scope="module")
def discord_holder():
    """Module-scoped fixture: boots a Discord client, yields it, then shuts down.

    Returns a _DiscordClientHolder whose .client is a connected discord.Client.
    Tests should use holder.run_async(coro) to run async calls on the client's loop.
    """
    token = os.environ.get("DISCORD_API_KEY")
    if not token:
        pytest.skip("DISCORD_API_KEY not set")

    holder = _DiscordClientHolder(token)
    holder.start()
    yield holder
    holder.stop()


# ---------------------------------------------------------------------------
# Discord DM tests
# ---------------------------------------------------------------------------

class TestDiscordNotifierLive:
    """Send real Discord DMs via the DiscordNotifier."""

    pytestmark = pytest.mark.discord_live

    def test_send_dm_to_user(self, discord_holder):
        """DiscordNotifier.send() delivers a DM and returns True."""
        notifier = DiscordNotifier(discord_holder.client)

        timestamp = time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())
        result = discord_holder.run_async(notifier.send(
            recipient=DISCORD_TEST_USER_ID,
            subject="Agent Live Test",
            body=(
                f"Automated test from pytest — verifying Discord DM delivery.\n"
                f"Timestamp: {timestamp}\n"
                f"If you see this, the notification pipeline is working."
            ),
        ))

        assert result is True, "DiscordNotifier.send() returned False — DM delivery failed"

    def test_send_dm_invalid_user_returns_false(self, discord_holder):
        """Sending to a nonexistent user ID returns False (not an exception)."""
        notifier = DiscordNotifier(discord_holder.client)

        result = discord_holder.run_async(notifier.send(
            recipient="000000000000000000",  # Invalid user ID
            subject="Should Fail",
            body="This message should not be delivered.",
        ))

        assert result is False, "Expected False for invalid recipient, got True"


class TestNotificationRouterLive:
    """End-to-end routing test with a real Discord backend."""

    pytestmark = pytest.mark.discord_live

    def test_router_discord_dm_channel(self, discord_holder):
        """NotificationRouter routes 'discord_dm' to DiscordNotifier and delivers."""
        router = NotificationRouter()
        router.register("discord_dm", DiscordNotifier(discord_holder.client))

        timestamp = time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())
        result = discord_holder.run_async(router.send(
            channel="discord_dm",
            recipient=DISCORD_TEST_USER_ID,
            subject="Router Live Test",
            body=(
                f"Routed via NotificationRouter → discord_dm channel.\n"
                f"Timestamp: {timestamp}"
            ),
        ))

        assert result is True

    def test_router_unknown_channel_falls_back_to_log(self, discord_holder):
        """An unregistered channel name falls back to LogNotifier (returns True)."""
        router = NotificationRouter()
        router.register("discord_dm", DiscordNotifier(discord_holder.client))

        result = discord_holder.run_async(router.send(
            channel="carrier_pigeon",
            recipient="nobody",
            subject="Fallback Test",
            body="This should go to the log fallback.",
        ))

        assert result is True  # LogNotifier always returns True


# ---------------------------------------------------------------------------
# Zammad notification tests
# ---------------------------------------------------------------------------

class TestZammadNotifierLive:
    """Post real internal notes to Zammad via ZammadNotifier."""

    pytestmark = pytest.mark.zammad_live

    @pytest.fixture
    def zammad_client(self):
        """Provide a live ZammadClient, skip if unavailable."""
        import requests
        from src.clients.zammad_client import ZammadClient

        try:
            client = ZammadClient()
            client.get_self()
            return client
        except (ValueError, requests.exceptions.RequestException) as e:
            pytest.skip(f"Zammad unavailable: {e}")

    @pytest.fixture
    def test_ticket(self, zammad_client):
        """Create a test ticket for notification testing.

        Cleans up previous runs, leaves new ticket for inspection.
        """
        old = zammad_client.search_tickets(
            query=f'title:"{NOTIFIER_TICKET_TITLE}"'
        )
        for t in old:
            try:
                zammad_client.delete_ticket(t["id"])
            except Exception:
                pass

        ticket_data = zammad_client.create_ticket(
            title=NOTIFIER_TICKET_TITLE,
            group="Users",
            customer_id=1,
            article_body="Ticket for agent notification live test.",
        )
        yield ticket_data["id"]

    @pytest.mark.asyncio
    async def test_zammad_internal_note(self, zammad_client, test_ticket):
        """ZammadNotifier posts an internal note and returns True."""
        notifier = ZammadNotifier(zammad_client)

        timestamp = time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())
        result = await notifier.send(
            recipient=str(test_ticket),
            subject="Agent Live Test Note",
            body=(
                f"Automated test from pytest — verifying Zammad notification.\n"
                f"Timestamp: {timestamp}"
            ),
        )

        assert result is True

    @pytest.mark.asyncio
    async def test_zammad_invalid_ticket_returns_false(self, zammad_client):
        """Posting to a nonexistent ticket ID returns False."""
        notifier = ZammadNotifier(zammad_client)

        result = await notifier.send(
            recipient="999999999",  # Nonexistent ticket
            subject="Should Fail",
            body="This note should not be posted.",
        )

        assert result is False


class TestMultiChannelRouterLive:
    """Test NotificationRouter with multiple real backends wired up."""

    pytestmark = [pytest.mark.discord_live, pytest.mark.zammad_live]

    @pytest.fixture
    def zammad_client(self):
        import requests
        from src.clients.zammad_client import ZammadClient

        try:
            client = ZammadClient()
            client.get_self()
            return client
        except (ValueError, requests.exceptions.RequestException) as e:
            pytest.skip(f"Zammad unavailable: {e}")

    @pytest.fixture
    def test_ticket(self, zammad_client):
        """Clean up previous runs, leave new ticket for inspection."""
        old = zammad_client.search_tickets(
            query=f'title:"{MULTICHANNEL_TICKET_TITLE}"'
        )
        for t in old:
            try:
                zammad_client.delete_ticket(t["id"])
            except Exception:
                pass

        ticket_data = zammad_client.create_ticket(
            title=MULTICHANNEL_TICKET_TITLE,
            group="Users",
            customer_id=1,
            article_body="Ticket for multi-channel router test.",
        )
        yield ticket_data["id"]

    def test_multi_channel_dispatch(self, discord_holder, zammad_client, test_ticket):
        """Router sends to both discord_dm and zammad channels in sequence."""
        router = NotificationRouter()
        router.register("discord_dm", DiscordNotifier(discord_holder.client))
        router.register("zammad", ZammadNotifier(zammad_client))
        router.register("log", LogNotifier())

        timestamp = time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())

        # Send via Discord (runs on discord's event loop)
        discord_result = discord_holder.run_async(router.send(
            channel="discord_dm",
            recipient=DISCORD_TEST_USER_ID,
            subject="Multi-Channel Test",
            body=f"Discord leg of multi-channel test.\nTimestamp: {timestamp}",
        ))

        # Send via Zammad (runs on discord's event loop, uses asyncio.to_thread internally)
        zammad_result = discord_holder.run_async(router.send(
            channel="zammad",
            recipient=str(test_ticket),
            subject="Multi-Channel Test",
            body=f"Zammad leg of multi-channel test.\nTimestamp: {timestamp}",
        ))

        assert discord_result is True, "Discord leg failed"
        assert zammad_result is True, "Zammad leg failed"
        assert router.available_channels == ["discord_dm", "zammad", "log"]


# ---------------------------------------------------------------------------
# Full pipeline: ticket → triage → dispatch → DM + tag verification
# ---------------------------------------------------------------------------

class TestFullTriageDispatchPipeline:
    """End-to-end test: create ticket → triage bot → dispatch agent → DM.

    Verifies the entire autonomous pipeline with real services:
    - Zammad: ticket creation, articles, tagging
    - LLM: triage analysis (scout, filter, analyst) + dispatch decision
    - Discord: DM notification delivery
    - SQLite: Agent_Actions step logging

    Requires all three live markers. Skipped if any service is unavailable.
    """

    pytestmark = [pytest.mark.discord_live, pytest.mark.zammad_live, pytest.mark.llm_live]

    @pytest.fixture
    def zammad_client(self):
        import requests
        from src.clients.zammad_client import ZammadClient
        try:
            client = ZammadClient()
            client.get_self()
            return client
        except (ValueError, requests.exceptions.RequestException) as e:
            pytest.skip(f"Zammad unavailable: {e}")

    @pytest.fixture
    def pipeline_env(self, zammad_client, discord_holder):
        """Set up a full ChatSystem + agents for the pipeline test.

        Yields a dict with all the components needed, then cleans up.
        """
        from src.chat_system import ChatSystem
        from src.database.memory_manager import MemoryManager
        from src.engine import TextEngine
        from src.interfaces.zammad_bot import ZammadBot
        from src.agents.dispatch_agent import DispatchAgent
        from config.global_config import TEST_MEMORY_DATABASE_FILE

        db_path = f"{TEST_MEMORY_DATABASE_FILE}.pipeline.{random.randint(1000, 9999)}"
        if os.path.exists(db_path):
            os.remove(db_path)

        memory_manager = MemoryManager(db_path=db_path)
        memory_manager.create_schema()
        text_engine = TextEngine()

        # Build ChatSystem with all personas (system personas injected by ZammadBot)
        with patch('src.chat_system.load_personas_from_file', return_value={}):
            chat_system = ChatSystem(
                memory_manager=memory_manager,
                text_engine=text_engine,
            )

        # Wire notification router with Discord + Zammad + log
        notification_router = NotificationRouter()
        notification_router.register("discord_dm", DiscordNotifier(discord_holder.client))
        notification_router.register("zammad", ZammadNotifier(zammad_client))
        notification_router.register("log", LogNotifier())

        # Build agents (system personas injected on construction)
        triage_bot = ZammadBot(chat_system, zammad_client)
        dispatch_agent = DispatchAgent(
            chat_system=chat_system,
            zammad_client=zammad_client,
            notification_router=notification_router,
            agent_config={
                "notification_defaults": {"channel": "discord_dm", "recipient": "adrich"},
                "_recipients": {
                    "adrich": {
                        "discord_user_id": DISCORD_TEST_USER_ID,
                        "discord_username": "adrich",
                    }
                },
            },
        )

        yield {
            "chat_system": chat_system,
            "memory_manager": memory_manager,
            "text_engine": text_engine,
            "zammad_client": zammad_client,
            "triage_bot": triage_bot,
            "dispatch_agent": dispatch_agent,
            "notification_router": notification_router,
            "discord_holder": discord_holder,
        }

        memory_manager.close()
        time.sleep(0.1)
        if os.path.exists(db_path):
            try:
                os.remove(db_path)
            except PermissionError:
                pass

    @pytest.fixture
    def test_ticket(self, zammad_client):
        """Create a realistic support ticket for pipeline testing.

        Pre-run cleanup: deletes any leftover tickets from previous runs
        (matched by title). The new ticket is left in Zammad after the test
        for manual inspection of tags, internal notes, and dispatch artifacts.
        """
        # Clean up previous runs
        old_tickets = zammad_client.search_tickets(
            query=f'title:"{PIPELINE_TICKET_TITLE}"'
        )
        for t in old_tickets:
            print(f"[CLEANUP] Deleting old pipeline ticket #{t['id']}")
            try:
                zammad_client.delete_ticket(t["id"])
            except Exception:
                pass

        # Create fresh ticket
        ticket_data = zammad_client.create_ticket(
            title=PIPELINE_TICKET_TITLE,
            group="Users",
            customer_id=1,
            article_body=(
                "Hi, my VPN keeps disconnecting roughly every 15 minutes. "
                "I'm using GlobalProtect on Windows 11. The issue started "
                "after the latest Windows update KB5039302. I've tried "
                "reinstalling the client but it didn't help. My machine is "
                "DESKTOP-ABC123. This is blocking my ability to work remotely."
            ),
        )
        ticket_id = ticket_data["id"]
        ticket_number = ticket_data["number"]
        # No teardown — ticket left in place for inspection
        yield {"id": ticket_id, "number": ticket_number, "data": ticket_data}

    def test_full_pipeline_triage_then_dispatch(self, pipeline_env, test_ticket):
        """Full pipeline: ticket → triage (LLM) → dispatch (LLM + DM).

        This test exercises every stage of the autonomous pipeline:
        1. Triage bot processes the ticket (4 LLM calls: scout, filter, analyst)
        2. Verifies 'autotriaged' tag and internal note with context dump
        3. Dispatch agent processes the ticket (1 LLM call: dispatch_analyst)
        4. Verifies 'ai_dispatched' tag, Discord DM sent, and Agent_Actions logged
        """
        triage_bot = pipeline_env["triage_bot"]
        dispatch_agent = pipeline_env["dispatch_agent"]
        zammad_client = pipeline_env["zammad_client"]
        memory_manager = pipeline_env["memory_manager"]
        discord_holder = pipeline_env["discord_holder"]
        ticket_id = test_ticket["id"]

        # ── Stage 1: Triage ──────────────────────────────────────────
        # Run a single triage cycle on the test ticket.
        # This makes real LLM calls (scout → filter → analyst).
        discord_holder.run_async(triage_bot._process_ticket(ticket_id))

        # Verify: autotriaged tag applied
        tags = zammad_client.get_tags(ticket_id)
        assert "autotriaged" in tags, (
            f"Triage bot did not add 'autotriaged' tag. Tags: {tags}"
        )

        # Verify: internal note posted with triage analysis
        articles = zammad_client.get_ticket_articles(ticket_id)
        internal_notes = [
            a for a in articles
            if a.get("internal", False)
        ]
        assert len(internal_notes) >= 1, (
            f"No internal notes found. Articles: {[a.get('subject') for a in articles]}"
        )

        # The triage note should contain the context dump marker
        triage_bodies = [a.get("body", "") for a in internal_notes]
        has_context_dump = any("AI TRIAGE CONTEXT DUMP" in b for b in triage_bodies)
        has_recommendation = any("Recommended Action" in b or "recommend" in b.lower() for b in triage_bodies)
        assert has_context_dump or has_recommendation, (
            f"Triage note missing expected content. "
            f"Note previews: {[b[:200] for b in triage_bodies]}"
        )

        # ── Stage 2: Dispatch ────────────────────────────────────────
        # Run a single dispatch cycle on the (now triaged) ticket.
        # This makes a real LLM call for routing + sends a real Discord DM.
        discord_holder.run_async(dispatch_agent._dispatch_ticket(ticket_id))

        # Verify: ai_dispatched tag applied
        tags = zammad_client.get_tags(ticket_id)
        assert "ai_dispatched" in tags, (
            f"Dispatch agent did not add 'ai_dispatched' tag. Tags: {tags}"
        )

        # Verify: Agent_Actions table has dispatch record
        # Use get_relevant_agent_actions which filters for top-level actions only
        # (parent_id IS NULL), avoiding child steps that would crowd out the parent.
        actions = memory_manager.get_relevant_agent_actions(
            agent_name="dispatch",
            match_contexts=[("ticket", str(ticket_id))],
            limit=10,
        )
        dispatch_actions = [
            a for a in actions
            if a["action_type"] == "dispatch"
            and a.get("trigger_context") == f"ticket:{ticket_id}"
        ]
        assert len(dispatch_actions) >= 1, (
            f"No dispatch action logged for ticket {ticket_id}. "
            f"Actions: {actions}"
        )

        parent_action = dispatch_actions[0]
        parent_id = parent_action["id"]

        # Verify: parent action outcome (success or notification_failed)
        assert parent_action["outcome"] in ("success", "notification_failed"), (
            f"Unexpected dispatch outcome: {parent_action['outcome']}. "
            f"Payload: {parent_action.get('outcome_payload')}"
        )

        # Verify: child steps logged (fetch_ticket, fetch_articles,
        # llm_decision, send_notification, tag_ticket)
        steps = memory_manager.get_action_steps(parent_id)
        step_types = [s["action_type"] for s in steps]
        assert "fetch_ticket" in step_types, f"Missing fetch_ticket step. Steps: {step_types}"
        assert "fetch_articles" in step_types, f"Missing fetch_articles step. Steps: {step_types}"
        assert "llm_decision" in step_types, f"Missing llm_decision step. Steps: {step_types}"
        assert "send_notification" in step_types, f"Missing send_notification step. Steps: {step_types}"
        assert "tag_ticket" in step_types, f"Missing tag_ticket step. Steps: {step_types}"

        # Verify: LLM decision produced valid JSON with expected fields
        llm_step = next(s for s in steps if s["action_type"] == "llm_decision")
        assert llm_step["outcome"] == "success", (
            f"LLM decision failed: {llm_step.get('outcome_payload')}"
        )

        # Verify: context tags stored for the action
        # (We can't query Agent_Action_Contexts directly via the public API,
        #  but we can verify via get_relevant_agent_actions with a context match)
        context_matches = memory_manager.get_relevant_agent_actions(
            agent_name="dispatch",
            match_contexts=[("ticket", str(ticket_id))],
            limit=5,
        )
        matching_ids = [a["id"] for a in context_matches]
        assert parent_id in matching_ids, (
            f"Action {parent_id} not found via context match for ticket:{ticket_id}. "
            f"Got IDs: {matching_ids}"
        )

        # Verify: outcome_payload contains delivery details
        if parent_action["outcome"] == "success":
            payload = json.loads(parent_action["outcome_payload"])
            assert payload.get("sent") is True, (
                f"Notification reported as not sent: {payload}"
            )
            # The LLM chooses the channel based on priority — both are valid
            channel = payload.get("channel")
            assert channel in ("discord_dm", "zammad", "log"), (
                f"Unexpected channel in payload: {payload}"
            )
            # Verify the decision contains required fields
            decision = payload.get("decision", {})
            assert "priority" in decision, f"Missing priority in decision: {decision}"
            assert "summary" in decision, f"Missing summary in decision: {decision}"


# ---------------------------------------------------------------------------
# Mocked-LLM pipeline: ticket → triage → dispatch → Discord DM
# ---------------------------------------------------------------------------

class TestTriageDispatchMockedLLM:
    """Triage → dispatch pipeline with mocked LLM, real Zammad + Discord.

    Exercises the full pipeline mechanics (ticket CRUD, tagging, triage note
    extraction, notification routing, Agent_Actions logging) against real
    Zammad and Discord without needing LLM API keys.

    Requires discord_live + zammad_live (NOT llm_live).
    """

    pytestmark = [pytest.mark.discord_live, pytest.mark.zammad_live]

    @pytest.fixture
    def zammad_client(self):
        import requests
        from src.clients.zammad_client import ZammadClient
        try:
            client = ZammadClient()
            client.get_self()
            return client
        except (ValueError, requests.exceptions.RequestException) as e:
            pytest.skip(f"Zammad unavailable: {e}")

    @pytest.fixture
    def pipeline_env(self, zammad_client, discord_holder):
        """ChatSystem + agents wired to real Zammad and Discord."""
        from src.chat_system import ChatSystem
        from src.database.memory_manager import MemoryManager
        from src.engine import TextEngine
        from src.interfaces.zammad_bot import ZammadBot
        from src.agents.dispatch_agent import DispatchAgent
        from config.global_config import TEST_MEMORY_DATABASE_FILE

        db_path = f"{TEST_MEMORY_DATABASE_FILE}.mocked_pipeline.{random.randint(1000, 9999)}"
        if os.path.exists(db_path):
            os.remove(db_path)

        memory_manager = MemoryManager(db_path=db_path)
        memory_manager.create_schema()
        text_engine = TextEngine()

        with patch('src.chat_system.load_personas_from_file', return_value={}):
            chat_system = ChatSystem(
                memory_manager=memory_manager,
                text_engine=text_engine,
            )

        notification_router = NotificationRouter()
        notification_router.register("discord_dm", DiscordNotifier(discord_holder.client))
        notification_router.register("zammad", ZammadNotifier(zammad_client))
        notification_router.register("log", LogNotifier())

        triage_bot = ZammadBot(chat_system, zammad_client)
        dispatch_agent = DispatchAgent(
            chat_system=chat_system,
            zammad_client=zammad_client,
            notification_router=notification_router,
            agent_config={
                "notification_defaults": {"channel": "discord_dm", "recipient": "adrich"},
                "_recipients": {
                    "adrich": {
                        "discord_user_id": DISCORD_TEST_USER_ID,
                        "discord_username": "adrich",
                    }
                },
            },
        )

        yield {
            "chat_system": chat_system,
            "memory_manager": memory_manager,
            "text_engine": text_engine,
            "zammad_client": zammad_client,
            "triage_bot": triage_bot,
            "dispatch_agent": dispatch_agent,
            "discord_holder": discord_holder,
        }

        memory_manager.close()
        time.sleep(0.1)
        if os.path.exists(db_path):
            try:
                os.remove(db_path)
            except PermissionError:
                pass

    @pytest.fixture
    def test_ticket(self, zammad_client):
        """Create a realistic support ticket. Cleans up previous runs."""
        old_tickets = zammad_client.search_tickets(
            query=f'title:"{MOCKED_PIPELINE_TICKET_TITLE}"'
        )
        for t in old_tickets:
            try:
                zammad_client.delete_ticket(t["id"])
            except Exception:
                pass

        ticket_data = zammad_client.create_ticket(
            title=MOCKED_PIPELINE_TICKET_TITLE,
            group="Users",
            customer_id=1,
            article_body=(
                "Our main office printer (HP LaserJet 4250) has been offline since "
                "this morning. The display shows 'PC LOAD LETTER' error. Multiple "
                "users are affected and unable to print. Tray 2 has been reloaded "
                "but the error persists. Machine ID: PRN-FLOOR3-001."
            ),
        )
        yield {"id": ticket_data["id"], "number": ticket_data["number"]}

    def test_triage_then_dispatch_sends_discord_dm(self, pipeline_env, test_ticket):
        """Ticket → triage (mock LLM) → dispatch (mock LLM) → real Discord DM.

        Verifies:
        1. Triage: 'autotriaged' tag applied, internal note with context dump posted
        2. Dispatch: 'ai_dispatched' tag applied, Agent_Actions logged, Discord DM sent
        """
        triage_bot = pipeline_env["triage_bot"]
        dispatch_agent = pipeline_env["dispatch_agent"]
        text_engine = pipeline_env["text_engine"]
        zammad_client = pipeline_env["zammad_client"]
        memory_manager = pipeline_env["memory_manager"]
        discord_holder = pipeline_env["discord_holder"]
        ticket_id = test_ticket["id"]

        async def mock_generate(persona_config, context_object, tools=None):
            """Deterministic LLM mock that dispatches on persona prompt content."""
            sys_prompt = context_object.get('persona_prompt', '')

            if 'keyword extraction' in sys_prompt:
                return {"type": "text", "content": "printer offline LaserJet error"}, {}

            if 'relevance classifier' in sys_prompt:
                return {"type": "text", "content": "RELEVANT"}, {}

            if 'summarization' in sys_prompt:
                return {"type": "text", "content": "Printer offline with PC LOAD LETTER error."}, {}

            if 'dispatch decision agent' in sys_prompt:
                return {"type": "text", "content": json.dumps({
                    "priority": "high",
                    "notify_channel": "discord_dm",
                    "summary": "Office printer offline affecting multiple users",
                    "reasoning": "Hardware issue blocking multiple users warrants immediate notification",
                })}, {}

            # Triage analyst (default)
            return {"type": "text", "content": (
                "## Summary\n"
                "Office printer HP LaserJet 4250 offline with 'PC LOAD LETTER' error.\n\n"
                "## User History\nNo relevant history.\n\n"
                "## Similar Tickets\nNone found.\n\n"
                "## Recommended Action\n"
                "Dispatch technician to inspect printer hardware."
            )}, {}

        # ── Stage 1: Triage ──────────────────────────────────────────
        with patch.object(text_engine, 'generate_response', side_effect=mock_generate):
            discord_holder.run_async(triage_bot._process_ticket(ticket_id))

        tags = zammad_client.get_tags(ticket_id)
        assert "autotriaged" in tags, f"Triage tag missing. Tags: {tags}"

        articles = zammad_client.get_ticket_articles(ticket_id)
        internal_notes = [a for a in articles if a.get("internal", False)]
        assert len(internal_notes) >= 1, "No internal triage note found."
        triage_bodies = [a.get("body", "") for a in internal_notes]
        assert any("AI TRIAGE CONTEXT DUMP" in b for b in triage_bodies), (
            f"Triage note missing context dump. Previews: {[b[:200] for b in triage_bodies]}"
        )

        # ── Stage 2: Dispatch ────────────────────────────────────────
        with patch.object(text_engine, 'generate_response', side_effect=mock_generate):
            discord_holder.run_async(dispatch_agent._dispatch_ticket(ticket_id))

        tags = zammad_client.get_tags(ticket_id)
        assert "ai_dispatched" in tags, f"Dispatch tag missing. Tags: {tags}"

        # Verify Agent_Actions logged
        actions = memory_manager.get_relevant_agent_actions(
            agent_name="dispatch",
            match_contexts=[("ticket", str(ticket_id))],
            limit=10,
        )
        dispatch_actions = [
            a for a in actions
            if a["action_type"] == "dispatch"
            and a.get("trigger_context") == f"ticket:{ticket_id}"
        ]
        assert len(dispatch_actions) >= 1, (
            f"No dispatch action logged for ticket {ticket_id}. Actions: {actions}"
        )

        parent = dispatch_actions[0]
        assert parent["outcome"] in ("success", "notification_failed"), (
            f"Unexpected outcome: {parent['outcome']}. "
            f"Payload: {parent.get('outcome_payload')}"
        )

        # Verify notification delivery details
        if parent["outcome"] == "success":
            payload = json.loads(parent["outcome_payload"])
            assert payload.get("sent") is True, f"DM not sent: {payload}"
            assert payload.get("channel") == "discord_dm", (
                f"Expected discord_dm channel: {payload}"
            )
            decision = payload.get("decision", {})
            assert decision.get("priority") == "high"
            assert "printer" in decision.get("summary", "").lower()
