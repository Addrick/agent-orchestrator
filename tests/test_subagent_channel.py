# tests/test_subagent_channel.py
"""Unit tests for the DP-230 direct subagent ↔ Discord channel.

Covers: the registry thread map, the transcript sink (thread create + progress
coalescing + event summaries), the wake split (question → human-in-thread with no
fixr LLM turn; done/error → fixr), the idle fallback, the inbound routing
shortcut, and the hidden agent-face system persona."""

import asyncio

from unittest.mock import AsyncMock, MagicMock

from config import global_config
from src.persona import Persona
from src.personas.store import load_system_personas_from_file
from src.self_edit.events import AgentEvent, DONE, PROGRESS, QUESTION, STARTED
from src.interfaces.discord_bot import route_agent_thread_message
from src.self_edit.integration import FixrIntegration
from src.self_edit.registry import AgentRecord, AgentRegistry
from tests.helpers import make_chat_system


# -- fakes -------------------------------------------------------------------

class FakeDiscord:
    """Implements the DiscordThreadClient protocol slice (DP-230)."""

    def __init__(self) -> None:
        self.threads_created: list = []
        self.posts: list = []
        self._next = 9001

    async def create_agent_thread(self, parent_channel_id: int, name: str):
        self.threads_created.append((parent_channel_id, name))
        tid = self._next
        self._next += 1
        return tid

    async def send_to_channel(self, channel_id: int, content: str) -> bool:
        self.posts.append((channel_id, content))
        return True


def _record(agent_id="DP-9-1", bug_id="DP-9", **kw):
    return AgentRecord(
        agent_id=agent_id, bug_id=bug_id, description="bug",
        worktree="/w", branch="bugfix/DP-9-fix",
        raw_log="/r", events_log="/e", **kw,
    )


def _ev(agent_id, type_, **payload):
    return AgentEvent(agent_id=agent_id, seq=0, type=type_, payload=payload)


def _make_integration(registry=None):
    chat = MagicMock()
    chat.generate_response = AsyncMock()
    notifier = MagicMock()
    integ = FixrIntegration(chat, notifier, registry=registry or AgentRegistry())
    return integ, chat


def _activate(integ, monkeypatch, *, idle_minutes=10.0, debounce=0.0):
    discord = FakeDiscord()
    integ.attach_discord(discord)
    monkeypatch.setattr(global_config, "CC_FIXR_AGENTS_CHANNEL_ID", "555")
    monkeypatch.setattr(global_config, "CC_FIXR_IDLE_MINUTES", idle_minutes)
    monkeypatch.setattr(global_config, "CC_FIXR_PROGRESS_DEBOUNCE_SECONDS", debounce)
    return discord


# -- registry thread map -----------------------------------------------------

async def test_discord_thread_id_round_trips_registry():
    registry = AgentRegistry()
    rec = _record()
    await registry.add(rec)
    assert await registry.get_by_thread("777") is None
    await registry.update(rec.agent_id, discord_thread_id="777")
    got = await registry.get_by_thread("777")
    assert got is not None and got.agent_id == rec.agent_id
    # round-trips through to_dict (registry persistence shape).
    assert (await registry.get(rec.agent_id)).to_dict()["discord_thread_id"] == "777"


# -- transcript sink ---------------------------------------------------------

async def test_on_event_creates_thread_on_first_event_and_posts(monkeypatch):
    integ, _ = _make_integration()
    discord = _activate(integ, monkeypatch)
    rec = _record()
    await integ.registry.add(rec)

    await integ._on_event(rec, _ev(rec.agent_id, STARTED))

    assert len(discord.threads_created) == 1
    assert discord.threads_created[0][0] == 555
    assert rec.discord_thread_id == "9001"
    # the started line was posted to the new thread
    assert discord.posts and discord.posts[0][0] == 9001
    # thread id persisted so inbound can resolve it
    assert (await integ.registry.get_by_thread("9001")) is not None


async def test_progress_coalesced_before_next_event(monkeypatch):
    integ, _ = _make_integration()
    discord = _activate(integ, monkeypatch)
    rec = _record()
    await integ.registry.add(rec)

    await integ._on_event(rec, _ev(rec.agent_id, STARTED))
    await integ._on_event(rec, _ev(rec.agent_id, PROGRESS, text="reading files"))
    await integ._on_event(rec, _ev(rec.agent_id, PROGRESS, text="editing"))
    # a non-progress event flushes buffered progress first, then posts itself
    await integ._on_event(rec, _ev(rec.agent_id, DONE, summary="fixed", pr_url="https://github.com/x/y/pull/3"))

    bodies = [c for _, c in discord.posts]
    coalesced = [b for b in bodies if "reading files" in b and "editing" in b]
    assert coalesced, f"progress not coalesced into one post: {bodies}"
    assert any(b.startswith("✅") and "pull/3" in b for b in bodies)


async def test_direct_channel_off_skips_thread(monkeypatch):
    integ, _ = _make_integration()
    # No discord attached / no channel id → feature off.
    monkeypatch.setattr(global_config, "CC_FIXR_AGENTS_CHANNEL_ID", "")
    rec = _record()
    await integ.registry.add(rec)
    await integ._on_event(rec, _ev(rec.agent_id, STARTED))
    assert rec.discord_thread_id is None


# -- wake split --------------------------------------------------------------

async def test_question_does_not_wake_fixr_when_direct_active(monkeypatch):
    integ, chat = _make_integration()
    _activate(integ, monkeypatch)
    rec = _record()
    await integ.registry.add(rec)

    await integ._on_wake(rec, _ev(rec.agent_id, QUESTION, text="which fix?"))

    chat.generate_response.assert_not_called()
    # an idle fallback timer was armed
    assert rec.agent_id in integ._idle_timers
    integ._cancel_idle(rec.agent_id)


async def test_done_wakes_fixr(monkeypatch):
    integ, chat = _make_integration()
    _activate(integ, monkeypatch)
    rec = _record()
    await integ.registry.add(rec)

    await integ._on_wake(rec, _ev(rec.agent_id, DONE, summary="done"))

    chat.generate_response.assert_awaited_once()


async def test_question_wakes_fixr_when_direct_off():
    integ, chat = _make_integration()
    # direct channel OFF (no attach, default empty channel id) → DP-227 behavior
    rec = _record()
    await integ.registry.add(rec)
    await integ._on_wake(rec, _ev(rec.agent_id, QUESTION, text="?"))
    chat.generate_response.assert_awaited_once()


# -- idle fallback -----------------------------------------------------------

async def test_idle_timer_fires_fixr_once_then_cancellable(monkeypatch):
    integ, chat = _make_integration()
    # ~0.03s idle window
    _activate(integ, monkeypatch, idle_minutes=0.0005)
    rec = _record()
    await integ.registry.add(rec)

    await integ._on_wake(rec, _ev(rec.agent_id, QUESTION, text="?"))
    await asyncio.sleep(0.12)
    chat.generate_response.assert_awaited_once()


async def test_idle_timer_cancelled_by_later_event(monkeypatch):
    integ, chat = _make_integration()
    _activate(integ, monkeypatch, idle_minutes=0.0005)
    rec = _record()
    await integ.registry.add(rec)

    await integ._on_wake(rec, _ev(rec.agent_id, QUESTION, text="?"))
    # agent resumes (human answered) → a new event cancels the idle fallback
    await integ._on_event(rec, _ev(rec.agent_id, PROGRESS, text="resuming"))
    await asyncio.sleep(0.12)
    chat.generate_response.assert_not_called()


# -- inbound routing ---------------------------------------------------------

async def test_inbound_thread_message_answers_agent_no_llm():
    registry = AgentRegistry()
    rec = _record(discord_thread_id="4242", session_id="sX")
    await registry.add(rec)
    dispatcher = MagicMock()
    dispatcher.answer_agent = AsyncMock()
    fixr = MagicMock(registry=registry, dispatcher=dispatcher)
    chat = MagicMock()
    chat.get_service = MagicMock(return_value=fixr)

    handled = await route_agent_thread_message(chat, "4242", "use option B")

    assert handled is True
    dispatcher.answer_agent.assert_awaited_once_with(rec.agent_id, "use option B")


async def test_inbound_unknown_thread_not_handled():
    registry = AgentRegistry()
    fixr = MagicMock(registry=registry, dispatcher=MagicMock())
    chat = MagicMock()
    chat.get_service = MagicMock(return_value=fixr)
    handled = await route_agent_thread_message(chat, "nope", "hi")
    assert handled is False


async def test_inbound_note_to_self_swallowed():
    registry = AgentRegistry()
    rec = _record(discord_thread_id="4242", session_id="sX")
    await registry.add(rec)
    dispatcher = MagicMock()
    dispatcher.answer_agent = AsyncMock()
    fixr = MagicMock(registry=registry, dispatcher=dispatcher)
    chat = MagicMock()
    chat.get_service = MagicMock(return_value=fixr)

    handled = await route_agent_thread_message(chat, "4242", "// just a note")

    assert handled is True  # handled (consumed) but not forwarded
    dispatcher.answer_agent.assert_not_called()


async def test_inbound_no_fixr_service_not_handled():
    chat = MagicMock()
    chat.get_service = MagicMock(return_value=None)
    assert await route_agent_thread_message(chat, "1", "hi") is False


def _chat_with_agent(thread_id="4242", *, answer_side_effect=None):
    """Wire a fake chat_system whose fixr service owns one threaded agent."""
    registry = AgentRegistry()
    rec = _record(discord_thread_id=thread_id, session_id="sX")
    dispatcher = MagicMock()
    dispatcher.answer_agent = AsyncMock(side_effect=answer_side_effect)
    fixr = MagicMock(registry=registry, dispatcher=dispatcher)
    chat = MagicMock()
    chat.get_service = MagicMock(return_value=fixr)
    return chat, registry, rec, dispatcher


async def test_inbound_acks_human_on_successful_resume():
    """A — the human gets a ✅ ack in the thread when their answer resumes the agent."""
    chat, registry, rec, dispatcher = _chat_with_agent()
    await registry.add(rec)
    channel = MagicMock()
    channel.send = AsyncMock()

    await route_agent_thread_message(chat, "4242", "do X", channel=channel)

    dispatcher.answer_agent.assert_awaited_once_with(rec.agent_id, "do X")
    channel.send.assert_awaited_once()
    assert "✅" in channel.send.await_args.args[0]


async def test_inbound_notifies_human_on_dispatcher_error():
    """A — a DispatcherError (e.g. agent not waiting) is surfaced into the thread."""
    from src.self_edit.dispatcher import DispatcherError
    chat, registry, rec, dispatcher = _chat_with_agent(
        answer_side_effect=DispatcherError("Agent a1 is not waiting (status=running)")
    )
    await registry.add(rec)
    channel = MagicMock()
    channel.send = AsyncMock()

    await route_agent_thread_message(chat, "4242", "do X", channel=channel)

    channel.send.assert_awaited_once()
    posted = channel.send.await_args.args[0]
    assert "⚠️" in posted and "not waiting" in posted


async def test_inbound_note_reacts_when_message_supplied():
    """D — a // note-to-self gets a 📝 reaction and is not forwarded."""
    chat, registry, rec, dispatcher = _chat_with_agent()
    await registry.add(rec)
    message = MagicMock()
    message.add_reaction = AsyncMock()

    handled = await route_agent_thread_message(
        chat, "4242", "// note", message=message
    )

    assert handled is True
    dispatcher.answer_agent.assert_not_called()
    message.add_reaction.assert_awaited_once_with("📝")


# -- hidden agent-face persona ----------------------------------------------

def test_fixr_agent_persona_is_a_system_persona():
    personas = load_system_personas_from_file()
    assert "fixr-agent" in personas


def test_attach_discord_late_bind_and_service_lookup():
    """Mirrors main.py wiring: resolve fixr via get_service, then attach_discord."""
    chat = make_chat_system()
    integ = FixrIntegration(chat, MagicMock())
    chat.register_service(integ)
    assert chat.get_service("fixr") is integ
    discord = FakeDiscord()
    chat.get_service("fixr").attach_discord(discord)
    assert integ._discord is discord


def test_system_persona_excluded_from_visible_personas():
    personas = {
        "fixr-agent": Persona(persona_name="fixr-agent", model_name="local", prompt=""),
        "testr": Persona(persona_name="testr", model_name="gemini-2.5-flash", prompt=""),
    }
    chat = make_chat_system(personas=personas, system_persona_names={"fixr-agent"})
    visible = chat.visible_personas()
    assert "testr" in visible
    assert "fixr-agent" not in visible
