# tests/integration/test_block_retain.py
#
# DP-495: history leaves the prompt in blocks, each retained whole. Driven
# through the real kernel and a real MemoryManager; only the LLM and the
# backend's `retain_block` / `retain_status` are faked.
#
# - The prompt start stays fixed while turns are added; a new user message
#   that takes the prompt past 90 % of the budget cuts the oldest history as
#   one block sized to ~50 %, ending on a user turn.
# - A cut block stays in the prompt until Hindsight reports its extraction
#   complete; past 100 % the oldest messages are dropped for that request.
# - 48 h idle trims in the background (pre-silence rows only, to ~50 %); 72 h
#   idle retains the rest as a final block that stays in the prompt and is
#   never sent again.

import re
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List

import pytest

from config.global_config import RETAIN_IDLE_FINAL_SECONDS, RETAIN_IDLE_TRIM_SECONDS
from src.memory.context_budget import estimate_tokens
from src.memory.retain_blocks import CONTEXT_PREAMBLE
from src.persona import MemoryMode

pytestmark = pytest.mark.integration

WORD = "x" * 96           # ~25 tokens a message, ~50 a turn
BUDGET = 1900             # max_context_tokens 2000 - tokens 100: ~34 turns to 90 %
_STAMP = re.compile(r"^\[\d{4}-\d{2}-\d{2} \d{2}:\d{2}\] ")


class FakeHindsight:
    """Records block sends; extraction completes only when told to."""

    def __init__(self) -> None:
        self.sends: List[Dict[str, Any]] = []
        self.accept = True
        self.status = "pending"

    async def retain_block(self, bank_id: str, items: List[Dict[str, Any]], *,
                           request_id: str) -> Any:
        if not self.accept:
            return None
        self.sends.append({"bank": bank_id, "items": items, "request_id": request_id})
        return [f"op-{len(self.sends)}"]

    async def retain_status(self, bank_id: str, op_ids: List[str], doc_ids: List[str]) -> str:
        return self.status

    def turns(self, n: int = -1) -> List[str]:
        """Block `n`'s turns, user time headers (DP-402) stripped."""
        return [_STAMP.sub("", t) for it in self.sends[n]["items"]
                for t in it["content"].split("\n\n")]


@pytest.fixture
def setup(mocked_chat_system):
    chat_system, memory_manager = mocked_chat_system
    persona = chat_system.personas["test_persona"]
    persona.set_history_messages(1000)
    persona.set_inject_timestamp(False)
    persona.set_enabled_tools([])
    persona.set_max_context_tokens(2000)
    persona.set_response_token_limit(100)
    hs = FakeHindsight()
    chat_system.memory_backend.retain_block = hs.retain_block  # type: ignore[assignment]
    chat_system.memory_backend.retain_status = hs.retain_status  # type: ignore[assignment]
    return chat_system, memory_manager, persona, hs


async def _turn(chat_system, message, *, user="u1", channel="chan"):
    chat_system.text_engine.generate_response.return_value = (
        {"type": "text", "content": f"re: {message}"}, {},
    )
    async for _ in chat_system.stream_response("test_persona", user, channel, message):
        pass
    await chat_system.turn_persistence.drain_flushes()
    return chat_system.text_engine.generate_response.call_args.args[1]["history"]


async def _until_cut(chat_system, hs, start=0, limit=60):
    """Turns `start`, `start+1`, ... until a new block is sent; returns the
    prompts of every turn, the last being the one that cut."""
    sent, prompts = len(hs.sends), []
    for i in range(start, start + limit):
        prompts.append(await _turn(chat_system, f"{i} {WORD}"))
        if len(hs.sends) > sent:
            return prompts
    raise AssertionError("no block was cut")


def _texts(history):
    return [m["content"] for m in history]


def _tokens(history):
    return sum(estimate_tokens(m["content"]) for m in history)


@pytest.mark.asyncio
async def test_prefix_is_stable_until_the_90_percent_cut(setup):
    """Until the watermark the prompt only grows at its end; the cut is one
    block, ends on a whole turn, and is sized so the rest fits 50 %."""
    chat_system, mm, persona, hs = setup
    prompts = await _until_cut(chat_system, hs)
    for earlier, later in zip(prompts, prompts[1:]):
        assert _texts(later)[:len(earlier) - 1] == _texts(earlier)[:-1]
    # The cutting turn's prompt crossed 90 %; the one before did not.
    assert _tokens(prompts[-1]) > 0.9 * BUDGET >= _tokens(prompts[-2])
    sent = hs.turns()
    assert sent[0] == f"u1: 0 {WORD}"
    assert sent[-1].startswith("test_persona: re:")         # ends on a whole turn
    assert len(sent) > 2                                     # a block, not a turn
    # What is left — up to the cutting turn's own message — fits half the budget.
    scope = chat_system.request_builder.load_history_window(
        persona, "u1", "chan", None, None, advance=False).scope
    start, _ = mm.retain_window(scope)
    left = mm.get_channel_history("chan", "test_persona", None, 1000, min_id=start)
    assert left[0]["author_role"] == "user"
    assert sum(estimate_tokens(r["content"]) for r in left[:-1]) <= BUDGET // 2


@pytest.mark.asyncio
async def test_cut_block_stays_in_the_prompt_until_extraction_completes(setup):
    chat_system, _, _, hs = setup
    prompts = await _until_cut(chat_system, hs)
    n = len(prompts)

    hs.status = "pending"
    history = await _turn(chat_system, f"{n} {WORD}")
    assert _texts(history)[0] == f"0 {WORD}"                # still shown

    hs.status = "completed"
    await chat_system.turn_persistence.poll_blocks()
    history = await _turn(chat_system, f"{n + 1} {WORD}")
    assert f"0 {WORD}" not in _texts(history)
    # The prompt now opens on the turn right after the block.
    last_cut = int(hs.turns()[-1].removeprefix("test_persona: re: ").split()[0])
    assert _texts(history)[0] == f"{last_cut + 1} {WORD}"


@pytest.mark.asyncio
async def test_slow_hindsight_does_not_cut_a_block_per_turn(setup):
    """While a block is still in flight, new turns do not cut more blocks
    until the live window itself crosses 90 % again."""
    chat_system, _, _, hs = setup
    prompts = await _until_cut(chat_system, hs)
    for i in range(len(prompts), len(prompts) + 8):
        await _turn(chat_system, f"{i} {WORD}")
    assert len(hs.sends) == 1


@pytest.mark.asyncio
async def test_full_budget_falls_back_to_dropping_oldest_while_block_in_flight(setup):
    """Hindsight down: the block stays queued and shown; once the prompt
    passes 100 % the oldest messages are dropped for that request — and stay
    queued, so nothing is lost."""
    chat_system, mm, _, hs = setup
    hs.accept = False
    for i in range(45):
        history = await _turn(chat_system, f"{i} {WORD}")
    assert hs.sends == []
    assert mm.retain_blocks(statuses=("queued",))
    assert _tokens(history) <= BUDGET
    assert f"0 {WORD}" not in _texts(history)

    hs.accept = True
    await chat_system.turn_persistence.flush_blocks()
    assert hs.turns(0)[0] == f"u1: 0 {WORD}"


@pytest.mark.asyncio
async def test_blocks_are_numbered_per_scope_with_first_message_time(setup):
    chat_system, _, _, hs = setup
    prompts = await _until_cut(chat_system, hs)
    hs.status = "completed"
    await chat_system.turn_persistence.poll_blocks()
    await _until_cut(chat_system, hs, start=len(prompts))
    docs = [s["items"][0]["document_id"] for s in hs.sends]
    assert len(docs) >= 2
    for n, doc in enumerate(docs, start=1):
        persona, scope, block, stamp = doc.split(":", 3)
        assert (persona, scope, block) == ("test_persona", "chan", f"b{n:04d}")
        datetime.strptime(stamp, "%Y-%m-%dT%H:%MZ")
    # Every block after the first carries the turns before it as background.
    ctx = hs.sends[1]["items"][0]["context"]
    assert ctx.startswith(CONTEXT_PREAMBLE)


@pytest.mark.asyncio
async def test_48h_idle_trims_to_half_without_a_new_message(setup):
    chat_system, mm, _, hs = setup
    for i in range(24):
        await _turn(chat_system, f"{i} {WORD}")              # ≈ 1200 tokens: over 50 %, under 90 %
    assert hs.sends == []
    later = datetime.now(timezone.utc) + timedelta(seconds=RETAIN_IDLE_TRIM_SECONDS + 60)
    await chat_system.retain_maintenance(now=later)
    assert len(hs.sends) == 1
    assert hs.turns()[0] == f"u1: 0 {WORD}"
    # Idempotent: what is left is under 50 %, so a second sweep cuts nothing.
    await chat_system.retain_maintenance(now=later)
    assert len(hs.sends) == 1


@pytest.mark.asyncio
async def test_48h_idle_leaves_a_short_conversation_alone(setup):
    chat_system, _, _, hs = setup
    await _turn(chat_system, "short")
    later = datetime.now(timezone.utc) + timedelta(seconds=RETAIN_IDLE_TRIM_SECONDS + 60)
    await chat_system.retain_maintenance(now=later)
    assert hs.sends == []


@pytest.mark.asyncio
async def test_72h_idle_retains_all_keeps_it_shown_and_never_resends(setup):
    chat_system, _, _, hs = setup
    await _turn(chat_system, "short one")
    later = datetime.now(timezone.utc) + timedelta(seconds=RETAIN_IDLE_FINAL_SECONDS + 60)
    await chat_system.retain_maintenance(now=later)
    assert hs.turns() == ["u1: short one", "test_persona: re: short one"]
    hs.status = "completed"
    await chat_system.turn_persistence.poll_blocks()

    # Picked up again: the retained turns are still in the prompt.
    history = await _turn(chat_system, f"back {WORD}")
    assert "short one" in _texts(history)[0]
    # A later cut over them sends only what was not sent before.
    await _until_cut(chat_system, hs)
    sent_after = [t for s in hs.sends[1:] for it in s["items"] for t in it["content"].split("\n\n")]
    assert sent_after and not any("short one" in t for t in sent_after)


@pytest.mark.asyncio
async def test_edit_and_delete_inside_the_prompt_are_honoured(setup):
    chat_system, mm, _, hs = setup
    await _turn(chat_system, f"draft {WORD}")
    rows = mm.get_channel_history("chan", "test_persona", None, 10)
    mm.update_interaction_content(rows[0]["interaction_id"], f"final {WORD}")
    mm.suppress_interaction(rows[1]["interaction_id"])
    await _until_cut(chat_system, hs)
    sent = hs.turns(0)
    assert sent[0] == f"u1: final {WORD}"
    assert not any("re: draft" in t for t in sent)


@pytest.mark.asyncio
async def test_retry_of_a_reply_in_a_sent_block_rebuilds_that_block(setup):
    """DP-409 per block: the regenerated reply replaces the block document
    (same id, new request id), not appended beside the discarded attempt."""
    chat_system, mm, _, hs = setup
    await _turn(chat_system, "only")
    later = datetime.now(timezone.utc) + timedelta(seconds=RETAIN_IDLE_FINAL_SECONDS + 60)
    await chat_system.retain_maintenance(now=later)
    first = hs.sends[-1]

    chat_system.text_engine.generate_response.return_value = (
        {"type": "text", "content": "better answer"}, {},
    )
    async for _ in chat_system.stream_response("test_persona", "u1", "chan", "", is_retry=True):
        pass
    await chat_system.turn_persistence.drain_flushes()
    rebuilt = hs.sends[-1]
    assert rebuilt is not first
    assert rebuilt["items"][0]["document_id"] == first["items"][0]["document_id"]
    assert rebuilt["request_id"] != first["request_id"]
    assert hs.turns() == ["u1: only", "test_persona: better answer"]


@pytest.mark.asyncio
async def test_global_mode_scope_spans_channels(setup):
    chat_system, _, persona, hs = setup
    persona.set_memory_mode(MemoryMode.GLOBAL)
    for i in range(40):
        await _turn(chat_system, f"{i} {WORD}", channel="a" if i % 2 else "b")
        if hs.sends:
            break
    assert hs.sends[0]["items"][0]["document_id"].split(":")[1] == "global"
    tags = hs.sends[0]["items"][0]["tags"]
    assert "channel:a" in tags and "channel:b" in tags


@pytest.mark.asyncio
async def test_ticket_mode_sends_each_turn_on_the_next(setup):
    chat_system, _, persona, hs = setup
    persona.set_memory_mode(MemoryMode.TICKET_ISOLATED)
    await _turn(chat_system, "first")
    assert hs.sends == []
    await _turn(chat_system, "second")
    assert hs.turns() == ["u1: first", "test_persona: re: first"]


@pytest.mark.asyncio
async def test_a_binding_count_cap_trims_in_blocks(setup):
    """`context 4`: when the cap starts hiding unsent turns the window is cut
    to half the cap — in one block, not a turn at a time."""
    chat_system, _, persona, hs = setup
    persona.set_history_messages(4)
    for i in range(3):
        await _turn(chat_system, f"t{i}")
    assert hs.sends == []                    # 4 rows shown, none hidden yet
    await _turn(chat_system, "t3")           # 6 rows: the cap hides t0's pair
    assert hs.turns() == ["u1: t0", "test_persona: re: t0", "u1: t1", "test_persona: re: t1"]


@pytest.mark.asyncio
async def test_maintenance_loop_runs_a_boot_pass_and_stops(setup):
    """The loop's first pass runs at once (the boot pass): a block a restart
    left queued goes out without any user turn. It exits on stop."""
    import asyncio

    chat_system, _, _, hs = setup
    hs.accept = False
    for i in range(36):
        await _turn(chat_system, f"{i} {WORD}")
    assert hs.sends == [] and chat_system.request_builder.memory_manager.retain_blocks()
    hs.accept = True

    task = asyncio.create_task(chat_system.run_retain_maintenance())
    for _ in range(50):
        await asyncio.sleep(0.01)
        if hs.sends:
            break
    chat_system.stop_retain_maintenance()
    await asyncio.wait_for(task, timeout=2)
    assert len(hs.sends) == 1


@pytest.mark.asyncio
async def test_sweep_acts_once_per_idle_mark(setup, monkeypatch):
    """After the boot pass, a sweep only replans scopes whose 48 h / 72 h
    mark fell since the previous sweep: a long-idle channel is not reloaded
    every 15 minutes."""
    chat_system, _, _, hs = setup
    for i in range(24):
        await _turn(chat_system, f"{i} {WORD}")
    calls = []
    real = chat_system.request_builder.idle_trim
    monkeypatch.setattr(chat_system.request_builder, "idle_trim",
                        lambda p, sc: calls.append(sc.key) or real(p, sc))
    t48 = datetime.now(timezone.utc) + timedelta(seconds=RETAIN_IDLE_TRIM_SECONDS + 60)
    await chat_system.idle_retain_sweep(t48, since=t48 - timedelta(minutes=15))
    assert calls == ["chan"]
    later = t48 + timedelta(minutes=15)
    await chat_system.idle_retain_sweep(later, since=t48)
    assert calls == ["chan"]                      # mark already handled
    await chat_system.idle_retain_sweep(later, since=None)
    assert calls == ["chan", "chan"]              # the boot pass checks everything
