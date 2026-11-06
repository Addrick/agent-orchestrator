# tests/integration/test_memory_rebuild.py
#
# DP-506: purge a persona's Hindsight bank and send its chat history back,
# through the DP-495 block path. Real kernel + real MemoryManager; only the
# LLM and the Hindsight calls are faked.
#
# - A finished block is sent again under its own document ids and a fresh
#   request id, and stays done: the prompt does not fall back to the start of
#   history.
# - Turns from before DP-495 (no Pending_Retain row) are adopted into blocks
#   below the window, as untrusted; those still in the window are queued for
#   the next cut, not sent.
# - A block that cannot be sent is re-queued, so it is in the prompt and on
#   the flush rather than nowhere.

from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List
from unittest.mock import AsyncMock

import pytest

from src.memory.backend.hindsight import HindsightBackend
from src.memory.retain_blocks import retain_scope
from src.persona import MemoryMode
from src.tools.definitions import get_all_tool_definitions

WORD = "x" * 96


class FakeHindsight(HindsightBackend):
    """A HindsightBackend (the rebuild checks for one) with no I/O."""

    def __init__(self) -> None:  # noqa: D107 — skip the real client setup
        self.sends: List[Dict[str, Any]] = []
        self.deleted_banks: List[str] = []
        self.accept = True
        self.status = "pending"

    async def delete_bank(self, bank_id: str) -> None:
        self.deleted_banks.append(bank_id)

    async def retain_block(self, bank_id: str, items: List[Dict[str, Any]], *,
                           request_id: str) -> Any:
        if not self.accept:
            return None
        self.sends.append({"bank": bank_id, "items": items, "request_id": request_id})
        return [f"op-{len(self.sends)}"]

    async def retain_status(self, bank_id: str, op_ids: List[str], doc_ids: List[str]) -> str:
        return self.status

    async def delete_document(self, bank_id: str, document_id: str) -> None:
        pass


@pytest.fixture
def setup(mocked_chat_system):
    chat_system, mm = mocked_chat_system
    persona = chat_system.personas["test_persona"]
    persona.set_history_messages(1000)
    persona.set_inject_timestamp(False)
    persona.set_enabled_tools([], get_all_tool_definitions())
    persona.set_max_context_tokens(2000)
    persona.set_response_token_limit(100)
    persona.set_long_term_memory(True)
    hs = FakeHindsight()
    chat_system.memory_backend = hs
    chat_system.turn_persistence.memory_backend = hs
    chat_system.provision_persona_memory = AsyncMock()  # type: ignore[method-assign]
    return chat_system, mm, persona, hs


async def _turn(chat_system, message, *, user="u1", channel="chan"):
    chat_system.text_engine.generate_response.return_value = (
        {"type": "text", "content": f"re: {message}"}, {},
    )
    async for _ in chat_system.stream_response("test_persona", user, channel, message):
        pass
    await chat_system.turn_persistence.drain_flushes()


async def _cut_one_block(chat_system, hs):
    for i in range(60):
        await _turn(chat_system, f"{i} {WORD}")
        if hs.sends:
            return
    raise AssertionError("no block was cut")


def _scope(persona):
    return retain_scope(MemoryMode.CHANNEL_ISOLATED, persona_name="test_persona",
                        channel="chan", user_identifier="u1", server_id=None,
                        legacy=15)


@pytest.mark.asyncio
async def test_done_block_is_resent_and_stays_out_of_the_prompt(setup):
    chat_system, mm, persona, hs = setup
    await _cut_one_block(chat_system, hs)
    hs.status = "completed"
    await chat_system.turn_persistence.poll_blocks()
    [block] = mm.retain_blocks(statuses=("done",), persona_name="test_persona")
    before = mm.retain_window(_scope(persona))
    first = hs.sends[-1]

    counts = await chat_system.rebuild_persona_memory("test_persona")

    assert hs.deleted_banks == ["test_persona"]
    chat_system.provision_persona_memory.assert_awaited_once_with("test_persona")
    resent = hs.sends[-1]
    assert [i["document_id"] for i in resent["items"]] == \
        [i["document_id"] for i in first["items"]]
    assert [i["content"] for i in resent["items"]] == [i["content"] for i in first["items"]]
    assert resent["request_id"] != first["request_id"]
    assert counts["blocks"] == 1 and counts["total"] == 1
    # Still done, and the prompt floor did not move back to the block.
    assert mm.retain_blocks(statuses=("done",), block_id=block["block_id"])
    assert mm.retain_window(_scope(persona)) == before


@pytest.mark.asyncio
async def test_pre_block_history_is_adopted_below_the_window_and_queued_inside(setup):
    chat_system, mm, persona, hs = setup
    # 40 turns from before DP-423/DP-495: rows with no Pending_Retain entry.
    t0 = datetime.now(timezone.utc) - timedelta(days=30)
    ids = []
    for i in range(40):
        role, name = ("user", "Adam") if i % 2 == 0 else ("assistant", "test_persona")
        ids.append(mm.log_message("u1", "test_persona", "chan", role, name,
                                  f"old {i}", t0 + timedelta(minutes=i)))
    counts = await chat_system.rebuild_persona_memory("test_persona")

    start, floor = mm.retain_window(_scope(persona))
    assert start == floor == ids[-15]                # DP-423's 15-row window
    sent = [t for s in hs.sends for it in s["items"] for t in it["content"].split("\n\n")]
    assert len(sent) == 25                            # everything below the window
    assert sent[0].endswith("Adam: old 0") and sent[-1].endswith("Adam: old 24")
    assert "test_persona: old 23" in sent
    assert all(it["untrusted"] for s in hs.sends for it in s["items"])
    assert counts["adopted"] == 1 and counts["pending"] == 15
    # The window's rows wait for the next cut, as live turns do.
    assert all(mm.is_retain_pending(i) for i in ids[-15:])
    assert not any(mm.is_retain_pending(i) for i in ids[:25])

    # A second rebuild sends the same documents again; nothing is adopted twice.
    first_ids = [it["document_id"] for s in hs.sends for it in s["items"]]
    hs.sends.clear()
    again = await chat_system.rebuild_persona_memory("test_persona")
    assert again["adopted"] == 0
    assert [it["document_id"] for s in hs.sends for it in s["items"]] == first_ids


@pytest.mark.asyncio
async def test_unsendable_block_is_requeued_and_the_run_stops(setup):
    chat_system, mm, persona, hs = setup
    await _cut_one_block(chat_system, hs)
    hs.status = "completed"
    await chat_system.turn_persistence.poll_blocks()
    [block] = mm.retain_blocks(statuses=("done",), persona_name="test_persona")

    hs.accept = False
    with pytest.raises(RuntimeError, match="stopped answering"):
        await chat_system.rebuild_persona_memory("test_persona")
    # Back in the queue (and so in the prompt) for the regular flush.
    assert mm.retain_blocks(statuses=("queued",), block_id=block["block_id"])


@pytest.mark.asyncio
async def test_rebuild_refuses_a_persona_without_long_term_memory(setup):
    chat_system, _, persona, hs = setup
    persona.set_long_term_memory(False)
    with pytest.raises(RuntimeError, match="no long-term memory"):
        await chat_system.rebuild_persona_memory("test_persona")
    assert hs.deleted_banks == []
    assert "test_persona" not in chat_system.memory_persona_names()


@pytest.mark.asyncio
async def test_in_window_dp423_row_keeps_its_taint_and_retain_text(setup):
    chat_system, mm, persona, hs = setup
    t0 = datetime.now(timezone.utc) - timedelta(days=30)
    row = mm.log_message("u1", "test_persona", "chan", "assistant", "test_persona",
                         "reply\n\n[tools: x]", t0)
    mm.queue_retain(row, "test_persona", "chan", untrusted=False,
                    retain_text="reply", source_content="reply\n\n[tools: x]")
    # DP-423 sent it with no block.
    mm._get_connection().execute(
        "UPDATE Pending_Retain SET sent_generation = generation WHERE interaction_id = ?",
        (row,))

    counts = await chat_system.rebuild_persona_memory("test_persona")

    assert counts["pending"] == 1 and mm.is_retain_pending(row)
    got = mm._get_connection().execute(
        "SELECT untrusted, retain_text FROM Pending_Retain WHERE interaction_id = ?",
        (row,)).fetchone()
    assert (got["untrusted"], got["retain_text"]) == (0, "reply")


@pytest.mark.asyncio
async def test_rebuild_stops_when_the_bank_cannot_be_created_again(setup):
    chat_system, mm, persona, hs = setup
    await _cut_one_block(chat_system, hs)
    sends = len(hs.sends)
    chat_system.provision_persona_memory = AsyncMock(return_value=False)  # type: ignore[method-assign]
    with pytest.raises(RuntimeError, match="could not be created"):
        await chat_system.rebuild_persona_memory("test_persona")
    assert len(hs.sends) == sends


@pytest.mark.asyncio
async def test_rebuild_drops_the_banks_ingest_cache(setup, tmp_path, monkeypatch):
    chat_system, mm, persona, hs = setup
    import src.chat_system as cs
    monkeypatch.setattr(cs, "INGEST_CACHE_DIR", tmp_path)
    cache = tmp_path / "test_persona.json"
    cache.write_text('{"a.md": {"sha256": "x"}}', encoding="utf-8")
    other = tmp_path / "other.json"
    other.write_text("{}", encoding="utf-8")
    await chat_system.rebuild_persona_memory("test_persona")
    assert not cache.exists() and other.exists()
