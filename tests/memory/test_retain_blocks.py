"""DP-495 block retain: the cut planner, the block splitter, document ids,
and TurnPersistence's send / poll of blocks (real MemoryManager, fake backend).

The end-to-end flow through the kernel is tests/integration/test_block_retain.py.
"""
import json
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock

import pytest

from config import global_config
from src.memory.backend.base import MemoryBackend, RetainRefused
from src.memory.context_budget import plan_block_cut
from src.memory.memory_manager import MemoryManager
from src.memory.retain_blocks import (
    CONTEXT_PREAMBLE, block_document_id, retain_scope, split_block,
)
from src.persona import MemoryMode, Persona
from src.request_builder import RequestBuilder
from src.turn_persistence import TurnPersistence

U, A = True, False  # a row the window may start on (user) / may not


# ---------- plan_block_cut ----------

def test_no_cut_under_the_high_watermark():
    rows = [(U, 100), (A, 100)] * 4                       # 800
    assert plan_block_cut(rows, budget=1000, fixed_tokens=50) is None


def test_cut_brings_window_to_low_and_snaps_to_a_turn():
    rows = [(U, 100), (A, 100)] * 5                       # 1000 + 50 > 900
    k = plan_block_cut(rows, budget=1000, fixed_tokens=50)
    # Remaining + fixed <= 500 → keep at most 450 → 4 rows, starting on a user.
    assert k == 6
    assert rows[k][0] is U
    assert sum(t for _, t in rows[k:]) + 50 <= 500


def test_snap_cuts_more_rather_than_start_mid_turn():
    # A long assistant row: keeping it would need the window to start on it.
    rows = [(U, 100), (A, 100), (U, 50), (A, 400), (U, 100)]
    k = plan_block_cut(rows, budget=900, fixed_tokens=100)
    assert k == 4 and rows[k][0] is U


def test_nothing_fits_cuts_everything():
    rows = [(U, 100), (A, 900)]
    assert plan_block_cut(rows, budget=1000, fixed_tokens=600) == 2


def test_idle_trim_has_no_high_watermark_and_leaves_short_windows():
    rows = [(U, 100), (A, 100)] * 3                       # 600: under 90 %, over 50 %
    assert plan_block_cut(rows, budget=1000, trigger=False) == 2
    assert plan_block_cut(rows[:4], budget=1000, trigger=False) is None


def test_keep_at_most_cuts_a_binding_count_cap_to_half():
    rows = [(U, 1), (A, 1)] * 5
    k = plan_block_cut(rows, budget=None, keep_at_most=5)
    assert k == 6 and len(rows) - k <= 5 and rows[k][0] is U


# ---------- RequestBuilder.plan_window_cut: rows are atomic ----------

def _builder(rows: List[Dict[str, Any]]) -> RequestBuilder:
    mm = MagicMock()
    mm.retain_window.return_value = (1, 1)
    mm.get_channel_history.return_value = rows
    return RequestBuilder(memory_manager=mm, memory_backend=MagicMock(),
                          tool_manager_lookup=MagicMock(), persona_lookup=lambda _n: None)


def test_tool_calls_and_their_results_are_never_split():
    """An assistant row's tool_context replays as tool_calls + tool messages;
    the cut works on rows, so the span leaves (or stays) whole."""
    tool_ctx = json.dumps([
        {"role": "assistant", "content": None, "tool_calls": [
            {"id": "c1", "type": "function",
             "function": {"name": "get_agent_status", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "c1", "content": "y" * 400},
    ])
    rows = [
        {"interaction_id": 1, "author_role": "user", "author_name": "u", "content": "a" * 400},
        {"interaction_id": 2, "author_role": "assistant", "author_name": "p",
         "content": "b" * 40, "tool_context": tool_ctx},
        {"interaction_id": 3, "author_role": "user", "author_name": "u", "content": "c" * 40},
        {"interaction_id": 4, "author_role": "assistant", "author_name": "p", "content": "d" * 40},
    ]
    builder = _builder(rows)
    persona = Persona("p", "m", "s", inject_timestamp=False)
    window = builder.load_history_window(persona, "u", "ch", None, None, advance=False)
    kinds = [m.get("role") for m in window.messages]
    assert "tool" in kinds                                  # the span replays
    new_start = builder.plan_window_cut(persona, window, budget=250, fixed_tokens=10,
                                        trigger=True)
    assert new_start == 3                                   # rows 1-2 (and the whole span) go


# ---------- split_block / document ids ----------

def test_block_document_id_format():
    ts = datetime(2026, 11, 2, 0, 34, 59, tzinfo=timezone.utc)
    assert block_document_id("stagr", "web_ui_pandora", 7, ts) == \
        "stagr:web_ui_pandora:b0007:2026-11-02T00:34Z"
    assert block_document_id("stagr", "web_ui_pandora", 7, ts, part=2) == \
        "stagr:web_ui_pandora:b0007p2:2026-11-02T00:34Z"


def test_scope_keys_follow_the_memory_mode():
    def key(mode, **kw):
        args = {"persona_name": "p", "channel": "ch", "user_identifier": "u", "server_id": None}
        args.update(kw)
        return retain_scope(mode, **args).key
    assert key(MemoryMode.CHANNEL_ISOLATED) == "channel-ch"
    assert key(MemoryMode.CHANNEL_ISOLATED, server_id="g") == "channel-ch@g"
    assert key(MemoryMode.GLOBAL) == "global"
    assert key(MemoryMode.SERVER_WIDE, server_id="g") == "server-g"
    assert key(MemoryMode.PERSONAL) == "user-u"
    assert key(MemoryMode.TICKET_ISOLATED) == "ticket-ch"
    # Two modes never share a key, whatever the channel is called: a persona
    # whose mode changes must not pick up the old mode's window row.
    for name in ("global", "user-u", "server-g", "ch"):
        keys = [key(m, channel=name, user_identifier="u", server_id=None)
                for m in MemoryMode if m != MemoryMode.SERVER_WIDE]
        keys.append(key(MemoryMode.SERVER_WIDE, channel=name, server_id="g"))
        assert len(set(keys)) == len(keys), keys
    assert not retain_scope(MemoryMode.TICKET_ISOLATED, persona_name="p", channel="t",
                            user_identifier="u", server_id=None).has_window


def test_small_block_is_one_part_with_the_prior_tail_as_context():
    parts = split_block(["a: 1", "b: 2"], 1000, prior=["old: x", "old: y"], tail_chars=100)
    assert len(parts) == 1
    assert parts[0]["content"] == "a: 1\n\nb: 2"
    assert parts[0]["context"] == CONTEXT_PREAMBLE + "old: x\n\nold: y"


def test_first_block_has_no_context():
    assert split_block(["a: 1"], 1000, tail_chars=100)[0]["context"] is None


def test_oversized_block_splits_on_turns_each_part_carrying_the_previous_tail():
    turns = [f"t{i}: " + "z" * 40 for i in range(10)]     # 44 chars each
    parts = split_block(turns, 100, prior=["before"], tail_chars=60)
    assert [p["start"] for p in parts] == [0, 2, 4, 6, 8]
    assert all(len(p["content"]) <= 100 for p in parts)
    assert "\n\n".join(p["content"] for p in parts) == "\n\n".join(turns)
    assert parts[0]["context"].endswith("before")
    # Part 2's background is the last turn(s) of part 1 that fit 60 chars.
    assert parts[1]["context"] == CONTEXT_PREAMBLE + turns[1]


def test_a_single_turn_larger_than_the_chunk_is_its_own_part():
    parts = split_block(["s", "L" * 500, "e"], 100, tail_chars=10)
    assert [p["content"] for p in parts] == ["s", "L" * 500, "e"]


# ---------- TurnPersistence: send and poll ----------

class FakeBackend(MemoryBackend):
    def __init__(self) -> None:
        self.sends: List[Dict[str, Any]] = []
        self.deleted: List[str] = []
        self.status = "pending"
        self.accept = True

    async def retain_block(self, bank_id, items, *, request_id) -> Optional[List[str]]:
        if not self.accept:
            return None
        self.sends.append({"items": items, "request_id": request_id})
        return [f"op{len(self.sends)}"]

    async def retain_status(self, bank_id, op_ids, document_ids) -> str:
        return self.status

    async def delete_document(self, bank_id, document_id) -> Dict[str, Any]:
        self.deleted.append(document_id)
        return {}

    # Unused abstract surface.
    def __getattr__(self, name):  # pragma: no cover
        raise AttributeError(name)


for _name in list(getattr(MemoryBackend, "__abstractmethods__", ())):
    if _name not in FakeBackend.__dict__:
        setattr(FakeBackend, _name, lambda self, *a, **k: None)
FakeBackend.__abstractmethods__ = frozenset()


class RefusingBackend(FakeBackend):
    """Hindsight up, refusing every block (a deterministic 4xx)."""

    def __init__(self) -> None:
        super().__init__()
        self.calls = 0

    async def retain_block(self, bank_id, items, *, request_id) -> Optional[List[str]]:
        self.calls += 1
        raise RetainRefused("Hindsight API Error 422: bad item")


@pytest.fixture
def mm():
    manager = MemoryManager(db_path=":memory:")
    manager.create_schema()
    yield manager
    manager.close()


@pytest.fixture
def tp(mm):
    return TurnPersistence(mm, FakeBackend())


@pytest.fixture(autouse=True)
def eastern(monkeypatch):
    monkeypatch.setattr(global_config, "LOCAL_TZ", "America/New_York")


T0 = datetime(2026, 11, 2, 0, 34, tzinfo=timezone.utc)   # Sun 19:34 EST
SCOPE = retain_scope(MemoryMode.CHANNEL_ISOLATED, persona_name="stagr",
                     channel="web_ui_pandora", user_identifier="portal", server_id=None)


def _turns(mm, *texts, untrusted=False):
    ids = []
    for n, text in enumerate(texts):
        role = "user" if n % 2 == 0 else "assistant"
        i = mm.log_message(user_identifier="portal", persona_name="stagr",
                           channel="web_ui_pandora", author_role=role,
                           author_name="Adam" if role == "user" else "stagr",
                           content=text, timestamp=T0 + timedelta(minutes=len(ids)))
        mm.queue_retain(i, "stagr", "web_ui_pandora", untrusted=untrusted)
        ids.append(i)
    return ids


def _cut(mm, below):
    start, _ = mm.retain_window(SCOPE)
    return mm.cut_retain_block(SCOPE, expected_start=start, new_start=below)


@pytest.mark.asyncio
async def test_block_is_sent_as_one_replace_document(mm, tp):
    ids = _turns(mm, "hi", "hello", "how are you", "fine")
    _cut(mm, ids[-1] + 1)
    assert await tp.flush_blocks() == 1
    (send,) = tp.memory_backend.sends
    (item,) = send["items"]
    assert item["document_id"] == "stagr:channel-web_ui_pandora:b0001:2026-11-02T00:34Z"
    assert item["content"] == (
        "[2026-11-01 19:34] Adam: hi\n\nstagr: hello\n\n"
        "[2026-11-01 19:36] Adam: how are you\n\nstagr: fine"
    )
    assert item["context"] is None
    assert item["untrusted"] is False
    assert "channel:web_ui_pandora" in item["tags"] and "persona:stagr" in item["tags"]
    block = mm.retain_blocks(statuses=("sent",))[0]
    assert json.loads(block["op_ids"]) == ["op1"]
    assert send["request_id"] == block["request_id"]


@pytest.mark.asyncio
async def test_second_block_carries_the_first_ones_tail(mm, tp):
    first = _turns(mm, "a", "b")
    _cut(mm, first[-1] + 1)
    second = _turns(mm, "c", "d")
    start, _ = mm.retain_window(SCOPE)
    mm.cut_retain_block(SCOPE, expected_start=start, new_start=second[-1] + 1)
    await tp.flush_blocks()
    item = tp.memory_backend.sends[1]["items"][0]
    assert item["document_id"].startswith("stagr:channel-web_ui_pandora:b0002:")
    assert item["context"] == CONTEXT_PREAMBLE + "[2026-11-01 19:34] Adam: a\n\nstagr: b"


@pytest.mark.asyncio
async def test_oversized_block_goes_as_numbered_parts(mm, tp, monkeypatch):
    monkeypatch.setattr("src.turn_persistence.HINDSIGHT_RETAIN_CHUNK_SIZE", 60)
    ids = _turns(mm, *[f"turn {i} " + "w" * 30 for i in range(4)])
    _cut(mm, ids[-1] + 1)
    await tp.flush_blocks()
    items = tp.memory_backend.sends[0]["items"]
    assert [i["document_id"].split(":")[2] for i in items] == ["b0001p1", "b0001p2", "b0001p3", "b0001p4"]
    assert items[0]["context"] is None
    assert all(i["context"].startswith(CONTEXT_PREAMBLE) for i in items[1:])
    assert items[1]["timestamp"] == T0 + timedelta(minutes=1)


@pytest.mark.asyncio
async def test_refused_block_stays_queued_and_in_the_prompt(mm, tp):
    ids = _turns(mm, "a", "b")
    _cut(mm, ids[-1] + 1)
    tp.memory_backend.accept = False
    assert await tp.flush_blocks() == 0
    assert mm.retain_blocks(statuses=("queued",))
    assert mm.retain_window(SCOPE)[1] < ids[-1] + 1        # still shown
    tp.memory_backend.accept = True
    assert await tp.flush_blocks() == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("status, outcome", [
    ("completed", "done"), ("pending", "sent"), ("unknown", "sent"),
    ("failed", "failed"), ("lost", "queued"),
])
async def test_poll_moves_the_block(mm, tp, status, outcome):
    ids = _turns(mm, "a", "b")
    block = _cut(mm, ids[-1] + 1)
    await tp.flush_blocks()
    tp.memory_backend.status = status
    await tp.poll_blocks()
    assert mm.retain_blocks(statuses=(outcome,), block_id=block)
    shown = outcome in ("sent", "queued")
    assert (mm.retain_window(SCOPE)[1] < ids[-1] + 1) is shown


@pytest.mark.asyncio
async def test_rebuild_with_fewer_parts_deletes_the_superseded_documents(mm, tp, monkeypatch):
    monkeypatch.setattr("src.turn_persistence.HINDSIGHT_RETAIN_CHUNK_SIZE", 60)
    ids = _turns(mm, *[f"turn {i} " + "w" * 30 for i in range(4)])
    _cut(mm, ids[-1] + 1)
    await tp.flush_blocks()
    first_ids = [i["document_id"] for i in tp.memory_backend.sends[0]["items"]]

    monkeypatch.setattr("src.turn_persistence.HINDSIGHT_RETAIN_CHUNK_SIZE", 10_000)
    mm.queue_retain(ids[1], "stagr", "web_ui_pandora", untrusted=False)   # a retry
    await tp.flush_blocks()
    (rebuilt,) = tp.memory_backend.sends[1]["items"]
    assert rebuilt["document_id"] == "stagr:channel-web_ui_pandora:b0001:2026-11-02T00:34Z"
    assert sorted(tp.memory_backend.deleted) == sorted(first_ids)


@pytest.mark.asyncio
async def test_untrusted_turn_taints_the_whole_block(mm, tp):
    ids = _turns(mm, "a")
    tainted = _turns(mm, "b", untrusted=True)
    _cut(mm, tainted[-1] + 1)
    await tp.flush_blocks()
    assert tp.memory_backend.sends[0]["items"][0]["untrusted"] is True
    assert ids


# ---------- Startup wiring and config ----------

def test_main_registers_the_retain_maintenance_loop():
    """The loop is the boot pass and the periodic sweep; without the
    registration no block is ever polled to completion or idle-trimmed."""
    import inspect

    import src.main as main

    registered = {}

    class App:
        def register_task(self, name, coro, stop=None, **_kw):
            registered[name] = (coro, stop)

    class Bot:
        async def run_retain_maintenance(self):
            return None

        def stop_retain_maintenance(self):
            return None

    bot = Bot()
    main._register_retain_maintenance(App(), bot)
    coro, stop = registered["retain_maintenance"]
    coro.close()
    assert stop == bot.stop_retain_maintenance
    assert "_register_retain_maintenance(app, bot)" in inspect.getsource(main.main)


@pytest.mark.parametrize("env, expected", [(None, "40000"), ("120000", "120000")])
def test_chunk_size_setting_default_and_env_override(env, expected):
    import os
    import subprocess
    import sys
    from pathlib import Path

    child_env = dict(os.environ)
    child_env.pop("HINDSIGHT_RETAIN_CHUNK_SIZE", None)
    if env is not None:
        child_env["HINDSIGHT_RETAIN_CHUNK_SIZE"] = env
    out = subprocess.run(
        [sys.executable, "-c",
         "from config import global_config as g; "
         "print(g.HINDSIGHT_RETAIN_CHUNK_SIZE, g.DEFAULT_HISTORY_MESSAGES)"],
        cwd=Path(__file__).resolve().parents[2], env=child_env,
        capture_output=True, text=True, check=True,
    )
    assert out.stdout.split() == [expected, "1000"]


# ---------- Hindsight down / refusing (DP-495) ----------

@pytest.mark.asyncio
async def test_refused_block_backs_off_alone_and_is_never_given_up(mm):
    """A 4xx is this block's problem, not an outage: it stays queued, its
    retry doubles from a minute to an hour, and Hindsight is not marked down."""
    backend = RefusingBackend()
    tp = TurnPersistence(mm, backend)
    ids = _turns(mm, "hi", "hello")
    _cut(mm, ids[-1] + 1)
    assert await tp.flush_blocks() == 0
    assert backend.calls == 1 and not tp._backend_down
    (block,) = mm.retain_blocks(statuses=("queued",))
    assert block["attempts"] == 1 and block["refused_since"] is not None
    # Not due again until its backoff runs out.
    assert await tp.flush_blocks() == 0 and backend.calls == 1
    due = datetime.fromisoformat(block["next_attempt_at"])
    assert mm.retain_blocks(statuses=("queued",), due_at=due) != []
    delays = [mm.mark_retain_block_refused(dict(block, attempts=n)) for n in range(10)]
    assert delays[0] == global_config.RETAIN_REFUSED_BACKOFF_MIN_SECONDS
    assert delays[1] == 2 * delays[0]
    assert delays[-1] == global_config.RETAIN_REFUSED_BACKOFF_MAX_SECONDS


@pytest.mark.asyncio
async def test_long_refused_block_leaves_the_prompt_but_stays_queued(mm):
    backend = RefusingBackend()
    tp = TurnPersistence(mm, backend)
    ids = _turns(mm, "hi", "hello", "more", "ok")
    _cut(mm, ids[2])
    await tp.flush_blocks()
    _, floor = mm.retain_window(SCOPE)
    assert floor == ids[0]                       # held in the prompt for now
    later = datetime.now(timezone.utc) + timedelta(
        seconds=global_config.RETAIN_REFUSED_PROMPT_SECONDS + 1)
    _, floor = mm.retain_window(SCOPE, now=later)
    assert floor == ids[2]                       # released from the prompt
    assert [b["status"] for b in mm.retain_blocks(statuses=("queued",))] == ["queued"]


@pytest.mark.asyncio
async def test_requeue_resets_the_refusal_backoff(mm):
    tp = TurnPersistence(mm, RefusingBackend())
    ids = _turns(mm, "hi", "hello")
    _cut(mm, ids[-1] + 1)
    await tp.flush_blocks()
    mm.queue_retain(ids[1], "stagr", "web_ui_pandora", untrusted=False)   # a retry
    (block,) = mm.retain_blocks(statuses=("queued",))
    assert block["attempts"] == 0 and block["next_attempt_at"] is None
    assert block["refused_since"] is None


@pytest.mark.asyncio
async def test_down_hindsight_pauses_turn_sends_and_the_loop_probes(mm):
    """Unreachable: the flush stops at the first block (the rest would wait
    out the same timeout), turns stop scheduling sends, and the next
    maintenance pass is the probe that resumes them."""
    backend = FakeBackend()
    tp = TurnPersistence(mm, backend)
    ids = _turns(mm, "a", "b", "c", "d", "e", "f")
    _cut(mm, ids[2])
    start, _ = mm.retain_window(SCOPE)
    mm.cut_retain_block(SCOPE, expected_start=start, new_start=ids[4])
    tried = []
    real = tp._deliver_block

    async def counting(block):
        tried.append(block["block_id"])
        return await real(block)

    tp._deliver_block = counting  # type: ignore[method-assign]
    backend.accept = False
    assert await tp.flush_blocks() == 0
    assert len(tried) == 1 and tp._backend_down
    tp.schedule_flush()
    assert not tp._flush_tasks                   # a turn starts nothing while down
    backend.accept = True
    assert await tp.flush_blocks() == 2          # the maintenance pass: probe, then all
    assert not tp._backend_down


@pytest.mark.asyncio
async def test_unreachable_poll_marks_down_and_stops(mm):
    backend = FakeBackend()
    tp = TurnPersistence(mm, backend)
    ids = _turns(mm, "a", "b", "c", "d")
    _cut(mm, ids[2])
    start, _ = mm.retain_window(SCOPE)
    mm.cut_retain_block(SCOPE, expected_start=start, new_start=ids[3])
    await tp.flush_blocks()
    asked = []

    async def status(bank_id, op_ids, document_ids):
        asked.append(op_ids)
        return "unreachable"

    backend.retain_status = status  # type: ignore[method-assign]
    assert await tp.poll_blocks() == 0
    assert len(asked) == 1 and tp._backend_down
    assert len(mm.retain_blocks(statuses=("sent",))) == 2


@pytest.mark.asyncio
async def test_turn_flushes_coalesce_to_one_waiting_pass(mm):
    tp = TurnPersistence(mm, FakeBackend())
    async with tp._flush_lock:                   # a pass is running
        for _ in range(5):
            tp.schedule_flush()
        assert len(tp._flush_tasks) == 1
    await tp.drain_flushes()
    tp.schedule_flush()                          # nothing waiting: a new one
    assert len(tp._flush_tasks) == 1
    await tp.drain_flushes()
