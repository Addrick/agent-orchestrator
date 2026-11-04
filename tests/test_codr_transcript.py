# tests/test_codr_transcript.py
#
# DP-484 — a ticket's opencode session folded into channel rows, and mirrored
# idempotently into the store.

import json

import aiohttp

from src.agents.codr_transcript import TranscriptMirror, fold_session, render_subagent
from src.engine.providers import opencode
from src.interfaces.transcript import build_transcript
from tests.test_codr_agent import FakeMemory, Persistence
from tests import test_opencode_provider as provider_tests
from tests.test_opencode_provider import ROOT

fake = provider_tests.fake  # the fake `opencode serve` fixture

AUTO = opencode.AUTO_FIELD


def _msg(mid, role, parts, **info):
    return {"info": {"id": mid, "role": role, "time": {"created": int(mid.split("_")[1]) * 1000}, **info},
            "parts": parts}


def _user(mid, text, **meta):
    return _msg(mid, "user", [{"type": "text", "text": text, **({"metadata": meta} if meta else {})}])


def _tool(call, tool="bash", status="completed", output="ok", inp=None, **state):
    return {"type": "tool", "callID": call, "tool": tool,
            "state": {"status": status, "input": inp or {"command": "ls"}, "output": output, **state}}


def _step(mid, *parts, reasoning=None):
    pre = [{"type": "reasoning", "text": reasoning}] if reasoning else []
    return _msg(mid, "assistant", [{"type": "step-start"}, *pre, *parts])


def test_a_turn_folds_into_a_prompt_row_and_one_reply_row_with_its_steps():
    msgs = [
        _user("msg_1", "[Ticket `CD-001`] Build it.", **{AUTO: {"task": "CD-001"}}),
        _step("msg_2", _tool("c1", inp={"command": "ls"}, output="a.py"), reasoning="look first"),
        _step("msg_3", {"type": "text", "text": "Now tests."}, _tool("c2", output="2 passed")),
        _step("msg_4", {"type": "text", "text": "Done."}, reasoning="all good"),
    ]
    user, reply = fold_session(msgs)
    assert (user.key, user.role, user.content) == ("msg_1", "user", "[Ticket `CD-001`] Build it.")
    assert (reply.key, reply.role, reply.content) == ("msg_2", "assistant", "Done.")
    assert reply.reasoning == "look first\n\nall good"
    steps = json.loads(reply.tool_context)
    assert [m["role"] for m in steps] == ["assistant", "tool", "assistant", "tool"]
    assert steps[2]["content"] == "Now tests."
    assert steps[3] == {"role": "tool", "tool_call_id": "c2", "content": "2 passed"}

    # The portal renders it as one reply: cards with each step's thinking,
    # prose and output, the closing thinking below them.
    rows = [{"interaction_id": i, "author_role": r.role, "content": r.content,
             "reasoning_content": r.reasoning, "tool_context": r.tool_context}
            for i, r in enumerate((user, reply), 1)]
    chunks = build_transcript(rows)["chunks"]
    cards = chunks[1]["tool_context"]
    assert [c["tool_name"] for c in cards] == ["bash", "bash"]
    assert cards[0]["arguments"] == {"command": "ls"} and cards[0]["result"] == "a.py"
    assert cards[0]["reasoning"] == "look first" and cards[1]["prose"] == "Now tests."


def test_adams_messages_and_the_turns_they_start_are_left_to_the_chat_path():
    msgs = [
        _user("msg_1", "Build it.", **{AUTO: {"task": "t"}}),
        _step("msg_2", _tool("c1")),
        # Delivered into the running turn: the turn goes on, in a new row after it.
        _user("msg_3", "User: also docs", **{opencode.KEY_FIELD: "k1", opencode.INJECTED_FIELD: True}),
        _step("msg_4", {"type": "text", "text": "Done, with docs."}),
        # After the turn: an ordinary chat turn, stored by the chat path.
        _user("msg_5", "User: thanks", **{opencode.KEY_FIELD: "k2"}),
        _step("msg_6", {"type": "text", "text": "You're welcome."}),
        # A compaction marker inside it changes nothing.
        _msg("msg_7", "user", [{"type": "compaction", "auto": True}]),
        _step("msg_8", {"type": "text", "text": "still the chat turn"}),
        # DERPR's next prompt is mirrored again.
        _user("msg_9", "Retry.", **{AUTO: {"task": "t", "attempt": 2}}),
    ]
    assert [(r.key, r.role) for r in fold_session(msgs)] == [
        ("msg_1", "user"), ("msg_2", "assistant"), ("msg_4", "assistant"), ("msg_9", "user")]


def test_a_window_that_starts_mid_turn_is_read_from_its_first_user_message():
    msgs = [_step("msg_1", _tool("c0")), _user("msg_2", "Go.", **{AUTO: {}}), _step("msg_3", _tool("c1"))]
    assert [r.key for r in fold_session(msgs)] == ["msg_2", "msg_3"]


def test_running_and_failed_calls_and_a_turn_error_show():
    msgs = [_user("msg_1", "Go.", **{AUTO: {}}),
            _step("msg_2", _tool("c1", status="running", output=""), _tool("c2", status="error", error="boom")),
            _msg("msg_3", "assistant", [], error={"name": "APIError", "data": {"message": "context full"}})]
    (_, reply) = fold_session(msgs)
    steps = json.loads(reply.tool_context)
    assert steps[1]["content"] == "(running…)"
    assert json.loads(steps[2]["content"]) == {"error": "boom"}
    assert reply.content == "[turn error: APIError context full]"


def test_a_subagent_session_is_nested_under_the_call_that_started_it():
    task = _tool("c1", tool="task", output="<task_result>found it</task_result>",
                 inp={"description": "find the bug", "subagent_type": "explore"},
                 metadata={"sessionId": "ses_child"})
    msgs = [_user("msg_1", "Go.", **{AUTO: {}}), _step("msg_2", task)]
    child = [_user("msg_10", "find the bug"),
             _step("msg_11", _tool("x1", tool="grep", output="src/a.py:3", title="grep bug"), reasoning="search")]
    rendered = render_subagent(child)
    assert "(thinking) search" in rendered and "▸ grep: grep bug\nsrc/a.py:3" in rendered
    (_, reply) = fold_session(msgs, {"ses_child": rendered})
    result = json.loads(reply.tool_context)[1]["content"]
    assert result.startswith("<task_result>found it</task_result>")
    assert "── subagent session ses_child ──" in result and "src/a.py:3" in result


def test_long_tool_output_is_cut():
    msgs = [_user("msg_1", "Go.", **{AUTO: {}}), _step("msg_2", _tool("c1", output="x" * 9000))]
    (_, reply) = fold_session(msgs)
    out = json.loads(reply.tool_context)[1]["content"]
    assert len(out) < 6100 and out.endswith("(3000 more chars)")


async def test_mirroring_twice_updates_rows_in_place(fake):
    sid = "ses_t"
    fake.sessions[sid] = {"id": sid, "directory": f"{ROOT}/codr", "title": "derpr:codr:CD-001",
                          "time": {"created": 1, "updated": 1}}
    fake.messages[sid] = []
    fake.add(sid, "user", [{"type": "text", "text": "Build it.", "metadata": {AUTO: {"task": "CD-001"}}}],
             time={"created": 1000})
    fake.add(sid, "assistant", [_tool("c1", status="running", output="")], time={"created": 2000})
    memory = FakeMemory()
    mirror = TranscriptMirror(Persistence(memory), "codr")
    async with aiohttp.ClientSession(headers=opencode.auth_headers()) as http:
        client = opencode.OpencodeClient(http, f"{ROOT}/codr")
        assert await mirror.sync(client, sid, "CD-001") == 2
        assert await mirror.sync(client, sid, "CD-001") == 0          # nothing changed
        fake.messages[sid][-1]["parts"][0]["state"].update(status="completed", output="ok")
        fake.add(sid, "assistant", [{"type": "text", "text": "Done."}], time={"created": 3000})
        assert await mirror.sync(client, sid, "CD-001") == 1
        # A restarted DERPR (a new mirror, no cache) finds the rows already there.
        assert await TranscriptMirror(Persistence(memory), "codr").sync(client, sid, "CD-001") == 0
    user, reply = memory.channel("task:CD-001")
    assert (user["author_role"], user["author_name"], user["content"]) == ("user", "DERPR", "Build it.")
    assert user["platform_message_id"].startswith("opencode:msg_")
    assert reply["content"] == "Done." and json.loads(reply["tool_context"])[1]["content"] == "ok"
    assert user["timestamp"] < reply["timestamp"]


def test_a_keyed_row_is_inserted_once_then_rewritten_in_place_in_sqlite():
    from datetime import datetime, timezone
    from unittest.mock import MagicMock

    from src.memory.memory_manager import MemoryManager
    from src.turn_persistence import TurnPersistence

    mm = MemoryManager(db_path=":memory:")
    mm.create_schema()
    tp = TurnPersistence(mm, MagicMock())
    row = dict(platform_message_id="opencode:msg_1", persona_name="codr", user_identifier="agent:codr",
               channel="task:CD-001", role="assistant", author_name="codr",
               timestamp=datetime(2026, 11, 3, tzinfo=timezone.utc))
    rid, written = tp.upsert_keyed_row(content="", tool_context_json='[{"role": "tool"}]', **row)
    assert written and mm.get_interaction_by_platform_id("opencode:msg_1")["interaction_id"] == rid
    assert tp.upsert_keyed_row(content="", tool_context_json='[{"role": "tool"}]', **row) == (rid, False)
    assert tp.upsert_keyed_row(content="Done.", reasoning_content="r", **row) == (rid, True)
    stored = mm.get_interaction_by_platform_id("opencode:msg_1")
    assert (stored["content"], stored["reasoning_content"], stored["tool_context"]) == ("Done.", "r", None)
    assert [r["channel"] for r in mm.get_distinct_channels("codr")] == ["task:CD-001"]
    mm.close()
