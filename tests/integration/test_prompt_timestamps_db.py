# tests/integration/test_prompt_timestamps_db.py
"""DP-431 through a real SQLite store: a user turn's prompt stamp survives the
write/read round trip byte-for-byte, so the next request's prefix matches."""

from datetime import datetime, timezone
from unittest.mock import AsyncMock

import pytest

from config import global_config
from src.engine import TextEngine
from src.memory.memory_manager import MemoryManager
from src.persona import Persona
from tests.helpers import make_chat_system, route_stream_through_generate_response

pytestmark = pytest.mark.integration


@pytest.fixture
def system(monkeypatch):
    monkeypatch.setattr(global_config, "LOCAL_TZ", "America/New_York")
    mm = MemoryManager(db_path=":memory:")
    mm.create_schema()
    engine = TextEngine()
    engine.generate_response = AsyncMock(  # type: ignore[method-assign]
        return_value=({'type': 'text', 'content': 'ok'}, {}),
    )
    route_stream_through_generate_response(engine)
    chat = make_chat_system(memory_manager=mm, text_engine=engine)
    chat.personas = {'p': Persona('p', 'mock_model', 'prompt', history_messages=10,
                                  inject_timestamp=True, long_term_memory=False)}
    yield chat, engine
    mm.close()


async def test_stamp_is_stable_across_turns(system):
    chat, engine = system
    t1 = datetime(2026, 10, 14, 18, 5, tzinfo=timezone.utc)
    t2 = datetime(2026, 10, 15, 13, 30, tzinfo=timezone.utc)

    await chat.generate_response("p", "u1", "c", "first", timestamp=t1)
    first_wire = engine.generate_response.call_args.args[1]['history']
    assert first_wire[-1] == {"role": "user", "content": "[Wed 2026-10-14 14:05] first"}

    await chat.generate_response("p", "u1", "c", "second", timestamp=t2)
    second_wire = engine.generate_response.call_args.args[1]['history']
    assert second_wire[:len(first_wire)] == first_wire   # prefix unchanged
    assert second_wire[len(first_wire):] == [
        {"role": "assistant", "content": "ok"},            # model's turn: no stamp
        {"role": "user", "content": "[Thu 2026-10-15 09:30] second"},
    ]
