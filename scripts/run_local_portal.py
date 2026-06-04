#!/usr/bin/env python3
"""Run the DERPR engine portal locally with a stubbed LLM, for fast iteration.

Spins up `KoboldEngineAdapter` on 127.0.0.1:5003 backed by an in-memory DB and a
fake text engine that streams a canned reply. No API keys, no network, no Discord
— just the portal + engine HTTP surface so the web UI (portal.html) connect/submit
flow can be exercised without redeploying to prod.

    .venv/bin/python scripts/run_local_portal.py
    # then open http://127.0.0.1:5003/  (or curl it)

The stub mirrors the prod "testr" persona ("only respond with 'success'") so a
submitted turn returns "success" without a real model.
"""

import os
import sys
from unittest.mock import AsyncMock, patch

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "src"))

PORT = int(os.environ.get("LOCAL_PORTAL_PORT", "5055"))

from memory.memory_manager import MemoryManager  # noqa: E402
from src.chat_system import ChatSystem  # noqa: E402
from src.engine import TextEngine  # noqa: E402
from src.interfaces.kobold_engine_adapter import KoboldEngineAdapter  # noqa: E402
from src.persona import Persona  # noqa: E402

PERSONA = "testr"
REPLY = "success"


async def _stub_stream(*args, **kwargs):
    """Fake LLM: emit one delta and a terminal done event."""
    yield {"type": "api_payload", "payload": {}}
    yield {"type": "text_delta", "text": REPLY}
    yield {"type": "done", "full_text": REPLY}


def main() -> int:
    mm = MemoryManager(db_path=":memory:")
    mm.create_schema()

    persona = Persona(
        persona_name=PERSONA,
        model_name="local",
        prompt="you only respond with 'success'",
        context_length=10,
    )

    text_engine = TextEngine()
    text_engine.stream_messages = _stub_stream  # type: ignore[method-assign]

    with patch("src.chat_system.load_personas_from_file", return_value={PERSONA: persona}):
        chat_system = ChatSystem(memory_manager=mm, text_engine=text_engine)
    chat_system.bot_logic.preprocess_message = AsyncMock(return_value=None)

    adapter = KoboldEngineAdapter(chat_system=chat_system)
    adapter.host = "127.0.0.1"
    adapter.port = PORT

    import uvicorn
    print(f"Local DERPR engine portal on http://127.0.0.1:{PORT}/  (persona={PERSONA}, reply={REPLY!r})")
    uvicorn.run(adapter.app, host="127.0.0.1", port=PORT, log_level="info")
    return 0


if __name__ == "__main__":
    sys.exit(main())
