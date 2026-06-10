# tests/test_chat_system_startup.py
"""DP-115: ChatSystem.startup() Hindsight bank provisioning.

The async startup hook (src/chat_system.py:1120) is the only place
ensure_bank is called per persona at boot. If it regresses, banks are
implicit on first retain — slower, and missing the per-persona
retain_mission / reflect_mission seeded by ensure_bank.
"""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from memory.memory_manager import MemoryManager
from src.bootstrap import create_chat_system
from src.chat_system import ChatSystem
from src.engine import TextEngine
from src.memory.backend.hindsight import HindsightBackend
from src.persona import Persona


def _make_system(backend: object, personas: dict[str, Persona]) -> ChatSystem:
    mm = MagicMock(spec=MemoryManager)
    mm.backend = backend
    text_engine = MagicMock(spec=TextEngine)
    with patch("src.bootstrap.load_personas_from_file", return_value=personas), \
         patch("src.bootstrap.load_system_personas_from_file", return_value={}):
        return create_chat_system(memory_manager=mm, text_engine=text_engine)


def _persona(name: str) -> Persona:
    return Persona(persona_name=name, model_name="m", prompt="p")


@pytest.mark.asyncio
async def test_startup_calls_ensure_bank_per_persona() -> None:
    backend = MagicMock(spec=HindsightBackend)
    backend.ensure_bank = AsyncMock()
    system = _make_system(backend, {"alice": _persona("alice"), "bob": _persona("bob")})

    await system.startup()

    called = {c.kwargs.get("bank_id") for c in backend.ensure_bank.await_args_list}
    assert called == {"alice", "bob"}
    assert backend.ensure_bank.await_count == 2


@pytest.mark.asyncio
async def test_startup_tolerates_per_bank_failures() -> None:
    """One failing bank must not prevent the next from being provisioned."""
    backend = MagicMock(spec=HindsightBackend)
    backend.ensure_bank = AsyncMock(side_effect=[RuntimeError("boom"), None])
    system = _make_system(backend, {"alice": _persona("alice"), "bob": _persona("bob")})

    await system.startup()  # must not raise

    assert backend.ensure_bank.await_count == 2


@pytest.mark.asyncio
async def test_startup_skips_when_backend_not_hindsight() -> None:
    """Sqlite / Null backends have an ensure_bank that's a noop, but the
    startup loop guards on isinstance to avoid even probing personas."""
    backend = MagicMock()  # not a HindsightBackend
    backend.ensure_bank = AsyncMock()
    system = _make_system(backend, {"alice": _persona("alice")})

    await system.startup()

    backend.ensure_bank.assert_not_awaited()
