# tests/test_chat_system_startup.py
"""DP-115: ChatSystem.startup() Hindsight bank provisioning.

The async startup hook (src/chat_system.py:1120) is the only place
ensure_bank is called per persona at boot. If it regresses, banks are
implicit on first retain — slower, and missing the per-persona
retain_mission / reflect_mission seeded by ensure_bank.
"""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from memory.memory_manager import MemoryManager
from src.chat_system import ChatSystem
from src.memory.backend.hindsight import HindsightBackend
from src.persona import Persona
from tests.helpers import make_chat_system


def _make_system(backend: object, personas: dict[str, Persona]) -> ChatSystem:
    mm = MagicMock(spec=MemoryManager)
    mm.backend = backend
    return make_chat_system(memory_manager=mm, personas=personas)


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


def _persona_tuned(name: str) -> Persona:
    return Persona(
        persona_name=name,
        model_name="m",
        prompt="p",
        retain_mission=f"retain-{name}",
        reflect_mission=f"reflect-{name}",
        enable_observations=True,
        observations_mission=f"obs-{name}",
        disposition={"skepticism": 4},
    )


@pytest.mark.asyncio
async def test_startup_passes_configured_missions_to_ensure_bank() -> None:
    """DP-255: per-persona retain/reflect/observations missions reach ensure_bank."""
    backend = MagicMock(spec=HindsightBackend)
    backend.ensure_bank = AsyncMock()
    # disposition goes through the live PATCH path, not ensure_bank.
    client = MagicMock()
    client.apatch_bank_config = AsyncMock()
    backend._get_client = MagicMock(return_value=client)

    system = _make_system(backend, {"sage": _persona_tuned("sage")})
    await system.startup()

    backend.ensure_bank.assert_awaited_once()
    kwargs = backend.ensure_bank.await_args.kwargs
    assert kwargs["bank_id"] == "sage"
    assert kwargs["retain_mission"] == "retain-sage"
    assert kwargs["reflect_mission"] == "reflect-sage"
    assert kwargs["enable_observations"] is True
    assert kwargs["observations_mission"] == "obs-sage"

    # disposition patched live with disposition_* keys.
    client.apatch_bank_config.assert_awaited_once_with("sage", {"disposition_skepticism": 4})


@pytest.mark.asyncio
async def test_startup_unset_missions_pass_none() -> None:
    """A persona that never configured missions passes None (bank keeps default),
    and disposition patch is NOT called when no disposition is set."""
    backend = MagicMock(spec=HindsightBackend)
    backend.ensure_bank = AsyncMock()
    client = MagicMock()
    client.apatch_bank_config = AsyncMock()
    backend._get_client = MagicMock(return_value=client)

    system = _make_system(backend, {"plain": _persona("plain")})
    await system.startup()

    kwargs = backend.ensure_bank.await_args.kwargs
    assert kwargs["retain_mission"] is None
    assert kwargs["reflect_mission"] is None
    assert kwargs["enable_observations"] is None
    assert kwargs["observations_mission"] is None
    client.apatch_bank_config.assert_not_awaited()
