"""Per-turn ContextVar for tools that need engine-side scope info.

Tools are dispatched as plain async callables (`handler(**args)`) — there's
no implicit handle on the surrounding ChatSystem turn. For tools that must
inherit the active turn's scope (persona, channel, user, server) without
exposing it as model-callable arguments, the engine sets this ContextVar at
the top of `_orchestrate` and the handler reads it.

Used by:
- `recall_memory` (DP-113) — to scope `MemoryBackend.recall` to the current
  bank + tag predicate without trusting the model to pass them.

Returns None outside an active turn (e.g., import-time tool registration,
unit tests that don't enter `_orchestrate`). Handlers must tolerate that.
"""
from __future__ import annotations

from contextvars import ContextVar
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class TurnContext:
    persona_name: str
    user_identifier: str
    channel: str
    server_id: Optional[str]


_turn_context: ContextVar[Optional[TurnContext]] = ContextVar(
    "derpr_turn_context", default=None,
)


def get_turn_context() -> Optional[TurnContext]:
    return _turn_context.get()


def set_turn_context(ctx: Optional[TurnContext]) -> object:
    """Set the current turn context. Returns a token usable for `reset`."""
    return _turn_context.set(ctx)


def reset_turn_context(token: object) -> None:
    _turn_context.reset(token)  # type: ignore[arg-type]
