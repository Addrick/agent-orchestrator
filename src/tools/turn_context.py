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

import contextlib
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Iterator, Optional


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


@contextlib.contextmanager
def turn_scope(ctx: Optional[TurnContext]) -> Iterator[None]:
    """Bind `ctx` for the duration of the `with` block, always restoring the
    prior value on exit — normal return, exception, or `GeneratorExit` raised
    into an enclosing async generator at a suspended `yield`.

    This is the only correct way to manage the turn ContextVar across the
    streaming kernel: a manually-paired set/reset is skipped whenever an exit
    path (post-loop exception, early consumer break) bypasses the trailing
    reset, leaking the scope into the next turn sharing the event-loop context.

    Restore-by-value (not `ContextVar.reset(token)`): `_orchestrate` is an
    async generator, so `set()` happens during one `__anext__` while cleanup
    can run during a later `aclose()` in a *different* Context — `reset(token)`
    raises "Token was created in a different Context" there. Plain `set` of the
    saved prior value has no such coupling.
    """
    prev = _turn_context.get()
    _turn_context.set(ctx)
    try:
        yield
    finally:
        _turn_context.set(prev)
