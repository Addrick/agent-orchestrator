"""Per-turn ContextVar for tools that need engine-side scope info.

Tools are dispatched as plain async callables (`handler(**args)`) — there's
no implicit handle on the surrounding ChatSystem turn. For tools that must
inherit the active turn's scope (persona, channel, user, server) without
exposing it as model-callable arguments, the engine sets this ContextVar at
the top of `_orchestrate` and the handler reads it.

Used by:
- `recall_memory` (DP-113) — to scope `MemoryBackend.recall` to the current
  bank + tag predicate without trusting the model to pass them. The predicate
  follows the persona's `memory_mode` (DP-407), the same scope auto-recall uses.

Returns None outside an active turn (e.g., import-time tool registration,
unit tests that don't enter `_orchestrate`). Handlers must tolerate that.
"""
from __future__ import annotations

import contextlib
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterator, Mapping, Optional

from src.persona import MemoryMode


@dataclass(frozen=True)
class TurnContext:
    persona_name: str
    user_identifier: str
    channel: str
    server_id: Optional[str]
    memory_mode: MemoryMode = MemoryMode.CHANNEL_ISOLATED


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


# ``bind_args`` sentinels (DP-475): an argument a tool entry takes away from
# the model and fills from the active turn instead. Allowlisted — a config
# value outside this map is rejected at load, never passed through literally.
BIND_SOURCES: Dict[str, Callable[[TurnContext], str]] = {
    "{PERSONA}": lambda ctx: ctx.persona_name,
}


def resolve_binds(binds: Mapping[str, str], ctx: TurnContext) -> Dict[str, str]:
    """``bind_args`` (arg -> sentinel) resolved against ``ctx``."""
    return {arg: BIND_SOURCES[source](ctx) for arg, source in binds.items()}


def bind_conflict(
    tool_name: str, binds: Mapping[str, str], args: Mapping[str, Any],
    ctx: Optional[TurnContext],
) -> Optional[str]:
    """Why a call to a bound tool must be refused, or None if it may run.

    A bound argument the model supplied with a different value is refused,
    never overwritten: overwriting runs an action other than the one the model
    believes it took and the approver was shown (an approval card reading
    ``bank_id='managr'`` for a delete that lands in the caller's own bank)."""
    if not binds:
        return None
    if ctx is None:
        return (
            f"tool '{tool_name}' needs an active chat turn to fill its bound "
            f"argument(s) {sorted(binds)}; not executed"
        )
    for arg, value in resolve_binds(binds, ctx).items():
        if arg in args and args[arg] != value:
            return (
                f"tool '{tool_name}': '{arg}' is fixed to '{value}' for this "
                f"persona and cannot be set to {args[arg]!r}; omit it"
            )
    return None
