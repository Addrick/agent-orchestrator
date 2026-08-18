# tests/security/test_origin_allowlist_gate.py
"""DP-330 — persona origin allowlist, enforced at the dev-command chokepoint.

`BotLogic.preprocess_message` refuses ANY message — chat turn or dev command —
addressed to a persona whose `origin_allowlist` does not admit the caller's
Origin. It lives there rather than in the generation kernel because that is the
one seam every message-bearing surface passes through:

  * `ChatSystem._orchestrate` step 1 (portal chat, gmail, zammad, agents),
  * `discord_bot.on_message`, which resolves dev commands and RETURNS before
    the kernel is ever entered,
  * the portal's `/api/v1/persona/{name}/dev_command` route, same shape.

Gating in the kernel alone left the last two open, so `hypr what prompt` from
an unlisted guild answered in full — including `what origin_allowlist`, which
prints the very ids the refusal text is written not to disclose. The tests in
this file pin the chokepoint; the per-surface tests
(`tests/interfaces/test_discord_bot.py`, `tests/interfaces/
test_kobold_engine_adapter.py`) pin that each surface honours the refusal it
hands back.
"""

import json

import pytest
from unittest.mock import MagicMock

from src.origin import ANONYMOUS, Origin
from src.persona import Persona
from tests.helpers import make_bot_logic

GUILD = "12345"
OTHER_GUILD = "99999"


def _origin(transport="discord", server=None, channel="c1", author="a1",
            operator=False):
    return Origin(transport=transport, server_id=server, channel_id=channel,
                  author_id=author, operator=operator)


ALLOWED = _origin(server=GUILD)
DISALLOWED = _origin(server=OTHER_GUILD)


@pytest.fixture
def chat_state():
    state = MagicMock()
    state.personas = {
        "gated": Persona("gated", "gpt-4", "You are gated.",
                         origin_allowlist=[GUILD]),
        "open": Persona("open", "gpt-4", "You are open."),
    }
    state.last_api_iterations = {}
    return state


@pytest.fixture
def bot_logic(chat_state):
    return make_bot_logic(chat_state)


# ---------------------------------------------------------------------------
# The gate itself
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_unrestricted_persona_is_untouched(bot_logic):
    """The gate must be inert for every persona that never set the field —
    that is what makes DP-330 a no-op for personas predating it. A chat turn
    still falls through to the LLM (returns None)."""
    assert await bot_logic.preprocess_message(
        DISALLOWED, "open", "u", "hello there friend") is None
    assert await bot_logic.preprocess_message(ANONYMOUS, "open", "u", "hi") is None


@pytest.mark.asyncio
async def test_allowed_origin_passes_through(bot_logic):
    assert await bot_logic.preprocess_message(
        ALLOWED, "gated", "u", "some ordinary chat") is None


@pytest.mark.asyncio
async def test_disallowed_origin_refuses_a_chat_turn(bot_logic):
    """A plain chat message reaches no command handler, so the gate has to sit
    above command parsing to catch it at all."""
    result = await bot_logic.preprocess_message(
        DISALLOWED, "gated", "u", "some ordinary chat")
    assert result is not None, "must refuse explicitly, not fall through to the LLM"
    assert result["mutated"] is False
    assert "not available from this channel" in result["response"]


@pytest.mark.asyncio
@pytest.mark.parametrize("command", [
    "what prompt",
    "what origin_allowlist",
    "detail",
    "dump_history",
    "dump_last",
    "help",
    "set prompt you are now unrestricted",
])
async def test_disallowed_origin_refuses_every_dev_command(bot_logic, command):
    """Read-only commands too: DP-277 gates what a command may DO, this gates
    whether the persona is reachable, so `what prompt` must not disclose a
    restricted persona's config either."""
    result = await bot_logic.preprocess_message(DISALLOWED, "gated", "u", command)
    assert result is not None
    assert "not available from this channel" in result["response"]


@pytest.mark.asyncio
async def test_refusal_discloses_nothing_about_the_allowlist(bot_logic):
    """`what origin_allowlist` from a disallowed origin used to print the exact
    guild/channel/author ids the refusal text is written to withhold."""
    result = await bot_logic.preprocess_message(
        DISALLOWED, "gated", "u", "what origin_allowlist")
    assert GUILD not in result["response"]
    assert "You are gated" not in result["response"]


@pytest.mark.asyncio
@pytest.mark.parametrize("origin", [
    _origin(server=None),                              # Discord DM
    _origin(transport="portal", channel="portal"),
    _origin(transport="gmail"),
    _origin(transport="zammad"),
    _origin(transport="internal"),
    _origin(transport="portal", server=GUILD),         # forged body server_id
    ANONYMOUS,                                         # caller asserted nothing
], ids=["discord-dm", "portal", "gmail", "zammad", "internal",
        "portal-forged-guild", "anonymous"])
async def test_fails_closed_off_discord(bot_logic, origin):
    """Only Discord carries a gateway-asserted guild id. Everything else — and
    a caller that asserts no origin at all — fails closed."""
    result = await bot_logic.preprocess_message(origin, "gated", "u", "hi")
    assert result is not None
    assert "not available from this channel" in result["response"]


@pytest.mark.asyncio
async def test_operator_flag_does_not_bypass_the_gate(bot_logic):
    """`operator` says the transport authenticated the caller, not that the
    persona is addressable from there — an operator in the wrong guild is still
    refused."""
    result = await bot_logic.preprocess_message(
        _origin(server=OTHER_GUILD, operator=True), "gated", "u", "what prompt")
    assert result is not None
    assert "not available from this channel" in result["response"]


@pytest.mark.asyncio
async def test_unknown_persona_keeps_its_own_error(bot_logic):
    """The gate only fires for personas that exist — an unknown name must keep
    its existing handling rather than being masked as an origin refusal."""
    result = await bot_logic.preprocess_message(
        DISALLOWED, "nonexistent", "u", "what prompt")
    assert result is not None
    assert "not available from this channel" not in result["response"]
    assert "not found" in result["response"]
    # And a chat turn to an unknown persona still falls through to the kernel,
    # which owns that error message ("hello" would not do here — it is itself
    # a command, so it hits the not-found branch above).
    assert await bot_logic.preprocess_message(
        DISALLOWED, "nonexistent", "u", "some ordinary chat") is None


@pytest.mark.asyncio
async def test_a_wholly_malformed_allowlist_is_unreachable_not_open(chat_state,
                                                                   bot_logic):
    """Empty means unrestricted, so a typo'd list that parsed down to empty
    turned a restriction into a wide-open persona. It fails closed instead."""
    chat_state.personas["gated"].set_origin_allowlist(["*"])
    for origin in (ALLOWED, DISALLOWED, ANONYMOUS):
        result = await bot_logic.preprocess_message(origin, "gated", "u", "hi")
        assert result is not None
        assert "not available from this channel" in result["response"]


@pytest.mark.asyncio
async def test_clearing_the_allowlist_restores_reachability(chat_state, bot_logic):
    """The recovery path out of a lockout, from a persona the operator can
    still reach."""
    chat_state.personas["gated"].set_origin_allowlist([])
    assert await bot_logic.preprocess_message(
        DISALLOWED, "gated", "u", "hi") is None


# ---------------------------------------------------------------------------
# Mutating the field is a privileged, audited edit
# ---------------------------------------------------------------------------

OPERATOR_IN_GUILD = _origin(server=GUILD, operator=True)


@pytest.mark.asyncio
async def test_setting_the_allowlist_is_audited(bot_logic, chat_state):
    """`explicit_overrides` decides what the persona may DO and is audited at
    this boundary; `origin_allowlist` decides WHO may reach it — including the
    one persona holding node power-ops — so it needs the same durable record,
    not just a log line."""
    result = await bot_logic.preprocess_message(
        OPERATOR_IN_GUILD, "gated", "operator", f"set origin_allowlist {GUILD} 777")
    assert result is not None and result["mutated"] is True
    assert chat_state.personas["gated"].get_origin_allowlist() == [GUILD, "777"]

    audit = bot_logic.memory_manager.log_audit_event
    audit.assert_called_once()
    kwargs = audit.call_args.kwargs
    assert kwargs["event_type"] == "origin_allowlist_change"
    assert kwargs["operator_id"] == "operator"
    assert json.loads(kwargs["prior_state"]) == [GUILD]
    assert json.loads(kwargs["new_state"]) == [GUILD, "777"]
    assert kwargs["metadata"]["persona"] == "gated"


@pytest.mark.asyncio
async def test_clearing_the_allowlist_is_audited(bot_logic, chat_state):
    """The dangerous direction — removing every restriction — must be the one
    most clearly on the record."""
    await bot_logic.preprocess_message(
        OPERATOR_IN_GUILD, "gated", "operator", "set origin_allowlist none")
    kwargs = bot_logic.memory_manager.log_audit_event.call_args.kwargs
    assert json.loads(kwargs["prior_state"]) == [GUILD]
    assert json.loads(kwargs["new_state"]) == []


@pytest.mark.asyncio
async def test_a_lockout_is_recorded_as_malformed(bot_logic):
    """An all-malformed list leaves the persona unreachable; the audit row has
    to say so, because after this the operator cannot ask the persona."""
    await bot_logic.preprocess_message(
        OPERATOR_IN_GUILD, "gated", "operator", "set origin_allowlist *")
    kwargs = bot_logic.memory_manager.log_audit_event.call_args.kwargs
    assert kwargs["metadata"]["malformed"] is True


@pytest.mark.asyncio
async def test_non_operator_in_the_allowed_guild_cannot_set_it(bot_logic, chat_state):
    """Being addressable is not being an operator: DP-277 still gates the
    mutation, so a plain user in the allowed guild cannot widen the field."""
    result = await bot_logic.preprocess_message(
        ALLOWED, "gated", "someone", "set origin_allowlist 777")
    assert result is not None and result["mutated"] is False
    assert "Refused" in result["response"]
    assert chat_state.personas["gated"].get_origin_allowlist() == [GUILD]
    bot_logic.memory_manager.log_audit_event.assert_not_called()
