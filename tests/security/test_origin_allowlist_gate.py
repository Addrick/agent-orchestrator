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
from src.persona_fields import _describe_origin_allowlist
from src.personas import store
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
    """Clearing the field re-opens the persona. Unit-level only — this asserts
    the setter's effect, NOT that any operator can still deliver it. See
    `test_clearing_a_different_persona_does_not_recover_the_locked_out_one`
    for what the routing actually does."""
    chat_state.personas["gated"].set_origin_allowlist([])
    assert await bot_logic.preprocess_message(
        DISALLOWED, "gated", "u", "hi") is None


# ---------------------------------------------------------------------------
# Fail-closed, for every shape a hand-edited JSON file can produce
#
# `[]` means UNRESTRICTED, so any input the normalizer diagnoses as unusable
# and then reduces to `[]` inverts the field. The first cut only fail-closed
# on entries that survived `str()`, which is none of the shapes below.
# ---------------------------------------------------------------------------

UNUSABLE_ALLOWLISTS = [
    pytest.param([None], id="null-entry"),
    pytest.param([True], id="bool-entry"),
    pytest.param([["12345"]], id="nested-list-entry"),
    pytest.param([None, ["12345"], True], id="all-entries-unstringifiable"),
    pytest.param(["", "   "], id="blank-entries"),
    pytest.param(["*"], id="wildcard-server"),
    pytest.param({"guild": "12345"}, id="dict-instead-of-list"),
    pytest.param(12345, id="bare-int-instead-of-list"),
]


@pytest.mark.parametrize("value", UNUSABLE_ALLOWLISTS)
def test_an_unusable_allowlist_is_unreachable_never_unrestricted(value):
    """The core inversion. Every one of these was diagnosed as bad AND then
    admitted every origin in the world, because the fail-closed branch keyed
    off entries that had survived stringification — which for most of these is
    none of them."""
    persona = Persona("p", "m", "pr", origin_allowlist=value)
    assert persona.origin_allowlist_is_unreachable() is True
    assert persona.origin_allowlist_is_malformed() is True
    for origin in (ALLOWED, DISALLOWED, ANONYMOUS, _origin(transport="portal")):
        assert persona.is_addressable_from(origin) is False


@pytest.mark.parametrize("value", UNUSABLE_ALLOWLISTS)
def test_an_unusable_allowlist_survives_a_save_load_round_trip(value):
    """A fail-closed persona must still be fail-closed after the next restart.
    `to_dict` used to write the NORMALIZED list, which drops what it cannot
    stringify — so `[null]` saved as `[]`, and `[]` is unrestricted. Any
    mutating dev command (`set temp 0.8`) rewrites personas.json, so the
    unreachable state lasted exactly until the next unrelated edit."""
    persona = Persona("p", "m", "pr", origin_allowlist=value)
    saved = store.to_dict({"p": persona})[0]
    assert "origin_allowlist" in saved
    assert saved["origin_allowlist"] != []

    reloaded = Persona("p", "m", "pr",
                       origin_allowlist=saved["origin_allowlist"])
    assert reloaded.origin_allowlist_is_unreachable() is True
    for origin in (ALLOWED, DISALLOWED, ANONYMOUS):
        assert reloaded.is_addressable_from(origin) is False


def test_a_usable_allowlist_still_round_trips_as_the_normalized_list():
    """The raw value is persisted ONLY when the persona failed closed —
    otherwise the normalized entries are still what lands on disk, so ints
    keep coming back as the strings everything downstream expects."""
    persona = Persona("p", "m", "pr", origin_allowlist=[12345, " 777 "])
    assert store.to_dict({"p": persona})[0]["origin_allowlist"] == ["12345", "777"]


def test_a_partly_malformed_allowlist_keeps_the_entries_that_parsed():
    """Malformed is not the same as unreachable: some entries in force and the
    rest dropped is a real, non-terminal state, and conflating the two is what
    made the reporting wrong."""
    persona = Persona("p", "m", "pr", origin_allowlist=[GUILD, "*", ""])
    assert persona.origin_allowlist_is_malformed() is True
    assert persona.origin_allowlist_is_unreachable() is False
    # The authored list KEEPS the rejected entry — it is what gets written back
    # to the file, and dropping the typo would hide what has to be fixed. Which
    # is exactly why `rejected` has to be tracked separately: `"*"` is stored
    # but is not a grant.
    assert persona.get_origin_allowlist() == [GUILD, "*"]
    assert persona.get_origin_allowlist_rejected() == ["*", ""]
    assert persona.is_addressable_from(ALLOWED) is True
    assert persona.is_addressable_from(DISALLOWED) is False


def test_a_partly_malformed_allowlist_does_not_report_dropped_entries_as_grants():
    """`what origin_allowlist` prints the authored list, which includes entries
    that were rejected. Printing it alone credits the persona with a grant it
    is not enforcing — the same 'describes a policy the persona is not running'
    failure as the unrestricted/unreachable mixup, one severity down."""
    persona = Persona("p", "m", "pr", origin_allowlist=[GUILD, "*"])
    described = _describe_origin_allowlist(persona)
    assert "NOT in force" in described
    assert "'*'" in described


# ---------------------------------------------------------------------------
# What the operator is TOLD about the state they are in
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("value", UNUSABLE_ALLOWLISTS)
def test_an_unreachable_persona_is_not_reported_as_unrestricted(value):
    """`what origin_allowlist` checked malformed only INSIDE its non-empty
    branch, so every shape above — including the ones where the persona
    matches no origin at all — printed 'unrestricted (any origin may address
    it)'. The operator was told the exact opposite of the truth.

    Called directly, not through `preprocess_message`, because the gate refuses
    the addressing attempt before the `what` handler runs: once a persona is
    unreachable you cannot ask it about itself either. The string still has one
    live delivery path — the setter's reply on the turn that caused it (see
    `test_locking_yourself_out_says_the_command_cannot_undo_it`) — and this
    pins the branch for every other caller that renders field state."""
    persona = Persona("p", "m", "pr", origin_allowlist=value)
    described = _describe_origin_allowlist(persona)
    assert "UNREACHABLE" in described
    assert "unrestricted (any origin may address it)" not in described


@pytest.mark.asyncio
async def test_an_unreachable_persona_cannot_even_be_queried(chat_state, bot_logic):
    """The corollary, pinned because it is the operator's actual experience and
    the reason the setter's reply has to carry the full recovery instructions:
    after a lockout, `what origin_allowlist` is refused like everything else."""
    chat_state.personas["gated"].set_origin_allowlist(["*"])
    result = await bot_logic.preprocess_message(
        OPERATOR_IN_GUILD, "gated", "operator", "what origin_allowlist")
    assert result is not None
    assert "not available from this channel" in result["response"]


@pytest.mark.asyncio
async def test_locking_yourself_out_says_the_command_cannot_undo_it(bot_logic):
    """The setter used to offer `set origin_allowlist none` as the fix. That
    command can only arrive through the persona it just made unreachable."""
    result = await bot_logic.preprocess_message(
        OPERATOR_IN_GUILD, "gated", "operator", "set origin_allowlist *")
    assert result is not None
    assert "UNREACHABLE" in result["response"]
    assert "data/personas.json" in result["response"]


@pytest.mark.asyncio
async def test_clearing_a_different_persona_does_not_recover_the_locked_out_one(
        chat_state, bot_logic):
    """`set origin_allowlist` targets the ADDRESSED persona — there is no
    cross-persona form. user_guide.md, architecture.md, the setter docstring
    and a test docstring all told operators to recover from a lockout by
    'running the command while addressing a different one', which removes a
    real restriction from an unrelated persona and leaves the locked-out one
    exactly as locked. Nothing exercised the routing, so the claim survived."""
    chat_state.personas["open"].set_origin_allowlist([GUILD])
    await bot_logic.preprocess_message(
        OPERATOR_IN_GUILD, "gated", "operator", "set origin_allowlist *")
    assert chat_state.personas["gated"].origin_allowlist_is_unreachable() is True

    # The documented "recovery", performed exactly as written.
    result = await bot_logic.preprocess_message(
        OPERATOR_IN_GUILD, "open", "operator", "set origin_allowlist none")
    assert result is not None and result["mutated"] is True

    # It cleared the WRONG persona and did nothing for the locked-out one.
    assert chat_state.personas["open"].get_origin_allowlist() == []
    assert chat_state.personas["gated"].origin_allowlist_is_unreachable() is True
    refused = await bot_logic.preprocess_message(
        ALLOWED, "gated", "u", "hi")
    assert refused is not None
    assert "not available from this channel" in refused["response"]


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
async def test_clearing_a_field_that_was_never_set_is_not_a_mutation(
        bot_logic, chat_state):
    """`mutated` drives the audit row AND a full personas.json rewrite, so it
    has to mean "the field changed", not "a set command ran".

    `set origin_allowlist none` on a persona that never carried the key used to
    report True: an audit row with identical prior/new state, and
    `origin_allowlist_is_declared()` flipped forever so `to_dict` began emitting
    `"origin_allowlist": []` for a persona whose policy did not change. Audit
    rows for changes that did not happen dilute the one signal this ticket
    added for spotting a real widening."""
    assert chat_state.personas["open"].origin_allowlist_is_declared() is False
    result = await bot_logic.preprocess_message(
        OPERATOR_IN_GUILD, "open", "operator", "set origin_allowlist none")
    assert result is not None and result["mutated"] is False
    bot_logic.memory_manager.log_audit_event.assert_not_called()
    # And the persona's on-disk shape is untouched — no key appears.
    assert chat_state.personas["open"].origin_allowlist_is_declared() is False
    assert "origin_allowlist" not in store.to_dict(
        {"open": chat_state.personas["open"]})[0]


@pytest.mark.asyncio
async def test_setting_the_same_allowlist_again_is_not_a_mutation(
        bot_logic, chat_state):
    """Re-asserting the value already in force changes no policy, so it earns
    no audit row and no file rewrite."""
    result = await bot_logic.preprocess_message(
        OPERATOR_IN_GUILD, "gated", "operator", f"set origin_allowlist {GUILD}")
    assert result is not None and result["mutated"] is False
    bot_logic.memory_manager.log_audit_event.assert_not_called()
    assert chat_state.personas["gated"].get_origin_allowlist() == [GUILD]


@pytest.mark.asyncio
async def test_an_explicitly_declared_empty_allowlist_still_round_trips(
        bot_logic, chat_state):
    """The no-op guard must not undo the reason `declared` exists: a persona
    SHIPPED with `"origin_allowlist": []` keeps the key, which is the
    operator's only in-file hint that the knob exists."""
    chat_state.personas["shipped"] = Persona("shipped", "m", "pr",
                                             origin_allowlist=[])
    await bot_logic.preprocess_message(
        OPERATOR_IN_GUILD, "shipped", "operator", "set origin_allowlist none")
    saved = store.to_dict({"shipped": chat_state.personas["shipped"]})[0]
    assert saved["origin_allowlist"] == []


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
