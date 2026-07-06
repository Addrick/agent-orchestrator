# tests/test_message_handler_edge_cases.py
"""DP-199 Batch 6: message_handler verb coverage edge cases.

Exercises the set/what/dump/verb/trust/untrust families on BotLogic,
including the dotted-path provider_extras setter (Phase E).

Pattern reused from tests/test_message_handler.py:
- MagicMock(spec=ChatSystem) with a real `personas` dict and real Persona objects.
- `preprocess_message` for end-to-end verb dispatch (note: it lowercases input).
- Direct handler calls (_set_X) when raw casing or signature exposure matters.
"""

from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from src.message_handler import BotLogic
from src.persona import ExecutionMode, MemoryMode, Persona
from tests.helpers import make_bot_logic, OPERATOR_ORIGIN


# --- Fixtures ---


@pytest.fixture
def chat_system():
    """Mutable state bucket BotLogic's explicit deps close over (DP-202)."""
    cs = MagicMock()
    cs.personas = {
        "derpr": Persona("derpr", "gpt-4", "You are derpr.", history_messages=20),
    }
    cs.models_available = {
        "From Google": ["gemini-2.5-flash", "gemini-2.5-pro"],
        "From Anthropic": ["claude-opus-4-7"],
    }
    cs.last_api_requests = {}
    cs.memory_manager = MagicMock()
    cs.tool_manager = MagicMock()
    return cs


@pytest.fixture
def bot(chat_system):
    return make_bot_logic(chat_system)


@pytest.fixture
def bot_with_tools(chat_system):
    """BotLogic with a mock tool_manager exposing several tool definitions."""
    tm = MagicMock()
    tm.get_tool_definitions.return_value = [
        {"type": "function", "function": {"name": "web_search"}},
        {"type": "function", "function": {"name": "create_ticket"}},
        {"type": "function", "function": {"name": "search_tickets"}},
        {"type": "function", "function": {"name": "google_grounding_search"}},
    ]
    chat_system.tool_manager = tm
    return make_bot_logic(chat_system)


def _persona(bot: BotLogic) -> Persona:
    return bot.personas()["derpr"]


# =============================================================================
# Set-family — dotted-path (Phase E) edge cases
# =============================================================================


@pytest.mark.asyncio
async def test_dotted_path_max_nesting_limit(bot):
    """Dotted path with multiple dots: partition('.') only splits on the first."""
    result = await bot.preprocess_message(OPERATOR_ORIGIN, "derpr", "u1", "set kobold.nested.key 42")
    assert result is not None and result["mutated"] is True
    # provider='kobold', key='nested.key' (everything after first dot)
    assert _persona(bot).get_provider_extra("kobold", "nested.key") == 42


@pytest.mark.asyncio
async def test_dotted_path_empty_provider(bot):
    """A leading dot like '.key' is routed to the dotted setter and rejected."""
    result = await bot.preprocess_message(OPERATOR_ORIGIN, "derpr", "u1", "set .key 1")
    assert result is not None and result["mutated"] is False
    assert "Invalid dotted path" in result["response"]


@pytest.mark.asyncio
async def test_set_provider_extra_clear_missing(bot):
    """Clearing an unset key returns mutated=False (not an error)."""
    result = await bot.preprocess_message(OPERATOR_ORIGIN, "derpr", "u1", "set kobold.never_set none")
    assert result is not None
    assert result["mutated"] is False
    assert "was not set" in result["response"]


def test_coerce_extra_value_type_chain():
    """_coerce_extra_value tries int → float → bool → str in order."""
    assert BotLogic._coerce_extra_value("42") == 42
    assert isinstance(BotLogic._coerce_extra_value("42"), int)
    assert BotLogic._coerce_extra_value("3.14") == 3.14
    assert isinstance(BotLogic._coerce_extra_value("3.14"), float)
    assert BotLogic._coerce_extra_value("true") is True
    assert BotLogic._coerce_extra_value("on") is True
    assert BotLogic._coerce_extra_value("yes") is True
    assert BotLogic._coerce_extra_value("false") is False
    assert BotLogic._coerce_extra_value("off") is False
    assert BotLogic._coerce_extra_value("no") is False
    assert BotLogic._coerce_extra_value("topk") == "topk"  # string fallback
    # Numeric-looking edge: leading '+' is accepted by int()
    assert BotLogic._coerce_extra_value("+5") == 5
    # Negative ints
    assert BotLogic._coerce_extra_value("-7") == -7


# =============================================================================
# Set-family — generic unknown / bare
# =============================================================================


@pytest.mark.asyncio
async def test_set_unknown_subcommand(bot):
    """Unknown set subcommand (no dot) returns error message."""
    result = await bot.preprocess_message(OPERATOR_ORIGIN, "derpr", "u1", "set bogus value")
    assert result is not None and result["mutated"] is False
    assert "Unknown 'set' command" in result["response"]


@pytest.mark.asyncio
async def test_set_bare_no_args(bot):
    """`set` with no subcommand falls through (returns None)."""
    result = await bot.preprocess_message(OPERATOR_ORIGIN, "derpr", "u1", "set")
    assert result is None


# =============================================================================
# Set-family — tokens / temp / top_p / top_k numeric handling
# =============================================================================


@pytest.mark.asyncio
async def test_set_tokens_non_numeric_fallback(bot):
    persona = _persona(bot)
    persona.set_response_token_limit(9999)  # known non-default
    result = await bot.preprocess_message(OPERATOR_ORIGIN, "derpr", "u1", "set tokens abc")
    assert result is not None and result["mutated"] is True
    assert "Non-numeric token limit" in result["response"]
    # Fallback resets away from the explicit override.
    assert persona.get_response_token_limit() != 9999


@pytest.mark.asyncio
async def test_set_temp_out_of_range(bot):
    result = await bot.preprocess_message(OPERATOR_ORIGIN, "derpr", "u1", "set temp 5.0")
    assert result is not None and result["mutated"] is False
    assert "between 0 and 2" in result["response"]


@pytest.mark.asyncio
async def test_set_temp_non_numeric_fallback(bot):
    result = await bot.preprocess_message(OPERATOR_ORIGIN, "derpr", "u1", "set temp hot")
    assert result is not None and result["mutated"] is True
    assert "Non-numeric temperature" in result["response"]
    assert _persona(bot).get_temperature() is None


@pytest.mark.asyncio
async def test_set_top_p_out_of_range(bot):
    result = await bot.preprocess_message(OPERATOR_ORIGIN, "derpr", "u1", "set top_p 1.5")
    assert result is not None and result["mutated"] is False
    assert "between 0 and 1" in result["response"]


@pytest.mark.asyncio
async def test_set_top_p_non_numeric_fallback(bot):
    result = await bot.preprocess_message(OPERATOR_ORIGIN, "derpr", "u1", "set top_p high")
    assert result is not None and result["mutated"] is True
    assert "Non-numeric Top P" in result["response"]
    assert _persona(bot).get_top_p() is None


@pytest.mark.asyncio
async def test_set_top_k_negative_no_validation(bot):
    """top_k accepts negative ints (no range guard) — documents current behavior.

    This is bounded behavior, not a latent bug: top_k is provider-validated downstream.
    """
    result = await bot.preprocess_message(OPERATOR_ORIGIN, "derpr", "u1", "set top_k -5")
    assert result is not None and result["mutated"] is True
    assert _persona(bot).get_top_k() == -5


@pytest.mark.asyncio
async def test_set_top_k_non_integer_fallback(bot):
    result = await bot.preprocess_message(OPERATOR_ORIGIN, "derpr", "u1", "set top_k 3.14")
    assert result is not None and result["mutated"] is True
    assert "Non-numeric Top K" in result["response"]
    assert _persona(bot).get_top_k() is None


# =============================================================================
# Set-family — enum modes
# =============================================================================


@pytest.mark.asyncio
async def test_set_execution_mode_invalid_enum(bot):
    result = await bot.preprocess_message(OPERATOR_ORIGIN, "derpr", "u1", "set execution_mode bogus")
    assert result is not None and result["mutated"] is False
    assert "Invalid execution mode" in result["response"]


@pytest.mark.asyncio
async def test_set_memory_mode_invalid_enum(bot):
    result = await bot.preprocess_message(OPERATOR_ORIGIN, "derpr", "u1", "set memory_mode nope")
    assert result is not None and result["mutated"] is False
    assert "Invalid memory mode" in result["response"]


# =============================================================================
# Set-family — tools
# =============================================================================


@pytest.mark.asyncio
async def test_set_tools_none_keyword(bot_with_tools):
    persona = _persona(bot_with_tools)
    persona.set_enabled_tools(["web_search"])
    result = await bot_with_tools.preprocess_message(OPERATOR_ORIGIN, "derpr", "u1", "set tools none")
    assert result["mutated"] is True
    assert persona.get_enabled_tools() == []


@pytest.mark.asyncio
async def test_set_tools_all_bare_names_error(bot_with_tools):
    """Bare names after 'all' (without '-') are an error."""
    result = await bot_with_tools.preprocess_message(OPERATOR_ORIGIN, 
        "derpr", "u1", "set tools all web_search"
    )
    assert result["mutated"] is False
    assert "'-' prefix" in result["response"]


@pytest.mark.asyncio
async def test_set_tools_excludes_without_all(bot_with_tools):
    """A bare '-tool' without 'all' yields the exclude-requires-all error."""
    result = await bot_with_tools.preprocess_message(OPERATOR_ORIGIN, 
        "derpr", "u1", "set tools -web_search"
    )
    assert result["mutated"] is False
    assert "requires 'all'" in result["response"]


@pytest.mark.asyncio
async def test_set_tools_all_wildcard(bot_with_tools):
    persona = _persona(bot_with_tools)
    result = await bot_with_tools.preprocess_message(OPERATOR_ORIGIN, "derpr", "u1", "set tools all")
    assert result["mutated"] is True
    assert persona.get_enabled_tools() == ["*"]


@pytest.mark.asyncio
async def test_set_tools_fuzzy_match(bot_with_tools):
    """A fuzzy name routes through the LLM selector and resolves."""
    persona = _persona(bot_with_tools)
    with patch.object(
        bot_with_tools,
        "_query_llm_for_tool_selection",
        new_callable=AsyncMock,
        return_value="google_grounding_search",
    ):
        result = await bot_with_tools.preprocess_message(OPERATOR_ORIGIN, 
            "derpr", "u1", "set tools grounding"
        )
    assert result["mutated"] is True
    assert persona.get_enabled_tools() == ["google_grounding_search"]
    assert "fuzzy" in result["response"]


# =============================================================================
# Set-family — service_bindings
# =============================================================================


@pytest.mark.asyncio
async def test_set_service_bindings_clear_keyword(bot):
    persona = _persona(bot)
    persona.set_service_bindings(["zammad", "agents"])
    result = await bot.preprocess_message(OPERATOR_ORIGIN, "derpr", "u1", "set service_bindings none")
    assert result["mutated"] is True
    assert persona.get_service_bindings() == []


def test_set_service_bindings_csv_whitespace(bot):
    """Direct handler call preserves casing; verifies CSV+whitespace parsing."""
    persona = _persona(bot)
    resp, mutated = bot.set_handlers["service_bindings"](
        ["service_bindings", "zammad, agents , notifier"], persona
    )
    assert mutated is True
    assert persona.get_service_bindings() == ["zammad", "agents", "notifier"]


# =============================================================================
# Set-family — long_term_memory / include_ambient_memory variants
# =============================================================================


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "value, expected",
    [("on", True), ("true", True), ("yes", True), ("1", True),
     ("off", False), ("false", False), ("no", False), ("0", False)],
)
async def test_set_long_term_memory_variants(bot, value, expected):
    result = await bot.preprocess_message(OPERATOR_ORIGIN, 
        "derpr", "u1", f"set long_term_memory {value}"
    )
    assert result["mutated"] is True
    assert _persona(bot).get_long_term_memory() is expected


@pytest.mark.asyncio
async def test_set_long_term_memory_invalid_value(bot):
    result = await bot.preprocess_message(OPERATOR_ORIGIN, 
        "derpr", "u1", "set long_term_memory maybe"
    )
    assert result["mutated"] is False
    assert "Invalid value" in result["response"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "value, expected",
    [("on", True), ("true", True), ("off", False), ("no", False)],
)
async def test_set_include_ambient_memory_variants(bot, value, expected):
    result = await bot.preprocess_message(OPERATOR_ORIGIN, 
        "derpr", "u1", f"set include_ambient_memory {value}"
    )
    assert result["mutated"] is True
    assert _persona(bot).get_include_ambient_memory() is expected


# =============================================================================
# Set-family — max_context_tokens
# =============================================================================


@pytest.mark.asyncio
async def test_set_max_context_tokens_non_numeric(bot):
    """Non-numeric input falls back to global default (still mutates)."""
    result = await bot.preprocess_message(OPERATOR_ORIGIN, 
        "derpr", "u1", "set max_context_tokens lots"
    )
    assert result["mutated"] is True
    from config import global_config
    assert _persona(bot).get_max_context_tokens() == global_config.DEFAULT_MAX_CONTEXT_TOKENS


@pytest.mark.asyncio
async def test_set_max_context_tokens_low_clamp(bot):
    """Values below 100 clamp to 100."""
    result = await bot.preprocess_message(OPERATOR_ORIGIN, "derpr", "u1", "set max_context_tokens 50")
    assert result["mutated"] is True
    assert _persona(bot).get_max_context_tokens() == 100


# =============================================================================
# Set-family — thinking_level / chat_template clear
# =============================================================================


@pytest.mark.asyncio
async def test_set_thinking_level_none_clear(bot):
    persona = _persona(bot)
    persona.set_thinking_level("minimal")
    result = await bot.preprocess_message(OPERATOR_ORIGIN, "derpr", "u1", "set thinking_level none")
    assert result["mutated"] is True
    assert persona.get_thinking_level() is None


@pytest.mark.asyncio
@pytest.mark.parametrize("keyword", ["none", "null", "clear"])
async def test_set_chat_template_clear_keyword(bot, keyword):
    persona = _persona(bot)
    persona.set_chat_template("chatml")
    result = await bot.preprocess_message(OPERATOR_ORIGIN, "derpr", "u1", f"set chat_template {keyword}")
    assert result["mutated"] is True
    assert persona.get_chat_template() is None


# =============================================================================
# Set-family — tool_policy
# =============================================================================


def test_set_tool_policy_invalid_json(bot):
    """Direct call: malformed JSON returns error, no mutation."""
    persona = _persona(bot)
    resp, mutated = bot.set_handlers["tool_policy"](["tool_policy", "{not json"], persona)
    assert mutated is False
    assert "Invalid JSON" in resp


def test_set_tool_policy_from_dict_error(bot):
    """A valid-JSON value that ToolPolicy.from_dict rejects (ValueError) surfaces error.

    The handler only catches ValueError; ToolPolicy accepts arbitrary kwargs and does
    not validate the 'default' field, so we simulate the error path by patching
    Persona.set_tool_policy to raise ValueError.
    """
    persona = _persona(bot)
    with patch.object(Persona, "set_tool_policy", side_effect=ValueError("bad policy")):
        resp, mutated = bot.set_handlers["tool_policy"](
            ["tool_policy", '{"default":"deny"}'], persona
        )
    assert mutated is False
    assert "bad policy" in resp


# =============================================================================
# What-family
# =============================================================================


@pytest.mark.asyncio
async def test_what_bare_no_args(bot):
    """`what` with no subcommand returns None (falls through)."""
    result = await bot.preprocess_message(OPERATOR_ORIGIN, "derpr", "u1", "what")
    assert result is None


@pytest.mark.asyncio
async def test_what_models_too_many_args(bot):
    """`what models foo bar` — more than 2 args returns None."""
    result = await bot.preprocess_message(OPERATOR_ORIGIN, "derpr", "u1", "what models google extra")
    assert result is None


@pytest.mark.asyncio
async def test_what_models_vendor_no_match(bot):
    """Vendor filter with no matching key returns None."""
    result = await bot.preprocess_message(OPERATOR_ORIGIN, "derpr", "u1", "what models nonexistent")
    assert result is None


# =============================================================================
# Dump / lifecycle verbs
# =============================================================================


@pytest.mark.asyncio
async def test_dump_last_no_history(bot):
    """dump_last with no recorded request returns informative message, not crash."""
    result = await bot.preprocess_message(OPERATOR_ORIGIN, "derpr", "u1", "dump_last")
    assert result is not None and result["mutated"] is False
    assert "No previous request" in result["response"]


@pytest.mark.asyncio
async def test_hello_with_args(bot):
    """`hello` accepts no args — falls through to LLM (returns None)."""
    result = await bot.preprocess_message(OPERATOR_ORIGIN, "derpr", "u1", "hello there")
    assert result is None


@pytest.mark.asyncio
async def test_goodbye_with_args(bot):
    """`goodbye` accepts no args — falls through."""
    result = await bot.preprocess_message(OPERATOR_ORIGIN, "derpr", "u1", "goodbye now")
    assert result is None


# =============================================================================
# update_models verb
# =============================================================================


@pytest.mark.asyncio
async def test_update_models_returns_none(bot, chat_system):
    """update_models returns mutated=False and refreshes models_available."""
    with patch(
        "src.message_handler.get_model_list",
        return_value={"From Google": ["gemini-9"]},
    ):
        result = await bot.preprocess_message(OPERATOR_ORIGIN, "derpr", "u1", "update_models")
    assert result is not None
    assert result["mutated"] is False
    assert "Model list updated" in result["response"]
    assert chat_system.models_available == {"From Google": ["gemini-9"]}


@pytest.mark.asyncio
async def test_update_models_with_args(bot):
    """update_models with extra args returns the Usage hint, no refresh."""
    with patch("src.message_handler.get_model_list") as mock_get:
        result = await bot.preprocess_message(OPERATOR_ORIGIN, "derpr", "u1", "update_models foo")
    assert result is not None and result["mutated"] is False
    assert "Usage" in result["response"]
    mock_get.assert_not_called()


# =============================================================================
# trust / untrust
# =============================================================================


@pytest.mark.asyncio
async def test_trust_non_integer_id(bot, chat_system):
    result = await bot.preprocess_message(OPERATOR_ORIGIN, "derpr", "u1", "trust abc some reason")
    assert result is not None and result["mutated"] is False
    assert "Invalid summary_id" in result["response"]
    chat_system.memory_manager.mark_trusted.assert_not_called()


@pytest.mark.asyncio
async def test_trust_not_found(bot, chat_system):
    """If mark_trusted returns False (not found), surface error and no mutation."""
    chat_system.memory_manager.mark_trusted.return_value = False
    result = await bot.preprocess_message(OPERATOR_ORIGIN, "derpr", "u1", "trust 999 some reason")
    assert result is not None and result["mutated"] is False
    assert "Could not find or update memory 999" in result["response"]
    chat_system.memory_manager.mark_trusted.assert_called_once()


@pytest.mark.asyncio
async def test_untrust_non_integer_id(bot, chat_system):
    """Symmetry check on untrust: non-int id rejected without backend call."""
    result = await bot.preprocess_message(OPERATOR_ORIGIN, "derpr", "u1", "untrust xyz reason")
    assert result is not None and result["mutated"] is False
    assert "Invalid summary_id" in result["response"]
    chat_system.memory_manager.mark_untrusted.assert_not_called()


@pytest.mark.asyncio
async def test_untrust_success_mutates(bot, chat_system):
    """Successful untrust marks mutated=True."""
    chat_system.memory_manager.mark_untrusted.return_value = True
    result = await bot.preprocess_message(OPERATOR_ORIGIN, "derpr", "u1", "untrust 7 stale data")
    assert result is not None and result["mutated"] is True
    assert "marked as UNTRUSTED" in result["response"]
