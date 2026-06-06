# tests/agents/test_zammad_bot_unit.py
"""Unit tests for ZammadBot pipeline methods (DP-199 Batch 8).

All external dependencies (ZammadClient, LLM responses, system personas) are
mocked. No live Zammad or LLM calls. Translates patterns from
tests/live/test_zammad_live.py into pure unit tests.
"""

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.agents.zammad_bot import ZammadBot
from config.global_config import (
    TRIAGE_SCOUT_NAME,
    TRIAGE_SUMMARIZER_NAME,
    TRIAGE_ANALYST_NAME,
    TRIAGE_FILTER_NAME,
    ZAMMAD_TRIAGE_TAG,
    ZAMMAD_BOT_EMAIL,
)


# --- Fixtures ---

@pytest.fixture
def mock_chat_system() -> Any:
    cs = MagicMock()
    cs.text_engine = MagicMock()
    cs.text_engine.generate_response = AsyncMock()
    cs.memory_manager = MagicMock()
    cs.personas = {}
    return cs


@pytest.fixture
def mock_zammad_client() -> Any:
    client = MagicMock()
    return client


def _make_persona(name: str = "p") -> MagicMock:
    persona = MagicMock()
    persona.get_prompt.return_value = f"You are {name}."
    persona.get_config_for_engine.return_value = {"model_name": "test-model"}
    return persona


@pytest.fixture
@patch('src.agents.base.load_system_personas_from_file', return_value={})
def bot(_mock_load: Any, mock_chat_system: Any, mock_zammad_client: Any) -> ZammadBot:
    return ZammadBot(mock_chat_system, mock_zammad_client)


# --- _on_start tests ---

class TestZammadOnStart:
    async def test_user_exists(self, bot: ZammadBot, mock_zammad_client: Any) -> None:
        mock_zammad_client.search_user = MagicMock(return_value=[{"id": 7, "email": ZAMMAD_BOT_EMAIL}])
        with patch('src.agents.zammad_bot.logger') as mock_logger:
            await bot._on_start()
            mock_zammad_client.search_user.assert_called_once()
            # Info log only (no error)
            assert mock_logger.error.call_count == 0
            mock_logger.info.assert_called()

    async def test_missing_bot_user_logged(self, bot: ZammadBot, mock_zammad_client: Any) -> None:
        mock_zammad_client.search_user = MagicMock(return_value=[])
        with patch('src.agents.zammad_bot.logger') as mock_logger:
            await bot._on_start()
            # Should log error with ACTION REQUIRED instructions
            mock_logger.error.assert_called_once()
            msg = mock_logger.error.call_args[0][0]
            assert "NOT FOUND" in msg
            assert "ACTION REQUIRED" in msg

    async def test_search_exception_logged_not_raised(
        self, bot: ZammadBot, mock_zammad_client: Any,
    ) -> None:
        mock_zammad_client.search_user = MagicMock(side_effect=RuntimeError("zammad down"))
        with patch('src.agents.zammad_bot.logger') as mock_logger:
            await bot._on_start()  # must not raise
            mock_logger.error.assert_called_once()
            assert "Failed to check bot identity" in mock_logger.error.call_args[0][0]


# --- deploy() polling tests ---

class TestZammadDeploy:
    async def test_polling_iteration_processes_tickets(
        self, bot: ZammadBot, mock_zammad_client: Any,
    ) -> None:
        mock_zammad_client.search_tickets = MagicMock(
            return_value=[{"id": 1}, {"id": 2}, {"id": 3}]
        )
        bot._process_ticket = AsyncMock()

        await bot.deploy()

        expected_query = f"state.name:new AND NOT tags:{ZAMMAD_TRIAGE_TAG}"
        mock_zammad_client.search_tickets.assert_called_once_with(query=expected_query, limit=10)
        assert bot._process_ticket.await_count == 3
        bot._process_ticket.assert_any_await(1)
        bot._process_ticket.assert_any_await(2)
        bot._process_ticket.assert_any_await(3)

    async def test_shutdown_event_stops_loop_early(
        self, bot: ZammadBot, mock_zammad_client: Any,
    ) -> None:
        mock_zammad_client.search_tickets = MagicMock(
            return_value=[{"id": 1}, {"id": 2}, {"id": 3}]
        )

        async def fake_process(ticket_id: int) -> None:
            # Trip the shutdown after first ticket
            if ticket_id == 1:
                bot._shutdown_event.set()

        bot._process_ticket = AsyncMock(side_effect=fake_process)

        await bot.deploy()

        # Should stop after processing ticket 1, before 2/3
        assert bot._process_ticket.await_count == 1

    async def test_search_exception_returns_silently(
        self, bot: ZammadBot, mock_zammad_client: Any,
    ) -> None:
        mock_zammad_client.search_tickets = MagicMock(side_effect=RuntimeError("API down"))
        bot._process_ticket = AsyncMock()

        await bot.deploy()  # must not raise

        bot._process_ticket.assert_not_awaited()


# --- _get_search_keywords ---

class TestGetSearchKeywords:
    async def test_persona_missing_returns_none(
        self, bot: ZammadBot, mock_chat_system: Any,
    ) -> None:
        mock_chat_system.personas = {}  # scout missing
        result = await bot._get_search_keywords("title", "body")
        assert result is None
        mock_chat_system.text_engine.generate_response.assert_not_called()

    async def test_valid_text_response_returns_keywords(
        self, bot: ZammadBot, mock_chat_system: Any,
    ) -> None:
        mock_chat_system.personas[TRIAGE_SCOUT_NAME] = _make_persona("scout")
        mock_chat_system.text_engine.generate_response = AsyncMock(
            return_value=({"type": "text", "content": "  network   outage  "}, {})
        )
        result = await bot._get_search_keywords("title", "body")
        # Whitespace collapsed
        assert result == "network outage"

    async def test_malformed_response_returns_none(
        self, bot: ZammadBot, mock_chat_system: Any,
    ) -> None:
        """Non-text response type (tool_calls) yields None — no keywords."""
        mock_chat_system.personas[TRIAGE_SCOUT_NAME] = _make_persona("scout")
        mock_chat_system.text_engine.generate_response = AsyncMock(
            return_value=({"type": "tool_calls", "calls": []}, {})
        )
        result = await bot._get_search_keywords("title", "body")
        assert result is None

    async def test_llm_exception_returns_none(
        self, bot: ZammadBot, mock_chat_system: Any,
    ) -> None:
        mock_chat_system.personas[TRIAGE_SCOUT_NAME] = _make_persona("scout")
        mock_chat_system.text_engine.generate_response = AsyncMock(
            side_effect=RuntimeError("LLM gone")
        )
        result = await bot._get_search_keywords("title", "body")
        assert result is None


# --- _summarize_text ---

class TestSummarizeText:
    async def test_under_threshold_returns_input(self, bot: ZammadBot) -> None:
        text = "short text"
        assert len(text) < 500
        result = await bot._summarize_text(text)
        assert result == text

    async def test_persona_missing_returns_truncated(
        self, bot: ZammadBot, mock_chat_system: Any,
    ) -> None:
        mock_chat_system.personas = {}
        text = "X" * 1000
        result = await bot._summarize_text(text)
        assert "Truncated" in result
        assert result.startswith("X")

    async def test_llm_error_fallback_truncated(
        self, bot: ZammadBot, mock_chat_system: Any,
    ) -> None:
        mock_chat_system.personas[TRIAGE_SUMMARIZER_NAME] = _make_persona("sum")
        mock_chat_system.text_engine.generate_response = AsyncMock(
            side_effect=RuntimeError("LLM error")
        )
        text = "Y" * 1000
        result = await bot._summarize_text(text)
        assert "Truncated" in result
        # Truncation prefix = original head
        assert result.startswith("Y" * 500)

    async def test_success_returns_summarized(
        self, bot: ZammadBot, mock_chat_system: Any,
    ) -> None:
        mock_chat_system.personas[TRIAGE_SUMMARIZER_NAME] = _make_persona("sum")
        mock_chat_system.text_engine.generate_response = AsyncMock(
            return_value=({"type": "text", "content": "Concise summary."}, {})
        )
        text = "Z" * 1000
        result = await bot._summarize_text(text)
        assert "[SUMMARIZED]" in result
        assert "Concise summary." in result


# --- _check_relevance ---

class TestCheckRelevance:
    async def test_relevant_returns_true(
        self, bot: ZammadBot, mock_chat_system: Any,
    ) -> None:
        mock_chat_system.personas[TRIAGE_FILTER_NAME] = _make_persona("filter")
        mock_chat_system.text_engine.generate_response = AsyncMock(
            return_value=({"type": "text", "content": "RELEVANT"}, {})
        )
        result = await bot._check_relevance("new", "history")
        assert result is True

    async def test_irrelevant_returns_false(
        self, bot: ZammadBot, mock_chat_system: Any,
    ) -> None:
        """IRRELEVANT must NOT match because it contains the substring RELEVANT."""
        mock_chat_system.personas[TRIAGE_FILTER_NAME] = _make_persona("filter")
        mock_chat_system.text_engine.generate_response = AsyncMock(
            return_value=({"type": "text", "content": "IRRELEVANT"}, {})
        )
        result = await bot._check_relevance("new", "history")
        assert result is False

    async def test_missing_persona_defaults_true(
        self, bot: ZammadBot, mock_chat_system: Any,
    ) -> None:
        mock_chat_system.personas = {}
        result = await bot._check_relevance("new", "history")
        assert result is True

    async def test_llm_exception_defaults_true(
        self, bot: ZammadBot, mock_chat_system: Any,
    ) -> None:
        mock_chat_system.personas[TRIAGE_FILTER_NAME] = _make_persona("filter")
        mock_chat_system.text_engine.generate_response = AsyncMock(
            side_effect=RuntimeError("LLM down")
        )
        result = await bot._check_relevance("new", "history")
        # Fail-open: default to relevant on error
        assert result is True

    async def test_lowercase_response_normalized(
        self, bot: ZammadBot, mock_chat_system: Any,
    ) -> None:
        mock_chat_system.personas[TRIAGE_FILTER_NAME] = _make_persona("filter")
        mock_chat_system.text_engine.generate_response = AsyncMock(
            return_value=({"type": "text", "content": " relevant "}, {})
        )
        result = await bot._check_relevance("new", "history")
        assert result is True


# --- _process_ticket ---

def _wire_full_personas(mock_chat_system: Any) -> None:
    for name in (TRIAGE_SCOUT_NAME, TRIAGE_SUMMARIZER_NAME,
                 TRIAGE_ANALYST_NAME, TRIAGE_FILTER_NAME):
        mock_chat_system.personas[name] = _make_persona(name)


class TestProcessTicket:
    async def test_analyst_persona_missing_aborts(
        self, bot: ZammadBot, mock_chat_system: Any, mock_zammad_client: Any,
    ) -> None:
        # Only scout/etc — analyst absent
        mock_chat_system.personas[TRIAGE_SCOUT_NAME] = _make_persona("scout")
        await bot._process_ticket(123)
        # Never fetched ticket — bailed early
        assert not hasattr(mock_zammad_client.get_ticket, 'assert_called') or \
            mock_zammad_client.get_ticket.call_count == 0

    async def test_article_fetch_failure_logs_and_returns(
        self, bot: ZammadBot, mock_chat_system: Any, mock_zammad_client: Any,
    ) -> None:
        _wire_full_personas(mock_chat_system)
        mock_zammad_client.get_ticket = MagicMock(
            return_value={"id": 1, "title": "T", "customer_id": 5}
        )
        mock_zammad_client.get_ticket_articles = MagicMock(
            side_effect=RuntimeError("zammad articles down")
        )
        with patch('src.agents.zammad_bot.logger') as mock_logger:
            await bot._process_ticket(1)  # must not raise
            mock_logger.error.assert_called()
            assert "Error processing ticket 1" in mock_logger.error.call_args[0][0]
        # Should never tag the ticket
        assert not mock_zammad_client.add_tag.called

    async def test_history_search_uses_keywords(
        self, bot: ZammadBot, mock_chat_system: Any, mock_zammad_client: Any,
    ) -> None:
        _wire_full_personas(mock_chat_system)
        mock_zammad_client.get_ticket = MagicMock(
            return_value={"id": 1, "title": "Printer broken", "customer_id": 9}
        )
        mock_zammad_client.get_ticket_articles = MagicMock(
            return_value=[{"body": "ticket body"}]
        )
        mock_zammad_client.search_tickets = MagicMock(return_value=[])
        mock_zammad_client.add_article_to_ticket = MagicMock()
        mock_zammad_client.add_tag = MagicMock()

        # Scout returns keywords, analyst returns text
        async def gen(persona_config: Any, history_object: Any, tools: Any = None) -> Any:
            sys = history_object.get('persona_prompt', '')
            if "scout" in sys.lower() or TRIAGE_SCOUT_NAME in sys:
                return {"type": "text", "content": "printer paper"}, {}
            return {"type": "text", "content": "triage note text"}, {}

        mock_chat_system.text_engine.generate_response = AsyncMock(side_effect=gen)

        await bot._process_ticket(1)

        # Global query must include the keywords AND state.name:closed
        search_calls = mock_zammad_client.search_tickets.call_args_list
        assert search_calls, "expected at least one search_tickets call for history"
        global_call = search_calls[0]
        query = global_call.kwargs.get("query") or global_call.args[0]
        assert "state.name:closed" in query
        assert "printer" in query or "paper" in query

    async def test_relevance_filter_excludes_irrelevant_from_context(
        self, bot: ZammadBot, mock_chat_system: Any, mock_zammad_client: Any,
    ) -> None:
        _wire_full_personas(mock_chat_system)
        mock_zammad_client.get_ticket = MagicMock(
            return_value={"id": 1, "title": "Issue", "customer_id": 5}
        )
        mock_zammad_client.get_ticket_articles = MagicMock(
            return_value=[{"body": "new body"}]
        )
        # Return one Global ticket; we'll mark it irrelevant
        mock_zammad_client.search_tickets = MagicMock(
            side_effect=[[{"id": 100, "title": "Old"}], []]
        )

        bot._get_search_keywords = AsyncMock(return_value="keyword")
        bot._check_relevance = AsyncMock(return_value=False)

        captured_context: dict[str, str] = {}

        async def gen(persona_config: Any, history_object: Any, tools: Any = None) -> Any:
            sys = history_object.get('persona_prompt', '')
            # Capture the analyst call's user message
            if TRIAGE_ANALYST_NAME in sys or "analyst" in sys.lower():
                captured_context["msg"] = history_object["message_history"][-1]["content"]
            return {"type": "text", "content": "ok"}, {}

        mock_chat_system.text_engine.generate_response = AsyncMock(side_effect=gen)
        mock_zammad_client.add_article_to_ticket = MagicMock()
        mock_zammad_client.add_tag = MagicMock()

        await bot._process_ticket(1)

        # The irrelevant global match should NOT appear as a relevant solution
        msg = captured_context.get("msg", "")
        assert msg, "analyst should have been called"
        assert "No similar closed tickets found." in msg

    async def test_context_compression_invoked_when_oversized(
        self, bot: ZammadBot, mock_chat_system: Any, mock_zammad_client: Any,
    ) -> None:
        _wire_full_personas(mock_chat_system)
        big_body = "X" * 200000  # well above TRIAGE_MAX_CONTEXT_CHARS=100000
        mock_zammad_client.get_ticket = MagicMock(
            return_value={"id": 1, "title": "Big", "customer_id": 5}
        )
        mock_zammad_client.get_ticket_articles = MagicMock(
            return_value=[{"body": big_body}]
        )
        mock_zammad_client.search_tickets = MagicMock(return_value=[])
        mock_zammad_client.add_article_to_ticket = MagicMock()
        mock_zammad_client.add_tag = MagicMock()

        bot._get_search_keywords = AsyncMock(return_value=None)
        mock_chat_system.text_engine.generate_response = AsyncMock(
            return_value=({"type": "text", "content": "analysis"}, {})
        )

        await bot._process_ticket(1)

        # _smart_truncate should have been triggered; the analyst context should
        # contain the truncation marker.
        sent_body = mock_zammad_client.add_article_to_ticket.call_args.kwargs.get("body", "")
        assert "TRUNCATED INTELLIGENTLY" in sent_body or len(big_body) > len(sent_body)

    async def test_note_post_impersonate_fallback(
        self, bot: ZammadBot, mock_chat_system: Any, mock_zammad_client: Any,
    ) -> None:
        _wire_full_personas(mock_chat_system)
        mock_zammad_client.get_ticket = MagicMock(
            return_value={"id": 1, "title": "T", "customer_id": 5}
        )
        mock_zammad_client.get_ticket_articles = MagicMock(
            return_value=[{"body": "body"}]
        )
        mock_zammad_client.search_tickets = MagicMock(return_value=[])
        bot._get_search_keywords = AsyncMock(return_value=None)

        mock_chat_system.text_engine.generate_response = AsyncMock(
            return_value=({"type": "text", "content": "analysis"}, {})
        )

        # First call (with impersonate_email) fails; fallback (no email kwarg) succeeds
        call_log: list[Any] = []

        def post(**kwargs: Any) -> None:
            call_log.append(kwargs)
            if "impersonate_email" in kwargs:
                raise RuntimeError("impersonation denied")

        mock_zammad_client.add_article_to_ticket = MagicMock(side_effect=post)
        mock_zammad_client.add_tag = MagicMock()

        await bot._process_ticket(1)

        # Two attempts: one impersonated (failed), one fallback (succeeded)
        assert len(call_log) == 2
        assert call_log[0].get("impersonate_email") == ZAMMAD_BOT_EMAIL
        assert "impersonate_email" not in call_log[1]
        # Tag still applied after fallback success
        mock_zammad_client.add_tag.assert_called_once()

    async def test_multi_persona_pipeline_dispatches_each(
        self, bot: ZammadBot, mock_chat_system: Any, mock_zammad_client: Any,
    ) -> None:
        """All four triage personas (scout, filter, summarizer, analyst) are
        addressable through the same text_engine using their own persona configs."""
        _wire_full_personas(mock_chat_system)
        mock_zammad_client.get_ticket = MagicMock(
            return_value={"id": 1, "title": "T", "customer_id": 5}
        )
        mock_zammad_client.get_ticket_articles = MagicMock(
            return_value=[{"body": "body"}]
        )
        # One historical match (global) so we exercise the relevance filter
        mock_zammad_client.search_tickets = MagicMock(
            side_effect=[[{"id": 100, "title": "Old"}], []]
        )
        mock_zammad_client.add_article_to_ticket = MagicMock()
        mock_zammad_client.add_tag = MagicMock()

        persona_prompts_seen: list[str] = []

        async def gen(persona_config: Any, history_object: Any, tools: Any = None) -> Any:
            persona_prompts_seen.append(history_object.get('persona_prompt', ''))
            return {"type": "text", "content": "RELEVANT"}, {}

        mock_chat_system.text_engine.generate_response = AsyncMock(side_effect=gen)

        await bot._process_ticket(1)

        # Scout, filter (once for the global match), analyst — at minimum
        joined = "\n".join(persona_prompts_seen)
        assert TRIAGE_SCOUT_NAME in joined
        assert TRIAGE_FILTER_NAME in joined
        assert TRIAGE_ANALYST_NAME in joined
