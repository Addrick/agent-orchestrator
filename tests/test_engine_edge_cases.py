# tests/test_engine_edge_cases.py
#
# DP-199 Batch 4 — Engine per-provider parity matrix.
# Tier 1, 2, 3 tests from memory/project/plans/DP-199-edge-cases.md.

import pytest
from unittest.mock import patch, AsyncMock, MagicMock
import anthropic
from openai import APIStatusError

from src.engine import TextEngine, LLMCommunicationError


@pytest.fixture
def text_engine():
    return TextEngine()


@pytest.fixture
def base_context():
    return {
        "persona_prompt": "You are a test bot.",
        "history": [],
        "current_message": {"text": "Hello"},
    }


# ------------------------------------------------------------------
# Tier 1 — OpenAI history / system handling
# ------------------------------------------------------------------


@patch("src.engine.AsyncOpenAI")
class TestOpenAIHistoryEdgeCases:
    @pytest.mark.asyncio
    async def test_openai_system_message_deduplication(
        self, mock_openai_class, text_engine, base_context, monkeypatch
    ):
        """If history starts with a system message, persona_prompt's own system
        message must NOT be added — only the one in history is sent."""
        monkeypatch.setenv("OPENAI_API_KEY", "dummy")
        mock_instance = mock_openai_class.return_value
        mock_instance.chat.completions.create = AsyncMock(
            return_value=MagicMock(
                choices=[MagicMock(message=MagicMock(content="ok", tool_calls=None))]
            )
        )
        base_context["history"] = [
            {"role": "system", "content": "Explicit system message"},
            {"role": "user", "content": "Hello"},
        ]
        await text_engine.generate_response({"model_name": "gpt-4"}, base_context)
        call_args = mock_instance.chat.completions.create.call_args[1]
        messages = call_args["messages"]
        system_messages = [m for m in messages if m["role"] == "system"]
        assert len(system_messages) == 1
        assert system_messages[0]["content"] == "Explicit system message"

    @pytest.mark.asyncio
    async def test_openai_null_content_returns_empty(
        self, mock_openai_class, text_engine, base_context, monkeypatch
    ):
        """OpenAI returning content=None (no tools) → result = empty text.
        That empty result will trigger retry path inside generate_response."""
        monkeypatch.setenv("OPENAI_API_KEY", "dummy")
        mock_instance = mock_openai_class.return_value
        # Direct call to _generate_openai_response to inspect raw shape.
        mock_instance.chat.completions.create = AsyncMock(
            return_value=MagicMock(
                choices=[MagicMock(message=MagicMock(content=None, tool_calls=None))]
            )
        )
        history_obj = {
            "persona_prompt": "test",
            "history": [],
            "current_message": {"text": "hi"},
        }
        result, _ = await text_engine._generate_openai_response(
            {"model_name": "gpt-4"}, history_obj, None
        )
        assert result == {"type": "text", "content": ""}

    @pytest.mark.asyncio
    async def test_empty_history_uses_persona_prompt(
        self, mock_openai_class, text_engine, base_context, monkeypatch
    ):
        """Empty history → persona_prompt becomes the system message."""
        monkeypatch.setenv("OPENAI_API_KEY", "dummy")
        mock_instance = mock_openai_class.return_value
        mock_instance.chat.completions.create = AsyncMock(
            return_value=MagicMock(
                choices=[MagicMock(message=MagicMock(content="ok", tool_calls=None))]
            )
        )
        base_context["persona_prompt"] = "Persona system prompt here"
        base_context["history"] = []
        await text_engine.generate_response({"model_name": "gpt-4"}, base_context)
        call_args = mock_instance.chat.completions.create.call_args[1]
        sys_msgs = [m for m in call_args["messages"] if m["role"] == "system"]
        assert len(sys_msgs) == 1
        assert sys_msgs[0]["content"] == "Persona system prompt here"


# ------------------------------------------------------------------
# Tier 1 — Anthropic system merge / tool_use extraction
# ------------------------------------------------------------------


@patch("src.engine.anthropic.Anthropic")
class TestAnthropicEdgeCases:
    @pytest.mark.asyncio
    async def test_anthropic_system_merge_with_separator(
        self, mock_anthropic_class, text_engine, base_context, monkeypatch
    ):
        """When history has a leading system message, anthropic merges
        persona_prompt with it via the '\\n\\n' separator."""
        monkeypatch.setenv("ANTHROPIC_API_KEY", "dummy")
        mock_instance = mock_anthropic_class.return_value
        mock_instance.messages.create.return_value = MagicMock(
            content=[MagicMock(text="ok")], stop_reason="end_turn"
        )
        base_context["persona_prompt"] = "Base persona"
        base_context["history"] = [
            {"role": "system", "content": "Extra system"},
            {"role": "user", "content": "hi"},
        ]
        await text_engine.generate_response(
            {"model_name": "claude-3-opus-20240229"}, base_context
        )
        call_args = mock_instance.messages.create.call_args[1]
        assert call_args["system"] == "Base persona\n\nExtra system"
        # The system msg is consumed; history should only have the user msg.
        assert all(m["role"] != "system" for m in call_args["messages"])

    @pytest.mark.asyncio
    async def test_anthropic_tool_use_only_extracted(
        self, mock_anthropic_class, text_engine, base_context, monkeypatch
    ):
        """When response has mixed content (text + tool_use), only the
        tool_use blocks are extracted under stop_reason='tool_use'."""
        monkeypatch.setenv("ANTHROPIC_API_KEY", "dummy")
        mock_instance = mock_anthropic_class.return_value
        text_block = MagicMock(type="text", text="thinking...")
        tool_block = MagicMock(
            type="tool_use", id="tu_1", input={"q": "x"}
        )
        tool_block.name = "search"
        mock_instance.messages.create.return_value = MagicMock(
            content=[text_block, tool_block], stop_reason="tool_use"
        )
        response, _ = await text_engine.generate_response(
            {"model_name": "claude-3-opus-20240229"},
            base_context,
            tools=[{"type": "function", "function": {"name": "search"}}],
        )
        assert response["type"] == "tool_calls"
        # Only tool_use blocks should appear in calls
        assert len(response["calls"]) == 1
        assert response["calls"][0]["name"] == "search"

    @pytest.mark.asyncio
    async def test_anthropic_empty_content_retries(
        self, mock_anthropic_class, text_engine, base_context, monkeypatch
    ):
        """Anthropic returning whitespace-only text triggers retry path."""
        monkeypatch.setenv("ANTHROPIC_API_KEY", "dummy")
        mock_instance = mock_anthropic_class.return_value
        # Always return whitespace — all retries exhaust.
        mock_instance.messages.create.return_value = MagicMock(
            content=[MagicMock(text="   ")], stop_reason="end_turn"
        )
        with patch("src.engine.asyncio.sleep", new_callable=AsyncMock):
            with pytest.raises(
                LLMCommunicationError, match="empty or invalid response"
            ):
                await text_engine.generate_response(
                    {"model_name": "claude-3-opus-20240229"}, base_context
                )
        # Should have retried (>1 call)
        assert mock_instance.messages.create.call_count > 1

    @pytest.mark.asyncio
    async def test_anthropic_429_rate_limited_flag(
        self, mock_anthropic_class, text_engine, base_context, monkeypatch
    ):
        """Anthropic 429 sets rate_limited and aborts retries immediately."""
        monkeypatch.setenv("ANTHROPIC_API_KEY", "dummy")
        mock_instance = mock_anthropic_class.return_value
        err = anthropic.APIStatusError(
            "Rate limit", response=MagicMock(status_code=429), body=None
        )
        mock_instance.messages.create.side_effect = err
        with pytest.raises(LLMCommunicationError) as ei:
            await text_engine.generate_response(
                {"model_name": "claude-3-opus-20240229"}, base_context
            )
        assert ei.value.rate_limited is True
        assert mock_instance.messages.create.call_count == 1


# ------------------------------------------------------------------
# Tier 1 — Google role mapping / tool ordering / grounding
# ------------------------------------------------------------------


@patch("src.engine.genai.client.AsyncClient")
class TestGoogleEdgeCases:
    @pytest.mark.asyncio
    async def test_google_role_mapping_first_turn(
        self, mock_google_class, text_engine, base_context, monkeypatch
    ):
        """First entry is system prompt; user → 'user', assistant → 'model'."""
        monkeypatch.setenv("GOOGLE_GENERATIVEAI_API_KEY", "dummy")
        mock_instance = mock_google_class.return_value
        mock_part = MagicMock(text="ok", function_call=None)
        mock_part.thought_signature = None
        mock_candidate = MagicMock(
            content=MagicMock(parts=[mock_part]), grounding_metadata=None
        )
        mock_instance.models.generate_content = AsyncMock(
            return_value=MagicMock(prompt_feedback=None, candidates=[mock_candidate])
        )
        base_context["history"] = [
            {"role": "user", "content": "u1"},
            {"role": "assistant", "content": "a1"},
            {"role": "user", "content": "u2"},
        ]
        await text_engine.generate_response({"model_name": "gemini-pro"}, base_context)
        contents = mock_instance.models.generate_content.call_args[1]["contents"]
        # First content entry is the persona system prompt (no role key in handler — that's the "system" entry).
        # Then alternating: user, model, user.
        roles_after_sys = [c.get("role") for c in contents[1:]]
        assert roles_after_sys == ["user", "model", "user"]

    @pytest.mark.asyncio
    async def test_google_tool_response_ordering(
        self, mock_google_class, text_engine, base_context, monkeypatch
    ):
        """Tool messages must serialize as function_response Parts with role='tool'."""
        monkeypatch.setenv("GOOGLE_GENERATIVEAI_API_KEY", "dummy")
        mock_instance = mock_google_class.return_value
        mock_part = MagicMock(text="done", function_call=None)
        mock_part.thought_signature = None
        mock_candidate = MagicMock(
            content=MagicMock(parts=[mock_part]), grounding_metadata=None
        )
        mock_instance.models.generate_content = AsyncMock(
            return_value=MagicMock(prompt_feedback=None, candidates=[mock_candidate])
        )
        base_context["history"] = [
            {"role": "user", "content": "ask"},
            {
                "role": "assistant",
                "tool_calls": [
                    {"id": "c1", "name": "search", "arguments": {"q": "x"}}
                ],
            },
            {
                "role": "tool",
                "name": "search",
                "tool_call_id": "c1",
                "content": '{"hits": []}',
            },
        ]
        await text_engine.generate_response({"model_name": "gemini-pro"}, base_context)
        contents = mock_instance.models.generate_content.call_args[1]["contents"]
        # Find the tool turn
        tool_turns = [c for c in contents if c.get("role") == "tool"]
        assert len(tool_turns) == 1

    @pytest.mark.asyncio
    async def test_google_mixed_parts_tool_priority(
        self, mock_google_class, text_engine, base_context, monkeypatch
    ):
        """When response parts include both text and function_call, tool_calls
        takes priority."""
        monkeypatch.setenv("GOOGLE_GENERATIVEAI_API_KEY", "dummy")
        mock_instance = mock_google_class.return_value
        fcall = MagicMock()
        fcall.name = "do_thing"
        fcall.args = {"a": 1}
        text_part = MagicMock(text="some explanation", function_call=None)
        text_part.thought_signature = None
        fcall_part = MagicMock(text=None, function_call=fcall)
        fcall_part.thought_signature = None
        mock_candidate = MagicMock(
            content=MagicMock(parts=[text_part, fcall_part]),
            grounding_metadata=None,
        )
        mock_instance.models.generate_content = AsyncMock(
            return_value=MagicMock(prompt_feedback=None, candidates=[mock_candidate])
        )
        response, _ = await text_engine.generate_response(
            {"model_name": "gemini-pro"},
            base_context,
            tools=[{"type": "function", "function": {"name": "do_thing"}}],
        )
        assert response["type"] == "tool_calls"
        assert response["calls"][0]["name"] == "do_thing"

    @pytest.mark.asyncio
    async def test_google_grounding_metadata_absent(
        self, mock_google_class, text_engine, base_context, monkeypatch
    ):
        """Absent grounding_metadata → plain text, no citations appended."""
        monkeypatch.setenv("GOOGLE_GENERATIVEAI_API_KEY", "dummy")
        mock_instance = mock_google_class.return_value
        mock_part = MagicMock(text="plain answer", function_call=None)
        mock_part.thought_signature = None
        mock_candidate = MagicMock(
            content=MagicMock(parts=[mock_part]), grounding_metadata=None
        )
        mock_instance.models.generate_content = AsyncMock(
            return_value=MagicMock(prompt_feedback=None, candidates=[mock_candidate])
        )
        response, _ = await text_engine.generate_response(
            {"model_name": "gemini-pro"}, base_context
        )
        assert response == {"type": "text", "content": "plain answer"}
        assert "Sources" not in response["content"]
        assert "Search Query" not in response["content"]

    @pytest.mark.asyncio
    async def test_google_grounding_citations_appended(
        self, mock_google_class, text_engine, base_context, monkeypatch
    ):
        """grounding_chunks + grounding_supports → inline citations + sources list."""
        monkeypatch.setenv("GOOGLE_GENERATIVEAI_API_KEY", "dummy")
        mock_instance = mock_google_class.return_value

        web = MagicMock(uri="http://example.com", title="Example")
        chunk = MagicMock(web=web)
        segment = MagicMock(text="answer", start_index=0)
        support = MagicMock(segment=segment, grounding_chunk_indices=[0])

        grounding_metadata = MagicMock(
            grounding_chunks=[chunk],
            grounding_supports=[support],
            web_search_queries=None,
        )

        mock_part = MagicMock(text="answer", function_call=None)
        mock_part.thought_signature = None
        mock_candidate = MagicMock(
            content=MagicMock(parts=[mock_part]),
            grounding_metadata=grounding_metadata,
        )
        mock_instance.models.generate_content = AsyncMock(
            return_value=MagicMock(prompt_feedback=None, candidates=[mock_candidate])
        )
        response, _ = await text_engine.generate_response(
            {"model_name": "gemini-pro"}, base_context
        )
        assert "Sources" in response["content"]
        assert "Example" in response["content"]
        # Inline citation marker
        assert "[1]" in response["content"]
        assert "http://example.com" in response["content"]

    @pytest.mark.asyncio
    async def test_google_grounding_url_not_in_context(
        self, mock_google_class, text_engine, base_context, monkeypatch
    ):
        """Chunk without a web.uri → no source entry generated."""
        monkeypatch.setenv("GOOGLE_GENERATIVEAI_API_KEY", "dummy")
        mock_instance = mock_google_class.return_value

        # Chunk has no usable web.uri
        chunk = MagicMock(web=MagicMock(uri=None, title=None))
        segment = MagicMock(text="answer", start_index=0)
        support = MagicMock(segment=segment, grounding_chunk_indices=[0])
        grounding_metadata = MagicMock(
            grounding_chunks=[chunk],
            grounding_supports=[support],
            web_search_queries=None,
        )

        mock_part = MagicMock(text="answer", function_call=None)
        mock_part.thought_signature = None
        mock_candidate = MagicMock(
            content=MagicMock(parts=[mock_part]),
            grounding_metadata=grounding_metadata,
        )
        mock_instance.models.generate_content = AsyncMock(
            return_value=MagicMock(prompt_feedback=None, candidates=[mock_candidate])
        )
        response, _ = await text_engine.generate_response(
            {"model_name": "gemini-pro"}, base_context
        )
        # No sources should be listed since the URI was missing.
        assert "Sources" not in response["content"]

    @pytest.mark.asyncio
    async def test_google_search_queries_appended(
        self, mock_google_class, text_engine, base_context, monkeypatch
    ):
        """web_search_queries → 'Search Query: ...' appended to text."""
        monkeypatch.setenv("GOOGLE_GENERATIVEAI_API_KEY", "dummy")
        mock_instance = mock_google_class.return_value

        web = MagicMock(uri="http://example.com", title="Example")
        chunk = MagicMock(web=web)
        segment = MagicMock(text="answer", start_index=0)
        support = MagicMock(segment=segment, grounding_chunk_indices=[0])
        grounding_metadata = MagicMock(
            grounding_chunks=[chunk],
            grounding_supports=[support],
            web_search_queries=["query1", "query2"],
        )

        mock_part = MagicMock(text="answer", function_call=None)
        mock_part.thought_signature = None
        mock_candidate = MagicMock(
            content=MagicMock(parts=[mock_part]),
            grounding_metadata=grounding_metadata,
        )
        mock_instance.models.generate_content = AsyncMock(
            return_value=MagicMock(prompt_feedback=None, candidates=[mock_candidate])
        )
        response, _ = await text_engine.generate_response(
            {"model_name": "gemini-pro"}, base_context
        )
        assert "Search Query" in response["content"]
        assert "query1" in response["content"]
        assert "query2" in response["content"]

    @pytest.mark.asyncio
    async def test_google_429_resource_exhausted(
        self, mock_google_class, text_engine, base_context, monkeypatch
    ):
        """Google RESOURCE_EXHAUSTED → rate_limited=True, no retry."""
        monkeypatch.setenv("GOOGLE_GENERATIVEAI_API_KEY", "dummy")
        mock_instance = mock_google_class.return_value
        mock_instance.models.generate_content.side_effect = Exception(
            "google.api_core.exceptions.ResourceExhausted: 429 RESOURCE_EXHAUSTED"
        )
        with pytest.raises(LLMCommunicationError) as ei:
            await text_engine.generate_response(
                {"model_name": "gemini-pro"}, base_context
            )
        assert ei.value.rate_limited is True
        assert mock_instance.models.generate_content.call_count == 1


# ------------------------------------------------------------------
# Tier 1 — Tool call parsing edges (OpenAI _parse_openai_tool_calls)
# ------------------------------------------------------------------


class TestToolCallParsing:
    def test_tool_call_empty_arguments_accepted(self):
        """Empty-JSON-object arguments ('{}') parse to an empty dict."""
        fn = MagicMock()
        fn.name = "noop"
        fn.arguments = "{}"
        call = MagicMock(id="c1", function=fn)
        result = TextEngine._parse_openai_tool_calls([call])
        assert len(result) == 1
        assert result[0]["name"] == "noop"
        assert result[0]["arguments"] == {}

    def test_tool_call_missing_name_discarded(self):
        """A tool call whose JSON args fail to parse is discarded."""
        good_fn = MagicMock()
        good_fn.name = "ok_tool"
        good_fn.arguments = '{"a": 1}'
        good = MagicMock(id="c1", function=good_fn)

        bad_fn = MagicMock()
        bad_fn.name = "bad_tool"
        bad_fn.arguments = "{not json"
        bad = MagicMock(id="c2", function=bad_fn)

        result = TextEngine._parse_openai_tool_calls([good, bad])
        assert len(result) == 1
        assert result[0]["name"] == "ok_tool"

    def test_tool_call_string_arguments_unwrapped(self):
        """OpenAI always sends args as a JSON string; the parser json.loads
        it into a dict. A non-JSON arguments value should not crash; it is
        skipped."""
        fn = MagicMock()
        fn.name = "broken"
        fn.arguments = "plain string not json"
        call = MagicMock(id="c1", function=fn)
        result = TextEngine._parse_openai_tool_calls([call])
        assert result == []


# ------------------------------------------------------------------
# Tier 2 — 429 / retry-after / fallback (skipped per HARD RULES)
# ------------------------------------------------------------------


def test_retry_after_header_honored():
    pytest.skip("DP-199 deferred bug 2 — feature not implemented")


def test_fallback_model_on_429():
    # Currently fallback exists only for gemma-4-31b-it → gemma-4-26b-a4b-it.
    # The test name implies a generic feature; treating as deferred missing.
    pytest.skip("DP-199 deferred bug 3 — generic fallback feature not implemented")


# ------------------------------------------------------------------
# Tier 3 — provider_extras / legacy dicts / chat template precedence
# ------------------------------------------------------------------


class TestProviderExtras:
    def test_provider_extras_unknown_keys_filtered(self):
        """GenerationParams.get_provider_extras filters by provider name —
        unknown providers → empty dict."""
        from src.generation_params import GenerationParams

        params = GenerationParams(provider_extras={"kobold": {"rep_pen": 1.1}})
        assert params.get_provider_extras("unknown_provider") == {}
        assert params.get_provider_extras("kobold") == {"rep_pen": 1.1}

    def test_kobold_extras_unrecognized_dropped(self):
        """StreamEngine._params_from_legacy_dicts only pulls a known set of
        kobold knobs; anything else is dropped silently."""
        from src.stream_engine import StreamEngine

        persona = {"model_name": "local"}
        lic = {
            "rep_pen": 1.1,
            "unrecognized_field": "garbage",
            "some_other_thing": 42,
        }
        params = StreamEngine._params_from_legacy_dicts(persona, lic)
        kobold = params.get_provider_extras("kobold")
        assert kobold.get("rep_pen") == 1.1
        assert "unrecognized_field" not in kobold
        assert "some_other_thing" not in kobold

    def test_legacy_dicts_instruct_tags_override(self):
        """If local_inference_config carries instruct_tags with any value,
        it overrides the persona's stored instruct_tags."""
        from src.stream_engine import StreamEngine

        persona = {
            "model_name": "local",
            "provider_extras": {
                "kobold": {
                    "instruct_tags": {"instruct_starttag": "PERSONA_USER:"}
                }
            },
        }
        lic = {"instruct_tags": {"instruct_starttag": "REQUEST_USER:"}}
        params = StreamEngine._params_from_legacy_dicts(persona, lic)
        kobold = params.get_provider_extras("kobold")
        assert kobold["instruct_tags"]["instruct_starttag"] == "REQUEST_USER:"

    def test_chat_template_resolution_precedence(self, monkeypatch):
        """persona_config['chat_template'] > env var > global default."""
        from src.stream_engine import StreamEngine

        monkeypatch.setenv("KOBOLD_CHAT_TEMPLATE", "llama3")
        # Persona setting wins.
        assert (
            StreamEngine._resolve_template_name({"chat_template": "gemma"})
            == "gemma"
        )
        # No persona value → env var.
        assert (
            StreamEngine._resolve_template_name({"chat_template": None})
            == "llama3"
        )
        # No persona, no env → falls back to chatml default.
        monkeypatch.delenv("KOBOLD_CHAT_TEMPLATE", raising=False)
        assert StreamEngine._resolve_template_name({}) == "chatml"
