# tests/test_message_handler.py

import pytest
from unittest.mock import MagicMock, AsyncMock, patch

from src.message_handler import BotLogic
from src.persona import Persona
from src.chat_system import ChatSystem


@pytest.fixture
def mock_chat_system_with_state():
    """Creates a mock ChatSystem that has a real dictionary for personas.json."""
    chat_system = MagicMock(spec=ChatSystem)
    # Start with a real dictionary to track state changes
    chat_system.personas = {
        "derpr": Persona("derpr", "gpt-4", "You are derpr.", context_length=20),
        "testbot": Persona("testbot", "gpt-3", "You are testbot.")
    }
    return chat_system


@pytest.fixture
def bot_logic(mock_chat_system_with_state):
    """Creates a BotLogic instance connected to the stateful mock ChatSystem."""
    return BotLogic(mock_chat_system_with_state)


# --- Test Cases for State Mutation (add/delete) ---

@pytest.mark.asyncio
async def test_handle_add_persona_success(bot_logic, mock_chat_system_with_state):
    """Tests that the 'add' command successfully adds a new persona to the chat system's state."""
    assert "new_persona" not in mock_chat_system_with_state.personas
    result = await bot_logic.preprocess_message("derpr", "user1", "add new_persona")
    assert "new_persona" in mock_chat_system_with_state.personas
    assert isinstance(mock_chat_system_with_state.personas["new_persona"], Persona)
    assert result is not None and result["mutated"] is True and "Added 'new_persona'" in result["response"]


@pytest.mark.asyncio
async def test_handle_add_persona_already_exists(bot_logic, mock_chat_system_with_state):
    """Tests that adding a persona that already exists returns an error and does not mutate state."""
    initial_persona_count = len(mock_chat_system_with_state.personas)
    result = await bot_logic.preprocess_message("derpr", "user1", "add derpr")
    assert len(mock_chat_system_with_state.personas) == initial_persona_count
    assert result is not None and result["mutated"] is False and "already exists" in result["response"]


@pytest.mark.asyncio
async def test_handle_delete_persona_success(bot_logic, mock_chat_system_with_state):
    """Tests that the 'delete' command successfully removes a persona from the chat system's state."""
    assert "testbot" in mock_chat_system_with_state.personas
    result = await bot_logic.preprocess_message("derpr", "user1", "delete testbot")
    assert "testbot" not in mock_chat_system_with_state.personas
    assert result is not None and result["mutated"] is True and "Deleted persona 'testbot'" in result["response"]


@pytest.mark.asyncio
async def test_handle_delete_persona_not_found(bot_logic, mock_chat_system_with_state):
    """Tests that deleting a non-existent persona returns an error and does not mutate state."""
    initial_persona_count = len(mock_chat_system_with_state.personas)
    result = await bot_logic.preprocess_message("derpr", "user1", "delete fake_persona")
    assert len(mock_chat_system_with_state.personas) == initial_persona_count
    assert result is not None and result["mutated"] is False and "not found" in result["response"]


@pytest.mark.asyncio
async def test_command_fall_through_on_bad_syntax(bot_logic):
    """Tests that a command with incorrect syntax returns None, allowing it to be processed by the LLM."""
    assert await bot_logic.preprocess_message("derpr", "user1", "add") is None
    assert await bot_logic.preprocess_message("derpr", "user1", "set") is None


@pytest.mark.asyncio
async def test_non_mutating_command(bot_logic):
    """Tests that a read-only command like 'detail' does not set the mutated flag."""
    result = await bot_logic.preprocess_message("derpr", "user1", "detail")
    assert result is not None and result["mutated"] is False


# --- Test Cases for Dynamic Context Management ---

def test_dynamic_context_lifecycle(bot_logic, mock_chat_system_with_state):
    """
    Tests the full lifecycle of dynamic context: hello, growth, and goodbye.
    """
    persona = mock_chat_system_with_state.personas["derpr"]
    assert persona.get_context_length() == 20, "Initial state should be the default context length."

    bot_logic._handle_start_conversation([], persona, "user1")
    assert persona.get_context_length() == 0, "First call after 'hello' should return 0."
    assert persona.get_context_length() == 2, "Second call should return 2."
    assert persona.get_context_length() == 4, "Third call should return 4."

    bot_logic._handle_stop_conversation([], persona, "user1")
    assert persona.get_context_length() == 20, "After 'goodbye', should revert to default."
    assert persona.get_context_length() == 20, "Should stay at default after reverting."


def test_set_context_command_variations(bot_logic, mock_chat_system_with_state):
    """Tests the enhanced `set context` command for static and dynamic modes."""
    persona = mock_chat_system_with_state.personas["derpr"]

    # Test 1: Set a new static context length
    bot_logic._set_context(["context", "50"], persona)
    assert persona.get_current_effective_context_length() == 50
    assert persona.is_in_dynamic_context() is False

    # Test 2: Switch to dynamic mode, inheriting the current value (50)
    bot_logic._set_context(["context", "dynamic"], persona)
    assert persona.is_in_dynamic_context() is True
    assert persona.get_context_length() == 50, "It should start at the captured value of 50."
    assert persona.get_context_length() == 52, "Then it should grow to 52."

    # Test 3: Set a new static context, which should disable dynamic mode
    bot_logic._set_context(["context", "30"], persona)
    assert persona.is_in_dynamic_context() is False
    assert persona.get_current_effective_context_length() == 30

    # Test 4: Switch to dynamic mode with a specific start value
    bot_logic._set_context(["context", "dynamic", "8"], persona)
    assert persona.is_in_dynamic_context() is True
    assert persona.get_context_length() == 8
    assert persona.get_context_length() == 10

    # Test 5 (Your Refinement): Verify dynamic start from a dynamic value
    # The current context is 12 (from the previous step: 8 -> 10 -> 12)
    assert persona.get_current_effective_context_length() == 12
    # Now, set dynamic again. It should capture 12.
    bot_logic._set_context(["context", "dynamic"], persona)
    assert persona.is_in_dynamic_context() is True
    assert persona.get_context_length() == 12
    assert persona.get_context_length() == 14


@pytest.mark.asyncio
async def test_handle_dump_context_returns_file_response_format(bot_logic, mock_chat_system_with_state):
    """
    Tests that the dump_context command returns the special FILE_RESPONSE string
    and correctly formats the context into a string.
    """
    # 1. Setup a mock API payload with tools
    user_identifier = "user1"
    persona_name = "derpr"
    mock_tools = [
        {
            "type": "function",
            "is_write": False,
            "service_binding": "zammad",
            "function": {
                "name": "search_tickets",
                "description": "Search for tickets in Zammad.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Search query string."}
                    },
                    "required": ["query"]
                }
            }
        },
        {
            "type": "function",
            "is_write": True,
            "service_binding": "agents",
            "function": {
                "name": "manage_agent",
                "description": "Start, stop, or restart an agent.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "agent_name": {"type": "string", "description": "The agent name."},
                        "action": {"type": "string", "description": "The action to perform."}
                    },
                    "required": ["agent_name", "action"]
                }
            }
        }
    ]
    mock_payload = {
        'model': 'test-model',
        'config': {'max_output_tokens': 1024},
        '_tools_for_llm': mock_tools,
        'contents': [
            {'role': 'user', 'parts': [{'text': 'Hello there'}]},
            {'role': 'assistant', 'parts': [{'text': 'General Kenobi'}]}
        ]
    }
    mock_chat_system_with_state.last_api_requests = {user_identifier: {persona_name: mock_payload}}
    current_persona = mock_chat_system_with_state.personas[persona_name]

    # 2. Action
    response, mutated = bot_logic._handle_dump_context(args=[], persona=current_persona,
                                                       user_identifier=user_identifier)

    # 3. Assertions
    assert mutated is False
    assert response.startswith("FILE_RESPONSE::context_dump.txt::")

    file_content = response.split("::", 2)[2]

    # Persona config section
    assert "=== Context Dump for derpr ===" in file_content
    assert "Enabled Tools:" in file_content
    assert "Service Bindings:" in file_content

    # Tool definitions section — full schemas, not just names
    assert "--- Tools Sent to LLM (2 total) ---" in file_content
    assert "[search_tickets]" in file_content
    assert "Search for tickets in Zammad." in file_content
    assert "Service Binding: zammad" in file_content
    assert "Write Operation: False" in file_content
    assert "query: string (required)" in file_content

    assert "[manage_agent]" in file_content
    assert "Service Binding: agents" in file_content
    assert "Write Operation: True" in file_content
    assert "agent_name: string (required)" in file_content

    # API config section
    assert "--- API Request Config ---" in file_content
    assert "max_output_tokens: 1024" in file_content

    # Conversation history
    assert "--- Context Sent to Model ---" in file_content
    assert "[Message 1 - ROLE: USER]" in file_content
    assert "Hello there" in file_content
    assert "[Message 2 - ROLE: ASSISTANT]" in file_content
    assert "General Kenobi" in file_content


# --- Test Cases for _extract_message_content ---

class TestExtractMessageContent:
    """Tests for the multi-provider message content extraction helper."""

    def test_openai_string_content(self):
        msg = {"role": "user", "content": "Hello world"}
        assert BotLogic._extract_message_content(msg) == "Hello world"

    def test_google_parts_format(self):
        msg = {"role": "user", "parts": [{"text": "Hello from Gemini"}]}
        assert BotLogic._extract_message_content(msg) == "Hello from Gemini"

    def test_openai_multimodal_content(self):
        msg = {"role": "user", "content": [
            {"type": "text", "text": "What is this?"},
            {"type": "image_url", "image_url": {"url": "https://example.com/img.png"}}
        ]}
        result = BotLogic._extract_message_content(msg)
        assert "What is this?" in result
        assert "[IMAGE]" in result

    def test_tool_calls_message(self):
        msg = {"role": "assistant", "tool_calls": [
            {"id": "call_1", "name": "search_tickets", "arguments": {"query": "printer"}}
        ]}
        result = BotLogic._extract_message_content(msg)
        assert "[TOOL CALLS]" in result
        assert "search_tickets" in result
        assert "printer" in result

    def test_tool_result_message(self):
        msg = {"role": "tool", "name": "search_tickets", "content": '{"tickets": []}'}
        result = BotLogic._extract_message_content(msg)
        assert "[TOOL RESULT: search_tickets]" in result
        assert '{"tickets": []}' in result

    def test_empty_message(self):
        msg = {"role": "user"}
        assert BotLogic._extract_message_content(msg) == "[NO TEXT CONTENT]"

    def test_anthropic_image_content(self):
        msg = {"role": "user", "content": [
            {"type": "text", "text": "Describe this"},
            {"type": "image", "source": {"type": "base64", "data": "..."}}
        ]}
        result = BotLogic._extract_message_content(msg)
        assert "Describe this" in result
        assert "[IMAGE]" in result


@pytest.mark.asyncio
async def test_dump_context_with_tool_calls_in_history(bot_logic, mock_chat_system_with_state):
    """dump_context renders tool call and tool result messages in conversation history."""
    user_identifier = "user1"
    persona_name = "derpr"
    mock_payload = {
        'model': 'test-model',
        'config': {},
        '_tools_for_llm': [],
        'messages': [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Search for printer tickets"},
            {"role": "assistant", "tool_calls": [
                {"id": "call_1", "name": "search_tickets", "arguments": {"query": "printer"}}
            ]},
            {"role": "tool", "name": "search_tickets", "content": '{"count": 3}'},
            {"role": "assistant", "content": "I found 3 printer tickets."}
        ]
    }
    mock_chat_system_with_state.last_api_requests = {user_identifier: {persona_name: mock_payload}}
    current_persona = mock_chat_system_with_state.personas[persona_name]

    response, mutated = bot_logic._handle_dump_context(args=[], persona=current_persona,
                                                       user_identifier=user_identifier)

    file_content = response.split("::", 2)[2]

    # System prompt extracted from OpenAI format
    assert "[System Prompt]" in file_content
    assert "You are a helpful assistant." in file_content

    # User message
    assert "Search for printer tickets" in file_content

    # Tool call rendered
    assert "[TOOL CALLS]" in file_content
    assert "search_tickets" in file_content

    # Tool result rendered
    assert "[TOOL RESULT: search_tickets]" in file_content
    assert '{"count": 3}' in file_content

    # Final assistant response
    assert "I found 3 printer tickets." in file_content


@pytest.mark.asyncio
async def test_dump_context_no_tools(bot_logic, mock_chat_system_with_state):
    """dump_context handles payloads with no tools gracefully."""
    user_identifier = "user1"
    persona_name = "derpr"
    mock_payload = {
        'model': 'test-model',
        'config': {},
        'contents': [
            {'role': 'user', 'parts': [{'text': 'Just a question'}]}
        ]
    }
    mock_chat_system_with_state.last_api_requests = {user_identifier: {persona_name: mock_payload}}
    current_persona = mock_chat_system_with_state.personas[persona_name]

    response, _ = bot_logic._handle_dump_context(args=[], persona=current_persona,
                                                  user_identifier=user_identifier)

    file_content = response.split("::", 2)[2]
    assert "--- Tools Sent to LLM (0 total) ---" in file_content
    assert "No tools were sent to the LLM." in file_content


@pytest.mark.asyncio
async def test_dump_context_openai_payload_shows_api_config(bot_logic, mock_chat_system_with_state):
    """dump_context shows API config from OpenAI-shaped payloads (flat top-level keys, no 'config')."""
    user_identifier = "user1"
    persona_name = "derpr"
    mock_payload = {
        'model': 'gpt-4',
        'max_tokens': 2048,
        'temperature': 0.7,
        'top_p': 0.9,
        'tool_choice': 'auto',
        'messages': [
            {"role": "user", "content": "Hello"},
        ],
        '_tools_for_llm': [],
    }
    mock_chat_system_with_state.last_api_requests = {user_identifier: {persona_name: mock_payload}}
    current_persona = mock_chat_system_with_state.personas[persona_name]

    response, _ = bot_logic._handle_dump_context(args=[], persona=current_persona,
                                                  user_identifier=user_identifier)
    file_content = response.split("::", 2)[2]

    assert "--- API Request Config ---" in file_content
    assert "max_tokens: 2048" in file_content
    assert "temperature: 0.7" in file_content
    assert "top_p: 0.9" in file_content
    # Internal keys should be excluded
    assert "tool_choice" not in file_content
    assert "_tools_for_llm" not in file_content
    assert "messages" not in file_content


@pytest.mark.asyncio
async def test_dump_context_google_payload_shows_api_config(bot_logic, mock_chat_system_with_state):
    """dump_context shows API config from Google-shaped payloads (nested 'config' dict)."""
    user_identifier = "user1"
    persona_name = "derpr"
    mock_payload = {
        'model': 'gemini-2.5-pro',
        'config': {
            'max_output_tokens': 4096,
            'temperature': 0.5,
            'tools': ['search_tickets'],
            'safety_settings': {'block': 'none'},
        },
        'contents': [
            {'role': 'user', 'parts': [{'text': 'Hello'}]},
        ],
        '_tools_for_llm': [],
    }
    mock_chat_system_with_state.last_api_requests = {user_identifier: {persona_name: mock_payload}}
    current_persona = mock_chat_system_with_state.personas[persona_name]

    response, _ = bot_logic._handle_dump_context(args=[], persona=current_persona,
                                                  user_identifier=user_identifier)
    file_content = response.split("::", 2)[2]

    assert "--- API Request Config ---" in file_content
    assert "max_output_tokens: 4096" in file_content
    assert "temperature: 0.5" in file_content
    # Internal config keys should be excluded
    assert "safety_settings" not in file_content
    assert "tools: [" not in file_content


@pytest.mark.asyncio
async def test_dump_last_shows_tools_from_openai_payload(bot_logic, mock_chat_system_with_state):
    """dump_last reads tools from _tools_for_llm for OpenAI payloads."""
    user_identifier = "user1"
    persona_name = "derpr"
    mock_payload = {
        'model': 'gpt-4',
        'max_tokens': 1024,
        'temperature': 0.7,
        'tools': ['search_tickets', 'web_search'],  # stripped names (engine post-processing)
        '_tools_for_llm': [
            {"type": "function", "function": {"name": "search_tickets"}},
            {"type": "function", "function": {"name": "web_search"}},
        ],
        'messages': [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there"},
        ],
    }
    mock_chat_system_with_state.last_api_requests = {user_identifier: {persona_name: mock_payload}}
    current_persona = mock_chat_system_with_state.personas[persona_name]

    response, _ = bot_logic._handle_dump_last(args=[], persona=current_persona,
                                               user_identifier=user_identifier)
    assert "search_tickets" in response
    assert "web_search" in response


@pytest.mark.asyncio
async def test_dump_last_falls_back_to_stripped_names(bot_logic, mock_chat_system_with_state):
    """dump_last falls back to stripped tool names when _tools_for_llm is absent."""
    user_identifier = "user1"
    persona_name = "derpr"
    mock_payload = {
        'model': 'gemini-2.5-pro',
        'config': {
            'max_output_tokens': 1024,
            'temperature': 0.5,
            'tools': ['search_tickets', 'manage_agent'],
        },
        'contents': [
            {'role': 'user', 'parts': [{'text': 'Hello'}]},
        ],
    }
    mock_chat_system_with_state.last_api_requests = {user_identifier: {persona_name: mock_payload}}
    current_persona = mock_chat_system_with_state.personas[persona_name]

    response, _ = bot_logic._handle_dump_last(args=[], persona=current_persona,
                                               user_identifier=user_identifier)
    assert "search_tickets" in response
    assert "manage_agent" in response


def test_dump_persona_config_shows_enabled_tools_wildcard(bot_logic, mock_chat_system_with_state):
    """Persona config dump shows '*' when all tools are enabled."""
    persona = mock_chat_system_with_state.personas["derpr"]
    persona.set_enabled_tools(['*'])
    lines = []
    BotLogic._dump_persona_config(lines, persona)
    output = "\n".join(lines)
    assert "Enabled Tools: *" in output


def test_dump_persona_config_shows_enabled_tools_specific(bot_logic, mock_chat_system_with_state):
    """Persona config dump shows specific tool names."""
    persona = mock_chat_system_with_state.personas["derpr"]
    persona.set_enabled_tools(['search_tickets', 'web_search'])
    lines = []
    BotLogic._dump_persona_config(lines, persona)
    output = "\n".join(lines)
    assert "Enabled Tools: search_tickets, web_search" in output


def test_dump_persona_config_shows_enabled_tools_none(bot_logic, mock_chat_system_with_state):
    """Persona config dump shows 'none' when no tools are enabled."""
    persona = mock_chat_system_with_state.personas["derpr"]
    persona.set_enabled_tools([])
    lines = []
    BotLogic._dump_persona_config(lines, persona)
    output = "\n".join(lines)
    assert "Enabled Tools: none" in output


class TestExtractMessageContentGoogleNative:
    """Tests for Google-native function_call/function_response part handling."""

    def test_google_function_call_parts(self):
        msg = {"role": "model", "parts": [
            {"function_call": {"name": "search_tickets", "args": {"query": "printer"}}}
        ]}
        result = BotLogic._extract_message_content(msg)
        assert "[TOOL CALL] search_tickets" in result
        assert "printer" in result

    def test_google_function_response_parts(self):
        msg = {"role": "tool", "parts": [
            {"function_response": {"name": "search_tickets", "response": {"count": 3}}}
        ]}
        result = BotLogic._extract_message_content(msg)
        assert "[TOOL RESULT: search_tickets]" in result
        assert "3" in result

    def test_google_mixed_parts(self):
        """Parts list with text and function_call renders both."""
        msg = {"role": "model", "parts": [
            {"text": "Let me search for that."},
            {"function_call": {"name": "web_search", "args": {"query": "test"}}}
        ]}
        result = BotLogic._extract_message_content(msg)
        assert "Let me search for that." in result
        assert "[TOOL CALL] web_search" in result


# --- Test Cases for Fuzzy Tool Selection & Exclude Syntax ---

@pytest.fixture
def bot_logic_with_tools(mock_chat_system_with_state):
    """Creates a BotLogic with a mock tool manager that returns tool definitions."""
    mock_tool_manager = MagicMock()
    mock_tool_manager.get_tool_definitions.return_value = [
        {"type": "function", "function": {"name": "web_search"}},
        {"type": "google_grounding", "function": {"name": "google_grounding_search"}},
        {"type": "function", "function": {"name": "create_ticket"}},
        {"type": "function", "function": {"name": "search_tickets"}},
    ]
    mock_chat_system_with_state.tool_manager = mock_tool_manager
    return BotLogic(mock_chat_system_with_state)


@pytest.mark.asyncio
async def test_set_tools_all(bot_logic_with_tools, mock_chat_system_with_state):
    persona = mock_chat_system_with_state.personas["derpr"]
    result = await bot_logic_with_tools.preprocess_message("derpr", "user1", "set tools all")
    assert result["mutated"] is True
    assert persona.get_enabled_tools() == ['*']


@pytest.mark.asyncio
async def test_set_tools_none(bot_logic_with_tools, mock_chat_system_with_state):
    persona = mock_chat_system_with_state.personas["derpr"]
    persona.set_enabled_tools(['*'])
    result = await bot_logic_with_tools.preprocess_message("derpr", "user1", "set tools none")
    assert result["mutated"] is True
    assert persona.get_enabled_tools() == []


@pytest.mark.asyncio
async def test_set_tools_exact_names(bot_logic_with_tools, mock_chat_system_with_state):
    persona = mock_chat_system_with_state.personas["derpr"]
    result = await bot_logic_with_tools.preprocess_message("derpr", "user1", "set tools web_search create_ticket")
    assert result["mutated"] is True
    assert persona.get_enabled_tools() == ["web_search", "create_ticket"]


@pytest.mark.asyncio
async def test_set_tools_invalid_name_returns_error(bot_logic_with_tools, mock_chat_system_with_state):
    """An unresolvable tool name should return an error."""
    with patch.object(bot_logic_with_tools, '_query_llm_for_tool_selection', new_callable=AsyncMock, return_value=None):
        result = await bot_logic_with_tools.preprocess_message("derpr", "user1", "set tools nonexistent_tool")
    assert result["mutated"] is False
    assert "Could not match" in result["response"]


@pytest.mark.asyncio
async def test_set_tools_all_with_excludes(bot_logic_with_tools, mock_chat_system_with_state):
    persona = mock_chat_system_with_state.personas["derpr"]
    result = await bot_logic_with_tools.preprocess_message(
        "derpr", "user1", "set tools all -google_grounding_search -web_search"
    )
    assert result["mutated"] is True
    enabled = persona.get_enabled_tools()
    assert "google_grounding_search" not in enabled
    assert "web_search" not in enabled
    assert "create_ticket" in enabled
    assert "search_tickets" in enabled


@pytest.mark.asyncio
async def test_set_tools_exclude_without_all_returns_error(bot_logic_with_tools, mock_chat_system_with_state):
    result = await bot_logic_with_tools.preprocess_message("derpr", "user1", "set tools -web_search")
    assert result["mutated"] is False
    assert "requires 'all'" in result["response"]


@pytest.mark.asyncio
async def test_set_tools_fuzzy_match(bot_logic_with_tools, mock_chat_system_with_state):
    """Fuzzy matching should resolve partial names via the LLM selector."""
    persona = mock_chat_system_with_state.personas["derpr"]
    with patch.object(bot_logic_with_tools, '_query_llm_for_tool_selection',
                      new_callable=AsyncMock, return_value="google_grounding_search"):
        result = await bot_logic_with_tools.preprocess_message("derpr", "user1", "set tools grounding")
    assert result["mutated"] is True
    assert persona.get_enabled_tools() == ["google_grounding_search"]
    assert "fuzzy" in result["response"]


@pytest.mark.asyncio
async def test_set_tools_all_with_fuzzy_exclude(bot_logic_with_tools, mock_chat_system_with_state):
    """Fuzzy matching should work for excludes too."""
    persona = mock_chat_system_with_state.personas["derpr"]
    with patch.object(bot_logic_with_tools, '_query_llm_for_tool_selection',
                      new_callable=AsyncMock, return_value="google_grounding_search"):
        result = await bot_logic_with_tools.preprocess_message("derpr", "user1", "set tools all -grounding")
    assert result["mutated"] is True
    enabled = persona.get_enabled_tools()
    assert "google_grounding_search" not in enabled
    assert "web_search" in enabled
    assert "fuzzy" in result["response"]


@pytest.mark.asyncio
async def test_set_tools_bare_name_after_all_returns_error(bot_logic_with_tools, mock_chat_system_with_state):
    """Bare names after 'all' (without '-') should return a usage error."""
    result = await bot_logic_with_tools.preprocess_message("derpr", "user1", "set tools all web_search")
    assert result["mutated"] is False
    assert "'-' prefix" in result["response"]


# --- Handler completeness tests ---
# These ensure that every configurable Persona property is exposed via
# the detail, what, and set commands. If a new set_*/get_* method is
# added to Persona without updating the handlers, these tests fail.

# Maps Persona setter method names → expected command name in set_handlers.
# When you add a new set_* to Persona, add it here too — the test will
# tell you if you forget.
_SETTER_TO_COMMAND = {
    'set_prompt': 'prompt',
    'set_model_name': 'model',
    'set_response_token_limit': 'tokens',
    'set_context_length': 'context',
    'set_temperature': 'temp',
    'set_top_p': 'top_p',
    'set_top_k': 'top_k',
    'set_display_name_in_chat': 'display_name',
    'set_execution_mode': 'execution_mode',
    'set_enabled_tools': 'tools',
    'set_memory_mode': 'memory_mode',
    'set_service_bindings': 'service_bindings',
    'set_long_term_memory': 'long_term_memory',
    'set_include_ambient_memory': 'include_ambient_memory',
    'set_thinking_level': 'thinking_level',
}

# Maps Persona getter method names → expected command name in what_handlers.
# Not every getter needs a what command (e.g. get_name), so only include
# properties that users should be able to query.
_GETTER_TO_COMMAND = {
    'get_prompt': 'prompt',
    'get_model_name': 'model',
    'get_response_token_limit': 'tokens',
    'get_context_length': 'context',
    'get_temperature': 'temp',
    'get_top_p': 'top_p',
    'get_top_k': 'top_k',
    'get_execution_mode': 'execution_mode',
    'get_enabled_tools': 'tools',
    'get_memory_mode': 'memory_mode',
    'get_service_bindings': 'service_bindings',
    'get_long_term_memory': 'long_term_memory',
    'get_include_ambient_memory': 'include_ambient_memory',
    'get_thinking_level': 'thinking_level',
}

# Getters that intentionally have no what command (internal/derived values).
_GETTER_EXCEPTIONS = {
    'get_name',
    'get_base_context_length',
    'get_current_effective_context_length',
    'get_config_for_engine',
}


def test_all_persona_setters_have_commands(bot_logic):
    """Every set_* method on Persona must map to a set command."""
    persona_setters = {
        name for name in dir(Persona)
        if name.startswith('set_') and callable(getattr(Persona, name))
    }

    unmapped = persona_setters - set(_SETTER_TO_COMMAND.keys())
    assert not unmapped, (
        f"Persona has set_* methods with no entry in _SETTER_TO_COMMAND (and thus no 'set' command): {unmapped}. "
        f"Add them to _SETTER_TO_COMMAND in this test AND to set_handlers in BotLogic."
    )

    for setter_name, command_name in _SETTER_TO_COMMAND.items():
        assert command_name in bot_logic.set_handlers, (
            f"Persona.{setter_name}() exists but 'set {command_name}' is not in set_handlers"
        )


def test_all_persona_getters_have_what_commands(bot_logic):
    """Every get_* method on Persona (except known exceptions) must map to a what command."""
    persona_getters = {
        name for name in dir(Persona)
        if name.startswith('get_') and callable(getattr(Persona, name))
    }

    expected_getters = persona_getters - _GETTER_EXCEPTIONS
    unmapped = expected_getters - set(_GETTER_TO_COMMAND.keys())
    assert not unmapped, (
        f"Persona has get_* methods with no entry in _GETTER_TO_COMMAND or _GETTER_EXCEPTIONS: {unmapped}. "
        f"Add them to _GETTER_TO_COMMAND (if queryable) or _GETTER_EXCEPTIONS (if internal)."
    )

    for getter_name, command_name in _GETTER_TO_COMMAND.items():
        assert command_name in bot_logic.what_handlers, (
            f"Persona.{getter_name}() exists but 'what {command_name}' is not in what_handlers"
        )


@pytest.mark.asyncio
async def test_detail_shows_all_properties(bot_logic):
    """The detail command output must mention every user-facing persona property."""
    result = await bot_logic.preprocess_message("derpr", "user1", "detail")
    detail_text = result["response"].lower()

    # Each entry is a substring that must appear in the detail output.
    required_fields = [
        "model:",
        "memory mode:",
        "execution mode:",
        "service bindings:",
        "enabled tools:",
        "context length:",
        "display name",
        "response token limit:",
        "temperature:",
        "top p:",
        "top k:",
    ]
    missing = [f for f in required_fields if f.lower() not in detail_text]
    assert not missing, (
        f"detail command is missing these fields: {missing}\n\nFull output:\n{result['response']}"
    )


# --- Tests for the tool-call-based selection helper ---

@pytest.fixture
def bot_logic_with_selector(mock_chat_system_with_state):
    """BotLogic with model_selector + tool_selector personas and a mocked text_engine."""
    mock_chat_system_with_state.personas["model_selector"] = Persona(
        "model_selector", "gemma-4-31b-it", "Pick a model via select_model."
    )
    mock_chat_system_with_state.personas["tool_selector"] = Persona(
        "tool_selector", "gemma-4-31b-it", "Pick a tool via select_tool."
    )
    mock_chat_system_with_state.models_available = {
        "From Google": ["gemini-2.5-flash-lite", "gemini-2.5-flash"],
        "From Anthropic": ["claude-opus-4-7"],
    }
    mock_chat_system_with_state.text_engine = MagicMock()
    mock_chat_system_with_state.text_engine.generate_response = AsyncMock()
    return BotLogic(mock_chat_system_with_state)


@pytest.mark.asyncio
async def test_model_selection_returns_enum_choice(bot_logic_with_selector, mock_chat_system_with_state):
    mock_chat_system_with_state.text_engine.generate_response.return_value = (
        {"type": "tool_calls", "calls": [
            {"name": "select_model", "arguments": {"model": "gemini-2.5-flash"}}
        ]},
        None,
    )
    result = await bot_logic_with_selector._query_llm_for_model_selection("gemini 2.5")
    assert result == "gemini-2.5-flash"


@pytest.mark.asyncio
async def test_model_selection_default_sentinel_returns_none(bot_logic_with_selector, mock_chat_system_with_state):
    mock_chat_system_with_state.text_engine.generate_response.return_value = (
        {"type": "tool_calls", "calls": [
            {"name": "select_model", "arguments": {"model": "DEFAULT"}}
        ]},
        None,
    )
    assert await bot_logic_with_selector._query_llm_for_model_selection("gibberish") is None


@pytest.mark.asyncio
async def test_model_selection_text_response_returns_none(bot_logic_with_selector, mock_chat_system_with_state):
    """If the model ignores the tool and returns plain text, no match."""
    mock_chat_system_with_state.text_engine.generate_response.return_value = (
        {"type": "text", "content": "gemini-2.5-flash"},
        None,
    )
    assert await bot_logic_with_selector._query_llm_for_model_selection("gemini") is None


@pytest.mark.asyncio
async def test_model_selection_off_list_value_rejected(bot_logic_with_selector, mock_chat_system_with_state):
    """Enum violation by the provider should not leak through."""
    mock_chat_system_with_state.text_engine.generate_response.return_value = (
        {"type": "tool_calls", "calls": [
            {"name": "select_model", "arguments": {"model": "gpt-9"}}
        ]},
        None,
    )
    assert await bot_logic_with_selector._query_llm_for_model_selection("gpt") is None


@pytest.mark.asyncio
async def test_model_selection_string_arguments_parsed(bot_logic_with_selector, mock_chat_system_with_state):
    """Some providers return tool arguments as a JSON string; parse it."""
    mock_chat_system_with_state.text_engine.generate_response.return_value = (
        {"type": "tool_calls", "calls": [
            {"name": "select_model", "arguments": '{"model": "claude-opus-4-7"}'}
        ]},
        None,
    )
    result = await bot_logic_with_selector._query_llm_for_model_selection("opus")
    assert result == "claude-opus-4-7"


@pytest.mark.asyncio
async def test_model_selection_case_insensitive(bot_logic_with_selector, mock_chat_system_with_state):
    mock_chat_system_with_state.text_engine.generate_response.return_value = (
        {"type": "tool_calls", "calls": [
            {"name": "select_model", "arguments": {"model": "CLAUDE-OPUS-4-7"}}
        ]},
        None,
    )
    result = await bot_logic_with_selector._query_llm_for_model_selection("opus")
    assert result == "claude-opus-4-7"


@pytest.mark.asyncio
async def test_model_selection_missing_persona_returns_none(bot_logic_with_selector, mock_chat_system_with_state):
    del mock_chat_system_with_state.personas["model_selector"]
    assert await bot_logic_with_selector._query_llm_for_model_selection("anything") is None
    mock_chat_system_with_state.text_engine.generate_response.assert_not_called()


@pytest.mark.asyncio
async def test_tool_selection_returns_choice(bot_logic_with_selector, mock_chat_system_with_state):
    mock_chat_system_with_state.text_engine.generate_response.return_value = (
        {"type": "tool_calls", "calls": [
            {"name": "select_tool", "arguments": {"tool": "web_search"}}
        ]},
        None,
    )
    result = await bot_logic_with_selector._query_llm_for_tool_selection(
        "search", ["web_search", "create_ticket"]
    )
    assert result == "web_search"


@pytest.mark.asyncio
async def test_tool_selection_none_sentinel(bot_logic_with_selector, mock_chat_system_with_state):
    mock_chat_system_with_state.text_engine.generate_response.return_value = (
        {"type": "tool_calls", "calls": [
            {"name": "select_tool", "arguments": {"tool": "NONE"}}
        ]},
        None,
    )
    assert await bot_logic_with_selector._query_llm_for_tool_selection(
        "unrelated", ["web_search"]
    ) is None


@pytest.mark.asyncio
async def test_tool_selection_empty_list_returns_none(bot_logic_with_selector, mock_chat_system_with_state):
    assert await bot_logic_with_selector._query_llm_for_tool_selection("x", []) is None
    mock_chat_system_with_state.text_engine.generate_response.assert_not_called()


@pytest.mark.asyncio
async def test_selection_tool_schema_enum_includes_sentinel(bot_logic_with_selector, mock_chat_system_with_state):
    """Verify the tool schema passed to the engine has the sentinel in its enum."""
    mock_chat_system_with_state.text_engine.generate_response.return_value = (
        {"type": "tool_calls", "calls": [
            {"name": "select_model", "arguments": {"model": "DEFAULT"}}
        ]},
        None,
    )
    await bot_logic_with_selector._query_llm_for_model_selection("q")
    kwargs = mock_chat_system_with_state.text_engine.generate_response.call_args.kwargs
    tools = kwargs["tools"]
    assert len(tools) == 1
    enum_values = tools[0]["function"]["parameters"]["properties"]["model"]["enum"]
    assert "DEFAULT" in enum_values
    assert "gemini-2.5-flash" in enum_values
    assert "claude-opus-4-7" in enum_values


@pytest.mark.asyncio
async def test_selection_engine_exception_returns_none(bot_logic_with_selector, mock_chat_system_with_state):
    mock_chat_system_with_state.text_engine.generate_response.side_effect = RuntimeError("boom")
    assert await bot_logic_with_selector._query_llm_for_model_selection("q") is None
