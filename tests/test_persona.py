# tests/persona/test_persona.py

import pytest
from src.persona import Persona
from config import global_config

# Use a known value for the default limit in tests
TEST_DEFAULT_LIMIT = 15

@pytest.fixture(autouse=True)
def patch_global_config(monkeypatch):
    """Fixture to ensure a consistent default context limit for all tests."""
    monkeypatch.setattr(global_config, 'DEFAULT_CONTEXT_LIMIT', TEST_DEFAULT_LIMIT)

@pytest.fixture
def base_persona_args():
    """Provides a dictionary of basic arguments to create a Persona."""
    return {
        "persona_name": "tester",
        "model_name": "test_model",
        "prompt": "You are a test persona."
    }

@pytest.fixture
def persona(base_persona_args):
    """Provides a standard Persona instance for tests."""
    return Persona(**base_persona_args)


# --- Initialization Tests ---

def test_persona_initialization_with_all_values(base_persona_args):
    """Tests that a Persona is created correctly when all values are provided."""
    p = Persona(
        **base_persona_args,
        context_length=10,
        memory_type="personal",
        display_name_in_chat=True
    )
    assert p.get_name() == "tester"
    assert p.get_context_length() == 10
    assert p.get_memory_type() == "personal"
    assert p.should_display_name_in_chat() is True

def test_persona_initialization_defaults_context_length(base_persona_args):
    """
    Tests that context_length defaults to the global config if the argument is None.
    This is a critical test for our new logic.
    """
    p = Persona(**base_persona_args, context_length=None)
    assert p.get_context_length() == TEST_DEFAULT_LIMIT

def test_persona_initialization_uses_provided_zero_context(base_persona_args):
    """Tests that an explicit context_length of 0 is respected during initialization."""
    p = Persona(**base_persona_args, context_length=0)
    assert p.get_context_length() == 0

# --- Setter Tests for context_length ---

def test_set_context_length_valid(persona):
    """Tests setting a valid, non-zero context length."""
    result = persona.set_context_length(5)
    assert result == 5
    assert persona.get_context_length() == 5

def test_set_context_length_zero(persona):
    """Tests that setting context length to 0 is a valid operation."""
    result = persona.set_context_length(0)
    assert result == 0
    assert persona.get_context_length() == 0

def test_set_context_length_invalid(persona):
    """
    Tests that setting an invalid context length (e.g., a string) causes it to
    revert to the global default. This is the fix for the failing test.
    """
    persona.set_context_length(99) # Start with a known value
    result = persona.set_context_length("invalid_string")
    assert result == TEST_DEFAULT_LIMIT
    assert persona.get_context_length() == TEST_DEFAULT_LIMIT

# --- Setter Tests for Other Attributes ---

def test_set_memory_type(persona):
    """Tests the setter for memory_type."""
    assert persona.get_memory_type() == "auto"
    assert persona.set_memory_type("personal") is True
    assert persona.get_memory_type() == "personal"
    assert persona.set_memory_type("invalid_type") is False
    assert persona.get_memory_type() == "personal"

def test_set_display_name(persona):
    """Tests the setter for display_name_in_chat."""
    assert persona.should_display_name_in_chat() is False
    persona.set_display_name_in_chat(True)
    assert persona.should_display_name_in_chat() is True

def test_set_prompt(persona):
    """Tests the setter for the prompt."""
    new_prompt = "This is a new prompt."
    persona.set_prompt(new_prompt)
    assert persona.get_prompt() == new_prompt

def test_set_model_name(persona):
    """Tests the setter for the model name."""
    new_model = "gpt-5"
    persona.set_model_name(new_model)
    assert persona.get_model_name() == new_model

def test_set_response_token_limit(persona):
    """Tests valid, invalid, and edge cases for setting token limit."""
    assert persona.set_response_token_limit(500) == 500
    assert persona.get_response_token_limit() == 500
    # Test the minimum value enforcement
    assert persona.set_response_token_limit(50) == 100
    assert persona.get_response_token_limit() == 100
    # Test invalid value
    assert persona.set_response_token_limit("invalid") is None
    assert persona.get_response_token_limit() is None

def test_set_temperature(persona):
    """Tests valid and invalid values for temperature."""
    assert persona.set_temperature(0.8) == 0.8
    assert persona.get_temperature() == 0.8
    assert persona.set_temperature("invalid") is None
    assert persona.get_temperature() is None

def test_set_top_p(persona):
    """Tests valid and invalid values for top_p."""
    assert persona.set_top_p(0.9) == 0.9
    assert persona.get_top_p() == 0.9
    assert persona.set_top_p("invalid") is None
    assert persona.get_top_p() is None

def test_set_top_k(persona):
    """Tests valid and invalid values for top_k."""
    assert persona.set_top_k(40) == 40
    assert persona.get_top_k() == 40
    assert persona.set_top_k("invalid") is None
    assert persona.get_top_k() is None

# --- Utility Method Tests ---

def test_append_to_prompt(persona):
    """Tests appending text to the existing prompt."""
    initial_prompt = persona.get_prompt()
    persona.append_to_prompt(" More text.")
    assert persona.get_prompt() == initial_prompt + " More text."

def test_get_config_for_engine(base_persona_args):
    """Tests that the engine config dictionary is assembled correctly."""
    p = Persona(
        **base_persona_args,
        token_limit=1024,
        temperature=0.8,
        top_p=0.9,
        top_k=40
    )
    expected_config = {
        "model_name": "test_model",
        "max_output_tokens": 1024,
        "temperature": 0.8,
        "top_p": 0.9,
        "top_k": 40,
    }
    assert p.get_config_for_engine() == expected_config
