from src.persona import Persona

def test_persona_initialization(default_persona: Persona):
    """Tests that the Persona class initializes with correct values."""
    assert default_persona.get_name() == "tester"
    assert default_persona.get_model_name() == "test-model"
    assert default_persona.get_prompt() == "You are a test bot."

def test_set_response_token_limit_valid(default_persona: Persona):
    """Tests setting a valid integer token limit."""
    default_persona.set_response_token_limit(500)
    assert default_persona.get_response_token_limit() == 500

def test_set_response_token_limit_invalid(default_persona: Persona):
    """Tests setting an invalid (non-integer) token limit."""
    default_persona.set_response_token_limit("not a number")
    assert default_persona.get_response_token_limit() is None