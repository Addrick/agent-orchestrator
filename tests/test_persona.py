import pytest
from src.persona import Persona
from config import global_config


@pytest.fixture
def default_persona():
    """Provides a default Persona instance for testing."""
    return Persona(
        persona_name="tester",
        model_name="test_model",
        prompt="This is a test prompt.",
        token_limit=500,
        context_length=10,
        temperature=0.5,
        top_p=0.9,
        top_k=40
    )


def test_persona_initialization(default_persona):
    """Tests that the Persona class initializes attributes correctly."""
    assert default_persona.get_name() == "tester"
    assert default_persona.get_model_name() == "test_model"
    assert default_persona.get_prompt() == "This is a test prompt."
    assert default_persona.get_response_token_limit() == 500
    assert default_persona.get_context_length() == 10
    assert default_persona.get_temperature() == 0.5
    assert default_persona.get_top_p() == 0.9
    assert default_persona.get_top_k() == 40


def test_persona_initialization_with_defaults():
    """Tests initialization with only required parameters, relying on defaults."""
    persona = Persona(persona_name="default_tester", model_name="default_model", prompt="default prompt")
    assert persona.get_name() == "default_tester"
    assert persona.get_model_name() == "default_model"
    assert persona.get_prompt() == "default prompt"
    assert persona.get_response_token_limit() == global_config.DEFAULT_TOKEN_LIMIT
    assert persona.get_context_length() == global_config.DEFAULT_CONTEXT_LIMIT
    assert persona.get_temperature() is None
    assert persona.get_top_p() is None
    assert persona.get_top_k() is None


def test_set_model_name(default_persona):
    """Tests setting the model name."""
    default_persona.set_model_name("new_test_model")
    assert default_persona.get_model_name() == "new_test_model"


def test_set_prompt(default_persona):
    """Tests setting the prompt."""
    default_persona.set_prompt("This is a new test prompt.")
    assert default_persona.get_prompt() == "This is a new test prompt."


def test_append_to_prompt(default_persona):
    """Tests appending text to the prompt."""
    initial_prompt = default_persona.get_prompt()
    default_persona.append_to_prompt(" Appended text.")
    assert default_persona.get_prompt() == initial_prompt + " Appended text."


def test_set_response_token_limit_valid(default_persona):
    """Tests setting a valid response token limit."""
    default_persona.set_response_token_limit(1024)
    assert default_persona.get_response_token_limit() == 1024


def test_set_response_token_limit_invalid(default_persona):
    """Tests setting an invalid response token limit."""
    default_persona.set_response_token_limit("not a number")
    assert default_persona.get_response_token_limit() is None


def test_set_response_token_limit_too_low(default_persona):
    """Tests that setting a token limit below 100 defaults to 100."""
    default_persona.set_response_token_limit(50)
    assert default_persona.get_response_token_limit() == 100


def test_set_context_length_valid(default_persona):
    """Tests setting a valid context length."""
    default_persona.set_context_length(20)
    assert default_persona.get_context_length() == 20


def test_set_context_length_invalid(default_persona):
    """Tests setting an invalid context length."""
    default_persona.set_context_length("twenty")
    assert default_persona.get_context_length() is None


def test_set_temperature_valid(default_persona):
    """Tests setting a valid temperature."""
    default_persona.set_temperature(0.99)
    assert default_persona.get_temperature() == 0.99


def test_set_temperature_invalid(default_persona):
    """Tests setting an invalid temperature."""
    default_persona.set_temperature("hot")
    assert default_persona.get_temperature() is None


def test_set_top_p_valid(default_persona):
    """Tests setting a valid top_p value."""
    default_persona.set_top_p(0.95)
    assert default_persona.get_top_p() == 0.95


def test_set_top_p_invalid(default_persona):
    """Tests setting an invalid top_p value."""
    default_persona.set_top_p("ninety-five")
    assert default_persona.get_top_p() is None


def test_set_top_k_valid(default_persona):
    """Tests setting a valid top_k value."""
    default_persona.set_top_k(50)
    assert default_persona.get_top_k() == 50


def test_set_top_k_invalid(default_persona):
    """Tests setting an invalid top_k value."""
    default_persona.set_top_k("fifty")
    assert default_persona.get_top_k() is None


def test_get_config_for_engine(default_persona):
    """Tests that the engine config dictionary is created correctly."""
    config = default_persona.get_config_for_engine()
    expected_config = {
        "model_name": "test_model",
        "max_output_tokens": 500,
        "temperature": 0.5,
        "top_p": 0.9,
        "top_k": 40,
    }
    assert config == expected_config

    # Change a value and check again
    default_persona.set_model_name("updated_model")
    default_persona.set_temperature(None)
    config = default_persona.get_config_for_engine()

    expected_config["model_name"] = "updated_model"
    expected_config["temperature"] = None
    assert config == expected_config
