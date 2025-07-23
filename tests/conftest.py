import pytest
from src.persona import Persona

@pytest.fixture
def default_persona():
    """Provides a default Persona instance for tests."""
    return Persona(
        persona_name="tester",
        model_name="test-model",
        prompt="You are a test bot."
    )