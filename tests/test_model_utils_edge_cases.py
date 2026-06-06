# tests/test_model_utils_edge_cases.py
"""
DP-199 Batch 7 — model_utils edge cases.

Covers:
  * refresh_available_* with no key, network error, malformed payload
  * get_model_list update vs no-update paths
  * check_model_available casing / malformed list / missing list
  * _is_chat_model blacklist + prefix filtering
  * get_model_prefix unknown
"""

from types import SimpleNamespace
from unittest.mock import patch, MagicMock

import pytest

from src.utils import model_utils


# ---------------------------------------------------------------------------
# refresh_available_openai_models
# ---------------------------------------------------------------------------

def test_refresh_openai_no_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    assert model_utils.refresh_available_openai_models() == []


def test_refresh_openai_network_error(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "dummy")
    fake_openai = MagicMock()
    fake_client = MagicMock()
    fake_client.models.list.side_effect = ConnectionError("upstream down")
    fake_openai.OpenAI.return_value = fake_client
    with patch.dict("sys.modules", {"openai": fake_openai}):
        assert model_utils.refresh_available_openai_models() == []


# ---------------------------------------------------------------------------
# refresh_available_google_models
# ---------------------------------------------------------------------------

def test_refresh_google_no_key(monkeypatch):
    monkeypatch.delenv("GOOGLE_GENERATIVEAI_API_KEY", raising=False)
    assert model_utils.refresh_available_google_models() == []


def test_refresh_google_malformed_model(monkeypatch):
    """Malformed entries (no name / no supported_actions / wrong prefix) are filtered."""
    monkeypatch.setenv("GOOGLE_GENERATIVEAI_API_KEY", "dummy")

    models = [
        # No name → filtered
        SimpleNamespace(name=None, supported_actions=["generateContent"]),
        # No generateContent → filtered
        SimpleNamespace(name="models/gemini-2.5-flash", supported_actions=["embedContent"]),
        # No supported_actions at all → filtered
        SimpleNamespace(name="models/gemini-other", supported_actions=None),
        # Non-chat prefix → filtered by _is_chat_model
        SimpleNamespace(name="models/text-bison-001", supported_actions=["generateContent"]),
        # Valid
        SimpleNamespace(name="models/gemini-2.5-flash", supported_actions=["generateContent"]),
        # Duplicate of the valid one → de-duped
        SimpleNamespace(name="models/gemini-2.5-flash", supported_actions=["generateContent"]),
        # Another valid one
        SimpleNamespace(name="models/gemma-3-27b-it", supported_actions=["generateContent"]),
    ]

    fake_client = MagicMock()
    fake_client.models.list.return_value = iter(models)
    fake_genai = MagicMock()
    fake_genai.Client.return_value = fake_client
    fake_google_module = MagicMock()
    fake_google_module.genai = fake_genai
    with patch.dict("sys.modules", {"google": fake_google_module, "google.genai": fake_genai}):
        result = model_utils.refresh_available_google_models()
    assert result == ["gemini-2.5-flash", "gemma-3-27b-it"]


def test_refresh_google_error(monkeypatch):
    """An exception from the Google client returns []."""
    monkeypatch.setenv("GOOGLE_GENERATIVEAI_API_KEY", "dummy")
    fake_genai = MagicMock()
    fake_genai.Client.side_effect = RuntimeError("boom")
    fake_google_module = MagicMock()
    fake_google_module.genai = fake_genai
    with patch.dict("sys.modules", {"google": fake_google_module, "google.genai": fake_genai}):
        assert model_utils.refresh_available_google_models() == []


# ---------------------------------------------------------------------------
# refresh_available_anthropic_models
# ---------------------------------------------------------------------------

def test_refresh_anthropic_no_key(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    assert model_utils.refresh_available_anthropic_models() == []


def test_refresh_anthropic_error(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "dummy")
    fake_anthropic = MagicMock()
    fake_client = MagicMock()
    fake_client.models.list.side_effect = TimeoutError("slow")
    fake_anthropic.Anthropic.return_value = fake_client
    with patch.dict("sys.modules", {"anthropic": fake_anthropic}):
        assert model_utils.refresh_available_anthropic_models() == []


# ---------------------------------------------------------------------------
# get_model_list
# ---------------------------------------------------------------------------

def test_get_model_list_no_update_missing_file():
    """When update=False and the underlying file is missing, returns None."""
    with patch("src.utils.model_utils.save_utils.load_models_from_file", return_value=None):
        assert model_utils.get_model_list(update=False) is None


def test_get_model_list_update_saves():
    """update=True calls all three refreshers and saves the combined result."""
    with patch("src.utils.model_utils.refresh_available_openai_models", return_value=["gpt-4"]) as m_o, \
         patch("src.utils.model_utils.refresh_available_google_models", return_value=["gemini-2.5"]) as m_g, \
         patch("src.utils.model_utils.refresh_available_anthropic_models", return_value=["claude-3"]) as m_a, \
         patch("src.utils.model_utils.save_utils.save_models_to_file") as m_save:
        result = model_utils.get_model_list(update=True)
    m_o.assert_called_once()
    m_g.assert_called_once()
    m_a.assert_called_once()
    expected = {
        "From OpenAI": ["gpt-4"],
        "From Google": ["gemini-2.5"],
        "From Anthropic": ["claude-3"],
        "Local": ["local"],
    }
    m_save.assert_called_once_with(expected)
    assert result == expected


# ---------------------------------------------------------------------------
# check_model_available
# ---------------------------------------------------------------------------

def test_check_model_available_no_list():
    """If get_model_list returns None or empty, check_model_available returns False."""
    with patch("src.utils.model_utils.get_model_list", return_value=None):
        assert model_utils.check_model_available("anything") is False
    with patch("src.utils.model_utils.get_model_list", return_value={}):
        assert model_utils.check_model_available("anything") is False


def test_check_model_available_malformed_list():
    """Scalars in the model-list values are accepted (coerced to string)."""
    with patch(
        "src.utils.model_utils.get_model_list",
        return_value={"WeirdProvider": "single-string-model", "Local": ["local"]},
    ):
        # The scalar gets stringified and lowered into the lookup pool.
        assert model_utils.check_model_available("single-string-model") is True
        assert model_utils.check_model_available("local") is True


def test_check_model_available_case_insensitive():
    with patch(
        "src.utils.model_utils.get_model_list",
        return_value={"From OpenAI": ["GPT-4"], "Local": ["local"]},
    ):
        assert model_utils.check_model_available("gpt-4") is True
        assert model_utils.check_model_available("GpT-4") is True


def test_check_model_available_not_found():
    with patch(
        "src.utils.model_utils.get_model_list",
        return_value={"From OpenAI": ["gpt-4"], "Local": ["local"]},
    ):
        assert model_utils.check_model_available("not-a-real-model") is False


# ---------------------------------------------------------------------------
# _is_chat_model / get_model_prefix
# ---------------------------------------------------------------------------

def test_is_chat_model_blacklist():
    """Blacklisted substrings always lose, even if the prefix is otherwise allowed."""
    assert model_utils._is_chat_model("text-embedding-ada") is False
    assert model_utils._is_chat_model("gpt-4-whisper") is False
    assert model_utils._is_chat_model("claude-3-latest") is False
    assert model_utils._is_chat_model("gpt-4-dall-e") is False


def test_is_chat_model_no_prefix():
    """Models without a recognized chat prefix are rejected."""
    assert model_utils._is_chat_model("random-model-name") is False
    assert model_utils._is_chat_model("text-bison") is False


def test_get_model_prefix_unknown():
    assert model_utils.get_model_prefix("some-novel-vendor-model") == "unknown"
    assert model_utils.get_model_prefix("gpt-4") == "gpt"
    assert model_utils.get_model_prefix("claude-3-opus") == "claude"
    assert model_utils.get_model_prefix("gemini-2.5-flash") == "gemini"
    assert model_utils.get_model_prefix("gemma-3-27b-it") == "gemma"
    assert model_utils.get_model_prefix("local") == "local"
