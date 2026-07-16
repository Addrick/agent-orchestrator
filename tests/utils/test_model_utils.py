import pytest
from unittest.mock import patch, Mock
from src.utils import model_utils


@patch('src.utils.model_utils.persona_store.load_models_from_file')
def test_get_model_list_no_update(mock_load_models):
    """update=False reads the cache and merges the static (non-API) models in."""
    mock_load_models.return_value = {"From OpenAI": ["gpt-4"]}

    result = model_utils.get_model_list(update=False)

    mock_load_models.assert_called_once()
    # API group from the cache is preserved...
    assert result["From OpenAI"] == ["gpt-4"]
    # ...and the static groups are always present.
    assert result["Antigravity (OAuth tier)"] == ["agy-flash"]
    assert result["Local"] == ["local"]


@patch('src.utils.model_utils.persona_store.load_models_from_file')
def test_get_model_list_no_update_stale_cache_gets_static_models(mock_load_models):
    """A cache written before agy existed still exposes agy/local on read.

    This is the bug the fix targets: the web UI dropdown and `what models`
    both read the cache, so static models must not depend on update_models
    having been run after they were introduced.
    """
    mock_load_models.return_value = {
        "From OpenAI": ["gpt-4"],
        "From Anthropic": ["claude-3"],
        "Local": ["local"],
        # NOTE: no 'Antigravity (OAuth tier)' group — pre-agy cache shape.
    }

    result = model_utils.get_model_list(update=False)

    assert "Antigravity (OAuth tier)" in result
    assert result["Antigravity (OAuth tier)"] == ["agy-flash"]
    # The pre-existing API/local groups are untouched.
    assert result["From OpenAI"] == ["gpt-4"]


@patch('src.utils.model_utils.persona_store.load_models_from_file')
def test_get_model_list_no_update_empty_cache(mock_load_models):
    """No cache at all (fresh install) still surfaces the static models."""
    mock_load_models.return_value = None

    result = model_utils.get_model_list(update=False)

    assert result == model_utils.STATIC_MODELS
    assert result["Antigravity (OAuth tier)"] == ["agy-flash"]
    assert result["Local"] == ["local"]


@patch('src.utils.model_utils.persona_store.save_models_to_file')
@patch('src.utils.model_utils.refresh_available_anthropic_models')
@patch('src.utils.model_utils.refresh_available_google_models')
@patch('src.utils.model_utils.refresh_available_openai_models')
def test_get_model_list_with_update(mock_openai, mock_google, mock_anthropic, mock_save):
    """Tests get_model_list when update=True, ensuring it calls refresh and save utilities."""
    mock_openai.return_value = ["gpt-4"]
    mock_google.return_value = ["gemini-pro"]
    mock_anthropic.return_value = ["claude-3"]

    expected_combined_dict = {
        'From OpenAI': ["gpt-4"],
        'From Google': ["gemini-pro"],
        'From Anthropic': ["claude-3"],
        'Antigravity (OAuth tier)': ['agy-flash'],
        'Claude Code (sandboxed)': ['cc-sonnet', 'cc-opus', 'cc-haiku'],
        'Local': ['local']
    }

    result = model_utils.get_model_list(update=True)

    mock_openai.assert_called_once()
    mock_google.assert_called_once()
    mock_anthropic.assert_called_once()
    mock_save.assert_called_once_with(expected_combined_dict)
    assert result == expected_combined_dict


@patch('src.utils.model_utils.get_model_list')
def test_check_model_available(mock_get_list):
    """Tests the check_model_available utility."""
    mock_get_list.return_value = {
        "ProviderA": ["model-a1", "model-a2"],
        "ProviderB": ["model-b1"],
        "Local": ["local-model"]
    }

    assert model_utils.check_model_available("model-b1") is True
    assert model_utils.check_model_available("local-model") is True
    assert model_utils.check_model_available("non-existent-model") is False


@pytest.mark.parametrize("model_name,expected", [
    ("agy-flash", "agy"),
    ("AGY-Flash", "agy"),
    ("gpt-4", "gpt"),
    ("claude-3-opus", "claude"),
    ("gemini-3.1-pro", "gemini-3.1"),
    ("gemini-pro", "gemini"),
    ("gemma-4-31b-it", "gemma"),
    ("local", "local"),
    ("something-else", "unknown"),
])
def test_get_model_prefix(model_name, expected):
    """The agy family must resolve so routing/registration recognise it."""
    assert model_utils.get_model_prefix(model_name) == expected


@pytest.mark.parametrize("model_name,expected_template", [
    # Gemma 4 models
    ("gemma-4-31b-it", "gemma4-think"),
    ("GEMMA-4-31B-IT", "gemma4-think"),
    ("gemma-4-26b-a4b-it", "gemma4-think"),
    ("gemma-4-e2b", "gemma4-e-nothink"),
    ("gemma-4-e4b", "gemma4-e-nothink"),
    ("gemma-4", "gemma4-think"),
    # Gemma 2 and 3 models
    ("gemma-2-9b", "gemma"),
    ("gemma-3-9b", "gemma"),
    ("gemma", "gemma"),
    # Qwen models
    ("qwen", "chatml"),
    ("qwen-32b", "chatml"),
    # Llama models — both hyphenated GGUF names and the unhyphenated spelling
    ("llama-3-70b", "llama3"),
    ("llama-4-70b", "llama4"),
    ("llama2-70b", "llama2"),
    ("llama-2-7b-chat", "llama2"),  # the common GGUF form must not fall to chatml
    ("Llama-2-13B", "llama2"),      # case-insensitive
    # ChatML family
    ("mistral-7b", "chatml"),
    ("hermes-2", "chatml"),
    ("chatml-test", "chatml"),
    # No match
    ("unknown-model", None),
    (None, None),
    ("", None),
])
def test_get_chat_template_for_model(model_name, expected_template):
    """Model name pattern matching returns correct chat templates."""
    assert model_utils.get_chat_template_for_model(model_name) == expected_template


# --- get_current_kobold_model (async, /api/v1/model, cached) ---------------
# These mock the REAL endpoint /api/v1/model whose shape is
# {"result": "koboldcpp/<name>"}. The retired implementation queried
# /api/extra/version (which carries no model field) and always returned None —
# the feature was dead. Each test clears the per-base-URL cache first.

class _FakeGetClient:
    """Minimal httpx.AsyncClient stand-in exposing only .get()."""

    def __init__(self, *, status=200, payload=None, exc=None):
        self._status = status
        self._payload = payload if payload is not None else {}
        self._exc = exc
        self.calls = []

    async def get(self, url, timeout=None):
        self.calls.append(url)
        if self._exc is not None:
            raise self._exc
        resp = Mock()
        resp.status_code = self._status
        resp.json.return_value = self._payload
        return resp


@pytest.mark.asyncio
async def test_get_current_kobold_model_success():
    """Reads data['result'] from /api/v1/model; koboldcpp/ prefix kept intact."""
    model_utils._KOBOLD_MODEL_CACHE.clear()
    client = _FakeGetClient(payload={"result": "koboldcpp/gemma-4-31b-it"})
    result = await model_utils.get_current_kobold_model(client, "http://kobold:5001")
    assert result == "koboldcpp/gemma-4-31b-it"
    assert client.calls == ["http://kobold:5001/api/v1/model"]


@pytest.mark.asyncio
async def test_get_current_kobold_model_missing_result():
    """200 with no 'result' field → None."""
    model_utils._KOBOLD_MODEL_CACHE.clear()
    client = _FakeGetClient(payload={"version": "1.115.2"})
    result = await model_utils.get_current_kobold_model(client, "http://kobold:5001")
    assert result is None


@pytest.mark.asyncio
async def test_get_current_kobold_model_connection_failure():
    """Transport error → None (best-effort, swallowed)."""
    model_utils._KOBOLD_MODEL_CACHE.clear()
    client = _FakeGetClient(exc=Exception("Connection refused"))
    result = await model_utils.get_current_kobold_model(client, "http://kobold:5001")
    assert result is None


@pytest.mark.asyncio
async def test_get_current_kobold_model_non_200():
    """Non-200 response → None."""
    model_utils._KOBOLD_MODEL_CACHE.clear()
    client = _FakeGetClient(status=404, payload={"result": "nope"})
    result = await model_utils.get_current_kobold_model(client, "http://kobold:5001")
    assert result is None


@pytest.mark.asyncio
async def test_get_current_kobold_model_cached():
    """Second call within TTL is served from cache — no second HTTP round-trip."""
    model_utils._KOBOLD_MODEL_CACHE.clear()
    client = _FakeGetClient(payload={"result": "koboldcpp/qwen3-40b"})
    base = "http://kobold-cache:5001"
    first = await model_utils.get_current_kobold_model(client, base)
    second = await model_utils.get_current_kobold_model(client, base)
    assert first == second == "koboldcpp/qwen3-40b"
    assert len(client.calls) == 1  # cache hit on the second call


@pytest.mark.asyncio
async def test_get_current_kobold_model_negative_result_short_ttl(monkeypatch):
    """A None (failure) result caches only for the short negative TTL, so a
    transient outage does not pin the default template for the full 60s."""
    model_utils._KOBOLD_MODEL_CACHE.clear()
    base = "http://kobold-neg:5001"

    fake_now = {"t": 1000.0}
    monkeypatch.setattr(model_utils.time, "monotonic", lambda: fake_now["t"])

    # First call fails → None cached at t=1000.
    failing = _FakeGetClient(exc=Exception("Connection refused"))
    assert await model_utils.get_current_kobold_model(failing, base) is None
    assert len(failing.calls) == 1

    # Within the negative TTL → served from cache, no re-query.
    fake_now["t"] = 1000.0 + model_utils._KOBOLD_MODEL_CACHE_NEG_TTL - 0.1
    assert await model_utils.get_current_kobold_model(failing, base) is None
    assert len(failing.calls) == 1

    # Past the negative TTL (but well within the 60s success TTL) → re-query,
    # and kobold is now up so detection recovers.
    fake_now["t"] = 1000.0 + model_utils._KOBOLD_MODEL_CACHE_NEG_TTL + 0.1
    recovered = _FakeGetClient(payload={"result": "koboldcpp/gemma-4-31b-it"})
    assert (
        await model_utils.get_current_kobold_model(recovered, base)
        == "koboldcpp/gemma-4-31b-it"
    )
    assert len(recovered.calls) == 1


@pytest.mark.asyncio
async def test_get_current_kobold_model_success_holds_full_ttl(monkeypatch):
    """A successful detection stays cached for the full (long) TTL — the short
    negative TTL must not shorten successful entries."""
    model_utils._KOBOLD_MODEL_CACHE.clear()
    base = "http://kobold-pos:5001"

    fake_now = {"t": 2000.0}
    monkeypatch.setattr(model_utils.time, "monotonic", lambda: fake_now["t"])

    client = _FakeGetClient(payload={"result": "koboldcpp/qwen3-40b"})
    assert await model_utils.get_current_kobold_model(client, base) == "koboldcpp/qwen3-40b"

    # Past the negative TTL but before the success TTL → still cached.
    fake_now["t"] = 2000.0 + model_utils._KOBOLD_MODEL_CACHE_NEG_TTL + 1.0
    assert await model_utils.get_current_kobold_model(client, base) == "koboldcpp/qwen3-40b"
    assert len(client.calls) == 1  # no re-query
