# src/utils/model_utils.py

import logging
import os
import time
from typing import List, Dict, Any, Optional, Tuple

import httpx

from src.personas import store as persona_store

logger = logging.getLogger(__name__)

# --- Filtering Configuration ---
# Only models starting with these prefixes will be included in the chat-focused list.
CHAT_MODEL_PREFIXES = [
    "gpt-4", "gpt-3.5", "o1-", "o3-",  # OpenAI
    "claude-",                         # Anthropic
    "gemini-", "gemma-",               # Google
]

# Explicitly exclude models containing these substrings (e.g., embeddings, tts, etc.)
MODEL_BLACKLIST = [
    "embedding", "whisper", "tts", "dall-e", "babbage", "davinci",
    "vision-only", "search-internal", "safety-force", "001", "latest"
]


def get_model_prefix(model_name: str) -> str:
    """Return the model family prefix for routing and compatibility checks."""
    name_lower = model_name.lower()
    # cc-* (Claude Code) is checked before the `"claude" in name_lower` branch
    # so a future alias like `cc-claude-*` can't be misclassified as the
    # Anthropic API family — mirrors the precedence in engine._get_provider_route.
    if name_lower.startswith("cc-"):
        return "cc"
    elif name_lower.startswith("gpt"):
        return "gpt"
    elif "claude" in name_lower:
        return "claude"
    elif "gemma" in name_lower:
        return "gemma"
    elif "gemini-3.1" in name_lower:
        return "gemini-3.1"
    elif "gemini" in name_lower:
        return "gemini"
    elif name_lower.startswith("agy"):
        return "agy"
    elif name_lower == "local":
        return "local"
    return "unknown"


def _is_chat_model(model_id: str) -> bool:
    """Check if a model ID matches chat/generative patterns and isn't blacklisted."""
    model_id = model_id.lower()
    if any(blacklisted in model_id for blacklisted in MODEL_BLACKLIST):
        return False
    return any(model_id.startswith(prefix) for prefix in CHAT_MODEL_PREFIXES)


def refresh_available_openai_models() -> List[str]:
    """# OpenAI API query to get current list of active models with filtering."""
    import openai
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        logger.warning("OPENAI_API_KEY not set — skipping OpenAI model refresh.")
        return []
    try:
        client = openai.OpenAI(api_key=api_key)
        openai_models = client.models.list()
        filtered_list = [m.id for m in openai_models if _is_chat_model(m.id)]
        logger.debug(f"OpenAI models after filtering: {filtered_list}")
        return filtered_list
    except Exception as e:
        logger.error(f"Error refreshing OpenAI models: {e}")
        return []


def refresh_available_google_models() -> List[str]:
    """Uses the google-genai SDK to list available generative models with filtering."""
    from google import genai

    api_key = os.environ.get("GOOGLE_GENERATIVEAI_API_KEY")
    if not api_key:
        logger.warning("GOOGLE_GENERATIVEAI_API_KEY not set — skipping Google model refresh.")
        return []
    try:
        client = genai.Client(api_key=api_key)
        google_models: List[str] = []
        for model in client.models.list():
            if not model.name:
                continue
            # Basic validation of generative capability
            if model.supported_actions and 'generateContent' in model.supported_actions:
                model_id = model.name.split("/")[-1]  # remove preceding 'models/'
                if _is_chat_model(model_id) and model_id not in google_models:
                    google_models.append(model_id)
        return google_models
    except Exception as e:
        logger.error(f"Error refreshing Google models: {e}")
        return []


def refresh_available_anthropic_models() -> List[str]:
    """Anthropic API query to get model list with filtering."""
    import anthropic
    api_key: Optional[str] = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        logger.warning("ANTHROPIC_API_KEY not set — skipping Anthropic model refresh.")
        return []
    try:
        client = anthropic.Anthropic(api_key=api_key)
        models = client.models.list(limit=20)
        filtered_ids = [m.id for m in models.data if _is_chat_model(m.id)]
        return filtered_ids
    except Exception as e:
        logger.error(f"Error refreshing Anthropic models: {e}")
        return []


# Models that exist by code, not by any provider API. They must always appear
# in the model list regardless of whether `update_models` (the slow API refresh)
# has ever run, and must survive a cache written before they were introduced.
# Keyed by display group so they slot into the same shape as the API groups.
STATIC_MODELS: Dict[str, List[str]] = {
    'Antigravity (OAuth tier)': ['agy-flash'],
    'Claude Code (sandboxed)': ['cc-sonnet', 'cc-opus', 'cc-haiku'],
    'Local': ['local'],
}


def get_model_list(update: bool = False) -> Optional[Dict[str, Any]]:
    """Get available models, optionally refreshing from APIs.

    Static (non-API) models in `STATIC_MODELS` are always merged in — on the
    update path and on the cached-read path — so both the `what models` command
    and the web UI model dropdown list them even if the cache predates them or
    `update_models` was never run.
    """
    if update:
        logger.info('Updating available models from API...')
        all_available_models: Dict[str, List[str]] = {
            'From OpenAI': refresh_available_openai_models(),
            'From Google': refresh_available_google_models(),
            'From Anthropic': refresh_available_anthropic_models(),
            **STATIC_MODELS,
        }
        persona_store.save_models_to_file(all_available_models)
        return all_available_models

    cached = persona_store.load_models_from_file()
    if cached is None:
        # No cache yet (fresh install / before first update): still expose the
        # static models so agy/local are selectable without an API round-trip.
        return dict(STATIC_MODELS)
    # Merge statics over the cache without clobbering refreshed API groups.
    for group, names in STATIC_MODELS.items():
        cached[group] = names
    return cached


def check_model_available(model_to_check: str) -> bool:
    """Check if a specific model is available (case-insensitive)."""
    model_list = get_model_list()
    if not model_list:
        return False

    all_names = []
    for value in model_list.values():
        if isinstance(value, list):
            all_names.extend([v.lower() for v in value])
        else:
            all_names.append(str(value).lower())

    check_lower = model_to_check.lower()
    if check_lower in all_names:
        # Return the original casing from the list if possible, but for this check, boolean is enough
        return True

    logger.info(f"Model '{model_to_check}' not found in available models.")
    return False


# --- Model-to-Chat-Template Lookup ---
# Maps model name patterns (case-insensitive) to the correct chat template.
# This ensures that when a local model is served via koboldcpp, the correct
# tagging scheme is used (e.g., Gemma uses <start_of_turn>, not ChatML tags).
# Patterns are checked in order; first match wins. More specific patterns must
# come before their less-specific counterparts.
MODEL_TO_CHAT_TEMPLATE: List[Tuple[str, str]] = [
    # Gemma 4 models (26B/31B with thinking, E2B/E4B without)
    ("gemma-4-31b-it", "gemma4-think"),
    ("gemma-4-26b-a4b-it", "gemma4-think"),
    ("gemma-4-e2b", "gemma4-e-nothink"),
    ("gemma-4-e4b", "gemma4-e-nothink"),
    ("gemma-4", "gemma4-think"),
    # Gemma 2 and 3 models
    ("gemma-2", "gemma"),
    ("gemma-3", "gemma"),
    ("gemma", "gemma"),
    # Qwen models use ChatML
    ("qwen", "chatml"),
    # Llama 3 and 4 models
    ("llama-3", "llama3"),
    ("llama-4", "llama4"),
    ("llama2", "llama2"),
    # ChatML family (Mistral, Hermes, etc.)
    ("chatml", "chatml"),
    ("mistral", "chatml"),
    ("hermes", "chatml"),
]


def get_chat_template_for_model(model_name: Optional[str]) -> Optional[str]:
    """Determine the appropriate chat template for a given model name.

    Args:
        model_name: The name of the model to check (case-insensitive, can be None).

    Returns:
        The template name (e.g., "gemma", "chatml", "llama3") or None if no match.
    """
    if not model_name:
        return None

    model_lower = model_name.lower()
    for pattern, template in MODEL_TO_CHAT_TEMPLATE:
        if pattern in model_lower:
            return template

    return None


# Per-base-URL cache of the detected model name. The loaded model rarely
# changes (a systemctl swap on the inference host), so we cache to avoid an
# HTTP round-trip on every message. Short TTL so a model swap is picked up
# within the window without a restart.
_KOBOLD_MODEL_CACHE: Dict[str, Tuple[float, Optional[str]]] = {}
_KOBOLD_MODEL_CACHE_TTL = 60.0


async def get_current_kobold_model(
    client: httpx.AsyncClient, base_url: str
) -> Optional[str]:
    """Query koboldcpp for the currently loaded model name (async, cached).

    Hits ``{base_url}/api/v1/model``, whose response is
    ``{"result": "koboldcpp/<modelname>"}`` — the ``koboldcpp/`` prefix is
    harmless for the substring matching in `get_chat_template_for_model`.
    (The older `/api/extra/version` endpoint carries NO model field — it
    returns ``{"result": "KoboldCpp", "version": ...}`` — so it must not be
    used for detection.)

    Best-effort: any failure (unreachable, non-200, missing field) returns
    None so the caller falls back to its default template. Results — including
    None — are cached per ``base_url`` for ``_KOBOLD_MODEL_CACHE_TTL`` seconds
    to keep this off the per-message hot path.

    Takes the caller's ``httpx.AsyncClient`` so we never block the event loop
    with a synchronous request and never spin up a second client.
    """
    now = time.monotonic()
    cached = _KOBOLD_MODEL_CACHE.get(base_url)
    if cached is not None and now - cached[0] < _KOBOLD_MODEL_CACHE_TTL:
        return cached[1]

    result: Optional[str] = None
    try:
        response = await client.get(f"{base_url}/api/v1/model", timeout=2.0)
        if response.status_code == 200:
            raw = response.json().get("result")
            if raw:
                result = str(raw)
                logger.debug(f"Detected koboldcpp model: {result}")
    except Exception as e:
        logger.debug(f"Could not query koboldcpp model: {e}")

    _KOBOLD_MODEL_CACHE[base_url] = (now, result)
    return result
