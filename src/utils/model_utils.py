# src/utils/model_utils.py

import logging
import os
from typing import List, Dict, Any, Optional

from src.utils import save_utils

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
    if name_lower.startswith("gpt"):
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


def get_model_list(update: bool = False) -> Optional[Dict[str, Any]]:
    """Get available models, optionally refreshing from APIs."""
    if update:
        logger.info('Updating available models from API...')
        all_available_models: Dict[str, List[str]] = {
            'From OpenAI': refresh_available_openai_models(),
            'From Google': refresh_available_google_models(),
            'From Anthropic': refresh_available_anthropic_models(),
            'Antigravity (OAuth tier)': ['agy-flash'],
            'Local': ['local']
        }
        save_utils.save_models_to_file(all_available_models)
        return all_available_models
    else:
        return save_utils.load_models_from_file()


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

