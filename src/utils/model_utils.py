# src/utils/model_utils.py

import logging
import os
from typing import List, Dict, Any, Optional

from src.utils import save_utils

logger = logging.getLogger(__name__)


def refresh_available_openai_models() -> List[str]:
    """# OpenAI API query to get current list of active models"""
    import openai
    client = openai.OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    openai_models = client.models.list()
    trimmed_list: List[str] = [model.id for model in openai_models]
    logger.debug(trimmed_list)
    return trimmed_list


def refresh_available_google_models() -> List[str]:
    """
    # Google
    # a bit hacked together as actual generation requests are using the vertexai package which lacks an api call to list models
    # instead uses google.generativeai which lists different models than what vertexai has available, TODO: find better way to list available Google models
    # vertexai models can be viewed at https://console.cloud.google.com/vertex-ai/model-garden
    # model garden includes tons of shit, incl non-google models if I wanted to run them on google hardware I guess. Also fine tuning"""
    import google.generativeai as genai
    genai.configure(api_key=os.environ.get("GOOGLE_GENERATIVEAI_API_KEY"))
    google_models: List[str] = []
    for model in genai.list_models():
        if 'generateContent' in model.supported_generation_methods:  # remove non-genai models
            version_tag: str = model.name.split("-")[-1]
            if version_tag != '001' and version_tag != 'latest':
                model_name: str = model.name.split("/")[-1]  # remove preceding 'models/' from name
                if model_name not in google_models:
                    google_models.append(model_name)
    return google_models


def refresh_available_anthropic_models() -> List[str]:
    import anthropic
    api_key: Optional[str] = os.environ.get("ANTHROPIC_API_KEY")  # Assumes .env is loaded at app startup

    client = anthropic.Anthropic(api_key=api_key)
    # The actual return type is a Page, but we can treat it as a generic iterable of Model objects
    models: Any = client.models.list(limit=20)
    model_infos: List[Any] = models.data
    # Extract the 'id' field from each ModelInfo object
    model_ids: List[str] = [model_info.id for model_info in model_infos]
    return model_ids


def get_model_list(update: bool = False) -> Optional[Dict[str, Any]]:
    """
    # get_model_list(update=False): If the update parameter is set to True, the function queries the API to update
    # and print the list of available models from OpenAI and Google. If update is False, it will return the models
    # saved in gloabl_config
    """
    if update:
        logger.info('Updating available models from API...')
        all_available_models: Dict[str, List[str]] = {'From OpenAI': refresh_available_openai_models(),
                                                      'From Google': refresh_available_google_models(),
                                                      'From Anthropic': refresh_available_anthropic_models(),
                                                      'Local': ['local']
                                                      }
        logger.debug(all_available_models)
        save_utils.save_models_to_file(all_available_models)
        logger.info('Current available models set from API.')
        return all_available_models
    else:
        return save_utils.load_models_from_file()


def check_model_available(model_to_check: str) -> bool:
    """Check if a specific model is available."""
    model_list = get_model_list()
    if not model_list:
        return False

    lowest_order_items: List[str] = []
    for value in model_list.values():
        if isinstance(value, list):
            lowest_order_items.extend(value)
        else:
            lowest_order_items.append(str(value))

    if model_to_check in lowest_order_items:
        return True
    else:
        logger.info(f"The value '{model_to_check}' is not found.")
        return False
