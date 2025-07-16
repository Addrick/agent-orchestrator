import config.api_keys as api_keys
import logging
from src.utils import save_utils
logger = logging.getLogger(__name__)


def refresh_available_openai_models():
    """# OpenAI API query to get current list of active models"""
    import openai
    client = openai.OpenAI(api_key=api_keys.openai)
    openai_models = client.models.list()
    trimmed_list = [model.id for model in openai_models]
    # trimmed_list = [model.id for model in openai_models if 'gpt-3' in model.id or 'gpt-4' in model.id]
    logger.debug(trimmed_list)
    return trimmed_list


def refresh_available_google_models():
    """
    # Google
    # a bit hacked together as actual generation requests are using the vertexai package which lacks an api call to list models
    # instead uses google.generativeai which lists different models than what vertexai has available, TODO: find better way to list available Google models
    # vertexai models can be viewed at https://console.cloud.google.com/vertex-ai/model-garden
    # model garden includes tons of shit, incl non-google models if I wanted to run them on google hardware I guess. Also fine tuning"""
    import google.generativeai as genai
    genai.configure(api_key=api_keys.google)
    google_models = []
    for model in genai.list_models():
        if 'generateContent' in model.supported_generation_methods: # remove non-genai models
            version_tag = model.name.split("-")[-1]
            if version_tag != '001' and version_tag != 'latest':
                if model not in google_models:
                    google_models.append(model.name.split("/")[-1]) # remove preceding 'models/' from name
    return google_models


def refresh_available_anthropic_models():
    import anthropic
    from dotenv import load_dotenv
    import os

    # load .env for api key
    current_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.join(current_dir, r'..\..')
    load_dotenv(os.path.join(root_dir, '.env'))
    api_key = os.environ.get("ANTHROPIC_API_KEY")

    client = anthropic.Anthropic(api_key=api_key)
    models = client.models.list(limit=20)
    model_infos = models.data
    # Extract the 'id' field from each ModelInfo object
    model_ids = [model_info.id for model_info in model_infos]
    return model_ids


def get_model_list(update=False):
    """
    # get_model_list(update=False): If the update parameter is set to True, the function queries the API to update
    # and print the list of available models from OpenAI and Google. If update is False, it will return the models
    # saved in gloabl_config
    """
    if update:
        logger.info('Updating available models from API...')
        all_available_models = {'From OpenAI': refresh_available_openai_models(),
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


def check_model_available(model_to_check):
    """Check if a specific model is available."""
    lowest_order_items = []
    for value in get_model_list().values():
        if isinstance(value, list):
            lowest_order_items.extend(value)
        else:
            lowest_order_items.append(value)

    if model_to_check in lowest_order_items:
        return True
    else:
        logger.info(f"The value '{model_to_check}' is not found.")
        return False
