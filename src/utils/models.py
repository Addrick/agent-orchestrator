import config.api_keys

from src.message_handler import *
from src.utils import config
import openai


def refresh_available_openai_models():
    """# OpenAI API query to get current list of active models"""
    client = openai.OpenAI(api_key=api_keys.openai)
    openai_models = client.models.list()
    trimmed_list = [model.id for model in openai_models if 'gpt-3' in model.id or 'gpt-4' in model.id]
    logging.debug(trimmed_list)
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
    """# Anthropic
    # TODO: can't find api call, some other way to get this information dynamically?
    # can maybe dig names out of the python library: https://github.com/anthropics/anthropic-sdk-python/blob/0336233fc076f20017b28433df9e3d9dd56ffa8d/src/anthropic/types/message_create_params.py#L127
    #     anthropic-sdk-python/src/anthropic/types/message_create_params.py"""
    models = [
        "claude-3-5-sonnet-20240620",
        "claude-3-opus-20240229",
        "claude-3-sonnet-20240229",
        "claude-3-haiku-20240307",
        "claude-2.1",
        "claude-2.0",
        "claude-instant-1.2"]
    return models


def get_model_list(update=False):
    """
    # get_model_list(update=False): If the update parameter is set to True, the function queries the API to update
    # and print the list of available models from OpenAI and Google. If update is False, it will return the models
    # saved in gloabl_config
    """
    if update:
        logging.info('Updating available models from API...')
        all_available_models = {'From OpenAI': refresh_available_openai_models(),
                                'From Google': refresh_available_google_models(),
                                'From Anthropic': refresh_available_anthropic_models(),
                                'Local': ['local']
                                }
        logging.debug(all_available_models)
        utils.config.save_models_to_file(all_available_models)
        return all_available_models
    else:
        return config.load_models_from_file()


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
        logging.info(f"The value '{model_to_check}' is not found.")
        return False
