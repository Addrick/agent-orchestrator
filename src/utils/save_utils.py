# src/utils/save_utils.py

import json
import logging
import os
from config.global_config import PERSONA_SAVE_FILE

logger = logging.getLogger(__name__)


def load_models_from_file(file_path=PERSONA_SAVE_FILE):
    if not os.path.exists(file_path):
        logger.warning(f"File '{file_path}' does not exist.")
        return None
    with open(file_path, "r") as file:
        data = json.load(file)
        return data.get('models', {})


def save_models_to_file(models_dict, file_path_override=None):
    """Save the models dictionary to the JSON file."""
    save_file = file_path_override if file_path_override is not None else PERSONA_SAVE_FILE
    try:
        with open(save_file, 'r') as file:
            save_data = json.load(file)
    except (FileNotFoundError, json.JSONDecodeError):
        # If file doesn't exist or is empty/corrupt, start with a default structure
        save_data = {"personas": [], "models": {}}

    save_data['models'] = models_dict
    with open(save_file, 'w') as file:
        json.dump(save_data, file, indent=4)
    logger.debug(f"Updated model save to {save_file}.")


def save_personas_to_file(personas, file_path_override=None):
    """Save all personas to the JSON file."""
    save_file = file_path_override if file_path_override is not None else PERSONA_SAVE_FILE

    # Ensure the directory exists
    save_dir = os.path.dirname(save_file)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    try:
        with open(save_file, 'r') as file:
            # Handle empty file case
            content = file.read()
            if not content:
                save_data = {"personas": [], "models": {}}
            else:
                save_data = json.loads(content)
    except (FileNotFoundError, json.JSONDecodeError):
        # If file doesn't exist or is corrupt, start with a default structure
        save_data = {"personas": [], "models": {}}

    persona_dict = to_dict(personas)
    save_data['personas'] = persona_dict

    with open(save_file, 'w') as file:
        json.dump(save_data, file, indent=4)
    logger.debug(f"Updated persona save to {save_file}.")


def to_dict(personas):
    """Convert a dictionary of Persona objects to a list of dictionaries for JSON serialization."""
    persona_list = []
    for persona_name, persona in personas.items():
        persona_json = {
            "name": persona.get_name(),
            "prompt": persona.get_prompt(),
            "model_name": persona.get_model_name(),
            "context_limit": persona.get_context_length(),
            "token_limit": persona.get_response_token_limit(),
            "temperature": persona.get_temperature(),
            "top_p": persona.get_top_p(),
            "top_k": persona.get_top_k(),
            "display_name_in_chat": persona.should_display_name_in_chat(),
        }
        persona_list.append(persona_json)
    return persona_list


def load_personas_from_file(file_path=PERSONA_SAVE_FILE):
    from src.persona import Persona
    """Load personas from a JSON-formatted file into a dictionary."""
    if not os.path.exists(file_path):
        logger.warning(f"File '{file_path}' does not exist.")
        return None
    try:
        with open(file_path, "r") as file:
            content = file.read()
            if not content:
                logger.warning(f"File '{file_path}' is empty.")
                return {}  # Return an empty dict if file is empty
            persona_data = json.loads(content)

        personas = {}
        # Ensure 'personas' key exists and is a list
        for new_persona in persona_data.get('personas', []):
            name = new_persona.get("name")
            if not name:
                logger.warning(f"Skipping persona with no name in '{file_path}'.")
                continue

            personas[name] = Persona(
                persona_name=name,
                model_name=new_persona.get("model_name"),
                prompt=new_persona.get("prompt"),
                token_limit=new_persona.get("token_limit"),
                context_length=new_persona.get("context_limit"),
                temperature=new_persona.get("temperature"),
                top_p=new_persona.get("top_p"),
                top_k=new_persona.get("top_k"),
                display_name_in_chat=new_persona.get("display_name_in_chat", False),
            )
        return personas
    except json.JSONDecodeError as e:
        logger.error(f"Error decoding JSON file '{file_path}': {str(e)}")
        return None
    except Exception as e:
        logger.error(f"An unexpected error occurred while loading personas from '{file_path}': {e}", exc_info=True)
        return None
