import json
import logging
import os
from config.global_config import PERSONA_SAVE_FILE
from src.persona import Persona


def load_models_from_file():
    if not os.path.exists(PERSONA_SAVE_FILE):
        logging.warning(f"File '{PERSONA_SAVE_FILE}' does not exist.")
        return
    with open(PERSONA_SAVE_FILE, "r") as file:
        data = json.load(file)
        return data['models']


def save_models_to_file(models_dict):  #  TODO: use self.models_dict
    with open(PERSONA_SAVE_FILE, 'r') as file:
        save_data = json.load(file)
    save_data['models'] = models_dict
    with open(PERSONA_SAVE_FILE, 'w') as file:
        json.dump(save_data, file, indent=4)
    logging.info(f"Updated persona save.")


def save_personas_to_file(personas):
    """Save all personas to the JSON file."""
    # Check if the file exists
    if not os.path.exists(PERSONA_SAVE_FILE):
        # Create the file
        with open(PERSONA_SAVE_FILE, 'w') as file:
            pass  # Just create an empty file
        print(f"File '{PERSONA_SAVE_FILE}' created.")
    with open(PERSONA_SAVE_FILE, 'r') as file:
        save_data = json.load(file)
    persona_dict = to_dict(personas)
    save_data['personas'] = persona_dict
    with open(PERSONA_SAVE_FILE, 'w') as file:
        json.dump(save_data, file, indent=4)
    logging.info(f"Updated persona save.")


def to_dict(personas):
    """Convert personas to a list of dictionaries for JSON serialization."""
    persona_dict = []
    for persona_name, persona in personas.items():
        persona_json = {
            "name": persona.persona_name,
            "prompt": persona.prompt,
            "model_name": persona.model.model_name,
            "context_limit": persona.context_length,
            "token_limit": persona.response_token_limit,
        }
        persona_dict.append(persona_json)
    return persona_dict


def load_personas_from_file(file_path=PERSONA_SAVE_FILE):
    """Load personas from a JSON-formatted file."""
    if not os.path.exists(file_path):
        logging.warning(f"File '{file_path}' does not exist.")
        return
    with open(file_path, "r") as file:
        persona_data = json.load(file)
        personas = {}
        for new_persona in persona_data['personas']:
            try:
                name = new_persona["name"]
                model_name = new_persona["model_name"]
                prompt = new_persona["prompt"]
                context_limit = new_persona["context_limit"]
                token_limit = new_persona["token_limit"]
                personas[name] = Persona(name, model_name, prompt, context_limit, token_limit)
            except json.JSONDecodeError as e:
                logging.info(f"Error decoding JSON file '{file_path}': {str(e)}")
                return
        return personas
