# src/utils/save_utils.py

import json
import logging
import os
from typing import Dict, Any, Optional, List, cast
from config.global_config import PERSONA_SAVE_FILE

logger = logging.getLogger(__name__)


def load_models_from_file(file_path: str = PERSONA_SAVE_FILE) -> Optional[Dict[str, Any]]:
    if not os.path.exists(file_path):
        logger.warning(f"File '{file_path}' does not exist.")
        return None
    with open(file_path, "r") as file:
        data: Dict[str, Any] = json.load(file)
        return cast(Optional[Dict[str, Any]], data.get('models', {}))


def save_models_to_file(models_dict: Dict[str, Any], file_path_override: Optional[str] = None) -> None:
    """Save the models dictionary to the JSON file."""
    save_file: str = file_path_override if file_path_override is not None else PERSONA_SAVE_FILE
    save_data: Dict[str, Any]
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


def save_personas_to_file(personas: Dict[str, Any], file_path_override: Optional[str] = None) -> None:
    """Save all personas to the JSON file."""
    save_file = file_path_override if file_path_override is not None else PERSONA_SAVE_FILE

    # Ensure the directory exists
    save_dir: str = os.path.dirname(save_file)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    save_data: Dict[str, Any]
    try:
        with open(save_file, 'r') as file:
            # Handle empty file case
            content: str = file.read()
            if not content:
                save_data = {"personas": [], "models": {}}
            else:
                save_data = json.loads(content)
    except (FileNotFoundError, json.JSONDecodeError):
        # If file doesn't exist or is corrupt, start with a default structure
        save_data = {"personas": [], "models": {}}

    persona_dict: List[Dict[str, Any]] = to_dict(personas)
    save_data['personas'] = persona_dict

    with open(save_file, 'w') as file:
        json.dump(save_data, file, indent=4)
    logger.debug(f"Updated persona save to {save_file}.")


def to_dict(personas: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Convert a dictionary of Persona objects to a list of dictionaries for JSON serialization."""
    persona_list: List[Dict[str, Any]] = []
    for persona_name, persona in personas.items():
        persona_json: Dict[str, Any] = {
            "name": persona.get_name(),
            "prompt": persona.get_prompt(),
            "model_name": persona.get_model_name(),
            "context_limit": persona.get_context_length(),
            "token_limit": persona.get_response_token_limit(),
            "temperature": persona.get_temperature(),
            "top_p": persona.get_top_p(),
            "top_k": persona.get_top_k(),
            "display_name_in_chat": persona.should_display_name_in_chat(),
            "execution_mode": persona.get_execution_mode().name,
            "enabled_tools": persona.get_enabled_tools(),
        }
        persona_list.append(persona_json)
    return persona_list


def load_personas_from_file(file_path: str = PERSONA_SAVE_FILE) -> Optional[Dict[str, Any]]:
    from src.persona import Persona, ExecutionMode
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
            persona_data: Dict[str, Any] = json.loads(content)

        personas: Dict[str, Persona] = {}
        # Ensure 'personas' key exists and is a list
        for new_persona in persona_data.get('personas', []):
            name: Optional[str] = new_persona.get("name")
            if not name:
                logger.warning(f"Skipping persona with no name in '{file_path}'.")
                continue

            # Handle loading execution_mode
            execution_mode_str: Optional[str] = new_persona.get("execution_mode")
            execution_mode: ExecutionMode = ExecutionMode.SILENT_ANALYSIS
            if execution_mode_str and isinstance(execution_mode_str, str):
                try:
                    execution_mode = ExecutionMode[execution_mode_str.upper()]
                except KeyError:
                    logger.warning(f"Invalid execution_mode '{execution_mode_str}' for persona '{name}'. "
                                   f"Defaulting to SILENT_ANALYSIS.")

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
                execution_mode=execution_mode,
                enabled_tools=new_persona.get("enabled_tools", [])  # Default to empty list
            )
        return personas
    except json.JSONDecodeError as e:
        logger.error(f"Error decoding JSON file '{file_path}': {str(e)}")
        return None
    except Exception as e:
        logger.error(f"An unexpected error occurred while loading personas from '{file_path}': {e}", exc_info=True)
        return None
