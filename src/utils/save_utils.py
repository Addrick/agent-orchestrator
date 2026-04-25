# src/utils/save_utils.py

import json
import logging
import os
import shutil
from pathlib import Path
from typing import Dict, Any, Optional, List

from config import global_config

logger = logging.getLogger(__name__)


def _get_persona_save_file_path() -> Path:
    """Returns the persona save file path from global config."""
    return global_config.PERSONA_SAVE_FILE


def load_models_from_file(file_path_override: Optional[str] = None) -> Optional[Dict[str, Any]]:
    file_path = file_path_override or _get_persona_save_file_path()
    if not os.path.exists(file_path):
        logger.warning(f"File '{file_path}' does not exist.")
        return None
    with open(file_path, "r") as file:
        data: Dict[str, Any] = json.load(file)
        models: Optional[Dict[str, Any]] = data.get('models', {})
        return models


def save_models_to_file(models_dict: Dict[str, Any], file_path_override: Optional[str] = None) -> None:
    """Save the models dictionary to the JSON file."""
    save_file = file_path_override or _get_persona_save_file_path()
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
    save_file = file_path_override or _get_persona_save_file_path()

    # Ensure the directory exists
    save_dir = os.path.dirname(save_file)
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
            "history_messages": persona.get_base_history_messages(),  # Save the base value, not the dynamic one
            "token_limit": persona.get_response_token_limit(),
            "temperature": persona.get_temperature(),
            "top_p": persona.get_top_p(),
            "top_k": persona.get_top_k(),
            "display_name_in_chat": persona.should_display_name_in_chat(),
            "execution_mode": persona.get_execution_mode().name,
            "enabled_tools": persona.get_enabled_tools(),
            "memory_mode": persona.get_memory_mode().name,
            "service_bindings": persona.get_service_bindings(),
            "include_ambient_memory": persona.get_include_ambient_memory(),
            "thinking_level": persona.get_thinking_level(),
            "long_term_memory": persona.get_long_term_memory(),
            "max_context_tokens": persona.get_max_context_tokens(),
        }
        persona_list.append(persona_json)
    return persona_list


def load_personas_from_file(file_path_override: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """
    Load personas from a JSON-formatted file into a dictionary of Persona objects.

    Auto-Seeding Logic:
    If the target persistent file (in /data) does not exist, this function will
    attempt to seed it by copying the default configuration from the application
    code (in /config). This ensures the bot starts with 'Factory Defaults' on
    a fresh deployment.
    """
    from src.persona import Persona

    file_path = file_path_override or _get_persona_save_file_path()

    # --- AUTO-SEEDING LOGIC ---
    # Only seed from defaults when using the standard path, not an explicit override.
    if not os.path.exists(file_path):
        if file_path_override:
            logger.warning(f"File '{file_path}' does not exist.")
            return None
        default_file = os.path.join(global_config.CONFIG_DIR, 'default_personas.json')
        if os.path.exists(default_file):
            logger.info(f"Auto-seeding personas from {default_file} to {file_path}")
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            shutil.copy(default_file, file_path)
        else:
            logger.warning(f"File '{file_path}' does not exist and no default_personas.json found to seed.")
            return None

    try:
        with open(file_path, "r") as file:
            content = file.read()
            if not content:
                logger.warning(f"File '{file_path}' is empty.")
                return {}  # Return an empty dict if file is empty
            persona_data: Dict[str, Any] = json.loads(content)

        personas: Dict[str, Persona] = {}
        for new_persona in persona_data.get('personas', []):
            name: Optional[str] = new_persona.get("name")
            if not name:
                logger.warning(f"Skipping malformed persona entry (missing name) in '{file_path}'.")
                continue

            personas[name] = Persona(
                persona_name=name,
                model_name=new_persona.get("model_name"),
                prompt=new_persona.get("prompt"),
                token_limit=new_persona.get("token_limit"),
                history_messages=new_persona.get("history_messages", new_persona.get("context_length")),
                temperature=new_persona.get("temperature"),
                top_p=new_persona.get("top_p"),
                top_k=new_persona.get("top_k"),
                display_name_in_chat=new_persona.get("display_name_in_chat", False),
                execution_mode=new_persona.get("execution_mode"),
                enabled_tools=new_persona.get("enabled_tools", []),
                memory_mode=new_persona.get("memory_mode"),
                service_bindings=new_persona.get("service_bindings"),
                include_ambient_memory=new_persona.get("include_ambient_memory", True),
                thinking_level=new_persona.get("thinking_level"),
                long_term_memory=new_persona.get("long_term_memory", True),
                max_context_tokens=new_persona.get("max_context_tokens"),
            )

        return personas

    except json.JSONDecodeError as e:
        logger.error(f"Corrupt JSON in file '{file_path}': {str(e)}")
        return None
    except Exception as e:
        logger.error(f"Critical error loading personas from '{file_path}': {e}", exc_info=True)
        return None


def load_system_personas_from_file() -> Dict[str, Any]:
    """
    Load system personas from the dedicated system configuration file.
    These are infrastructure agents required for bot functionality.
    """
    from src.persona import Persona

    file_path = global_config.SYSTEM_PERSONA_FILE
    if not os.path.exists(file_path):
        logger.error(f"System persona file not found at '{file_path}'. Bot functionality may be degraded.")
        return {}

    try:
        with open(file_path, "r") as file:
            content = file.read()
            if not content:
                return {}
            data = json.loads(content)

        persona_list = data if isinstance(data, list) else data.get('personas', [])

        personas: Dict[str, Persona] = {}

        for new_persona in persona_list:
            name: Optional[str] = new_persona.get("name")
            if not name:
                continue

            personas[name] = Persona(
                persona_name=name,
                model_name=new_persona.get("model_name", "local"),
                prompt=new_persona.get("prompt", ""),
                token_limit=new_persona.get("response_token_limit"),
                history_messages=new_persona.get("history_messages", new_persona.get("context_length", 0)),
                temperature=new_persona.get("temperature"),
                top_p=new_persona.get("top_p"),
                top_k=new_persona.get("top_k"),
                display_name_in_chat=new_persona.get("display_name_in_chat", False),
                execution_mode=new_persona.get("execution_mode"),
                enabled_tools=new_persona.get("enabled_tools", []),
                memory_mode=new_persona.get("memory_mode", "TICKET_ISOLATED"),
                service_bindings=new_persona.get("service_bindings"),
                include_ambient_memory=new_persona.get("include_ambient_memory", True),
                thinking_level=new_persona.get("thinking_level"),
                long_term_memory=new_persona.get("long_term_memory", True),
                max_context_tokens=new_persona.get("max_context_tokens"),
            )

        return personas

    except json.JSONDecodeError as e:
        logger.error(f"Corrupt JSON in system persona file '{file_path}': {str(e)}")
        return {}
    except Exception as e:
        logger.error(f"Critical error loading system personas: {e}", exc_info=True)
        return {}
