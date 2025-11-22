# src/utils/save_utils.py

import json
import logging
import os
import shutil
from pathlib import Path
from typing import Dict, Any, Optional, List, cast

from config.global_config import PERSONA_SAVE_FILE, TEST_PERSONA_SAVE_FILE, CONFIG_DIR

logger = logging.getLogger(__name__)


def _get_persona_save_file_path() -> str:
    """
    Determines the correct persona file path based on the execution context.
    If running under pytest, uses the test-specific file. Otherwise, uses production.
    """
    if os.getenv('PYTEST_CURRENT_TEST'):
        return TEST_PERSONA_SAVE_FILE
    return PERSONA_SAVE_FILE


def load_models_from_file(file_path_override: Optional[str] = None) -> Optional[Dict[str, Any]]:
    file_path = file_path_override or _get_persona_save_file_path()
    if not os.path.exists(file_path):
        logger.warning(f"File '{file_path}' does not exist.")
        return None
    with open(file_path, "r") as file:
        data: Dict[str, Any] = json.load(file)
        return cast(Optional[Dict[str, Any]], data.get('models', {}))


def save_models_to_file(models_dict: Dict[str, Any], file_path_override: Optional[str] = None) -> None:
    """Save the models dictionary to the JSON file."""
    save_file: str = file_path_override or _get_persona_save_file_path()
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
    from src.persona import Persona
    persona_list: List[Dict[str, Any]] = []
    for persona_name, persona in personas.items():
        persona_json: Dict[str, Any] = {
            "name": persona.get_name(),
            "prompt": persona.get_prompt(),
            "model_name": persona.get_model_name(),
            "context_length": persona.get_base_context_length(), # Save the base value, not the dynamic one
            "token_limit": persona.get_response_token_limit(),
            "temperature": persona.get_temperature(),
            "top_p": persona.get_top_p(),
            "top_k": persona.get_top_k(),
            "display_name_in_chat": persona.should_display_name_in_chat(),
            "execution_mode": persona.get_execution_mode().name,
            "enabled_tools": persona.get_enabled_tools(),
            "memory_mode": persona.get_memory_mode().name,
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
    from src.persona import Persona, ExecutionMode, MemoryMode

    # Determine the target file path
    if file_path_override:
        target_path = Path(file_path_override)
    else:
        # This typically points to data/personas.json via global_config
        target_path = _get_persona_save_file_path()

    # --- PERSISTENCE INITIALIZATION (MIGRATION) ---
    # If we are loading the main database (no override) and it doesn't exist yet:
    if not file_path_override and not target_path.exists():
        default_source = CONFIG_DIR / "personas.json"

        if default_source.exists():
            logger.info(f"First-run detected. Seeding database from defaults: {default_source} -> {target_path}")
            try:
                # Ensure the parent directory (data/) exists before copying
                target_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(default_source, target_path)
            except Exception as e:
                logger.error(f"Failed to seed persona database: {e}")
                return None
        else:
            logger.warning(f"No default persona file found at {default_source}. Starting with empty state.")

    # --- FILE LOADING ---
    if not target_path.exists():
        logger.warning(f"Persona file '{target_path}' does not exist.")
        return None

    try:
        with open(target_path, "r", encoding='utf-8') as file:
            content = file.read()
            if not content:
                logger.warning(f"File '{target_path}' is empty.")
                return {}
            persona_data: Dict[str, Any] = json.loads(content)

        personas: Dict[str, Persona] = {}

        # Iterate through the JSON list and instantiate Persona objects
        for new_persona in persona_data.get('personas', []):
            name: Optional[str] = new_persona.get("name")
            if not name:
                logger.warning(f"Skipping malformed persona entry (missing name) in '{target_path}'.")
                continue

            # Parse Execution Mode (Default: SILENT_ANALYSIS)
            execution_mode_str: Optional[str] = new_persona.get("execution_mode")
            execution_mode: ExecutionMode = ExecutionMode.SILENT_ANALYSIS
            if execution_mode_str and isinstance(execution_mode_str, str):
                try:
                    execution_mode = ExecutionMode[execution_mode_str.upper()]
                except KeyError:
                    logger.warning(f"Invalid execution_mode '{execution_mode_str}' for '{name}'. Defaulting to SILENT_ANALYSIS.")

            # Parse Memory Mode (Default: CHANNEL_ISOLATED)
            memory_mode_str: Optional[str] = new_persona.get("memory_mode")
            memory_mode: MemoryMode = MemoryMode.CHANNEL_ISOLATED
            if memory_mode_str and isinstance(memory_mode_str, str):
                try:
                    memory_mode = MemoryMode[memory_mode_str.upper()]
                except KeyError:
                    logger.warning(f"Invalid memory_mode '{memory_mode_str}' for '{name}'. Defaulting to CHANNEL_ISOLATED.")

            # Create the Persona instance
            personas[name] = Persona(
                persona_name=name,
                model_name=new_persona.get("model_name"),
                prompt=new_persona.get("prompt"),
                token_limit=new_persona.get("token_limit"),
                context_length=new_persona.get("context_length"),
                temperature=new_persona.get("temperature"),
                top_p=new_persona.get("top_p"),
                top_k=new_persona.get("top_k"),
                display_name_in_chat=new_persona.get("display_name_in_chat", False),
                execution_mode=execution_mode,
                enabled_tools=new_persona.get("enabled_tools", []),
                memory_mode=memory_mode
            )

        return personas

    except json.JSONDecodeError as e:
        logger.error(f"Corrupt JSON in file '{target_path}': {str(e)}")
        return None
    except Exception as e:
        logger.error(f"Critical error loading personas from '{target_path}': {e}", exc_info=True)
        return None