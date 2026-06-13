# src/utils/save_utils.py

import json
import logging
import os
import shutil
from pathlib import Path
from typing import Dict, Any, Iterable, Optional, List

from config import global_config
from src.tools.definitions import ALL_TOOL_DEFINITIONS
from src.tools.policy import ToolPolicy

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


def save_personas_to_file(
    personas: Dict[str, Any],
    exclude_names: Iterable[str],
    file_path_override: Optional[str] = None,
) -> None:
    """Save user personas to the JSON file.

    `exclude_names` MUST be the set of system-persona names (see
    `ChatSystem.system_persona_names`). System personas live in the same
    in-memory dict as user personas but are sourced from `system_personas.json`
    — writing them into the user file corrupts it: the next load injects them
    twice (once from each file) and any persona dropped by load-time validation
    gets permanently erased when this function next runs. Filtering here is the
    single chokepoint that prevents both.
    """
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

    excluded = set(exclude_names)
    user_personas = {name: p for name, p in personas.items() if name not in excluded}
    persona_dict: List[Dict[str, Any]] = to_dict(user_personas)
    save_data['personas'] = persona_dict

    with open(save_file, 'w') as file:
        json.dump(save_data, file, indent=4)
    logger.debug(f"Updated persona save to {save_file}.")


def to_dict(personas: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Convert a dictionary of Persona objects to a list of dictionaries for JSON serialization.

    Generation params live under a nested `params` block — see
    src/generation_params.py. The flat temperature/top_p/top_k/token_limit keys
    are neither written nor read (DP-221 retired the flat-key load fallback).
    """
    persona_list: List[Dict[str, Any]] = []
    for persona_name, persona in personas.items():
        persona_json: Dict[str, Any] = {
            "name": persona.get_name(),
            "prompt": persona.get_prompt(),
            "model_name": persona.get_model_name(),
            "history_messages": persona.get_base_history_messages(),  # Save the base value, not the dynamic one
            "params": persona.get_generation_params().to_dict(),
            "display_name_in_chat": persona.should_display_name_in_chat(),
            "execution_mode": persona.get_execution_mode().name,
            "enabled_tools": persona.get_enabled_tools(),
            "memory_mode": persona.get_memory_mode().name,
            "service_bindings": persona.get_service_bindings(),
            "include_ambient_memory": persona.get_include_ambient_memory(),
            "thinking_level": persona.get_thinking_level(),
            "long_term_memory": persona.get_long_term_memory(),
            "max_context_tokens": persona.get_max_context_tokens(),
            "tool_policy": persona.get_tool_policy().to_dict(),
            "meta_visible": persona.get_meta_visible(),
            "inject_timestamp": persona.get_inject_timestamp(),
        }
        persona_list.append(persona_json)
    return persona_list


def _resolve_params_kwargs(entry: Dict[str, Any]) -> Dict[str, Any]:
    """Map a persona JSON entry's nested `params` block to Persona() kwargs.

    `max_tokens` inside the block carries the response token limit. A file with
    no `params` block loads with default generation params.
    """
    params = entry.get("params")
    return {"params": params if isinstance(params, dict) else None}


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

            # --- Composition Validation ---
            policy_data = new_persona.get("tool_policy")
            temp_policy = ToolPolicy.from_dict(policy_data) if policy_data else ToolPolicy.from_legacy_list(new_persona.get("enabled_tools", []))
            
            # Get definitions for the tools this persona wants to use
            if temp_policy.default == "allow" and "*" in temp_policy.allow:
                persona_tools = ALL_TOOL_DEFINITIONS
            else:
                allowed_set = set(temp_policy.allow + temp_policy.ask)
                persona_tools = [t for t in ALL_TOOL_DEFINITIONS if t.get("function", {}).get("name") in allowed_set]
            
            # DP-128: a persona that fails composition validation is no longer
            # dropped — it loads *quarantined* (security_block_reasons set) so it
            # stays selectable/editable and the operator can fix its tools live
            # (`set tools` / web tools modal). Generation is refused downstream
            # until a live edit re-validates clean.
            validation_errors = temp_policy.validate_composition(persona_tools)
            if validation_errors:
                for err in validation_errors:
                    logger.critical(f"Persona '{name}' security violation: {err}")
                logger.warning(f"Quarantining persona '{name}' (loaded but generation blocked until fixed).")

            personas[name] = Persona(
                persona_name=name,
                model_name=new_persona.get("model_name"),
                prompt=new_persona.get("prompt"),
                history_messages=new_persona.get("history_messages"),
                display_name_in_chat=new_persona.get("display_name_in_chat", False),
                execution_mode=new_persona.get("execution_mode"),
                enabled_tools=new_persona.get("enabled_tools", []),
                memory_mode=new_persona.get("memory_mode"),
                service_bindings=new_persona.get("service_bindings"),
                include_ambient_memory=new_persona.get("include_ambient_memory", True),
                thinking_level=new_persona.get("thinking_level"),
                long_term_memory=new_persona.get("long_term_memory", True),
                max_context_tokens=new_persona.get("max_context_tokens"),
                tool_policy=new_persona.get("tool_policy"),
                meta_visible=new_persona.get("meta_visible", False),
                ingest_bank=new_persona.get("ingest_bank"),
                security_block_reasons=validation_errors,
                inject_timestamp=new_persona.get("inject_timestamp", True),
                **_resolve_params_kwargs(new_persona),
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

            # --- Composition Validation ---
            policy_data = new_persona.get("tool_policy")
            temp_policy = ToolPolicy.from_dict(policy_data) if policy_data else ToolPolicy.from_legacy_list(new_persona.get("enabled_tools", []))
            
            if temp_policy.default == "allow" and "*" in temp_policy.allow:
                persona_tools = ALL_TOOL_DEFINITIONS
            else:
                allowed_set = set(temp_policy.allow + temp_policy.ask)
                persona_tools = [t for t in ALL_TOOL_DEFINITIONS if t.get("function", {}).get("name") in allowed_set]
            
            # DP-128: quarantine instead of drop (see the user-persona path above).
            validation_errors = temp_policy.validate_composition(persona_tools)
            if validation_errors:
                for err in validation_errors:
                    logger.critical(f"System Persona '{name}' security violation: {err}")
                logger.warning(f"Quarantining system persona '{name}' (loaded but generation blocked until fixed).")

            params_kwargs = _resolve_params_kwargs(new_persona)
            personas[name] = Persona(
                persona_name=name,
                model_name=new_persona.get("model_name", "local"),
                prompt=new_persona.get("prompt", ""),
                history_messages=new_persona.get("history_messages", 0),
                display_name_in_chat=new_persona.get("display_name_in_chat", False),
                execution_mode=new_persona.get("execution_mode"),
                enabled_tools=new_persona.get("enabled_tools", []),
                memory_mode=new_persona.get("memory_mode", "TICKET_ISOLATED"),
                service_bindings=new_persona.get("service_bindings"),
                include_ambient_memory=new_persona.get("include_ambient_memory", True),
                thinking_level=new_persona.get("thinking_level"),
                long_term_memory=new_persona.get("long_term_memory", True),
                max_context_tokens=new_persona.get("max_context_tokens"),
                tool_policy=new_persona.get("tool_policy"),
                meta_visible=new_persona.get("meta_visible", False),
                ingest_bank=new_persona.get("ingest_bank"),
                security_block_reasons=validation_errors,
                inject_timestamp=new_persona.get("inject_timestamp", False),
                **params_kwargs,
            )

        return personas

    except json.JSONDecodeError as e:
        logger.error(f"Corrupt JSON in system persona file '{file_path}': {str(e)}")
        return {}
    except Exception as e:
        logger.error(f"Critical error loading system personas: {e}", exc_info=True)
        return {}
