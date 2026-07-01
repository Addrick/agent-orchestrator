# src/tools/definitions.py

import importlib
import logging
from typing import List, Dict, Any, Optional

from src.tools.tool_defs import (
    SEARCH_TOOLS,
    ZAMMAD_TOOLS,
    AGENT_TOOLS,
    MEMORY_TOOLS,
    FIXR_TOOLS,
    VOICE_TOOLS,
    PROXMOX_TOOLS,
)

logger = logging.getLogger(__name__)

"""
This file contains the definitions for all tools available to the LLM.
Each tool is defined as a JSON schema compatible with the function-calling
APIs of major providers like OpenAI, Google, and Anthropic.

These definitions serve as the "contract" that the LLM uses to understand
what a tool does, what parameters it requires, and what it returns.

The actual implementation of these tools is handled by the ToolManager.

Every tool definition carries a `capabilities` block driving the runtime
tool-security framework (see memory/project/plans/tool_security_framework.md):

    "capabilities": {
        "produces_untrusted": bool,    # result may carry attacker-controlled text
        "irreversible": bool,          # effect cannot be trivially undone
        "irreversible_if": str | None, # optional "module:function" classifier
    }

`produces_untrusted` is about *origin* of the data, not network — local
memory tools that surface previously-ingested external content count.
`irreversible_if` is a dotted path to a `(args: dict) -> bool` callable;
when present, runtime ORs its result with `irreversible`.
"""

# A list containing all tool definitions.
# The ToolManager will expose these to the ChatSystem.
#
# Special tool types:
#   "google_grounding" — Not a callable function. Signals the engine to enable
#                        Google's native grounding feature for Gemini models.
#                        Has no effect on other providers or Gemma models.
# Assembled from per-binding modules in src/tools/tool_defs/ (DP-248). Order is
# preserved, so this is byte-identical to the former inline literal. The split
# is navigability-only; all capability/helper logic below stays here.
ALL_TOOL_DEFINITIONS: List[Dict[str, Any]] = (
    SEARCH_TOOLS + ZAMMAD_TOOLS + AGENT_TOOLS + MEMORY_TOOLS + FIXR_TOOLS + VOICE_TOOLS
    + PROXMOX_TOOLS
)


def validate_tool_capabilities(tool: Dict[str, Any]) -> None:
    """
    Assert a tool definition carries the required `capabilities` block per
    the tool-security framework. Raises ValueError on any violation.

    - `capabilities` must be a dict with bool `produces_untrusted` and `irreversible`.
    - Optional `irreversible_if` must be a `"module:function"` dotted path that
      resolves to a callable at validation time.
    """
    name = tool.get("function", {}).get("name", "<unknown>")
    caps = tool.get("capabilities")
    if not isinstance(caps, dict):
        raise ValueError(f"Tool '{name}' missing 'capabilities' block")
    for required in ("produces_untrusted", "irreversible", "locality", "sensitivity"):
        if required not in caps:
            raise ValueError(
                f"Tool '{name}' capabilities missing required flag '{required}'"
            )
        if required in ("produces_untrusted", "irreversible"):
            if not isinstance(caps[required], bool):
                raise ValueError(
                    f"Tool '{name}' capability '{required}' must be bool, "
                    f"got {type(caps[required]).__name__}"
                )
        else:
            if not isinstance(caps[required], str):
                raise ValueError(
                    f"Tool '{name}' capability '{required}' must be str, "
                    f"got {type(caps[required]).__name__}"
                )
    classifier = caps.get("irreversible_if")
    if classifier is None:
        return
    if not isinstance(classifier, str) or ":" not in classifier:
        raise ValueError(
            f"Tool '{name}' irreversible_if must be a 'module:function' dotted path"
        )
    module_path, func_name = classifier.split(":", 1)
    try:
        module = importlib.import_module(module_path)
    except ImportError as e:
        raise ValueError(
            f"Tool '{name}' irreversible_if module '{module_path}' could not be imported: {e}"
        ) from e
    func = getattr(module, func_name, None)
    if not callable(func):
        raise ValueError(
            f"Tool '{name}' irreversible_if '{classifier}' did not resolve to a callable"
        )


# Validate at import time so any consumer (engine, tests, tooling) catches
# capability drift immediately.
for _tool in ALL_TOOL_DEFINITIONS:
    validate_tool_capabilities(_tool)


# Model prefixes that do NOT support each tool.
# Uses the same prefix logic as engine.py routing.
# Tools not listed here are compatible with all providers.
GROUNDING_INCOMPATIBLE_PREFIXES = {'gpt', 'claude', 'gemma', 'local', 'unknown'}

MODEL_INCOMPATIBLE_TOOLS = {
    'google_grounding_search': GROUNDING_INCOMPATIBLE_PREFIXES,
}

# Derived from tool metadata — no manual maintenance needed when adding new tools.
WRITE_TOOLS = {t['function']['name'] for t in ALL_TOOL_DEFINITIONS if t.get('is_write')}

# Tools that ALWAYS require confirmation, even in AUTONOMOUS mode.
# These are typically high-impact or destructive operations.
ALWAYS_CONFIRM_TOOLS = {"merge_tickets", "delete_user"}

# Name → definition index. ALL_TOOL_DEFINITIONS is static after import, and
# the lookups below run per tool call inside the tool loop — a linear scan
# per call adds up with batched calls over a 50+ tool catalog.
_TOOL_DEFINITION_INDEX: Dict[str, Dict[str, Any]] = {
    t.get("function", {}).get("name", ""): t for t in ALL_TOOL_DEFINITIONS
}

_DEFAULT_CAPABILITIES: Dict[str, Any] = {
    "produces_untrusted": False,
    "irreversible": False,
    "locality": "unknown",
    "sensitivity": "unknown",
}


def get_tool_capabilities(tool_name: str) -> Dict[str, Any]:
    """
    Retrieves the security capabilities block for a given tool name.
    Returns a default block (all False) if the tool is not found.
    """
    tool = _TOOL_DEFINITION_INDEX.get(tool_name)
    if tool is not None:
        from typing import cast
        return cast(Dict[str, Any], tool.get("capabilities", dict(_DEFAULT_CAPABILITIES)))
    return dict(_DEFAULT_CAPABILITIES)


def get_tool_definition(tool_name: str) -> Optional[Dict[str, Any]]:
    """
    Retrieves the full JSON definition for a given tool name.
    Returns None if the tool is not found.
    """
    return _TOOL_DEFINITION_INDEX.get(tool_name)


def is_irreversible(tool_name: str, args: Dict[str, Any]) -> bool:
    """
    Checks if a tool call is irreversible based on its name and arguments.
    """
    caps = get_tool_capabilities(tool_name)
    if caps.get("irreversible"):
        return True

    classifier = caps.get("irreversible_if")
    if not classifier:
        return False

    # Resolve "module:function"
    try:
        module_path, func_name = classifier.split(":", 1)
        module = importlib.import_module(module_path)
        func = getattr(module, func_name)
        return bool(func(args))
    except (ValueError, ImportError, AttributeError, Exception) as e:
        logger.error(f"Error resolving irreversible_if for {tool_name}: {e}")
        return False
