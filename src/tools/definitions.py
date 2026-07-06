# src/tools/definitions.py

import importlib
import logging
from typing import List, Dict, Any, Optional, cast

from src.tools.tool_defs import (
    SEARCH_TOOLS,
    ZAMMAD_TOOLS,
    AGENT_TOOLS,
    MEMORY_TOOLS,
    FIXR_TOOLS,
    VOICE_TOOLS,
    PROXMOX_TOOLS,
    MCP_TOOLS,
    PROPOSAL_TOOLS,
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
    + PROXMOX_TOOLS + MCP_TOOLS + PROPOSAL_TOOLS
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
    # Optional: exfil_capable (bool). When present and False, a network tool is
    # excluded from exfil-composition accounting (tool_policy Rules 2/3) — used
    # for constrained-arg egress to trusted infra that can't carry a payload out.
    if "exfil_capable" in caps and not isinstance(caps["exfil_capable"], bool):
        raise ValueError(
            f"Tool '{name}' capability 'exfil_capable' must be bool, "
            f"got {type(caps['exfil_capable']).__name__}"
        )

    _validate_irreversible_if(name, caps.get("irreversible_if"))


def _validate_irreversible_if(name: str, classifier: Any) -> None:
    """Assert an optional `irreversible_if` is a `module:function` dotted path
    that resolves to a callable. None is allowed (no classifier)."""
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


# Model prefixes that do NOT support each tool.
# Uses the same prefix logic as engine.py routing.
# Tools not listed here are compatible with all providers.
GROUNDING_INCOMPATIBLE_PREFIXES = {'gpt', 'claude', 'gemma', 'local', 'unknown'}

MODEL_INCOMPATIBLE_TOOLS = {
    'google_grounding_search': GROUNDING_INCOMPATIBLE_PREFIXES,
}

# Tools that ALWAYS require confirmation, even in AUTONOMOUS mode.
# These are typically high-impact or destructive operations.
ALWAYS_CONFIRM_TOOLS = {"merge_tickets", "delete_user"}

_DEFAULT_CAPABILITIES: Dict[str, Any] = {
    "produces_untrusted": False,
    "irreversible": False,
    "locality": "unknown",
    "sensitivity": "unknown",
}


class ToolDefinitionRegistry:
    """The runtime tool-definition catalog: the static `ALL_TOOL_DEFINITIONS`
    seed plus any definitions registered after import (DP-268: MCP-discovered
    tools). Everything that must see the *live* toolset — the tool loop,
    composition validation, ToolManager listing — reads through this registry
    (via the module-level accessors below); `ALL_TOOL_DEFINITIONS` itself
    remains the static seed only.

    The name index and write-tool set are maintained incrementally because the
    lookups run per tool call inside the tool loop — a linear scan per call
    adds up with batched calls over a 50+ tool catalog.
    """

    def __init__(self, seed: List[Dict[str, Any]]) -> None:
        self._definitions: List[Dict[str, Any]] = []
        self._index: Dict[str, Dict[str, Any]] = {}
        self._write_tools: set[str] = set()
        for tool in seed:
            self.register(tool)

    def register(self, tool: Dict[str, Any]) -> None:
        """Validate and add a tool definition to the live catalog.

        Raises ValueError on a missing/invalid `capabilities` block or on a
        name collision — a dynamically discovered tool must never shadow an
        existing one (namespacing is the caller's job, e.g. `mcp__<server>__`).
        Non-callable entries (no function name, e.g. `google_grounding`) are
        listed but not indexed.
        """
        validate_tool_capabilities(tool)
        name = tool.get("function", {}).get("name")
        if name:
            if name in self._index:
                raise ValueError(f"Tool '{name}' is already registered")
            self._index[name] = tool
            if tool.get("is_write"):
                self._write_tools.add(name)
        self._definitions.append(tool)

    def unregister(self, tool_name: str) -> bool:
        """Remove a dynamically registered definition from the live catalog.

        Only definitions carrying the ``dynamic`` marker (runtime-registered,
        e.g. MCP-discovered) may be removed — the static seed is permanent.
        Returns True if a definition was removed, False if the name is unknown.
        """
        tool = self._index.get(tool_name)
        if tool is None:
            return False
        if not tool.get("dynamic"):
            raise ValueError(
                f"Tool '{tool_name}' is a static definition and cannot be unregistered"
            )
        del self._index[tool_name]
        self._write_tools.discard(tool_name)
        self._definitions.remove(tool)
        return True

    def all_definitions(self) -> List[Dict[str, Any]]:
        """The live toolset (static seed + dynamic). Treat as read-only."""
        return self._definitions

    def get(self, tool_name: str) -> Optional[Dict[str, Any]]:
        return self._index.get(tool_name)

    def capabilities(self, tool_name: str) -> Dict[str, Any]:
        tool = self._index.get(tool_name)
        if tool is not None:
            caps = tool.get("capabilities")
            if caps is not None:
                return cast(Dict[str, Any], caps)
        return dict(_DEFAULT_CAPABILITIES)

    def is_write(self, tool_name: str) -> bool:
        return tool_name in self._write_tools


# Import-time seeding doubles as the validation pass: any capability drift in
# the static definitions raises immediately for every consumer (engine, tests,
# tooling), exactly like the former module-level validation loop.
_REGISTRY = ToolDefinitionRegistry(ALL_TOOL_DEFINITIONS)


def get_registry() -> ToolDefinitionRegistry:
    return _REGISTRY


def get_all_tool_definitions() -> List[Dict[str, Any]]:
    """The live toolset: static seed + dynamically registered definitions.
    Readers that must see runtime-registered tools (tool loop, composition,
    ToolManager) use this, NOT `ALL_TOOL_DEFINITIONS`."""
    return _REGISTRY.all_definitions()


def register_tool_definition(tool: Dict[str, Any]) -> None:
    """Register a definition discovered after import (validates; rejects
    duplicate names)."""
    _REGISTRY.register(tool)


def unregister_tool_definition(tool_name: str) -> bool:
    """Remove a runtime-registered (``dynamic``) definition. Raises on an
    attempt to remove a static seed definition."""
    return _REGISTRY.unregister(tool_name)


def is_write_tool(tool_name: str) -> bool:
    """Whether a tool is flagged `is_write` (parks for audit). Replaces the
    former import-time `WRITE_TOOLS` set so runtime-registered tools are
    covered."""
    return _REGISTRY.is_write(tool_name)


def get_tool_capabilities(tool_name: str) -> Dict[str, Any]:
    """
    Retrieves the security capabilities block for a given tool name.
    Returns a default block (all False) if the tool is not found.
    """
    return _REGISTRY.capabilities(tool_name)


def get_tool_definition(tool_name: str) -> Optional[Dict[str, Any]]:
    """
    Retrieves the full JSON definition for a given tool name.
    Returns None if the tool is not found.
    """
    return _REGISTRY.get(tool_name)


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
