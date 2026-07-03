# src/tools/composition.py
"""Tool-composition security validation (DP-128 quarantine seam).

DP-204 inverted the persona <-> tools dependency: personas no longer validate
their own tool composition (which forced src.persona to import
src.tools.definitions). Instead, tools/ exposes the validation as functions
personas are validated WITH:

- the persona loader (src/personas/store.py) calls
  ``validate_policy_composition`` at load and constructs quarantined personas
  (``security_block_reasons``) on errors;
- the operator-edit boundary (BotLogic ``set tools`` / ``set tool_policy``,
  which the web tools modal also routes through) calls
  ``revalidate_persona_security`` so a live edit can clear or trip the
  quarantine without a restart.

Persona itself stays a domain leaf: it stores the block reasons and exposes
``is_security_blocked()`` / ``set_security_block_reasons()``.
"""

import logging
from typing import TYPE_CHECKING, Any, Dict, List

from src.tool_policy import ToolPolicy
from src.tools.definitions import get_all_tool_definitions

if TYPE_CHECKING:
    from src.persona import Persona

logger = logging.getLogger(__name__)


def resolve_policy_tools(policy: ToolPolicy) -> List[Dict[str, Any]]:
    """The subset of the live tool catalog a policy exposes (allow + ask).

    Validation runs against the GLOBAL definitions (not binding-filtered),
    matching the original load-time check: ``['*']`` is always the full
    *static* toolset regardless of ``service_bindings``. Reads the registry
    (not the static ``ALL_TOOL_DEFINITIONS``) so runtime-registered tools are
    composition-checked too — but delegates the expansion to
    ``ToolPolicy.filter_tools`` so the wildcard's dynamic-tool exclusion
    (DP-268: ``['*']`` never auto-includes MCP-discovered defs) is the same
    here as at request time, by construction.
    """
    return policy.filter_tools(get_all_tool_definitions())


def validate_policy_composition(policy: ToolPolicy) -> List[str]:
    """Run the security composition invariants for a policy's toolset.

    Returns the list of violation messages ([] = clean)."""
    return policy.validate_composition(resolve_policy_tools(policy))


def revalidate_persona_security(persona: "Persona") -> bool:
    """Re-run tool composition validation against the persona's current policy
    and update its quarantine state. Called after any live tool edit so the
    operator can clear (or trip) the block without a restart. Returns the
    new ``is_security_blocked()`` value.
    """
    reasons = validate_policy_composition(persona.get_tool_policy())
    persona.set_security_block_reasons(reasons)
    if reasons:
        logger.warning(
            f"Persona '{persona.get_name()}' remains quarantined after edit: "
            f"{reasons}"
        )
    return persona.is_security_blocked()
