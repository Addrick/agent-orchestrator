# src/personas/policy_migration.py
"""Bring a stored tool policy up to DP-497 semantics, once.

Before DP-497 a persona's allow/ask was mostly ignored at call time: every
write asked whatever it was listed under (except stagr's `queue_codr_task` and
`copy_from_codr` under allow, DP-496), and every read ran, even under ask. Since
DP-497 the listed state is the only gate. Loading an old policy as-is would
therefore change behaviour — each write under allow would start running
unprompted — so the store rewrites it to what it actually did, logs each
change, and saves it stamped `policy_version: 2` (`ToolPolicy.to_dict`). A
stamped policy is never migrated again, so an ask on a read set afterwards is
kept.

The `"*"` wildcard is expanded on every load, whatever the version: to each
non-dynamic tool in the catalog at its default state, which is exactly how the
wildcard behaved (reads ran, writes asked; DP-496 never let it count as an
allow).

Lives in the store's layer, not in `src.tool_policy`, because it reads the
tool catalog and encodes one ticket's history (`DP496_UNPROMPTED_WRITES`).
"""

from typing import Any, Dict, Iterable, List, Optional, Tuple

from src.tool_policy import POLICY_VERSION, ToolPolicy, default_tool_state, definition_index

# The only writes DP-496 let run without a prompt (listed by name under allow,
# not under ask). The `auto_approvable` key that marked them is gone; the
# migration must still know what they did.
DP496_UNPROMPTED_WRITES = frozenset({"queue_codr_task", "copy_from_codr"})


def _stored_lists(entry: Dict[str, Any]) -> Tuple[List[str], List[str], List[str], int]:
    """allow, ask, capabilities_required and policy_version of a persona
    entry. A legacy entry with only `enabled_tools` is a version-1 policy
    with everything under allow (what `from_legacy_list` built before DP-497)."""
    policy = entry.get("tool_policy")
    if isinstance(policy, dict):
        version = policy.get("policy_version", 1)
        return (list(policy.get("allow") or []), list(policy.get("ask") or []),
                list(policy.get("capabilities_required") or []),
                version if isinstance(version, int) else 1)
    return list(entry.get("enabled_tools") or []), [], [], 1


def _v1_behaviour(name: str, definition: Optional[Dict[str, Any]], allow: List[str],
                  ask: List[str]) -> str:
    """What a call to a listed tool did before DP-497: `allow` (ran) or
    `ask` (parked). A tool the catalog does not know yet -- an MCP tool
    registers after personas load -- cannot be read, so it is ask (fails
    closed; the change is logged)."""
    if definition is None:
        return "ask"
    if not definition.get("is_write"):
        return "allow"
    if name in DP496_UNPROMPTED_WRITES and name in allow and name not in ask:
        return "allow"
    return "ask"


def resolve_stored_policy(
    entry: Dict[str, Any], definitions: Iterable[Dict[str, Any]],
) -> Tuple[ToolPolicy, List[str], bool]:
    """The persona entry's tool policy in DP-497 terms.

    Returns `(policy, changes, needs_save)`: `changes` describes each tool
    whose state moved (for the log), and `needs_save` is True when the stored
    form differs from what `policy.to_dict()` will write (a migration ran, a
    wildcard was expanded, or the version stamp is missing). Idempotent: a
    policy this returns, saved and loaded again, comes back unchanged with
    `needs_save` False.
    """
    allow, ask, caps_required, version = _stored_lists(entry)
    index = definition_index(definitions)
    changes: List[str] = []
    needs_save = version < POLICY_VERSION

    expanded: List[str] = []
    if "*" in allow or "*" in ask:
        listed = set(allow) | set(ask)
        expanded = [n for n, d in index.items() if not d.get("dynamic") and n not in listed]
        allow = [n for n in allow if n != "*"]
        ask = [n for n in ask if n != "*"]
        changes.append(f"'*' expanded to {len(expanded)} tool(s) at their defaults")
        needs_save = True

    if version < POLICY_VERSION:
        new_allow: List[str] = []
        new_ask: List[str] = []
        for name in dict.fromkeys(allow + ask):
            was = "ask" if name in ask else "allow"
            now = _v1_behaviour(name, index.get(name), allow, ask)
            (new_allow if now == "allow" else new_ask).append(name)
            if now != was:
                unknown = " (not in the tool catalog at load)" if name not in index else ""
                changes.append(f"{name}: {was} -> {now}{unknown}")
        allow, ask = new_allow, new_ask

    for name in expanded:
        (ask if default_tool_state(index[name]) == "ask" else allow).append(name)

    policy = ToolPolicy(default="deny", allow=allow, ask=ask,
                        capabilities_required=caps_required)
    return policy, changes, needs_save
