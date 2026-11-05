# src/personas/policy_migration.py
"""Bring a stored tool policy up to DP-497 semantics, once per tool.

Before DP-497 a persona's allow/ask was mostly ignored at call time: every
write asked whatever it was listed under (except stagr's `queue_codr_task` and
`copy_from_codr` under allow, DP-496), and every read ran, even under ask. Since
DP-497 the listed state is the only gate. Loading an old policy as-is would
therefore change behaviour — each write under allow would start running
unprompted — so the store rewrites each listed tool to what it actually did
(`tool_policy.pre_dp497_state`), logs each change, and saves the result
stamped `policy_version: 2`.

What a tool did depends on its definition, and an MCP tool's definition only
exists once its server connects — after personas load, or much later if the
server is down. Such a name cannot be classified at load, and guessing would be
wrong either way (ask makes every Hindsight `recall` prompt; allow lets a write
run unprompted). So the record is **per name**: the name stays exactly as
stored and is listed in the policy's `unmigrated`, and `policy_version: 2`
means "every listed name not in `unmigrated` has been migrated". The MCP client
finishes the job when the definitions register
(`ToolPolicy.classify_unmigrated`, from `MCPClientManager._revalidate_personas`,
the hook that already runs after every live registration). A name that has been
classified is never migrated again, so an ask on a read set afterwards is kept.

The `"*"` wildcard is expanded on every load, whatever the version: to each
non-dynamic tool in the catalog at its default state, which is exactly how the
wildcard behaved (reads ran, writes asked; DP-496 never let it count as an
allow).
"""

from typing import Any, Dict, Iterable, List, Tuple

from src.tool_policy import (
    POLICY_VERSION, ToolPolicy, default_tool_state, definition_index, pre_dp497_state,
)


def _stored(entry: Dict[str, Any]) -> Tuple[List[str], List[str], List[str], int, List[str]]:
    """allow, ask, capabilities_required, policy_version and unmigrated of a
    persona entry. A legacy entry with only `enabled_tools` is a version-1
    policy with everything under allow (what `from_legacy_list` built before
    DP-497)."""
    policy = entry.get("tool_policy")
    if isinstance(policy, dict):
        version = policy.get("policy_version", 1)
        return (list(policy.get("allow") or []), list(policy.get("ask") or []),
                list(policy.get("capabilities_required") or []),
                version if isinstance(version, int) else 1,
                list(policy.get("unmigrated") or []))
    return list(entry.get("enabled_tools") or []), [], [], 1, []


def resolve_stored_policy(
    entry: Dict[str, Any], definitions: Iterable[Dict[str, Any]],
) -> Tuple[ToolPolicy, List[str], bool]:
    """The persona entry's tool policy in DP-497 terms.

    Returns `(policy, changes, needs_save)`: `changes` describes what moved
    (for the log), and `needs_save` is True when the stored form differs from
    what `policy.to_dict()` will write. A name with no definition in
    `definitions` is left exactly as stored and recorded in
    `policy.unmigrated`. Idempotent: a policy this returns, saved and loaded
    again with the same catalog, comes back unchanged with `needs_save` False.
    """
    allow, ask, caps_required, version, unmigrated = _stored(entry)
    defs = list(definitions)
    index = definition_index(defs)
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
        # Every listed name is unclassified; `classify_unmigrated` below moves
        # each one the catalog knows, and leaves the rest as stored.
        unmigrated = list(dict.fromkeys(allow + ask))

    for name in expanded:
        (ask if default_tool_state(index[name]) == "ask" else allow).append(name)

    policy = ToolPolicy(default="deny", allow=allow, ask=ask,
                        capabilities_required=caps_required, unmigrated=unmigrated)
    for name, was, now in policy.classify_unmigrated(defs):
        needs_save = True
        if was != now:
            changes.append(f"{name}: {was} -> {now}")
    if policy.unmigrated:
        changes.append(
            f"{len(policy.unmigrated)} tool(s) not registered yet (an MCP server not "
            f"connected); left as stored until they register: {', '.join(policy.unmigrated)}"
        )
    return policy, changes, needs_save


__all__ = ["resolve_stored_policy", "pre_dp497_state"]
