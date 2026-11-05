# src/tool_policy.py
# Leaf value object (no src.* imports) — DP-204 moved it out of src/tools/
# so personas can hold a ToolPolicy without depending on the tools layer.

import logging
from typing import Any, Dict, FrozenSet, Iterable, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)

# The composition-invariant names that validate_composition honors as
# overrides. Anything else in explicit_overrides is inert (fails closed).
KNOWN_OVERRIDES = frozenset({
    "network_read_local_write",
    "untrusted_read_network_write",
    "pii_read_network_any",
})


# DP-497: the serialized policy's semantics version. Absent (1) = before
# DP-497, when allow/ask on most tools was ignored; 2 = allow/ask is the only
# approval gate. The persona store migrates a v1 policy once, on load
# (src/personas/policy_migration.py); every policy written carries 2.
POLICY_VERSION = 2

WILDCARD_RETIRED = (
    "The '*' wildcard is retired (DP-497): list each tool by name under "
    "allow or ask (`set tools all` does that at each tool's default)."
)


def declares_irreversible(definition: Optional[Dict[str, Any]]) -> bool:
    """Whether a tool definition says its effect may not be undoable:
    ``capabilities.irreversible``, or an ``irreversible_if`` classifier (it is
    for some arguments). The static label behind an irreversible tool's
    default of ask and the warning on setting it to allow (DP-497); the
    per-call answer is ``definitions.is_irreversible(name, args)``."""
    caps = (definition or {}).get("capabilities") or {}
    return bool(caps.get("irreversible") or caps.get("irreversible_if"))


def default_tool_state(definition: Optional[Dict[str, Any]]) -> str:
    """The state a tool gets when it is enabled without one (DP-497): a
    write or an irreversible tool -> ``ask``, anything else -> ``allow``. No
    definition -> ``ask``: what it does cannot be read, so it fails closed."""
    if definition is None:
        return "ask"
    if definition.get("is_write") or declares_irreversible(definition):
        return "ask"
    return "allow"


# The only writes DP-496 let run without a prompt (listed by name under allow,
# not under ask). The `auto_approvable` key that marked them is gone; the
# pre-DP-497 rule below must still know what they did.
DP496_UNPROMPTED_WRITES = frozenset({"queue_codr_task", "copy_from_codr"})


def pre_dp497_state(name: str, definition: Dict[str, Any], allow: List[str],
                    ask: List[str]) -> str:
    """What a call to a listed tool did before DP-497: ``allow`` (it ran) or
    ``ask`` (it parked). Reads ran whatever they were listed under; writes
    parked, except DP-496's two stagr tools listed under allow alone. Needs
    the tool's definition -- a name whose definition is not registered yet
    (an MCP tool before its server connects) cannot be classified, and
    callers leave it as stored (``ToolPolicy.unmigrated``)."""
    if not definition.get("is_write"):
        return "allow"
    if name in DP496_UNPROMPTED_WRITES and name in allow and name not in ask:
        return "allow"
    return "ask"


def definition_index(definitions: Iterable[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """Name -> definition for every entry with a callable name."""
    index: Dict[str, Dict[str, Any]] = {}
    for d in definitions:
        name = (d.get("function") or {}).get("name")
        if isinstance(name, str) and name:
            index[name] = d
    return index


class ToolPolicy:
    """
    A persona's (or agent's) tools and how each one runs.

    Each tool is in one of three states (DP-497): listed nowhere (off — never
    offered), under ``allow`` (runs with no prompt), or under ``ask`` (parks
    for approval). The state is the only approval gate, for reads and writes
    alike (``requires_approval``). A name under both lists is ask (fails
    closed).
    """

    def __init__(
            self,
            default: str = "deny",
            allow: Optional[List[str]] = None,
            ask: Optional[List[str]] = None,
            capabilities_required: Optional[List[str]] = None,
            explicit_overrides: Optional[List[str]] = None,
            unmigrated: Optional[List[str]] = None,
    ) -> None:
        self.default = default
        self.allow = allow or []
        self.ask = ask or []
        self.capabilities_required = capabilities_required or []
        self.explicit_overrides = explicit_overrides or []
        # DP-497: names still listed in their pre-DP-497 state because their
        # definition was not registered when the policy was migrated (an MCP
        # tool registers after personas load). Set by the persona store only;
        # emptied by `classify_unmigrated` once the definitions arrive.
        self.unmigrated: List[str] = list(unmigrated or [])

    @classmethod
    def from_legacy_list(
            cls, enabled_tools: List[str], definitions: Iterable[Dict[str, Any]],
    ) -> "ToolPolicy":
        """A policy enabling ``enabled_tools``, each at its default state
        (``default_tool_state``: reads allow; writes, irreversible and unknown
        tools ask).

        ``"*"`` expands to every tool in ``definitions`` by name, at its
        default (DP-497 retired the wildcard: it would have let a tool added
        later run unprompted, irreversible ones included, with nobody choosing
        that). Definitions carrying the ``dynamic`` marker (runtime-registered,
        e.g. MCP-discovered, DP-268) are never part of the expansion.
        ``definitions`` is the live catalog — the leaf cannot import it."""
        index = definition_index(definitions)
        names: List[str] = []
        for name in enabled_tools:
            if name == "*":
                names.extend(n for n, d in index.items() if not d.get("dynamic"))
            else:
                names.append(name)
        allow: List[str] = []
        ask: List[str] = []
        for name in dict.fromkeys(names):
            (ask if default_tool_state(index.get(name)) == "ask" else allow).append(name)
        return cls(default="deny", allow=allow, ask=ask)

    def to_dict(self) -> Dict[str, Any]:
        # explicit_overrides deliberately absent: it is a privileged persona
        # field with its own gated setter/persistence (DP-277), never part of
        # the generic policy dict that PATCH bodies / `set tool_policy` carry.
        return {
            "default": self.default,
            "allow": self.allow,
            "ask": self.ask,
            "capabilities_required": self.capabilities_required,
            "policy_version": POLICY_VERSION,
            **({"unmigrated": list(self.unmigrated)} if self.unmigrated else {}),
        }

    def state_of(self, tool_name: str) -> str:
        """``off``, ``allow`` or ``ask``; under both lists -> ``ask``."""
        if tool_name in self.ask:
            return "ask"
        if tool_name in self.allow:
            return "allow"
        return "off"

    def requires_approval(self, tool_name: str, definition: Optional[Dict[str, Any]]) -> bool:
        """Whether a call to ``tool_name`` parks for approval (DP-497): its
        state and nothing else — ask parks, allow runs, read or write.

        A tool listed nowhere is never offered from this policy, so a call to
        it is refused before it gets here; should one arrive anyway, its
        default state decides (a write still asks)."""
        if tool_name in self.unmigrated:
            # Stored before DP-497 and not classified yet: what it did then
            # (for any tool that can be unclassified -- never DP-496's two --
            # that is its default). Unregistered, it is not offered at all.
            return default_tool_state(definition) == "ask"
        state = self.state_of(tool_name)
        if state == "off":
            return default_tool_state(definition) == "ask"
        return state == "ask"

    def classify_unmigrated(
            self, definitions: Iterable[Dict[str, Any]],
    ) -> List[Tuple[str, str, str]]:
        """Finish the DP-497 migration for each `unmigrated` name whose
        definition is now in ``definitions``: move it to what it did before
        DP-497 (``pre_dp497_state``) and drop it from `unmigrated`. Names
        still unregistered stay exactly as stored. Returns ``(name, was,
        now)`` for every name classified, moved or not -- a non-empty result
        means the stored form changed and should be saved."""
        if not self.unmigrated:
            return []
        index = definition_index(definitions)
        done: List[Tuple[str, str, str]] = []
        for name in list(self.unmigrated):
            definition = index.get(name)
            if definition is None:
                continue
            was = self.state_of(name)
            now = pre_dp497_state(name, definition, self.allow, self.ask)
            if name in self.allow and name in self.ask or was != now:
                self.allow = [n for n in self.allow if n != name]
                self.ask = [n for n in self.ask if n != name]
                (self.allow if now == "allow" else self.ask).append(name)
            self.unmigrated.remove(name)
            done.append((name, was, now))
        return done

    def irreversible_allowed(self, definitions: Iterable[Dict[str, Any]]) -> List[str]:
        """Names under allow (not ask) whose definition declares them
        irreversible — what an edit warns about (DP-497)."""
        index = definition_index(definitions)
        return [n for n in self.allow
                if n not in self.ask and declares_irreversible(index.get(n))]

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ToolPolicy":
        # explicit_overrides in a generic policy dict is IGNORED (DP-277): it
        # is the kill switch for the composition invariants, so it must not be
        # settable via any caller-supplied dict (PATCH body, `set tool_policy`).
        # Legacy saved values are migrated by the persona store, not here —
        # as are a saved "*" and a pre-DP-497 policy (no `policy_version`):
        # the dict given here already means what it says (DP-497).
        if "explicit_overrides" in data:
            logger.warning(
                "Ignoring 'explicit_overrides' in generic tool_policy dict — "
                "overrides require the dedicated operator setter (DP-277)."
            )
        allow = data.get("allow")
        ask = data.get("ask")
        if "*" in (allow or []) or "*" in (ask or []):
            raise ValueError(WILDCARD_RETIRED)
        return cls(
            default="deny",
            allow=allow,
            ask=ask,
            capabilities_required=data.get("capabilities_required"),
        )

    @staticmethod
    def _egress_domain(tool: Dict[str, Any]) -> str:
        """Maps a tool to its *egress domain* — where its data can flow to/from.

        The lethal-trifecta danger is data reaching a sink *outside the domain
        it came from*; same-domain read+write (e.g. zammad→zammad) is a closed
        loop and not exfiltration. The mapping:

            network + service_binding  -> that binding (e.g. "zammad")
            network + no binding       -> "open"  (arbitrary, attacker-reachable sink)
            local                      -> "local"

        Derived from ``service_binding`` today. An explicit
        ``capabilities["domain"]`` is honored if present, leaving a seam for a
        tool whose egress domain diverges from its service identity — not used
        yet (YAGNI), but the single chokepoint for adding it.
        """
        caps = tool.get("capabilities", {})
        explicit = caps.get("domain")
        if explicit:
            return str(explicit)
        locality = caps.get("locality")
        if locality == "network":
            # Fail-safe: a network tool that forgot to declare a binding is
            # treated as arbitrary egress, not trusted-internal.
            return tool.get("service_binding") or "open"
        # local (or undeclared locality — no asserted network egress)
        return "local"

    def _gather_domain_caps(self, tool_definitions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Accumulates per-domain capability sets across the active toolset.

        Read-side provenance (untrusted / pii) is domain-tagged regardless of
        locality: a poisoned local memory hit is as much a foreign source to a
        zammad write as a web result is.
        """
        untrusted_read_domains: Set[str] = set()
        pii_read_domains: Set[str] = set()
        network_write_domains: Set[str] = set()
        network_tool_domains: Set[str] = set()  # every network tool, read or write
        has_network_read = False
        has_local_write = False

        for tool in tool_definitions:
            is_write = tool.get("is_write", False)
            caps = tool.get("capabilities", {})
            locality = caps.get("locality")
            domain = self._egress_domain(tool)

            # A network tool may opt out of exfil accounting with
            # exfil_capable=False: its egress carries no model-controlled payload
            # (constrained args to trusted infra), so it is not a data-exfil
            # vector and must not arm Rules 2/3. Destructive risk on such tools is
            # covered separately by is_write (parked for confirmation). Default
            # True keeps every existing tool unchanged.
            if locality == "network" and caps.get("exfil_capable", True):
                network_tool_domains.add(domain)
                if is_write:
                    network_write_domains.add(domain)
                else:
                    has_network_read = True
            elif locality == "local" and is_write:
                has_local_write = True

            if not is_write:
                if caps.get("produces_untrusted", False):
                    untrusted_read_domains.add(domain)
                if caps.get("sensitivity") == "pii":
                    pii_read_domains.add(domain)

        return {
            "untrusted_read_domains": untrusted_read_domains,
            "pii_read_domains": pii_read_domains,
            "network_write_domains": network_write_domains,
            "network_tool_domains": network_tool_domains,
            "has_network_read": has_network_read,
            "has_local_write": has_local_write,
        }

    def validate_composition(self, tool_definitions: List[Dict[str, Any]]) -> List[str]:
        """
        Validates a set of tools against security invariants.
        Returns a list of error messages (empty if valid).

        Rules 2 and 3 are *same-origin aware*: they fire only when untrusted or
        PII data could reach a domain it did not originate from. Reading and
        writing within a single ``service_binding`` (e.g. a zammad ticketing
        agent) is a closed loop and validates clean; the protection re-arms the
        instant a foreign-domain tool (e.g. ``web_search`` → ``open``) is added.
        """
        errors: List[str] = []
        caps = self._gather_domain_caps(tool_definitions)

        # Rule 1: network:read + local:write → DENY (local is always a foreign
        # domain to a network read; no same-origin exception possible).
        if (caps["has_network_read"] and caps["has_local_write"]
                and "network_read_local_write" not in self.explicit_overrides):
            errors.append("Insecure composition: network:read + local:write (potential disk rewrite via injection)")

        # Rule 2: untrusted:read + network:write to a FOREIGN domain → DENY.
        foreign_writes = caps["network_write_domains"] - caps["untrusted_read_domains"]
        if (caps["untrusted_read_domains"] and foreign_writes
                and "untrusted_read_network_write" not in self.explicit_overrides):
            errors.append(
                "Insecure composition: untrusted:read + network:write to foreign domain(s) "
                f"{sorted(foreign_writes)} (potential exfiltration via injection)"
            )

        # Rule 3: pii:read + network egress to a FOREIGN domain → DENY.
        foreign_network = caps["network_tool_domains"] - caps["pii_read_domains"]
        if (caps["pii_read_domains"] and foreign_network
                and "pii_read_network_any" not in self.explicit_overrides):
            errors.append(
                "Insecure composition: pii:read + network:* egress to foreign domain(s) "
                f"{sorted(foreign_network)} (potential PII exfiltration)"
            )

        return errors

    def filter_tools(self, all_tools: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Returns the subset of tools this policy lists (allow + ask).

        Naming a tool is the only way in (DP-497 retired the ``['*']``
        wildcard), so a runtime-registered tool (``dynamic``, e.g. MCP,
        DP-268) reaches a persona only when that persona names it.

        ``composition.resolve_policy_tools`` delegates here, so exposure
        filtering and composition validation see the same set by
        construction.
        """
        listed = set(self.allow) | set(self.ask)
        return [t for t in all_tools if t.get("function", {}).get("name") in listed]


def callable_tool_names(tools: Optional[Iterable[Dict[str, Any]]]) -> FrozenSet[str]:
    """Names a model can call from a list of tool definitions.

    Only `type == "function"` entries count: that is what every provider
    declares to the model (`google.build_google_tools` filters on it), so a
    flag entry like `google_grounding` — which carries a `function.name` but
    is never declared as a function — is not callable through a list that
    contains it. Shared by the tool loop's offered set and the MCP bridge's
    exposed set (DP-404), so the two cannot disagree on shape. Lives here, not
    in src.tools.definitions, so src.proposals can use it without importing
    upward (DP-407).
    """
    names = set()
    for t in tools or ():
        if t.get("type") != "function":
            continue
        name = (t.get("function") or {}).get("name")
        if isinstance(name, str) and name:
            names.add(name)
    return frozenset(names)
