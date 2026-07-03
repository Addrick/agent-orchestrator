# src/tool_policy.py
# Leaf value object (no src.* imports) — DP-204 moved it out of src/tools/
# so personas can hold a ToolPolicy without depending on the tools layer.

import logging
from typing import List, Dict, Any, Optional, Set

logger = logging.getLogger(__name__)


class ToolPolicy:
    """
    Manages tool permissions and security invariants for a persona or agent.
    """

    def __init__(
            self,
            default: str = "deny",
            allow: Optional[List[str]] = None,
            ask: Optional[List[str]] = None,
            capabilities_required: Optional[List[str]] = None,
            explicit_overrides: Optional[List[str]] = None
    ) -> None:
        self.default = default
        self.allow = allow or []
        self.ask = ask or []
        self.capabilities_required = capabilities_required or []
        self.explicit_overrides = explicit_overrides or []

    @classmethod
    def from_legacy_list(cls, enabled_tools: List[str]) -> "ToolPolicy":
        """
        Creates a ToolPolicy from a legacy flat list of tool names.
        """
        if "*" in enabled_tools:
            # The dangerous wildcard case from internal_tool_schema_cleanup.md
            return cls(default="allow", allow=["*"])
        return cls(default="deny", allow=enabled_tools)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "default": self.default,
            "allow": self.allow,
            "ask": self.ask,
            "capabilities_required": self.capabilities_required,
            "explicit_overrides": self.explicit_overrides
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ToolPolicy":
        return cls(
            default=data.get("default", "deny"),
            allow=data.get("allow"),
            ask=data.get("ask"),
            capabilities_required=data.get("capabilities_required"),
            explicit_overrides=data.get("explicit_overrides")
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
        Returns the subset of tools allowed by this policy.
        """
        if self.default == "allow" and "*" in self.allow:
            return all_tools

        allowed_names = set(self.allow)
        ask_names = set(self.ask)

        filtered = []
        for tool in all_tools:
            name = tool.get("function", {}).get("name")
            if name in allowed_names or name in ask_names:
                filtered.append(tool)

        return filtered
