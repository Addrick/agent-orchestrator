# src/tools/policy.py

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

    def validate_composition(self, tool_definitions: List[Dict[str, Any]]) -> List[str]:
        """
        Validates a set of tools against security invariants.
        Returns a list of error messages (empty if valid).
        """
        errors = []
        
        # Capability tags for the active toolset
        has_network_read = False
        has_local_write = False
        has_untrusted_read = False
        has_network_write = False
        has_pii_read = False
        has_network_any = False

        for tool in tool_definitions:
            name = tool.get("function", {}).get("name", "<unknown>")
            is_write = tool.get("is_write", False)
            caps = tool.get("capabilities", {})
            locality = caps.get("locality")
            sensitivity = caps.get("sensitivity")
            produces_untrusted = caps.get("produces_untrusted", False)

            if locality == "network":
                has_network_any = True
                if not is_write:
                    has_network_read = True
                else:
                    has_network_write = True
            
            if locality == "local" and is_write:
                has_local_write = True
            
            if produces_untrusted and not is_write:
                has_untrusted_read = True
            
            if sensitivity == "pii" and not is_write:
                has_pii_read = True

        # Rule 1: network:read + local:write → DENY
        if has_network_read and has_local_write:
            if "network_read_local_write" not in self.explicit_overrides:
                errors.append("Insecure composition: network:read + local:write (potential disk rewrite via injection)")

        # Rule 2: untrusted:read + network:write → DENY
        if has_untrusted_read and has_network_write:
            if "untrusted_read_network_write" not in self.explicit_overrides:
                errors.append("Insecure composition: untrusted:read + network:write (potential exfiltration via injection)")

        # Rule 3: pii:read + network:* → DENY
        if has_pii_read and has_network_any:
            if "pii_read_network_any" not in self.explicit_overrides:
                errors.append("Insecure composition: pii:read + network:* (potential PII exfiltration)")

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
