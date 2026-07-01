"""MCP server management tools (service_binding: mcp, DP-268).

Agent-driven setup for external MCP tool servers: add a server at runtime
(connect + discover + register its tools live, persist config), remove one,
or list what's configured. ``add_mcp_server`` / ``remove_mcp_server`` are
``is_write: True`` so they park for human confirmation — adding a server
installs a whole new capability surface and must never happen silently.

Capability notes:
- All three carry ``locality: "network"`` with the shared ``mcp`` egress
  domain: their data originates from / flows to the configured MCP servers,
  so management reads + writes form one same-origin closed loop under
  composition Rules 2/3 (the discovered tools themselves get their own
  per-server ``mcp:<server>`` domain).
- ``add_mcp_server``'s ``url`` argument is arbitrary model-controlled egress →
  exfil-capable (default). ``remove_mcp_server``/``list_mcp_servers`` take a
  configured name / no args → ``exfil_capable: False``.
- ``produces_untrusted: True`` where results carry server-authored text
  (discovered tool names/descriptions).
"""

from typing import Any, Dict, List


MCP_TOOLS: List[Dict[str, Any]] = [
    {
        "type": "function",
        "is_write": True,
        "service_binding": "mcp",
        "capabilities": {
            "produces_untrusted": True,
            "irreversible": False,
            "locality": "network",
            "sensitivity": "internal",
        },
        "function": {
            "name": "add_mcp_server",
            "description": (
                "Connect a new external MCP tool server (streamable-HTTP "
                "transport), discover its tools, and register them live under "
                "mcp__<name>__<tool>. Requires human approval. The server's "
                "tools get restrictive default security metadata (write, "
                "untrusted, irreversible, pii) until the operator downgrades "
                "them in the config file; personas must explicitly list the "
                "new tools and bind mcp:<name> to use them."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": (
                            "Short identifier for the server (lowercase "
                            "letters/digits/hyphens, e.g. \"home-assistant\"). "
                            "Becomes the mcp__<name>__ tool prefix and the "
                            "mcp:<name> service binding."
                        ),
                    },
                    "url": {
                        "type": "string",
                        "description": (
                            "Streamable-HTTP endpoint of the MCP server, e.g. "
                            "\"http://10.0.0.5:8000/mcp\"."
                        ),
                    },
                },
                "required": ["name", "url"],
            },
        },
    },
    {
        "type": "function",
        "is_write": True,
        "service_binding": "mcp",
        # exfil_capable=False: the only argument is a name already present in
        # the config map — no model-controlled payload can ride out.
        "capabilities": {
            "produces_untrusted": False,
            "irreversible": False,
            "locality": "network",
            "sensitivity": "internal",
            "exfil_capable": False,
        },
        "function": {
            "name": "remove_mcp_server",
            "description": (
                "Disconnect a configured MCP server, unregister all of its "
                "discovered tools, and delete it from the persisted config. "
                "Requires human approval."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "Server name from list_mcp_servers.",
                    },
                },
                "required": ["name"],
            },
        },
    },
    {
        "type": "function",
        "is_write": False,
        "service_binding": "mcp",
        # exfil_capable=False: no arguments at all — nothing can egress.
        # produces_untrusted=True: discovered tool names are server-authored.
        "capabilities": {
            "produces_untrusted": True,
            "irreversible": False,
            "locality": "network",
            "sensitivity": "internal",
            "exfil_capable": False,
        },
        "function": {
            "name": "list_mcp_servers",
            "description": (
                "List the configured MCP servers with their connection status "
                "and the tools each one currently provides."
            ),
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
]
