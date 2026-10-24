"""stagr setup-VM tools (service_binding: stagr, DP-456).

The setup VM has no LAN and reaches the internet only through the egress
proxy's domain allowlist, so ``stagr_shell`` is a read (ungated): the VM is the
boundary, not a review of each command. Everything that crosses out of it is
``is_write`` and parked for approval: widening the allowlist, and moving files
into or out of codr's offline sandbox.

``stagr_shell`` returns whatever the VM fetched from the web, so it is
``produces_untrusted``. Every tool here is in the one ``stagr`` domain, which
keeps ToolPolicy Rule 2 (untrusted read + foreign network write) closed: the
writes go to the same boxes the reads come from. Adding a tool from another
domain to the setup persona re-arms it, as it should.
"""

from typing import Any, Dict, List


def _caps(*, untrusted: bool = False) -> Dict[str, Any]:
    return {
        "produces_untrusted": untrusted,
        "irreversible": False,
        "locality": "network",
        "sensitivity": "internal",
    }


STAGR_TOOLS: List[Dict[str, Any]] = [
    {
        "type": "function",
        "is_write": False,
        "service_binding": "stagr",
        "capabilities": _caps(untrusted=True),
        "function": {
            "name": "stagr_shell",
            "description": (
                "Run a bash command on stagr, your setup VM (Debian 13, user agent with "
                "passwordless sudo, git, python3, pip, curl). It starts in /home/agent/work. "
                "It has no LAN, and its internet goes through a proxy that admits only "
                "allowlisted domains (GitHub, PyPI, Debian mirrors by default; see "
                "list_web_access). A refused domain fails with HTTP 403 from the proxy: ask "
                "for it with request_web_access instead of working around it. Returns the "
                "exit code and the output (stdout and stderr together; the middle of very "
                "long output is cut)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string", "description": "The bash to run. May be several lines."},
                    "timeout_seconds": {
                        "type": "integer",
                        "description": "Kill it after this long. Default 600, at most 3600.",
                    },
                },
                "required": ["command"],
            },
        },
    },
    {
        "type": "function",
        "is_write": False,
        "service_binding": "stagr",
        "capabilities": _caps(),
        "function": {
            "name": "list_web_access",
            "description": (
                "Show which domains stagr may reach: the defaults, then each approved grant "
                "with when it expires. A domain also admits its subdomains."
            ),
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "is_write": True,
        "service_binding": "stagr",
        "capabilities": _caps(),
        "function": {
            "name": "request_web_access",
            "description": (
                "Ask Adam to let stagr reach one more domain for a while. Needs his approval. "
                "Ask for the narrowest domain that works (e.g. files.example.org, not "
                "example.org) and the shortest time the job needs."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "domain": {"type": "string", "description": "A bare domain name: no scheme, port or path, no IP address."},
                    "minutes": {"type": "integer", "description": "How long, 1-1440."},
                    "reason": {"type": "string", "description": "What you need it for, in one sentence."},
                },
                "required": ["domain", "minutes", "reason"],
            },
        },
    },
    {
        "type": "function",
        "is_write": True,
        "service_binding": "stagr",
        "capabilities": _caps(),
        "function": {
            "name": "deliver_to_codr",
            "description": (
                "Copy a folder from stagr into codr's offline sandbox as tasks/<task>/. Needs "
                "Adam's approval. Any .git is left behind: the folder arrives as a fresh git "
                "repo with one commit, so no upstream history reaches codr. Remove anything "
                "else that would give the answer away (a changelog entry, a test from the fix) "
                "before delivering. Refused while codr is working, and if tasks/<task> already "
                "exists there."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Folder on stagr, relative to /home/agent/work."},
                    "task": {"type": "string", "description": "Task folder name in codr's workspace, e.g. 09-click-issue-2711."},
                },
                "required": ["path", "task"],
            },
        },
    },
    {
        "type": "function",
        "is_write": True,
        "service_binding": "stagr",
        "capabilities": _caps(),
        "function": {
            "name": "retrieve_from_codr",
            "description": (
                "Copy codr's tasks/<task>/ folder to stagr as from-codr/<task>/ (replacing any "
                "earlier copy), e.g. to review or test its work. Needs Adam's approval."
            ),
            "parameters": {
                "type": "object",
                "properties": {"task": {"type": "string", "description": "Task folder name in codr's workspace."}},
                "required": ["task"],
            },
        },
    },
]
