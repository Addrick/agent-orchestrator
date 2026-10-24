"""stagr setup-VM tools (service_binding: stagr, DP-456).

The setup VM has no LAN and reaches the internet only through the egress
proxy's domain allowlist, so ``stagr_shell`` is a read (ungated): the VM is the
boundary, not a review of each command. ``codr_shell`` is ungated for the same
reason (DP-462): codr's sandbox is offline and restorable, and the handler runs
it only between codr's turns. Everything that crosses a boundary is
``is_write`` and parked for approval: widening the allowlist, and copying files
into or out of codr's sandbox.

``stagr_shell`` returns whatever the VM fetched from the web, and
``codr_shell`` whatever was copied in from there, so both are
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
        "is_write": False,
        "service_binding": "stagr",
        "capabilities": _caps(untrusted=True),
        "function": {
            "name": "codr_shell",
            "description": (
                "Run a bash command in codr's workspace on codr's offline sandbox (Debian 13, "
                "user agent with passwordless sudo, Python 3.13, git; no internet, no LAN). Use "
                "it between tasks: install what a task needs from files you copied in (e.g. pip "
                "install --no-index --find-links wheels ...), check the environment, debug a test "
                "that fails offline. Refused while codr is working, and codr starts nothing "
                "while it runs. Returns the exit code and the output (stdout and stderr "
                "together; the middle of very long output is cut)."
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
        "is_write": True,
        "service_binding": "stagr",
        "capabilities": _caps(),
        "function": {
            "name": "copy_to_codr",
            "description": (
                "Copy a file or folder from stagr into codr's workspace. Needs Adam's approval. "
                "Any .git is left behind unless keep_git is true, so no upstream history reaches "
                "codr. Remove anything else that would give the answer away (a changelog entry, "
                "a test from the fix) before copying. Never overwrites: refused if dest exists, "
                "and while codr is working."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "source": {"type": "string", "description": "File or folder on stagr, relative to /home/agent/work."},
                    "dest": {
                        "type": "string",
                        "description": "New path in codr's workspace, e.g. tasks/09-click-issue-2711 or tools/mcp-foo.",
                    },
                    "keep_git": {"type": "boolean", "description": "Copy .git folders too. Default false."},
                },
                "required": ["source", "dest"],
            },
        },
    },
    {
        "type": "function",
        "is_write": True,
        "service_binding": "stagr",
        "capabilities": _caps(),
        "function": {
            "name": "copy_from_codr",
            "description": (
                "Copy a file or folder from codr's workspace to stagr, e.g. to review or test "
                "codr's work. Needs Adam's approval. Any .git is left behind unless keep_git is "
                "true. Never overwrites: refused if dest exists on stagr."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "source": {"type": "string", "description": "File or folder in codr's workspace, e.g. tasks/08-cron."},
                    "dest": {"type": "string", "description": "New path on stagr, relative to /home/agent/work."},
                    "keep_git": {"type": "boolean", "description": "Copy .git folders too, e.g. to diff codr's commits. Default false."},
                },
                "required": ["source", "dest"],
            },
        },
    },
]
