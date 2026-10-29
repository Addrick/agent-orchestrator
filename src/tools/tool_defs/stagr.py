"""stagr setup-VM tools (service_binding: stagr, DP-456).

The setup VM's LAN and WAN are switches, off by default, and its WAN goes only
through a host allowlist (DP-465), so ``stagr_shell`` is a read (ungated): the
network is gated, not each command. ``codr_shell`` is ungated for the same
reason (DP-462): codr's sandbox is offline and restorable, and the handler runs
it only between codr's turns. Everything that crosses a boundary is
``is_write`` and parked for approval: turning a switch on, approving a host,
copying files out of codr's sandbox, and queueing a codr task
(``queue_codr_task``: the approval is where the operator sees the hidden check). ``sandbox_net_status`` and
``set_sandbox_net`` are the portal panel's (the operator's click is the
approval there); no persona is given them. ``copy_to_codr`` is not: ``codr_shell`` can already write
any file there, so a gate on the copy would only be a convention.

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
                "Its LAN and WAN are switches, normally off (see list_web_access). With WAN on, "
                "the internet is web only (HTTP, HTTPS on 443) through a proxy that admits "
                "allowlisted hosts (GitHub, PyPI, Debian mirrors by default): clone over https, "
                "never git@. A refused host fails with HTTP 403 from the proxy: ask for it with "
                "request_web_access instead of working around it. Returns the "
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
                "Show stagr's two network switches (LAN, WAN: on or off) and the hosts its WAN "
                "lets through: the defaults (each admits its subdomains; '=' means that host "
                "only), then the hosts approved since WAN went on."
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
                "Ask Adam to let stagr reach one more host through the proxy. Needs his "
                "approval, and stagr's WAN must be on. The approval admits exactly that host "
                "(not its subdomains) until stagr's WAN goes off."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "host": {"type": "string", "description": "A bare host name: no scheme, port or path, no IP address."},
                    "reason": {"type": "string", "description": "What you need it for, in one sentence."},
                },
                "required": ["host", "reason"],
            },
        },
    },
    {
        "type": "function",
        "is_write": True,
        "service_binding": "stagr",
        "capabilities": _caps(),
        "function": {
            "name": "request_network",
            "description": (
                "Ask Adam to turn one of stagr's network switches on. Needs his approval. "
                "wan: the internet through the proxy allowlist (web only). lan: the shared folder "
                "on dt21 (~/work/dt21) and nothing else at home, not even DNS. Ask only for "
                "what the job needs; Adam turns it off again."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "zone": {"type": "string", "enum": ["lan", "wan"], "description": "Which switch."},
                    "reason": {"type": "string", "description": "What you need it for, in one sentence."},
                },
                "required": ["zone", "reason"],
            },
        },
    },
    {
        "type": "function",
        "is_write": False,
        "service_binding": "stagr",
        "capabilities": _caps(),
        "function": {
            "name": "sandbox_net_status",
            "description": (
                "Both sandbox boxes (codr, stagr): LAN and WAN switch state, the WAN allowlist, "
                "open LAN and WAN connections, and the proxy's recent connections and refusals."
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
            "name": "set_sandbox_net",
            "description": (
                "Turn a sandbox box's LAN or WAN switch on or off. Without confirm, turning off "
                "with connections open (or on with the other switch already on) changes nothing "
                "and answers status 'confirm' with what would happen."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "box": {"type": "string", "enum": ["codr", "stagr"]},
                    "zone": {"type": "string", "enum": ["lan", "wan"]},
                    "state": {"type": "string", "enum": ["on", "off"]},
                    "confirm": {"type": "boolean", "description": "Go ahead despite the warning. Default false."},
                },
                "required": ["box", "zone", "state"],
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
                "that fails offline. Never run opencode here: start codr's work with queue_codr_task. "
                "Refused while codr is working, and codr starts nothing "
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
        "is_write": False,
        "service_binding": "stagr",
        "capabilities": _caps(),
        "function": {
            "name": "copy_to_codr",
            "description": (
                "Copy a file or folder from stagr into codr's workspace. "
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
    {
        "type": "function",
        "is_write": True,
        "service_binding": "stagr",
        "capabilities": _caps(),
        "function": {
            "name": "queue_codr_task",
            "description": (
                "Put a task in codr's queue. Needs Adam's approval. DERPR starts it when codr is "
                "idle, in tasks/<name>/ of codr's workspace: copy the task's files there with "
                "copy_to_codr first, under the same name. task_md is what codr reads; check_sh "
                "is the hidden pass/fail check (exit 0 = pass), run in tasks/<name>/ after codr "
                "says it is done. codr never sees it. Refused if a task of that name is queued."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "The task's name, e.g. 09-click-issue-2711. Same as its folder in tasks/."},
                    "task_md": {"type": "string", "description": "The task for codr, in markdown."},
                    "check_sh": {"type": "string", "description": "The bash check. Exit code 0 means pass."},
                    "max_attempts": {"type": "integer", "description": "Attempts before it counts as failed. Default 3."},
                },
                "required": ["name", "task_md", "check_sh"],
            },
        },
    },
]
