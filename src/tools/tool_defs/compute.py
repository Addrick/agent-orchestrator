"""`run_python` (DP-522): no service_binding -- a core tool any persona may enable.

Handler: ``src/stagr/compute.py`` (registered by the stagr integration, whose
box it runs on). Not ``is_write``, so it runs without an approval prompt once
enabled. ``exfil_capable`` False because the runner gives the script no network
at all (``services/compute-sandbox/derpr-compute``): the code reaches stagr's
box but cannot carry anything past it, so it arms no exfiltration rule.
``produces_untrusted`` False: the output is a function of the model's own code,
and the unit cannot read /home, where stagr's fetched files live.
"""

from typing import Any, Dict, List

COMPUTE_TOOLS: List[Dict[str, Any]] = [
    {
        "type": "function",
        "is_write": False,
        "capabilities": {
            "produces_untrusted": False,
            "irreversible": False,
            "locality": "network",
            "sensitivity": "internal",
            "exfil_capable": False,
        },
        "function": {
            "name": "run_python",
            "description": (
                "Run a short Python 3 script and get its output back. Use it for arithmetic, "
                "statistics, dates, unit conversion or checking a formula instead of working it out "
                "in your head. print() what you want returned. It has the standard library (math, "
                "statistics, fractions, decimal, datetime, itertools, json, re, ...) but no network, "
                "no package installs and no files that survive the call. Returns exit_code and output "
                "(stdout and stderr together); exit_code 124 means it hit the time limit."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {"type": "string", "description": "The Python 3 script. At most 32 KB."},
                    "timeout_seconds": {
                        "type": "integer",
                        "description": "Stop it after this long. Default 30, at most 120.",
                    },
                },
                "required": ["code"],
            },
        },
    },
]
