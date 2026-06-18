# src/utils/claude_cli_env.py
"""Environment construction for headless ``claude`` CLI subprocesses.

Both the cc-* engine route (`engine._run_cc_cli`) and the fixr dispatcher
(`self_edit.dispatcher`) spawn `claude -p`. In non-interactive (`-p`) mode the
CLI's auth precedence puts ``ANTHROPIC_API_KEY`` / ``ANTHROPIC_AUTH_TOKEN``
ABOVE the subscription OAuth token (``CLAUDE_CODE_OAUTH_TOKEN`` / stored
`/login` creds) — the docs are explicit: "In non-interactive mode (-p), the key
is always used when present." So if the host has ``ANTHROPIC_API_KEY`` set (it
does — the in-process Anthropic provider needs it), every cc-* / fixr agent
silently bills the metered API instead of the Claude subscription.

The fix is to strip those keys from the *child* env so the CLI falls through to
the subscription. We only touch the subprocess env copy; the parent process (and
the in-process Anthropic SDK provider) keep their ``ANTHROPIC_API_KEY``.
"""

from __future__ import annotations

import os
from typing import Dict, Optional

from config import global_config

#: API-key env vars that outrank the subscription OAuth token in `-p` mode.
_API_AUTH_VARS = ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN")


def build_claude_cli_env(base: Optional[Dict[str, str]] = None) -> Dict[str, str]:
    """Return the env dict for a headless ``claude`` subprocess.

    When ``CC_USE_SUBSCRIPTION`` is on (default), drop the API-key vars so the
    CLI authenticates with the Claude subscription (``CLAUDE_CODE_OAUTH_TOKEN``
    or stored `/login` creds) rather than the metered API. Set the flag off to
    keep API-key billing (escape hatch)."""
    env = dict(os.environ if base is None else base)
    if global_config.CC_USE_SUBSCRIPTION:
        for var in _API_AUTH_VARS:
            env.pop(var, None)
    return env
