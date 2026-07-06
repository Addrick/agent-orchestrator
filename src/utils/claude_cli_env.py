# src/utils/claude_cli_env.py
"""Environment construction for spawned agent CLI subprocesses (`claude`, `agy`).

Two concerns are handled here, both about what the *child* process inherits:

1. **Billing (DP-232).** In non-interactive (`-p`) mode the `claude` CLI's auth
   precedence puts ``ANTHROPIC_API_KEY`` / ``ANTHROPIC_AUTH_TOKEN`` ABOVE the
   subscription OAuth token (``CLAUDE_CODE_OAUTH_TOKEN`` / stored `/login`
   creds) — the docs are explicit: "In non-interactive mode (-p), the key is
   always used when present." The host sets ``ANTHROPIC_API_KEY`` (the
   in-process Anthropic provider needs it), so without stripping it every cc-* /
   fixr agent silently bills the metered API instead of the subscription.

2. **Secret isolation (DP-277).** The cc-*/agy/fixr routes spawn agent harnesses
   that run attacker-influenceable content (a persona answering a phishing
   email, a ticket body, an injected instruction). Claude Code's OS sandbox
   confines the child's filesystem and network — but NOT its environment: a
   sandboxed `Bash` can still read ``os.environ``, and the agent's own reply
   text is an exfiltration channel that needs no network at all. So the child
   must not inherit derpr's machine secrets (the portal control token, the other
   providers' API keys, the Discord/Zammad creds). The child authenticates with
   its OWN credential (the Claude subscription OAuth token, or agy's harness
   auth) and needs none of ours. We strip ours from the child env copy only; the
   parent process and the in-process SDK providers keep theirs.

   This is an ALLOWLIST-of-what-to-keep at heart, but implemented as an explicit
   denylist of the secrets derpr manages, because a spawned CLI legitimately
   reads a long, host-specific tail of benign vars (PATH, HOME, LANG, TERM,
   TMPDIR, XDG_*, NODE_*, TLS CA bundles, the CLI's own OAuth) that an allowlist
   would have to enumerate and keep in sync per-platform. The denylist is the
   set that must never cross the boundary; keep it current as new host secrets
   are added (there is a test asserting the known refs are covered).
"""

from __future__ import annotations

import os
from typing import Dict, Iterable, Optional

from config import global_config

#: API-key env vars that outrank the subscription OAuth token in `-p` mode.
_API_AUTH_VARS = ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN")

#: derpr-host secrets a spawned agent must never inherit (DP-277). It runs
#: attacker-influenceable content and can exfiltrate an inherited value through
#: its own response text even inside the OS sandbox. The child authenticates
#: with its own credential, so none of these are needed.
#: ``GH_TOKEN`` is here too but is re-added explicitly by the fixr dispatcher,
#: the one route that legitimately pushes a branch (see ``keep``).
_DERPR_SECRET_VARS = (
    "DERPR_CONTROL_TOKEN",       # the portal control-plane gate itself (DP-277)
    "OPENAI_API_KEY",
    "GOOGLE_GENERATIVEAI_API_KEY",
    "ZAMMAD_API_KEY",
    "DISCORD_API_KEY",
    "GH_TOKEN",                  # fixr re-injects; every other route drops it
    "PVE_SSH_KEY",               # path to the proxmox host private key
    "GMAIL_TOKEN_FILE",          # path to the Gmail OAuth token file
    "GMAIL_CREDENTIALS_FILE",
)


def _scrub_derpr_secrets(env: Dict[str, str], keep: Iterable[str] = ()) -> None:
    """Remove derpr-managed secrets from ``env`` in place, except any in
    ``keep`` (the caller asserts the child genuinely needs them)."""
    keep_set = set(keep)
    for var in _DERPR_SECRET_VARS:
        if var not in keep_set:
            env.pop(var, None)


def build_claude_cli_env(
    base: Optional[Dict[str, str]] = None,
    *,
    keep_gh_token: bool = False,
) -> Dict[str, str]:
    """Return the env dict for a headless ``claude`` subprocess.

    When ``CC_USE_SUBSCRIPTION`` is on (default), drop the API-key vars so the
    CLI authenticates with the Claude subscription (``CLAUDE_CODE_OAUTH_TOKEN``
    or stored `/login` creds) rather than the metered API. Set the flag off to
    keep API-key billing (escape hatch).

    Always strips derpr's machine secrets (DP-277) so a sandboxed-but-untrusted
    agent cannot read or exfiltrate them. ``keep_gh_token=True`` retains
    ``GH_TOKEN`` for the fixr dispatcher, which must push its branch to open a
    PR; no other route sets it.
    """
    env = dict(os.environ if base is None else base)
    if global_config.CC_USE_SUBSCRIPTION:
        for var in _API_AUTH_VARS:
            env.pop(var, None)
    _scrub_derpr_secrets(env, keep=("GH_TOKEN",) if keep_gh_token else ())
    return env


def build_agy_cli_env(base: Optional[Dict[str, str]] = None) -> Dict[str, str]:
    """Return the env dict for a headless ``agy`` (Antigravity) subprocess.

    The agy route clamps tools off (text-only), but still runs untrusted
    content, so it gets the same DP-277 secret scrub. agy authenticates with its
    own harness OAuth, not derpr's provider keys; the Anthropic billing strip is
    cc-specific and deliberately not applied here.
    """
    env = dict(os.environ if base is None else base)
    _scrub_derpr_secrets(env)
    return env
