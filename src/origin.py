# src/origin.py
# Leaf value object (no src.* imports) — like src/tool_policy.py, so any
# layer can carry an Origin without dependency cycles.
"""Typed message origin for control-plane authorization (DP-277).

Every message entering the dev-command chokepoint
(``BotLogic.preprocess_message``) carries an ``Origin`` describing where it
came from and whether that transport authenticated it as the operator.

``operator`` is set by the interface adapter from TRANSPORT-AUTHENTICATED
facts only, never from caller-supplied request fields:

- **Discord** — guild/channel/author ids come from the gateway (unforgeable);
  the adapter matches them against ``OPERATOR_ALLOWLIST``.
- **Portal HTTP** — a validated operator token (``DERPR_CONTROL_TOKEN``);
  body/query fields like ``server_id`` are caller-supplied and worthless as
  auth.
- **Ticket bodies, Gmail, unauthenticated portal calls** — ``operator=False``,
  structurally data-plane: control commands are refused no matter what the
  message text says, which is the mitigation for injected-NL escalation
  (there is no reliable NL sanitization; the gate is architectural).
"""

import logging
from dataclasses import dataclass
from typing import List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Origin:
    """Where a message came from, per the transport — not per its content."""
    transport: str  # "discord" | "portal" | "gmail" | "zammad" | "internal" | "test"
    server_id: Optional[str] = None
    channel_id: Optional[str] = None
    author_id: Optional[str] = None
    operator: bool = False


# Secure default for callers that have no authenticated origin facts.
ANONYMOUS = Origin(transport="unknown", operator=False)


def parse_operator_allowlist(raw: str) -> List[Tuple[str, str, str]]:
    """Parse ``OPERATOR_ALLOWLIST`` into (server_id, channel_id, author_id)
    tuples. Entry format: ``server_id[/channel_id[/author_id]]`` separated by
    commas; a missing or ``*`` component matches anything at that level, so a
    whole-server grant is just the bare server id.

    Malformed entries are dropped with a warning (fail closed: a typo narrows
    access, never widens it).
    """
    entries: List[Tuple[str, str, str]] = []
    for chunk in raw.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        parts = [p.strip() for p in chunk.split("/")]
        if len(parts) > 3 or not parts[0] or parts[0] == "*":
            # a wildcard server would grant every guild the bot is in
            logger.warning(f"Ignoring malformed OPERATOR_ALLOWLIST entry {chunk!r}.")
            continue
        server = parts[0]
        channel = parts[1] if len(parts) > 1 and parts[1] else "*"
        author = parts[2] if len(parts) > 2 and parts[2] else "*"
        entries.append((server, channel, author))
    return entries


def is_discord_operator(
    allowlist: List[Tuple[str, str, str]],
    server_id: Optional[str],
    channel_id: Optional[str],
    author_id: Optional[str],
) -> bool:
    """True if the gateway-asserted (server, channel, author) matches an
    allowlist entry. DMs (no server) never match — operator grants are
    per-guild by design."""
    if not server_id:
        return False
    for srv, chan, auth in allowlist:
        if srv != server_id:
            continue
        if chan != "*" and chan != channel_id:
            continue
        if auth != "*" and auth != author_id:
            continue
        return True
    return False
