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


def split_allowlist_entries(raw: str) -> List[str]:
    """Split a user-typed allowlist string into entries on whitespace OR commas.

    One home for the separator grammar. It was copied verbatim into
    ``persona_fields._parse_string_list_arg`` and ``Persona._normalize_origin_
    allowlist`` — the two write paths that are documented as producing identical
    results — so a change to how entries are separated had to be made in three
    places or the CLI and the file would drift.
    """
    return [e for chunk in raw.split() for e in chunk.split(',') if e]


def parse_operator_allowlist_entry(entry: str) -> Optional[Tuple[str, str, str]]:
    """Parse ONE allowlist entry, ``server_id[/channel_id[/author_id]]``.
    Returns None if it is malformed.

    This is the entry grammar, and it is the only thing that knows it. Callers
    used to infer "the parser did not split this" from
    ``len(parse_operator_allowlist(text)) == 1 and ',' not in text``, which is a
    proxy for a rule that lives here: **a comma is the entry separator, so an
    entry containing one is two grants glued together** and honouring it would
    widen reachability by a typo. With the rule inlined at the call site, adding
    a separator here would silently turn those rejections into grants.

    A missing or ``*`` component matches anything at that level, so a
    whole-server grant is just the bare server id — but a ``*`` *server* is
    refused outright, since it would grant every guild the bot is in.
    """
    entry = entry.strip()
    if not entry or ',' in entry:
        return None
    parts = [p.strip() for p in entry.split("/")]
    if len(parts) > 3 or not parts[0] or parts[0] == "*":
        return None
    server = parts[0]
    channel = parts[1] if len(parts) > 1 and parts[1] else "*"
    author = parts[2] if len(parts) > 2 and parts[2] else "*"
    return (server, channel, author)


def parse_operator_allowlist(raw: str) -> List[Tuple[str, str, str]]:
    """Parse ``OPERATOR_ALLOWLIST`` into (server_id, channel_id, author_id)
    tuples. Entry format: ``server_id[/channel_id[/author_id]]`` separated by
    commas.

    Malformed entries are dropped with a warning (fail closed: a typo narrows
    access, never widens it).
    """
    entries: List[Tuple[str, str, str]] = []
    for chunk in raw.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        parsed = parse_operator_allowlist_entry(chunk)
        if parsed is None:
            logger.warning(f"Ignoring malformed OPERATOR_ALLOWLIST entry {chunk!r}.")
            continue
        entries.append(parsed)
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


def is_origin_allowed(
    allowlist: List[Tuple[str, str, str]],
    origin: Origin,
) -> bool:
    """DP-330: may this origin address a persona carrying ``allowlist``?

    An **empty allowlist is unrestricted** — every persona that never set the
    field is unchanged by construction.

    A non-empty allowlist is Discord-only: entries are guild ids, and no other
    transport has a gateway-asserted one, so portal / gmail / zammad /
    internal-agent turns and Discord DMs all fail closed. The transport check
    is belt-and-braces on top of that: ``is_discord_operator`` already returns
    False without a ``server_id``, and today no non-Discord adapter populates
    one — but the portal *does* receive a caller-supplied ``server_id`` in its
    request body, so an adapter that ever starts copying it into an Origin must
    not thereby widen a persona allowlist.
    """
    if not allowlist:
        return True
    if origin.transport != "discord":
        return False
    return is_discord_operator(
        allowlist, origin.server_id, origin.channel_id, origin.author_id,
    )
