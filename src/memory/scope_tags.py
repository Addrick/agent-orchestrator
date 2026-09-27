"""Memory scope tags — the tag predicates retain and recall scope by.

Lives in `src.memory` (not `src.request_builder`, its original home) so every
layer that retains or recalls — request_builder's auto-recall, turn_persistence's
retain, and the `recall_memory` tool in `src.tools` — shares one definition
without importing upward (DP-407).
"""
from typing import List, Optional, Tuple

from src.persona import MemoryMode


def build_scope_tags(
    *,
    channel: Optional[str],
    server_id: Optional[str],
    user_identifier: Optional[str],
    interface: Optional[str] = None,
) -> List[str]:
    """Plan §1.4 scope tags. Used for retain + recall tag predicates."""
    tags: List[str] = []
    if channel:
        tags.append(f"channel:{channel}")
    if user_identifier:
        tags.append(f"user:{user_identifier}")
    if server_id:
        tags.append(f"server:{server_id}")
    if interface:
        tags.append(f"interface:{interface}")
    return tags


def recall_scope_tags(
    mode: MemoryMode,
    *,
    channel: Optional[str],
    server_id: Optional[str],
    user_identifier: Optional[str],
) -> Tuple[List[str], str]:
    """Build the recall tag predicate + mode label for a persona's memory mode.

    DP-253 — recall must scope the same way history does (`fetch_raw_history`),
    not by blindly tagging the caller's channel. The returned tags are the
    *recall filter* (Hindsight scopes purely on these); the label is forwarded
    to the backend's `recall(memory_mode=...)` so the sqlite index applies the
    matching WHERE branch instead of its legacy channel-presence guess.

    - GLOBAL          → no scope tags (recall across all channels/servers)
    - SERVER_WIDE     → server tag only (server_id may be None for the portal)
    - PERSONAL        → user tag only
    - TICKET_ISOLATED → no tags ("ticket"; sqlite recall returns nothing here)
    - CHANNEL_ISOLATED (default) → channel (+ server + user), the prior behavior
    """
    if mode == MemoryMode.GLOBAL:
        return [], "global"
    if mode == MemoryMode.TICKET_ISOLATED:
        return [], "ticket"
    if mode == MemoryMode.SERVER_WIDE:
        tags: List[str] = []
        if server_id:
            tags.append(f"server:{server_id}")
        return tags, "server"
    if mode == MemoryMode.PERSONAL:
        tags = []
        if user_identifier:
            tags.append(f"user:{user_identifier}")
        return tags, "personal"
    # CHANNEL_ISOLATED / default — preserve the exact prior scoping.
    return build_scope_tags(
        channel=channel, server_id=server_id, user_identifier=user_identifier,
    ), "channel"
