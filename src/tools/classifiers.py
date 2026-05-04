# src/tools/classifiers.py

"""
Argument-aware reversibility classifiers for tools whose `irreversible`
status depends on call arguments rather than tool identity.

Each classifier has signature `(args: dict) -> bool` and returns True
when the call is irreversible (i.e. requires HITL approval under
tainted-turn conditions per the tool security framework).
"""

from typing import Any, Dict


def add_note_irreversible_check(args: Dict[str, Any]) -> bool:
    """
    `add_note_to_ticket` is irreversible when the note is customer-visible.

    A customer-visible note may trigger an outbound email and cannot be
    retracted from the recipient. Internal notes are operator-only and
    reversible by editing or deleting the article.
    """
    return not bool(args.get("internal", False))
