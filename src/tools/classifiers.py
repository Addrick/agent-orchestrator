# src/tools/classifiers.py

"""
Argument-aware reversibility classifiers for tools whose `irreversible`
status depends on call arguments rather than tool identity.

Each classifier has signature `(args: dict) -> bool` and returns True
when the call is irreversible (i.e. requires HITL approval under
tainted-turn conditions per the tool security framework).

Currently empty: `add_note_irreversible_check` was retired when
`add_note_to_ticket` was clamped to internal-only articles (never
customer-visible, so never an unretractable outbound email). If a tool
regains argument-dependent reversibility, its classifier lives here.
"""
