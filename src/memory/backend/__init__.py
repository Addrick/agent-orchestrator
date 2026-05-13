"""Memory backend layer.

Provides a swappable ABC for the semantic + episodic memory tier. Sprint 1
(DP-108) ships only the SQLite implementation as a thin wrapper around the
existing MemoryManager logic. Sprint 2 lands HindsightBackend.

The transcript layer (User_Interactions, suppression, version chevrons,
audit) is the system of record and stays on MemoryManager — not part of this
ABC.
"""
from src.memory.backend.base import (
    Experience,
    MemoryBackend,
    MemoryHit,
    MentalModel,
    ReflectResult,
)
from src.memory.backend.sqlite import SqliteSemanticBackend
from src.memory.backend.hindsight import HindsightBackend

__all__ = [
    "Experience",
    "MemoryBackend",
    "MemoryHit",
    "MentalModel",
    "ReflectResult",
    "SqliteSemanticBackend",
    "HindsightBackend",
]
