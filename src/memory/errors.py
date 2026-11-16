"""Transcript-layer errors.

Kept out of `memory_manager` so there is one class object: under pytest's
`pythonpath = src .` that module also loads as `memory.memory_manager`, and an
exception defined there would not match an `except` naming the other copy.
"""
from typing import Optional


class RetryTargetMismatch(Exception):
    """A retry's write target is not the row the caller's screen ends on
    (DP-523). Nothing was archived or written."""

    def __init__(self, expected: int, actual: Optional[int]) -> None:
        super().__init__(f"retry target mismatch: expected {expected}, latest {actual}")
        self.expected = expected
        self.actual = actual
