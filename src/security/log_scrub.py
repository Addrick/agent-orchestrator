# src/security/log_scrub.py
"""Secret-scrubbing logging formatter (DP-284).

`derpr.log` is a persistent, fixr-tailable artifact, and a logged exception
(`exc_info`) puts the exception string — which a provider SDK can populate with
an auth header or a key-in-URL — into the traceback tail. Registered secrets
(provider API keys, vault entries) must not land there permanently.

This formatter runs the *fully-formatted* record — message plus rendered
traceback — through the process-global scrubber before it is written. The
scrubber is resolved per-record (not at construction) so keys registered by the
vault at bootstrap, after logging is configured, are still covered. It never
raises, so a scrub fault can't lose a log line.
"""

from __future__ import annotations

import logging

from src.security.scrubber import get_scrubber


class ScrubbingFormatter(logging.Formatter):
    """A ``logging.Formatter`` that redacts registered secrets from the final
    formatted line, including any ``exc_info`` traceback."""

    def format(self, record: logging.LogRecord) -> str:
        formatted = super().format(record)
        return str(get_scrubber().scrub(formatted))
