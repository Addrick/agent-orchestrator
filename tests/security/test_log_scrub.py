# tests/security/test_log_scrub.py
"""DP-284: ScrubbingFormatter must keep registered secrets out of log output,
including exc_info tracebacks that land in the permanent derpr.log."""

import logging

from src.security.log_scrub import ScrubbingFormatter
from src.security.scrubber import get_scrubber

SECRET = "sk-supersecretapikey1234567890"


def _record(msg, exc_info=None):
    return logging.LogRecord(
        name="test", level=logging.ERROR, pathname=__file__, lineno=1,
        msg=msg, args=(), exc_info=exc_info,
    )


def test_scrubs_registered_secret_from_message():
    scrubber = get_scrubber()
    scrubber.register(SECRET, "TEST_KEY")
    try:
        out = ScrubbingFormatter("%(message)s").format(
            _record(f"auth failed with {SECRET}")
        )
    finally:
        scrubber.clear()
    assert SECRET not in out
    assert "[REDACTED:TEST_KEY]" in out


def test_scrubs_registered_secret_from_traceback():
    scrubber = get_scrubber()
    scrubber.register(SECRET, "TEST_KEY")
    try:
        try:
            raise RuntimeError(f"401 for key {SECRET}")
        except RuntimeError:
            import sys
            out = ScrubbingFormatter("%(message)s").format(
                _record("provider call failed", exc_info=sys.exc_info())
            )
    finally:
        scrubber.clear()
    # the traceback tail renders "RuntimeError: 401 for key <secret>"
    assert SECRET not in out
    assert "[REDACTED:TEST_KEY]" in out


def test_no_secret_registered_leaves_line_intact():
    get_scrubber().clear()
    out = ScrubbingFormatter("%(message)s").format(_record("ordinary log line"))
    assert out == "ordinary log line"
