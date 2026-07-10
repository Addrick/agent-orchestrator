# tests/test_generation_events.py
"""DP-280: format_internal_error must yield a diagnosable, bounded message."""

import re

from src.generation_events import format_internal_error


def test_includes_class_ref_and_detail():
    err_id, msg = format_internal_error(KeyError("ticket_id"))
    assert re.fullmatch(r"[0-9a-f]{8}", err_id)
    assert "[KeyError]" in msg
    assert f"ref {err_id}" in msg
    # KeyError stringifies with quotes; detail is carried through
    assert "ticket_id" in msg


def test_detail_is_single_line_and_truncated():
    exc = ValueError("line one\nline two   with    spaces\n" + "x" * 500)
    _, msg = format_internal_error(exc, max_detail=40)
    # detail sits between "): " and the trailing retry nudge
    detail = msg.split("): ", 1)[1].split(" — you may want", 1)[0]
    assert "\n" not in detail
    assert len(detail) <= 40
    assert detail.endswith("…")


def test_empty_message_omits_detail_suffix():
    err_id, msg = format_internal_error(RuntimeError())
    assert msg.startswith(f"Internal error [RuntimeError] (ref {err_id})")
    assert "): " not in msg  # no detail separator when the exception is empty


def test_message_nudges_user_to_retry():
    _, msg = format_internal_error(ValueError("boom"))
    assert "rephrase and try again" in msg


def test_registered_secret_is_scrubbed_from_message():
    """DP-225: an API key embedded in the exception must not reach the user."""
    from src.security.scrubber import get_scrubber

    scrubber = get_scrubber()
    secret = "sk-supersecretapikey1234567890"
    scrubber.register(secret, "TEST_KEY")
    try:
        _, msg = format_internal_error(
            RuntimeError(f"401 auth failed for key {secret}"),
            scrub=scrubber.scrub,
            max_detail=500,
        )
    finally:
        scrubber.clear()
    assert secret not in msg
    assert "[REDACTED:TEST_KEY]" in msg


def test_ids_are_unique_per_call():
    id1, _ = format_internal_error(Exception("boom"))
    id2, _ = format_internal_error(Exception("boom"))
    assert id1 != id2
