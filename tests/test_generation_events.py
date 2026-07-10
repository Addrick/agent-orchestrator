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
    detail = msg.split("): ", 1)[1]
    assert "\n" not in detail
    assert len(detail) <= 40
    assert detail.endswith("…")


def test_empty_message_omits_detail_suffix():
    err_id, msg = format_internal_error(RuntimeError())
    assert msg == f"Internal error [RuntimeError] (ref {err_id})"


def test_ids_are_unique_per_call():
    id1, _ = format_internal_error(Exception("boom"))
    id2, _ = format_internal_error(Exception("boom"))
    assert id1 != id2
