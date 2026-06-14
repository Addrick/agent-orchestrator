"""Unit tests for the egress secret scrubber."""

import pytest

from src.security.scrubber import (
    MAX_PATTERN_SCAN_LEN,
    MIN_SECRET_LEN,
    SecretScrubber,
    get_scrubber,
    reset_scrubber,
)


@pytest.fixture(autouse=True)
def fresh_scrubber():
    """Reset the process-global scrubber so singleton state never bleeds."""
    reset_scrubber()
    yield
    reset_scrubber()


def test_registered_value_redacted_in_plain_str():
    scrubber = SecretScrubber()
    scrubber.register("supersecretvalue", "OPENAI_API_KEY")
    out = scrubber.scrub("key is supersecretvalue here")
    assert out == "key is [REDACTED:OPENAI_API_KEY] here"


def test_redacted_inside_nested_dict_and_list():
    scrubber = SecretScrubber()
    scrubber.register("supersecretvalue", "ZAMMAD_API_KEY")
    payload = {
        "outer": ["plain", {"inner": "use supersecretvalue now"}],
        "n": 42,
    }
    out = scrubber.scrub(payload)
    assert out == {
        "outer": ["plain", {"inner": "use [REDACTED:ZAMMAD_API_KEY] now"}],
        "n": 42,
    }


def test_redacted_inside_tuple():
    scrubber = SecretScrubber()
    scrubber.register("supersecretvalue", "REF")
    out = scrubber.scrub(("a", "supersecretvalue"))
    assert out == ("a", "[REDACTED:REF]")


def test_longest_first_substring_secret_fully_redacted():
    scrubber = SecretScrubber()
    # "abcdefgh" is a substring of the longer secret. Both must fully redact.
    scrubber.register("abcdefgh", "SHORT_REF")
    scrubber.register("abcdefghIJKLMNOP", "LONG_REF")
    out = scrubber.scrub("token abcdefghIJKLMNOP end")
    assert out == "token [REDACTED:LONG_REF] end"
    # The shorter secret alone still redacts when it appears on its own.
    out2 = scrubber.scrub("token abcdefgh end")
    assert out2 == "token [REDACTED:SHORT_REF] end"


def test_short_value_not_registered():
    scrubber = SecretScrubber()
    short = "a" * (MIN_SECRET_LEN - 1)
    scrubber.register(short, "REF")
    assert scrubber.active_secret_count() == 0
    assert scrubber.scrub(f"value {short} stays") == f"value {short} stays"


def test_empty_value_not_registered():
    scrubber = SecretScrubber()
    scrubber.register("", "REF")
    assert scrubber.active_secret_count() == 0


def test_pattern_fallback_catches_unregistered_sk_key():
    scrubber = SecretScrubber()
    leaked = "sk-abcdefghijklmnopqrstuvwxyz0123"
    out = scrubber.scrub(f"oops {leaked} leaked")
    assert out == "oops [REDACTED:pattern] leaked"


def test_pattern_fallback_catches_zammad_token_header():
    scrubber = SecretScrubber()
    out = scrubber.scrub("Authorization: Token token=abc123DEF456")
    assert out == "Authorization: [REDACTED:pattern]"


def test_pattern_fallback_catches_bearer_token():
    scrubber = SecretScrubber()
    out = scrubber.scrub("Bearer abcdefghijklmnopqrstuvwxyz")
    assert out == "[REDACTED:pattern]"


def test_pattern_fallback_skipped_on_long_string_but_registered_still_redacts():
    scrubber = SecretScrubber()
    scrubber.register("supersecretvalue", "REF")
    leaked = "sk-abcdefghijklmnopqrstuvwxyz0123"
    # A string longer than the scan cap: the unregistered key-shape must NOT be
    # touched (fallback skipped), but the registered exact secret still redacts.
    blob = "A" * (MAX_PATTERN_SCAN_LEN + 1)
    text = f"{leaked} {blob} supersecretvalue"
    out = scrubber.scrub(text)
    assert leaked in out  # pattern fallback skipped on the long string
    assert "supersecretvalue" not in out
    assert "[REDACTED:REF]" in out


def test_pattern_fallback_runs_at_threshold_boundary():
    scrubber = SecretScrubber()
    leaked = "sk-abcdefghijklmnopqrstuvwxyz0123"
    pad = "A" * (MAX_PATTERN_SCAN_LEN - len(leaked))
    text = leaked + pad  # exactly at the cap → still scanned
    assert len(text) == MAX_PATTERN_SCAN_LEN
    out = scrubber.scrub(text)
    assert leaked not in out
    assert "[REDACTED:pattern]" in out


def test_non_str_scalars_pass_through():
    scrubber = SecretScrubber()
    assert scrubber.scrub(42) == 42
    assert scrubber.scrub(3.14) == 3.14
    assert scrubber.scrub(True) is True
    assert scrubber.scrub(None) is None


def test_clear_forgets_secrets():
    scrubber = SecretScrubber()
    scrubber.register("supersecretvalue", "REF")
    assert scrubber.active_secret_count() == 1
    scrubber.clear()
    assert scrubber.active_secret_count() == 0


def test_register_dedupes_by_value_newest_ref_wins():
    scrubber = SecretScrubber()
    scrubber.register("supersecretvalue", "OLD_REF")
    scrubber.register("supersecretvalue", "NEW_REF")
    assert scrubber.active_secret_count() == 1
    assert scrubber.scrub("supersecretvalue") == "[REDACTED:NEW_REF]"


def test_reset_scrubber_isolates_state():
    first = get_scrubber()
    first.register("supersecretvalue", "REF")
    assert first.active_secret_count() == 1
    reset_scrubber()
    second = get_scrubber()
    assert second is not first
    assert second.active_secret_count() == 0


def test_get_scrubber_is_singleton_until_reset():
    assert get_scrubber() is get_scrubber()
