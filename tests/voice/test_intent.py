# tests/voice/test_intent.py
import pytest

from src.voice.intent import KeywordTimerRouter, parse_duration


@pytest.mark.parametrize("text,expected", [
    ("10 minutes", 600),
    ("set a timer for 10 minutes", 600),
    ("30 seconds", 30),
    ("1 hour", 3600),
    ("ten minutes", 600),
    ("a minute", 60),
    ("five mins", 300),
    ("2 hrs", 7200),
    ("1.5 minutes", 90),
    ("for fifteen seconds", 15),
])
def test_parse_duration_ok(text, expected):
    assert parse_duration(text) == expected


@pytest.mark.parametrize("text", ["", "hello there", "set a timer", "no number here", "0 minutes"])
def test_parse_duration_none(text):
    assert parse_duration(text) is None


async def test_router_matches_timer_command():
    router = KeywordTimerRouter()
    intent = await router.route("set a timer for 10 minutes")
    assert intent is not None
    assert intent.seconds == 600


async def test_router_requires_timer_word_by_default():
    router = KeywordTimerRouter()
    assert await router.route("remind me in 10 minutes") is None  # no 'timer'/'alarm'


async def test_router_timer_word_off():
    router = KeywordTimerRouter(require_timer_word=False)
    intent = await router.route("in 10 minutes")
    assert intent is not None and intent.seconds == 600


async def test_router_wake_word_gate():
    router = KeywordTimerRouter(wake_word="derpr")
    assert await router.route("set a timer for 5 minutes") is None  # no wake word
    intent = await router.route("derpr set a timer for 5 minutes")
    assert intent is not None and intent.seconds == 300


async def test_router_extracts_label():
    router = KeywordTimerRouter()
    intent = await router.route("set a timer for 10 minutes for the pasta")
    assert intent is not None
    assert intent.seconds == 600
    assert intent.label == "the pasta"


async def test_router_no_spurious_label():
    router = KeywordTimerRouter()
    intent = await router.route("set a timer for 10 minutes")
    assert intent is not None
    assert intent.label is None
