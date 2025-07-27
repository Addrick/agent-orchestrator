# /tests/utils/test_message_utils.py

from src.utils.message_utils import split_string_by_limit

def test_split_string_shorter_than_limit():
    """Tests that a string shorter than the limit is not split."""
    text = "This is a short message."
    limit = 100
    result = split_string_by_limit(text, limit)
    assert result == ["This is a short message."]

def test_split_string_longer_than_limit():
    """Tests that a long string is correctly split into multiple chunks."""
    text = "This is a very long message that should definitely be split into multiple parts because it exceeds the character limit."
    limit = 50
    result = split_string_by_limit(text, limit)
    # This expected output is now corrected to match the function's greedy logic.
    expected = [
        "This is a very long message that should definitely",
        "be split into multiple parts because it exceeds",
        "the character limit."
    ]
    assert result == expected
    for chunk in result:
        assert len(chunk) <= limit

def test_split_string_empty_input():
    """Tests that an empty string results in a list with one empty string."""
    text = ""
    limit = 100
    result = split_string_by_limit(text, limit)
    assert result == [""]

def test_split_string_with_no_spaces():
    """Tests splitting a long word without spaces."""
    text = "averylongwordthatcannotbesplitnicely"
    limit = 20
    result = split_string_by_limit(text, limit)
    assert result == ["averylongwordthatcannotbesplitnicely"]

def test_split_string_on_exact_limit():
    """Tests splitting when adding a word meets the limit exactly."""
    text = "one two three four five"
    limit = 18 # "one two three four" is 18 chars
    result = split_string_by_limit(text, limit)
    expected = ["one two three four", "five"]
    assert result == expected
