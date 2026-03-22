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


def test_split_preserves_inline_code():
    """Tests that inline code spans are not split at internal spaces."""
    text = "Select New, then `DWORD (32-bit) Value` and continue."
    limit = 30
    result = split_string_by_limit(text, limit)
    # The inline code `DWORD (32-bit) Value` must stay intact
    for chunk in result:
        if '`' in chunk:
            assert chunk.count('`') % 2 == 0, f"Unmatched backtick in chunk: {chunk}"


def test_split_preserves_citation_blocks():
    """Tests that citation blocks like [[1](<url>), [2](<url>)] are not split."""
    url = "https://example.com/very-long-path"
    citation = f" [[1](<{url}>), [2](<{url}>)]"
    text = f"Some cited text.{citation} More text after citation."
    limit = 40
    result = split_string_by_limit(text, limit)
    # The citation block must appear intact in exactly one chunk
    found = [chunk for chunk in result if citation.strip() in chunk]
    assert len(found) == 1, f"Citation block was split across chunks: {result}"


def test_split_preserves_markdown_links():
    """Tests that markdown links [text](<url>) are not split."""
    link = "[click here](<https://example.com/page>)"
    text = f"Please {link} for more information about this topic."
    limit = 30
    result = split_string_by_limit(text, limit)
    found = [chunk for chunk in result if link in chunk]
    assert len(found) == 1, f"Markdown link was split across chunks: {result}"


def test_split_preserves_fenced_code_blocks():
    """Tests that fenced code blocks are not split."""
    text = "Here is code:\n```python\nprint('hello world')\n```\nEnd."
    limit = 40
    result = split_string_by_limit(text, limit)
    # The code block must be intact in one chunk
    found = [chunk for chunk in result if "```python" in chunk and "```\n" in chunk]
    assert len(found) >= 1, f"Code block was split: {result}"


def test_split_at_newlines():
    """Tests that newlines can serve as split points."""
    text = "first line\nsecond line\nthird line"
    limit = 11  # "first line\n" = 11, forces split at newline boundary
    result = split_string_by_limit(text, limit)
    assert result[0] == "first line"
    assert len(result) >= 2
    for chunk in result:
        assert len(chunk) <= limit
