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
    """Tests that a long word without spaces is force-split to fit the limit."""
    text = "averylongwordthatcannotbesplitnicely"
    limit = 20
    result = split_string_by_limit(text, limit)
    assert result == ["averylongwordthatcan", "notbesplitnicely"]
    for chunk in result:
        assert len(chunk) <= limit

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
    # Limit must be larger than the citation token itself so the test
    # actually verifies the token isn't broken at internal spaces.
    limit = len(citation) + 10
    result = split_string_by_limit(text, limit)
    # The citation block must appear intact in exactly one chunk
    found = [chunk for chunk in result if citation.strip() in chunk]
    assert len(found) == 1, f"Citation block was split across chunks: {result}"


def test_split_preserves_markdown_links():
    """Tests that markdown links [text](<url>) are not split."""
    link = "[click here](<https://example.com/page>)"
    text = f"Please {link} for more information about this topic."
    # Limit must be larger than the link token itself so the test
    # actually verifies the token isn't broken at internal spaces.
    limit = len(link) + 10
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


def test_split_oversized_code_block():
    """Tests that a fenced code block exceeding the limit is split into
    multiple properly-fenced chunks, each within the limit."""
    lines = [f"line {i}" for i in range(50)]
    code_block = "```python\n" + "\n".join(lines) + "\n```"
    limit = 100
    result = split_string_by_limit(code_block, limit)
    assert len(result) > 1, "Oversized code block should be split"
    for chunk in result:
        assert len(chunk) <= limit, f"Chunk exceeds limit ({len(chunk)} > {limit}): {chunk!r}"
        assert chunk.startswith("```python\n"), f"Chunk missing opening fence: {chunk!r}"
        assert chunk.endswith("\n```"), f"Chunk missing closing fence: {chunk!r}"


def test_split_oversized_code_block_preserves_content():
    """Tests that force-splitting a code block preserves all content lines."""
    lines = [f"line {i}" for i in range(20)]
    code_block = "```\n" + "\n".join(lines) + "\n```"
    limit = 60
    result = split_string_by_limit(code_block, limit)
    # Reassemble: strip fences from each chunk and join
    recovered_lines = []
    for chunk in result:
        inner = chunk.removeprefix("```\n").removesuffix("\n```")
        recovered_lines.extend(inner.split("\n"))
    assert recovered_lines == lines


def test_split_mixed_text_and_oversized_code_block():
    """Tests text before/after an oversized code block lands in separate chunks."""
    big_code = "```\n" + ("x" * 300) + "\n```"
    text = f"Before code\n{big_code}\nAfter code"
    limit = 200
    result = split_string_by_limit(text, limit)
    for chunk in result:
        assert len(chunk) <= limit
    full = " ".join(result)
    assert "Before code" in full
    assert "After code" in full


def test_all_chunks_within_discord_limit():
    """Realistic scenario: LLM response with a large code block must produce
    chunks that all fit within Discord's 2000-char limit."""
    big_code = "```python\n" + "\n".join(f"print({i})" for i in range(300)) + "\n```"
    text = f"Here is the code:\n{big_code}\nHope that helps!"
    limit = 2000
    result = split_string_by_limit(text, limit)
    for i, chunk in enumerate(result):
        assert len(chunk) <= limit, f"Chunk {i} is {len(chunk)} chars, exceeds {limit}"


def test_force_split_prefers_spaces_over_url():
    """An oversized token with spaces should split at space boundaries,
    keeping URLs intact rather than chopping through them."""
    url = "https://example.com/very/long/path?query=value"
    # Build a single oversized token (inline code with spaces)
    inner = f"some text before {url} and some text after {url} end"
    token = f"`{inner}`"
    limit = 60
    result = split_string_by_limit(token, limit)
    for chunk in result:
        assert len(chunk) <= limit
        # The URL must not be split mid-string
        if "https://" in chunk:
            assert url in chunk, f"URL was chopped in chunk: {chunk!r}"
