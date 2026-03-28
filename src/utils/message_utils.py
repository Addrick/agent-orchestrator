# src/utils/message_utils.py

import requests
import time
import logging
import re
from typing import List, Optional

logger = logging.getLogger(__name__)


def cleanse_message_for_history(text: str) -> str:
    """Removes metadata like [ [1](<url>)] from text for cleaner LLM history."""
    # This regex removes a space, then the citation block, e.g., " [[1](<url>)]"
    # It also handles multiple citations like [[1](<url1>), [2](<url2>)]
    text = re.sub(r"\s\[\s?\[\d+\]\(<.+?>\)(,\s?\[\d+\]\(<.+?>\))*\s?\]", "", text)
    # This regex removes the "Sources:\n..." and "Search Query: ..." sections
    text = re.sub(r"\n\nSources:\n.*", "", text, flags=re.DOTALL)
    text = re.sub(r"\n\nSearch Query:.*", "", text, flags=re.DOTALL)
    return text.strip()


def resolve_redirect_url(redirect_url: str, max_retries: int = 3, initial_delay: int = 5) -> str:
    """
    Follows a redirect URL using HEAD method to get the final URL,
    handles 429 retries, and returns the URL even on other final HTTP errors.

    Args:
        redirect_url: The initial URL to follow.
        max_retries: Maximum number of retries for 429 errors.
        initial_delay: Initial delay in seconds before the first retry.

    Returns:
        The final URL after all redirects, or the original URL if a non-HTTP error occurred
        before reaching a final URL, or if max 429 retries are exhausted.
    """
    retries: int = 0
    # Use a common browser User-Agent and adjust Accept header for HEAD
    headers: dict[str, str] = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Accept': '*/*',  # HEAD requests typically accept any content type header
        'Accept-Language': 'en-US,en;q=0.9',
        'Accept-Encoding': 'gzip, deflate, br'  # Servers might still compress headers
    }

    while retries <= max_retries:
        try:
            logger.debug(f"Attempting to resolve {redirect_url} using HEAD (Attempt {retries + 1}/{max_retries + 1})...")

            response: requests.Response = requests.head(redirect_url, allow_redirects=True, timeout=10, headers=headers)

            if response.status_code == 429:
                retries += 1
                if retries <= max_retries:
                    delay: float = float(initial_delay * (2 ** (retries - 1)))
                    logger.debug(f"Received 429 status for {redirect_url}. Retrying HEAD in {delay:.2f} seconds...")
                    retry_after_header: Optional[str] = response.headers.get('Retry-After')
                    if retry_after_header:
                        try:
                            server_delay: int = int(retry_after_header)
                            logger.debug(f"Server requested waiting {server_delay} seconds.")
                            time.sleep(max(delay, float(server_delay)))
                        except ValueError:
                            logger.debug(f"Could not parse Retry-After header '{retry_after_header}'. Using calculated delay.")
                            time.sleep(delay)
                    else:
                        time.sleep(delay)
                    continue
                else:
                    logger.debug(
                        f"Max 429 retries reached for {redirect_url}. Returning the last resolved URL from HEAD: {response.url}")
                    return response.url

            logger.debug(f"Resolved {redirect_url} to {response.url} with final status {response.status_code} using HEAD.")
            return response.url

        except requests.exceptions.RequestException as e:
            logger.debug(f"Request error resolving redirect {redirect_url} using HEAD: {e}")
            return redirect_url

        except Exception as e:
            logger.debug(f"An unexpected error occurred resolving redirect {redirect_url} using HEAD: {e}")
            return redirect_url

    logger.debug(f"Retry loop finished for {redirect_url} without returning.")
    return redirect_url


def break_and_recombine_string(input_string: str, substring_length: int, bumper_string: str) -> str:
    substrings: List[str] = [input_string[i:i + substring_length] for i in range(0, len(input_string), substring_length)]
    formatted_substrings: List[str] = [bumper_string + substring + bumper_string for substring in substrings]
    combined_string: str = ' '.join(formatted_substrings)
    return combined_string


def _force_split_token(token: str, char_limit: int) -> List[str]:
    """Split an oversized token into pieces that each fit within char_limit.

    For fenced code blocks, splits by lines and re-wraps each piece with
    the original fence markers.  For everything else, does a hard character split.
    """
    code_match = re.match(r'^```([^\n]*)\n([\s\S]*)```$', token)
    if code_match:
        lang = code_match.group(1)
        content = code_match.group(2)
        if content.endswith('\n'):
            content = content[:-1]
        fence_open = f"```{lang}\n"
        fence_close = "\n```"
        overhead = len(fence_open) + len(fence_close)
        inner_limit = max(char_limit - overhead, 1)

        lines = content.split('\n')
        pieces: List[str] = []
        current = ""
        for line in lines:
            candidate = (current + "\n" + line) if current else line
            if len(candidate) <= inner_limit:
                current = candidate
            else:
                if current:
                    pieces.append(fence_open + current + fence_close)
                if len(line) > inner_limit:
                    for i in range(0, len(line), inner_limit):
                        pieces.append(fence_open + line[i:i + inner_limit] + fence_close)
                    current = ""
                else:
                    current = line
        if current:
            pieces.append(fence_open + current + fence_close)
        return pieces if pieces else [token]

    # Non-code-block: hard character split
    return [token[i:i + char_limit] for i in range(0, len(token), char_limit)]


def split_string_by_limit(input_string: str, char_limit: int) -> List[str]:
    """
    Splits a string into chunks under a character limit.
    Respects markdown syntax: won't split inside inline code, code blocks,
    markdown links, or citation blocks.  Oversized tokens (e.g. large code
    blocks) are force-split so every returned chunk fits within the limit.
    """
    if not input_string:
        return [""]

    # Tokenize preserving markdown structure. Order matters: specific patterns first.
    token_pattern = re.compile(
        r'```[\s\S]*?```'                                              # fenced code blocks
        r'|`[^`\n]+`'                                                  # inline code
        r'|\[\s?\[\d+\]\(<[^>]+>\)(?:,\s?\[\d+\]\(<[^>]+>\))*\s?\]'  # citation blocks
        r'|\[[^\]]*\]\(<[^>]+>\)'                                      # markdown links
        r'|\n'                                                         # newline
        r'| '                                                          # space
        r'|[^ \n]+'                                                    # any non-whitespace text
    )
    tokens: List[str] = token_pattern.findall(input_string)

    chunks: List[str] = []
    current_chunk: str = ""

    def _assign_token(token: str) -> str:
        """Handle assigning a new token as current_chunk, force-splitting if oversized."""
        if len(token) > char_limit:
            pieces = _force_split_token(token, char_limit)
            chunks.extend(pieces[:-1])
            return pieces[-1]
        return token

    for token in tokens:
        if not current_chunk:
            if token in (" ", "\n"):
                continue
            current_chunk = _assign_token(token)
        elif len(current_chunk) + len(token) <= char_limit:
            current_chunk += token
        else:
            stripped = current_chunk.rstrip()
            if stripped:
                chunks.append(stripped)
            if token in (" ", "\n"):
                current_chunk = ""
            else:
                current_chunk = _assign_token(token)

    if current_chunk:
        stripped = current_chunk.rstrip()
        if stripped:
            chunks.append(stripped)

    return chunks if chunks else [""]
