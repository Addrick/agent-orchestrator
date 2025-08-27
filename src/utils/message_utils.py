# src/utils/message_utils.py

import requests
import time
import logging
import re
from typing import List, Optional

logger = logging.getLogger(__name__)


def cleanse_message_for_history(text: str) -> str:
    """Removes metadata like [ [1](<url>)] from text for cleaner LLM history."""
    if not isinstance(text, str):
        return ""
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


def split_string_by_limit(input_string: str, char_limit: int) -> List[str]:
    """
    Splits a string between words to create chunks under a character limit.
    This version correctly handles initialization and line-breaking logic.
    """
    if not input_string:
        return [""]

    words: List[str] = input_string.split(' ')
    lines: List[str] = []
    current_line: str = ""

    for word in words:
        if not current_line:
            current_line = word
            continue

        if len(current_line) + len(word) + 1 <= char_limit:
            current_line += " " + word
        else:
            lines.append(current_line)
            current_line = word

    lines.append(current_line)

    return lines
