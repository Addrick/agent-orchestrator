import requests
import time
import logging

logger = logging.getLogger(__name__)


def resolve_redirect_url(redirect_url: str, max_retries: int = 3, initial_delay: int = 5) -> str | None:
    """
    Follows a redirect URL using HEAD method to get the final URL,
    handles 429 retries, and returns the URL even on other final HTTP errors.

    Args:
        redirect_url: The initial URL to follow.
        max_retries: Maximum number of retries for 429 errors.
        initial_delay: Initial delay in seconds before the first retry.

    Returns:
        The final URL after all redirects, or None if a non-HTTP error occurred
        before reaching a final URL, or if max 429 retries are exhausted.
    """
    retries = 0
    # Use a common browser User-Agent and adjust Accept header for HEAD
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Accept': '*/*',  # HEAD requests typically accept any content type header
        'Accept-Language': 'en-US,en;q=0.9',
        'Accept-Encoding': 'gzip, deflate, br'  # Servers might still compress headers
    }

    # Using a Session can be slightly more efficient if you make many requests
    # with requests.Session() as session:
    #     session.headers.update(headers)
    #     ... use session.head(...) ...

    while retries <= max_retries:
        try:
            logger.debug(f"Attempting to resolve {redirect_url} using HEAD (Attempt {retries + 1}/{max_retries + 1})...")

            # Use requests.head() which follows redirects and doesn't download body
            # allow_redirects=True is default for HEAD too, but explicit is clear
            response = requests.head(redirect_url, allow_redirects=True, timeout=10, headers=headers)

            # The final URL is available in response.url *after* redirects are followed,
            # even if the final HEAD request results in a 4xx or 5xx status.

            # Check the status code of the final response
            if response.status_code == 429:
                retries += 1
                if retries <= max_retries:
                    # Retry logic for 429
                    delay = initial_delay * (2 ** (retries - 1))
                    logger.debug(f"Received 429 status for {redirect_url}. Retrying HEAD in {delay:.2f} seconds...")
                    retry_after = response.headers.get('Retry-After')
                    if retry_after:
                        try:
                            # Server specified how long to wait
                            server_delay = int(retry_after)
                            logger.debug(f"Server requested waiting {server_delay} seconds.")
                            time.sleep(max(delay,
                                           server_delay))  # Wait at least server_delay, but not less than our calculated delay
                        except ValueError:
                            # Handle date format Retry-After if necessary, or just use our delay
                            logger.debug(f"Could not parse Retry-After header '{retry_after}'. Using calculated delay.")
                            time.sleep(delay)
                    else:
                        time.sleep(delay)
                    continue  # Go back to the start of the while loop for the retry
                else:
                    # Max 429 retries reached. We did get a response and its URL on the last attempt.
                    logger.debug(
                        f"Max 429 retries reached for {redirect_url}. Returning the last resolved URL from HEAD: {response.url}")
                    return response.url  # Return the URL even if the last attempt was 429

            # If we are here, the status code is NOT 429 (or it was the last 429 attempt)
            # For any other status code (2xx, 3xx, 403, 404, 500, etc.), we successfully
            # completed the HEAD request to the final destination.
            # The user wants the final URL regardless of the final status code.
            logger.debug(f"Resolved {redirect_url} to {response.url} with final status {response.status_code} using HEAD.")
            return response.url  # Return the final URL


        except requests.exceptions.RequestException as e:
            # This block catches errors that occur *before* getting a response back from
            # the final destination or during the redirect process, like connection errors,
            # timeouts before any response, or malformed URLs that requests can't even start with.
            # In these cases, we couldn't successfully reach the final URL.
            logger.debug(f"Request error resolving redirect {redirect_url} using HEAD: {e}")
            # If you wanted to retry *any* RequestException, you'd adjust the loop and conditions here.
            # But based on your problem description (403 after getting the URL),
            # retrying only 429 and returning None on other RequestExceptions seems appropriate.
            return redirect_url

        except Exception as e:
            # Catch any other unexpected errors
            logger.debug(f"An unexpected error occurred resolving redirect {redirect_url} using HEAD: {e}")
            return redirect_url

    # This part should only be reached if max_retries for 429 are exhausted and the
    # last attempt was 429, in which case we already returned the URL inside the loop.
    # Defensive coding:
    logger.debug(f"Retry loop finished for {redirect_url} without returning.")
    return redirect_url


def break_and_recombine_string(input_string, substring_length, bumper_string):
    substrings = [input_string[i:i + substring_length] for i in range(0, len(input_string), substring_length)]
    formatted_substrings = [bumper_string + substring + bumper_string for substring in substrings]
    combined_string = ' '.join(formatted_substrings)
    return combined_string


def split_string_by_limit(input_string, char_limit):
    """Splits a string between words for easier to read long messages"""  # TODO: maybe split after a period to only send full sentences?
    words = input_string.split(" ")
    current_line = ""
    result = []

    for word in words:
        # Check if adding the next word would exceed the limit
        if len(current_line) + len(word) + 1 > char_limit - 1:
            result.append(current_line.strip())
            current_line = word
        else:
            current_line += " " + word

    # Add the last line if there's any content left
    if current_line:
        result.append(current_line.strip())

    return result
