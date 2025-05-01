import requests
import time

def resolve_redirect_url(redirect_url: str, max_retries: int = 3, initial_delay: int = 1) -> str | None:
    """
    Follows a redirect URL with retries for 429 errors.

    Args:
        redirect_url: The initial URL to follow.
        max_retries: Maximum number of retries for 429 errors.
        initial_delay: Initial delay in seconds before the first retry.

    Returns:
        The final URL after all redirects, or None if errors persist or a non-retryable error occurs.
    """
    retries = 0
    # Use a common browser User-Agent
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
    }
    while retries <= max_retries:
        try:
            # Set a timeout to prevent hanging indefinitely
            # allow_redirects is True by default, but being explicit is fine
            # Using a Session can be more efficient if making multiple requests
            # to the same domain, but for a single resolve, get is fine.
            response = requests.get(redirect_url, allow_redirects=True, timeout=10, headers=headers)
            response.raise_for_status()  # Raise an exception for bad status codes (4xx or 5xx)

            # If successful, return the final URL
            return response.url

        except requests.exceptions.HTTPError as e:
            if e.response is not None and e.response.status_code == 429:
                retries += 1
                if retries <= max_retries:
                    # Exponential backoff: delay doubles each retry
                    delay = initial_delay * (2**(retries - 1))
                    print(f"Received 429 status for {redirect_url}. Retrying {retries}/{max_retries} in {delay:.2f} seconds...")
                    # Check for Retry-After header if available
                    retry_after = e.response.headers.get('Retry-After')
                    if retry_after:
                        try:
                            # Server specified how long to wait (e.g., '60' seconds or a date)
                            # Simple case: integer seconds
                            server_delay = int(retry_after)
                            print(f"Server requested waiting {server_delay} seconds.")
                            time.sleep(max(delay, server_delay)) # Wait at least server_delay, but not less than our calculated delay
                        except ValueError:
                            # Handle date format Retry-After if necessary, or just use our delay
                            print(f"Could not parse Retry-After header '{retry_after}'. Using calculated delay.")
                            time.sleep(delay)
                    else:
                         time.sleep(delay)
                    continue # Go back to the start of the while loop for the retry
                else:
                    # Max retries reached
                    print(f"Failed to resolve redirect {redirect_url} after {max_retries} retries due to 429 error.")
                    return redirect_url

            else:
                # Handle other HTTP errors (4xx or 5xx other than 429)
                print(f"HTTP error resolving redirect {redirect_url}: {e}")
                return redirect_url

        except requests.exceptions.RequestException as e:
            # Handle other requests-related errors (connection, timeout, etc.)
            print(f"Request error resolving redirect {redirect_url}: {e}")
            return redirect_url

        except Exception as e:
            # Catch any other unexpected errors
            print(f"An unexpected error occurred resolving redirect {redirect_url}: {e}")
            return redirect_url

    # If the loop finishes without returning (shouldn't happen with the `continue`),
    # it means retries were exhausted.
    return None


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
