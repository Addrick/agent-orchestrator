# src/llm_errors.py
"""Provider-communication error type (DP-206b).

A leaf module so the engine (`src.engine`) and anything below it can share
the exception without an import cycle (it was split out for the kobold-native
`src.stream_engine`, retired in DP-417). Importers may keep
using the historical `from src.engine import LLMCommunicationError` re-export.
"""

from typing import Any, Dict, Optional


class LLMCommunicationError(Exception):
    """Raised when the engine cannot communicate with an LLM provider, or the
    provider's response is unusable after all retries.

    Attributes:
        api_payload: the (log-safe) request payload associated with the failed
            call, when one was built — used by orchestration for dump_history.
        rate_limited: True for HTTP 429 / quota errors; routing uses this to
            trigger model fallback instead of burning the retry budget.
        status_code: the provider's HTTP status, when it answered with one.
    """

    def __init__(self, message: str, api_payload: Optional[Dict[str, Any]] = None,
                 rate_limited: bool = False, status_code: Optional[int] = None):
        super().__init__(message)
        self.api_payload = api_payload
        self.rate_limited = rate_limited
        self.status_code = status_code

    @property
    def permanent(self) -> bool:
        """The provider answered and refused the request itself (a 4xx other
        than 408, 409 and 429: bad request, context too long, auth): sending
        the same request again gets the same answer (DP-518). The AI SDK's
        `isRetryable` draws the same line. A failure with no status (no
        connection, a timeout, an empty reply) is not permanent."""
        code = self.status_code
        return code is not None and 400 <= code < 500 and code not in (408, 409, 429)
