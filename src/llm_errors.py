# src/llm_errors.py
"""Provider-communication error type (DP-206b).

A leaf module so both the engine (`src.engine`) and its kobold-native local
provider (`src.stream_engine`) can share the exception without an import
cycle — the engine owns and constructs the StreamEngine, so the old
stream_engine → engine import direction had to invert. Importers may keep
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
    """

    def __init__(self, message: str, api_payload: Optional[Dict[str, Any]] = None,
                 rate_limited: bool = False):
        super().__init__(message)
        self.api_payload = api_payload
        self.rate_limited = rate_limited
