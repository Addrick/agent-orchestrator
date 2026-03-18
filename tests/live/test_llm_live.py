# tests/live/test_llm_live.py
#
# Live LLM API tests. These make real API calls and cost money.
# Auto-skipped when no LLM API keys are set.

import pytest

pytestmark = pytest.mark.llm_live


@pytest.mark.asyncio
async def test_llm_api_connectivity():
    """Placeholder: verify that at least one LLM provider is reachable."""
    # TODO: implement with real TextEngine call using LLM_LIVE_MODEL and LLM_LIVE_MAX_TOKENS
    pytest.skip("LLM live tests not yet implemented")
