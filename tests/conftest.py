import os
from pathlib import Path

import pytest
from dotenv import load_dotenv

# Load test-specific environment variables, overriding any production .env values.
# This ensures live tests always hit the test Zammad instance, never production.
_project_root = Path(__file__).resolve().parent.parent
_env_test_path = _project_root / ".env.test"
if _env_test_path.exists():
    load_dotenv(dotenv_path=_env_test_path, override=True)


def pytest_collection_modifyitems(config, items):
    """Auto-skip live tests when required env vars are missing."""
    has_zammad = bool(os.environ.get("ZAMMAD_URL") and os.environ.get("ZAMMAD_API_KEY"))
    has_llm_keys = bool(
        os.environ.get("OPENAI_API_KEY")
        or os.environ.get("GOOGLE_API_KEY")
        or os.environ.get("ANTHROPIC_API_KEY")
    )

    skip_zammad = pytest.mark.skip(reason="ZAMMAD_URL/ZAMMAD_API_KEY not set")
    skip_llm = pytest.mark.skip(reason="No LLM API keys set (OPENAI_API_KEY, GOOGLE_API_KEY, ANTHROPIC_API_KEY)")

    for item in items:
        if "zammad_live" in item.keywords and not has_zammad:
            item.add_marker(skip_zammad)
        if "llm_live" in item.keywords and not has_llm_keys:
            item.add_marker(skip_llm)
