import os

# Set testing environment flag immediately to ensure path redirection in global_config.py
os.environ["APP_ENV"] = "testing"


import pytest
from dotenv import find_dotenv, load_dotenv

# Load test-specific environment variables, overriding any production .env values.
# This ensures live tests always hit the test Zammad instance, never production.
# Uses find_dotenv to walk up parent directories — necessary because git worktrees
# resolve to a different directory tree than the main repo, so a path relative to
# __file__ would miss .env.test when running from a worktree.
_env_test_path = find_dotenv(filename=".env.test", usecwd=True)
if _env_test_path:
    load_dotenv(dotenv_path=_env_test_path, override=True)


def pytest_collection_modifyitems(config, items):
    """Auto-skip live tests when required env vars are missing."""
    has_zammad = bool(os.environ.get("ZAMMAD_URL") and os.environ.get("ZAMMAD_API_KEY"))
    has_llm_keys = bool(
        os.environ.get("OPENAI_API_KEY")
        or os.environ.get("GOOGLE_API_KEY")
        or os.environ.get("ANTHROPIC_API_KEY")
    )
    has_discord = bool(os.environ.get("DISCORD_API_KEY"))

    skip_zammad = pytest.mark.skip(reason="ZAMMAD_URL/ZAMMAD_API_KEY not set")
    skip_llm = pytest.mark.skip(reason="No LLM API keys set (OPENAI_API_KEY, GOOGLE_API_KEY, ANTHROPIC_API_KEY)")
    skip_discord = pytest.mark.skip(reason="DISCORD_API_KEY not set")

    for item in items:
        if "zammad_live" in item.keywords and not has_zammad:
            item.add_marker(skip_zammad)
        if "llm_live" in item.keywords and not has_llm_keys:
            item.add_marker(skip_llm)
        if "discord_live" in item.keywords and not has_discord:
            item.add_marker(skip_discord)
