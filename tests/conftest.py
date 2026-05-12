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


@pytest.fixture(autouse=True)
def _force_sqlite_backend_in_tests(monkeypatch):
    """DP-114 followup: keep MemoryManager()-constructing tests on the
    SQLite backend even though production defaults to Hindsight.

    The 67 legacy tests in tests/memory, tests/test_memory_retrieval, etc.
    exercise SqliteSemanticBackend's contract via MemoryManager's selector.
    Flipping the production default to "hindsight" (DP-114) routes those
    constructions to HindsightBackend, whose legacy methods raise
    NotImplementedError by design ("migrate the caller before flipping").

    Tests that intend to exercise HindsightBackend instantiate it directly,
    so this autouse patch only affects MemoryManager-via-selector paths.
    """
    # `pythonpath = src .` in pytest.ini causes `memory.memory_manager` and
    # `src.memory.memory_manager` to load as two distinct module objects.
    # Patch both so MemoryManager picks SQLite regardless of which import
    # path the test used.
    import sys
    for mod_name in ("memory.memory_manager", "src.memory.memory_manager"):
        if mod_name in sys.modules:
            monkeypatch.setattr(
                f"{mod_name}.SEMANTIC_BACKEND", "sqlite", raising=False,
            )


def pytest_collection_modifyitems(config, items):
    """Auto-skip live tests when required env vars are missing."""
    has_zammad = bool(os.environ.get("ZAMMAD_URL") and os.environ.get("ZAMMAD_API_KEY"))
    has_llm_keys = bool(
        os.environ.get("OPENAI_API_KEY")
        or os.environ.get("GOOGLE_API_KEY")
        or os.environ.get("GOOGLE_GENERATIVEAI_API_KEY")
        or os.environ.get("ANTHROPIC_API_KEY")
    )
    has_discord = bool(os.environ.get("DISCORD_API_KEY") or os.environ.get("DISCORD_BOT_TOKEN"))
    has_hindsight = bool(os.environ.get("HINDSIGHT_LIVE_URL"))

    skip_zammad = pytest.mark.skip(reason="ZAMMAD_URL/ZAMMAD_API_KEY not set")
    skip_llm = pytest.mark.skip(reason="No LLM API keys set (OPENAI_API_KEY, GOOGLE_API_KEY, ANTHROPIC_API_KEY)")
    skip_discord = pytest.mark.skip(reason="DISCORD_API_KEY not set")
    skip_hindsight = pytest.mark.skip(reason="HINDSIGHT_LIVE_URL not set")

    for item in items:
        if "zammad_live" in item.keywords and not has_zammad:
            item.add_marker(skip_zammad)
        if "llm_live" in item.keywords and not has_llm_keys:
            item.add_marker(skip_llm)
        if "discord_live" in item.keywords and not has_discord:
            item.add_marker(skip_discord)
        if "hindsight_live" in item.keywords and not has_hindsight:
            item.add_marker(skip_hindsight)
