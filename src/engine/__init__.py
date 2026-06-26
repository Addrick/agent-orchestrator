# src/engine/ — provider-agnostic LLM orchestration engine (DP-244).
#
# `engine.py` (a 1586-loc god module) became this package: `driver.py` holds the
# slimmed `TextEngine` (routing/rate-limit/retry/429 policy), `providers/` holds
# the `Provider` ABC family, `registry.py` the ordered router. The public import
# surface is unchanged — `from src.engine import TextEngine, LLMCommunicationError`.

from .driver import (  # noqa: F401
    TextEngine,
    AGY_CALL_TIMEOUT_SECONDS,
    CC_CALL_TIMEOUT_SECONDS,
)
from src.llm_errors import LLMCommunicationError  # noqa: F401

# Back-compat for the test suite, which patches SDK/stdlib module *attributes*
# through `src.engine.<module>` (mutation-style, e.g.
# `monkeypatch.setattr(src.engine.os, "name", ...)`,
# `patch("src.engine.asyncio.sleep")`, `patch("src.engine.anthropic.AsyncAnthropic")`,
# `patch("src.engine.genai.client.AsyncClient")`). Re-export the shared module
# objects so those paths resolve; the driver imports the same singletons.
import os  # noqa: F401,E402
import asyncio  # noqa: F401,E402
import shutil  # noqa: F401,E402
import tempfile  # noqa: F401,E402
import anthropic  # noqa: F401,E402
from google import genai  # noqa: F401,E402

__all__ = [
    "TextEngine",
    "LLMCommunicationError",
    "AGY_CALL_TIMEOUT_SECONDS",
    "CC_CALL_TIMEOUT_SECONDS",
]
