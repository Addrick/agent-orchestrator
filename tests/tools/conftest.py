"""Shared test fixtures for tests/tools/.

The dispatcher (scripts/gemini_acp_dispatch.py) spawns a real gemini-cli in
production; tests swap in ``fake_gemini.py`` via the ``spawn_fn`` seam on
``run_dispatch``. This file exposes a fixture that returns a spawn callable
configured with whatever scenario env vars the test wants.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import Callable

import pytest

FAKE_GEMINI = Path(__file__).parent / "fixtures" / "fake_gemini.py"
FAKE_AGY = Path(__file__).parent / "fixtures" / "fake_agy.py"


@pytest.fixture
def fake_gemini_spawner() -> Callable[..., Callable[[Path], subprocess.Popen]]:
    """Return a factory that builds a ``spawn_fn`` for run_dispatch.

    Usage::

        def test_x(fake_gemini_spawner):
            spawn = fake_gemini_spawner(prompt_delay=0.0, final_message="...")
            run_dispatch(..., spawn_fn=spawn)
    """

    def factory(
        *,
        final_message: str | None = None,
        stop_reason: str = "end_turn",
        permission_requests: int = 0,
        prompt_delay: float = 0.0,
        fail_init: bool = False,
    ) -> Callable[[Path], subprocess.Popen]:
        env = os.environ.copy()
        if final_message is not None:
            env["FAKE_GEMINI_FINAL_MESSAGE"] = final_message
        env["FAKE_GEMINI_STOP_REASON"] = stop_reason
        env["FAKE_GEMINI_PERMISSION_REQUESTS"] = str(permission_requests)
        env["FAKE_GEMINI_PROMPT_DELAY"] = str(prompt_delay)
        if fail_init:
            env["FAKE_GEMINI_FAIL_INIT"] = "1"

        def spawn(worktree: Path) -> subprocess.Popen:
            return subprocess.Popen(
                [sys.executable, str(FAKE_GEMINI)],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
                cwd=str(worktree.resolve()),
                env=env,
            )

        return spawn

    return factory


@pytest.fixture
def fake_agy_spawner() -> Callable[..., Callable[[Path], subprocess.Popen]]:
    """Return a factory that builds a ``spawn_fn`` for the agy dispatcher.

    Usage::

        def test_x(fake_agy_spawner):
            spawn = fake_agy_spawner(prompt_delay=0.0, final_message="...")
            run_dispatch(..., spawn_fn=spawn)
    """

    def factory(
        *,
        final_message: str | None = None,
        stderr: str = "",
        prompt_delay: float = 0.0,
        exit_code: int = 0,
    ) -> Callable[[Path], subprocess.Popen]:
        env = os.environ.copy()
        if final_message is not None:
            env["FAKE_AGY_FINAL_MESSAGE"] = final_message
        env["FAKE_AGY_STDERR"] = stderr
        env["FAKE_AGY_PROMPT_DELAY"] = str(prompt_delay)
        env["FAKE_AGY_EXIT_CODE"] = str(exit_code)

        def spawn(worktree: Path) -> subprocess.Popen:
            return subprocess.Popen(
                [sys.executable, str(FAKE_AGY)],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                cwd=str(worktree.resolve()),
                env=env,
            )

        return spawn

    return factory
