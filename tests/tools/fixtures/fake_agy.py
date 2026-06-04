"""Fake agy CLI for testing scripts/agy_dispatch.py.

agy's transport is trivial compared to gemini's ACP: ``agy -p`` just prints
its narration + final ``## Result`` block to stdout and exits. So this fake
ignores its argv, optionally sleeps, optionally writes stderr, prints a
message to stdout, and exits with a configurable code. Scenario is controlled
via env vars so a test can spawn it exactly like the real binary:

  FAKE_AGY_FINAL_MESSAGE
      Text printed to stdout (agy's whole turn). Default: a well-formed
      ``## Result`` block.

  FAKE_AGY_STDERR
      Text written to stderr (agy's narration/log). Default: empty.

  FAKE_AGY_PROMPT_DELAY
      Float seconds to sleep before printing. Used to exercise wall-clock and
      SIGTERM cancellation. Default 0.

  FAKE_AGY_EXIT_CODE
      Integer process exit code. Default 0.
"""

from __future__ import annotations

import os
import sys
import time

_DEFAULT_RESULT = (
    "Did the thing.\n\n"
    "## Result\n"
    "stop_reason: ok\n"
    "files_modified: src/foo.py\n"
    "key_changes: implemented foo\n"
    "acceptance_self_check: ran -- pass\n"
    "blockers: none\n"
)


def main() -> int:
    delay = float(os.environ.get("FAKE_AGY_PROMPT_DELAY", "0") or "0")
    if delay > 0:
        time.sleep(delay)

    stderr_text = os.environ.get("FAKE_AGY_STDERR", "")
    if stderr_text:
        sys.stderr.write(stderr_text)
        sys.stderr.flush()

    sys.stdout.write(os.environ.get("FAKE_AGY_FINAL_MESSAGE", _DEFAULT_RESULT))
    sys.stdout.flush()

    return int(os.environ.get("FAKE_AGY_EXIT_CODE", "0") or "0")


if __name__ == "__main__":
    sys.exit(main())
