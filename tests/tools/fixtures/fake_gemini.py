"""Fake gemini ACP server for testing scripts/gemini_acp_dispatch.py.

Speaks line-delimited JSON-RPC 2.0 over stdio. Scenario is controlled via
env vars so a test can spawn this exactly like the real gemini binary:

  FAKE_GEMINI_FINAL_MESSAGE
      Text streamed back as agent_message_chunk notifications before the
      session/prompt response. Default: a well-formed ``## Result`` block.

  FAKE_GEMINI_STOP_REASON
      The ``stopReason`` in the session/prompt response. Default ``end_turn``.

  FAKE_GEMINI_PERMISSION_REQUESTS
      Integer count of ``session/request_permission`` server-to-client
      requests to emit before the prompt response. Used to exercise the
      dispatcher's auto-approve path. Default 0.

  FAKE_GEMINI_PROMPT_DELAY
      Float seconds to sleep before responding to session/prompt. Used to
      exercise wall-clock and SIGTERM cancellation. Default 0.

  FAKE_GEMINI_FAIL_INIT
      If set to "1", reply to initialize with an error and exit.
"""

from __future__ import annotations

import json
import os
import sys
import time
import uuid

_DEFAULT_RESULT = (
    "Did the thing.\n\n"
    "## Result\n"
    "stop_reason: ok\n"
    "files_modified: src/foo.py\n"
    "key_changes: implemented foo\n"
    "acceptance_self_check: ran -- pass\n"
    "blockers: none\n"
)


def _send(obj: dict) -> None:
    sys.stdout.write(json.dumps(obj) + "\n")
    sys.stdout.flush()


def _read_msg():
    line = sys.stdin.readline()
    if not line:
        return None
    return json.loads(line)


def _handle_initialize(msg: dict) -> None:
    if os.environ.get("FAKE_GEMINI_FAIL_INIT") == "1":
        _send({
            "jsonrpc": "2.0",
            "id": msg["id"],
            "error": {"code": -32000, "message": "fake init failure"},
        })
        sys.exit(0)
    _send({
        "jsonrpc": "2.0",
        "id": msg["id"],
        "result": {
            "protocolVersion": 1,
            "authMethods": [{"id": "oauth-personal", "name": "OAuth", "description": ""}],
            "agentInfo": {"name": "fake-gemini", "title": "Fake", "version": "0.0.1"},
            "agentCapabilities": {
                "loadSession": False,
                "promptCapabilities": {},
                "mcpCapabilities": {},
            },
        },
    })


def _handle_session_new(msg: dict) -> str:
    sid = str(uuid.uuid4())
    _send({
        "jsonrpc": "2.0",
        "id": msg["id"],
        "result": {"sessionId": sid, "modes": {"availableModes": []}},
    })
    return sid


def _emit_permission_requests(sid: str, count: int) -> None:
    """Send count server→client request_permission calls.

    We don't block on the dispatcher's responses (the dispatcher's
    auto-approve path is exercised regardless), but we do increment our
    own id space so the dispatcher's response ids don't collide with our
    request ids.
    """
    for i in range(count):
        _send({
            "jsonrpc": "2.0",
            "id": 1000 + i,
            "method": "session/request_permission",
            "params": {
                "sessionId": sid,
                "toolCall": {"name": "edit", "args": {"path": f"f{i}.py"}},
                "options": [
                    {"optionId": "allow_always", "name": "Allow"},
                    {"optionId": "reject", "name": "Reject"},
                ],
            },
        })


def _stream_assistant_text(sid: str, text: str) -> None:
    # Split into a few chunks so the dispatcher's concatenation logic is
    # exercised, not just one big chunk.
    chunks: list[str] = []
    if len(text) <= 32:
        chunks = [text]
    else:
        step = max(1, len(text) // 3)
        chunks = [text[i:i + step] for i in range(0, len(text), step)]
    for chunk in chunks:
        _send({
            "jsonrpc": "2.0",
            "method": "session/update",
            "params": {
                "sessionId": sid,
                "update": {
                    "sessionUpdate": "agent_message_chunk",
                    "content": {"type": "text", "text": chunk},
                },
            },
        })


def _handle_session_prompt(msg: dict, sid: str) -> None:
    delay = float(os.environ.get("FAKE_GEMINI_PROMPT_DELAY", "0") or "0")
    if delay > 0:
        time.sleep(delay)

    perm_count = int(os.environ.get("FAKE_GEMINI_PERMISSION_REQUESTS", "0") or "0")
    if perm_count > 0:
        _emit_permission_requests(sid, perm_count)

    text = os.environ.get("FAKE_GEMINI_FINAL_MESSAGE", _DEFAULT_RESULT)
    _stream_assistant_text(sid, text)

    stop = os.environ.get("FAKE_GEMINI_STOP_REASON", "end_turn")
    _send({
        "jsonrpc": "2.0",
        "id": msg["id"],
        "result": {"stopReason": stop, "_meta": {}},
    })


def _handle_session_cancel(msg: dict) -> None:
    # session/cancel is a request with an id; respond OK so the dispatcher
    # doesn't time out waiting on a response (it currently fires-and-forgets,
    # but cooperative response is the spec-correct behavior).
    if "id" in msg:
        _send({"jsonrpc": "2.0", "id": msg["id"], "result": {}})


def main() -> int:
    session_id = ""
    while True:
        msg = _read_msg()
        if msg is None:
            return 0
        method = msg.get("method")
        if method == "initialize":
            _handle_initialize(msg)
        elif method == "session/new":
            session_id = _handle_session_new(msg)
        elif method == "session/prompt":
            _handle_session_prompt(msg, session_id)
        elif method == "session/cancel":
            _handle_session_cancel(msg)
        elif "id" in msg:
            # Unknown request — respond empty so dispatcher doesn't hang.
            _send({"jsonrpc": "2.0", "id": msg["id"], "result": {}})
        # Responses from dispatcher (e.g. permission approval) — drop.


if __name__ == "__main__":
    sys.exit(main())
