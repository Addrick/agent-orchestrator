# tests/stagr/test_compute.py
#
# DP-522 — run_python, against a fake stagr `opencode serve` whose shell runs
# the script in a real bash. The box-side runner (systemd, user `compute`) is
# replaced by a fake that runs the script with this interpreter; its guards are
# verified live (services/compute-sandbox/README.md) and pinned below.

import asyncio
import re
import sys
from pathlib import Path

import pytest
from aiohttp.test_utils import TestServer

from config import global_config
from src.stagr import StagrIntegration, compute
from src.stagr.compute import ComputeToolHandler
from src.tool_policy import ToolPolicy
from src.tools.definitions import ALL_TOOL_DEFINITIONS
from src.tools.tool_defs.compute import COMPUTE_TOOLS
from src.tools.tool_manager import ToolManager
from tests.test_opencode_provider import FakeOpencode, real_bash

_RUNNER = Path(__file__).resolve().parents[2] / "services" / "compute-sandbox" / "derpr-compute"


@pytest.fixture
async def box(monkeypatch, tmp_path):
    fake = FakeOpencode()
    fake.shell_output, fake.file_root = real_bash(tmp_path), tmp_path
    server = TestServer(fake.app())
    await server.start_server()
    monkeypatch.setattr(global_config, "STAGR_SERVER_URL", str(server.make_url("")).rstrip("/"))
    monkeypatch.setattr(global_config, "STAGR_SERVER_PASSWORD", "spw")
    runner = tmp_path / "fake-runner"
    # Records the SECONDS it was given, then runs stdin as the real one would
    # (UTF-8 and \n line ends, as on the Linux box, wherever the suite runs).
    runner.write_bytes(
        f'#!/bin/bash\necho "$1" >> "{tmp_path.as_posix()}/seconds"\n'
        f'"{Path(sys.executable).as_posix()}" -X utf8 -I - | tr -d "\\r"\n'
        'exit "${PIPESTATUS[0]}"\n'.encode())
    monkeypatch.setattr(compute, "RUNNER", f"bash {runner.as_posix()}")
    yield tmp_path
    await server.close()


async def test_runs_the_script_and_returns_its_output(box):
    result = await ComputeToolHandler().run_python("import math\nprint(math.comb(52, 5))")
    assert result == {"status": "ok", "exit_code": 0, "output": "2598960\n"}


async def test_the_script_arrives_byte_for_byte(box):
    # $, quotes, backslashes and a heredoc marker: nothing on the way may expand them.
    code = 'x = "$HOME `id` \\\\n \'q\' EOF"\nprint(x)\nprint("ünï")'
    result = await ComputeToolHandler().run_python(code)
    assert result["output"] == "$HOME `id` \\n 'q' EOF\nünï\n"


async def test_stderr_and_the_exit_code_come_back(box):
    result = await ComputeToolHandler().run_python("import sys\nprint('out')\nsys.exit('boom')")
    assert result["exit_code"] == 1
    assert "out" in result["output"] and "boom" in result["output"]


async def test_timeout_is_defaulted_and_clamped(box):
    handler = ComputeToolHandler()
    for asked in (None, "5", 0, 10_000):
        await handler.run_python("pass", asked)
    assert (box / "seconds").read_text().split() == ["30", "5", "1", str(compute.MAX_SECONDS)]


async def test_a_runner_timeout_is_reported(box, monkeypatch):
    monkeypatch.setattr(compute, "RUNNER", "bash -c 'exit 124'")
    result = await ComputeToolHandler().run_python("pass", 7)
    assert result["exit_code"] == 124 and result["note"] == "timed out after 7 s"


@pytest.mark.parametrize("code, timeout, expect", [
    ("", None, "code is empty"),
    ("   \n", None, "code is empty"),
    ("#" * (compute.MAX_CODE_BYTES + 1), None, "over"),
    ("print(1)", "soon", "whole number"),
], ids=["empty", "blank", "too-big", "bad-timeout"])
async def test_bad_arguments_send_nothing(box, code, timeout, expect):
    result = await ComputeToolHandler().run_python(code, timeout)
    assert result["status"] == "error" and expect in result["message"]
    assert not (box / "seconds").exists()


async def test_refuses_when_stagr_is_not_configured(monkeypatch):
    monkeypatch.setattr(global_config, "STAGR_SERVER_URL", "")
    result = await ComputeToolHandler().run_python("print(1)")
    assert result["status"] == "error" and "not configured" in result["message"]


async def test_at_most_two_calls_are_in_flight(monkeypatch):
    monkeypatch.setattr(global_config, "STAGR_SERVER_URL", "http://stagr.invalid")
    live, peak = 0, 0

    async def run_shell(self, script, **_):
        nonlocal live, peak
        live += 1
        peak = max(peak, live)
        await asyncio.sleep(0.05)
        live -= 1
        return "__STAGR_EXIT=0\n"

    monkeypatch.setattr(compute.OpencodeClient, "run_shell", run_shell)
    handler = ComputeToolHandler()
    results = await asyncio.gather(*(handler.run_python("print(1)") for _ in range(6)))
    assert all(r["exit_code"] == 0 for r in results)
    assert peak == compute.MAX_CONCURRENT == 2


def test_the_stagr_integration_registers_it():
    manager = ToolManager.__new__(ToolManager)
    names = []
    manager.register = lambda name, *a, **k: names.append(name)  # type: ignore[method-assign]
    StagrIntegration().register_tools(manager)
    assert "run_python" in names


def test_definition_is_a_core_unprompted_read():
    (tool,) = COMPUTE_TOOLS
    assert tool in ALL_TOOL_DEFINITIONS
    assert "service_binding" not in tool  # any persona, no binding
    assert tool["is_write"] is False  # no approval prompt
    assert tool["capabilities"]["exfil_capable"] is False


def test_it_arms_no_exfiltration_rule():
    """A persona holding PII and untrusted reads plus a write elsewhere: adding
    run_python changes nothing, because it can carry nothing off its box."""
    by_name = {t["function"]["name"]: t for t in ALL_TOOL_DEFINITIONS if "function" in t}
    pii = next(t for t in ALL_TOOL_DEFINITIONS if t.get("capabilities", {}).get("sensitivity") == "pii"
               and not t.get("is_write"))
    base = [pii, by_name["web_search"]]
    policy = ToolPolicy.from_dict({"default": "deny", "allow": [], "ask": []})
    assert policy.validate_composition(base + COMPUTE_TOOLS) == policy.validate_composition(base)


def test_the_runner_keeps_its_guards():
    """The no-approval decision rests on these lines of the box-side runner;
    removing one is a security change, not a refactor."""
    text = _RUNNER.read_text(encoding="utf-8")
    for prop in ("--uid=compute", "PrivateNetwork=yes", "IPAddressDeny=any",
                 "RestrictAddressFamilies=AF_UNIX", "TemporaryFileSystem=/run:ro", "ProtectHome=yes", "ProtectSystem=strict",
                 "PrivateTmp=yes", "NoNewPrivileges=yes", "MemoryMax=", "TasksMax=", "RuntimeMaxSec=",
                 "CapabilityBoundingSet= "):
        assert prop in text, prop
    assert re.search(r"seconds < 1 \|\| seconds > 120", text)
    assert compute.MAX_SECONDS == 120
