# tests/stagr/test_sandbox_net.py
#
# DP-465 — the sandbox boxes' LAN/WAN switches, against a fake node that keeps
# switch state the way agent-net (host derpr) and egress-ctl (CT110) do. The
# argvs these tests produce are fed through the real wrappers in
# tests/services/test_pve_wrapper.py and test_net_wrapper.py.

from typing import Dict, List, Sequence, Tuple

import pytest

from config import global_config
from src.proxmox.ssh import SSHResult
from src.stagr.handler import StagrToolHandler
from src.stagr.net import AGENT_NET, EGRESS_VERB

pytestmark = pytest.mark.asyncio


class FakeNode:
    """Both node targets at once. `conns` holds the open connections per
    (box, zone); turning a switch off clears them, as the node does."""

    def __init__(self) -> None:
        self.on: Dict[Tuple[str, str], bool] = {}
        self.conns: Dict[Tuple[str, str], List[str]] = {}
        self.approved: Dict[str, List[str]] = {}
        self.calls: List[List[str]] = []

    async def run(self, argv: Sequence[str]) -> SSHResult:
        a = list(argv)
        self.calls.append(a)
        if a[0] == AGENT_NET:
            box, zone, verb = a[1], "lan", a[3]
        else:
            assert a[0] == EGRESS_VERB
            box, zone, verb = a[1], "wan", a[2]
        key = (box, zone)
        if verb == "status":
            return SSHResult(0, f"{zone}: {'on' if self.on.get(key) else 'off'}", "")
        if verb == "conns":
            return SSHResult(0, "\n".join(self.conns.get(key, [])), "")
        if verb == "on":
            self.on[key] = True
            return SSHResult(0, f"{zone}: on", "")
        if verb == "off":
            self.on[key] = False
            closed = "".join(f"closed {c}\n" for c in self.conns.pop(key, []))
            self.approved.pop(box, None) if zone == "wan" else None
            return SSHResult(0, f"{closed}{zone}: off", "")
        if verb == "allow":
            if not self.on.get(key):
                return SSHResult(1, "", f"egress-ctl: {box}'s WAN is off; turn it on first")
            self.approved.setdefault(box, []).append(a[3])
            return SSHResult(0, f"allowed {a[3]} until {box}'s WAN goes off", "")
        if verb == "list":
            return SSHResult(0, "default:\n  github.com\napproved:\n" +
                             "".join(f"  {h}\n" for h in self.approved.get(box, [])), "")
        if verb == "log":
            return SSHResult(0, 'CONNECT [1]: Established connection to host "pypi.org" using file descriptor 7.', "")
        raise AssertionError(a)


@pytest.fixture
def node(monkeypatch) -> FakeNode:
    monkeypatch.setattr(global_config, "PVE_TOOLS_ENABLED", True)
    return FakeNode()


def _handler(node: FakeNode) -> StagrToolHandler:
    return StagrToolHandler(node, node)  # type: ignore[arg-type]


async def test_lan_goes_to_host_derpr_and_wan_to_the_egress_verb(node):
    """Two node targets: LAN is agent-net over the second key, WAN is
    derpr-egress over the first. Separate runners keep them apart."""
    pve, net = FakeNode(), FakeNode()
    handler = StagrToolHandler(pve, net)  # type: ignore[arg-type]
    assert (await handler._set_sandbox_net("codr", "lan", "on"))["status"] == "ok"
    assert (await handler._set_sandbox_net("codr", "wan", "on", confirm=True))["status"] == "ok"
    assert all(c[0] == AGENT_NET for c in net.calls) and net.calls
    assert all(c[0] == EGRESS_VERB for c in pve.calls) and pve.calls


async def test_off_with_open_connections_asks_first_then_cuts(node):
    node.on[("stagr", "wan")] = True
    node.conns[("stagr", "wan")] = ["140.82.112.3:443 github.com"]
    handler = _handler(node)
    ask = await handler._set_sandbox_net("stagr", "wan", "off")
    assert ask["status"] == "confirm" and ask["connections"] == ["140.82.112.3:443 github.com"]
    assert node.on[("stagr", "wan")] is True, "a confirm answer must change nothing"
    done = await handler._set_sandbox_net("stagr", "wan", "off", confirm="true")
    assert done["status"] == "ok" and done["closed"] == ["140.82.112.3:443 github.com"]
    assert node.on[("stagr", "wan")] is False


async def test_off_with_nothing_open_just_goes_off(node):
    node.on[("codr", "lan")] = True
    result = await _handler(node)._set_sandbox_net("codr", "lan", "off")
    assert result == {"status": "ok", "box": "codr", "lan": False, "changed": True}


async def test_the_second_switch_of_a_box_warns_before_going_on(node):
    node.on[("stagr", "lan")] = True
    handler = _handler(node)
    ask = await handler._set_sandbox_net("stagr", "wan", "on")
    assert ask["status"] == "confirm" and "both on" in ask["message"]
    assert not node.on.get(("stagr", "wan"))
    assert (await handler._set_sandbox_net("stagr", "wan", "on", confirm=True))["status"] == "ok"
    assert node.on[("stagr", "wan")] is True


async def test_an_unchanged_switch_does_not_touch_the_node_twice(node):
    result = await _handler(node)._set_sandbox_net("codr", "wan", "off")
    assert result["changed"] is False
    assert [c[2] for c in node.calls] == ["status"]


@pytest.mark.parametrize("args", [("dt21", "lan", "on"), ("codr", "usb", "on"), ("codr", "lan", "maybe")])
async def test_bad_switch_arguments_never_reach_the_node(node, args):
    assert (await _handler(node)._set_sandbox_net(*args))["status"] == "error"
    assert node.calls == []


async def test_request_network_turns_stagrs_switch_on_and_only_stagrs(node):
    """The parked approval is the confirmation, so a LAN+WAN warning does not
    stop it a second time; and the persona can only ask for its own box."""
    node.on[("stagr", "lan")] = True
    result = await _handler(node)._request_network("WAN", "pip install")
    assert result["status"] == "ok" and node.on[("stagr", "wan")] is True
    assert {c[1] for c in node.calls} == {"stagr"}
    assert (await _handler(node)._request_network("usb", "x"))["status"] == "error"


async def test_web_access_needs_wan_on_and_sends_only_a_checked_host(node):
    handler = _handler(node)
    off = await handler._request_web_access("files.example.org", "docs")
    assert off["status"] == "error" and "WAN is off" in off["stderr"]
    node.on[("stagr", "wan")] = True
    ok = await handler._request_web_access("Files.Example.org.", "docs")
    assert ok["status"] == "ok"
    assert node.calls[-1] == [EGRESS_VERB, "stagr", "allow", "files.example.org"]
    before = len(node.calls)
    for bad in ("10.0.0.70", "https://example.org", "example.org/x", "localhost", "a;b.com", ""):
        assert (await handler._request_web_access(bad, "x"))["status"] == "error", bad
    assert len(node.calls) == before


async def test_a_web_access_call_parked_before_dp465_still_resolves(node):
    """An approval parked under the DP-456 shape (domain, minutes) runs after
    the deploy: minutes is ignored, there is no timer any more."""
    node.on[("stagr", "wan")] = True
    result = await _handler(node)._request_web_access(domain="files.example.org", minutes=30, reason="x")
    assert result["status"] == "ok"
    assert node.calls[-1] == [EGRESS_VERB, "stagr", "allow", "files.example.org"]


async def test_list_web_access_shows_stagrs_switches_and_allowlist(node):
    node.on[("stagr", "wan")] = True
    node.approved["stagr"] = ["files.example.org"]
    result = await _handler(node)._list_web_access()
    assert result["lan"] == "off" and result["wan"] == "on"
    assert "files.example.org" in result["allowlist"]
    assert {c[1] for c in node.calls} == {"stagr"}


async def test_status_covers_both_boxes_and_a_dead_host_fails_alone(monkeypatch):
    monkeypatch.setattr(global_config, "PVE_TOOLS_ENABLED", True)

    class DeadHost:
        async def run(self, argv):
            return SSHResult(255, "", "ssh: connect to host 10.0.0.74 port 22: No route to host")

    pve = FakeNode()
    pve.on[("codr", "wan")] = True
    result = await StagrToolHandler(pve, DeadHost())._sandbox_net_status()  # type: ignore[arg-type]
    boxes = {b["box"]: b for b in result["boxes"]}
    assert set(boxes) == {"codr", "stagr"}
    assert boxes["codr"]["wan"] is True and boxes["codr"]["lan"] is None
    assert "lan" in boxes["codr"]["errors"] and boxes["codr"]["log"]
