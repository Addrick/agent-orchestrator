# tests/tools/test_notify_integration.py
#
# DP-512: `notify_adam` as a native tool, the shared sender behind it and
# behind codr's relay, and the guard against a send on a router channel that
# nothing registers (how fixr's `send_discord` posted nowhere).

import ast
import json
from pathlib import Path

import pytest

from config import global_config
from src.clients.notification import (
    HourlyLimiter, NotificationRouter, Notifier, post_as_persona,
)
from src.persona import MemoryMode
from src.tool_policy import ToolPolicy
from src.tools.definitions import ALL_TOOL_DEFINITIONS
from src.tools.notify_integration import NotifyIntegration
from src.tools.tool_manager import ToolManager
from src.tools.turn_context import TurnContext, set_turn_context

SRC = Path(__file__).resolve().parents[2] / "src"


class _Channel(Notifier):
    """A chat channel: every recipient files under ("conv-<recipient>", "srv")."""

    def __init__(self, ok=True):
        self.ok = ok
        self.sent = []

    async def send(self, recipient, subject, body):
        self.sent.append((recipient, subject, body))
        return self.ok

    async def conversation_key(self, recipient):
        return (f"conv-{recipient}", "srv")


class _Dm(Notifier):
    """Not a chat conversation: nothing is stored for it."""

    def __init__(self):
        self.sent = []

    async def send(self, recipient, subject, body):
        self.sent.append((recipient, subject, body))
        return True


class _Persistence:
    def __init__(self):
        self.rows = []
        self.retained = []

    def commit_or_update_assistant(self, **kw):
        self.rows.append(kw)
        return len(self.rows)

    def queue_retain_safe(self, **kw):
        self.retained.append(kw)


CONFIG = {
    "recipients": {
        "adrich": {"discord_user_id": "111"},
        "ops": {"discord_channel_id": "222"},
    },
    "notify": {
        "per_hour": 2,
        "targets": {
            "default": {"channel": "discord_dm", "recipient": "adrich"},
            "stagr": {"channel": "discord_channel", "recipient": "ops"},
        },
    },
}


def _wired(config=CONFIG, channel_ok=True):
    router = NotificationRouter()
    channel, dm = _Channel(channel_ok), _Dm()
    router.register("discord_channel", channel)
    router.register("discord_dm", dm)
    persistence = _Persistence()
    integration = NotifyIntegration(router, persistence, lambda: config)
    return integration, channel, dm, persistence


@pytest.fixture
def turn():
    """Start a persona's turn. Called inside the test, so the context var is
    set in that test's own task context and ends with it."""
    def start(persona="stagr", channel="somewhere", server_id="srv"):
        set_turn_context(TurnContext(
            persona_name=persona, user_identifier="u1", channel=channel,
            server_id=server_id, memory_mode=MemoryMode.CHANNEL_ISOLATED))
    return start


async def test_sent_to_the_personas_own_target_and_stored_as_its_reply(turn):
    integration, channel, dm, persistence = _wired()
    turn("stagr")
    out = await integration._notify_adam("  the build is red  ")
    assert out["status"] == "sent"
    assert channel.sent == [("222", "", "stagr: the build is red")]
    assert dm.sent == []
    (row,) = persistence.rows
    assert (row["persona_name"], row["channel"], row["server_id"], row["final_text"]) == (
        "stagr", "conv-222", "srv", "the build is red")
    assert persistence.retained[0]["untrusted"] is False


async def test_a_persona_without_its_own_target_uses_the_default(turn):
    integration, channel, dm, persistence = _wired()
    turn("managr")
    out = await integration._notify_adam("hello")
    assert out["status"] == "sent"
    assert dm.sent == [("111", "", "managr: hello")]
    # A DM is not a chat conversation, so nothing is written to history.
    assert persistence.rows == []


@pytest.mark.parametrize("message", ["", "   ", None, 7])
async def test_an_empty_message_is_not_sent(turn, message):
    integration, channel, dm, _ = _wired()
    turn()
    out = await integration._notify_adam(message)
    assert out["status"] == "not_sent" and "non-empty" in out["message"]
    assert channel.sent == [] and dm.sent == []


@pytest.mark.parametrize("config", [
    {},                                                       # old agents.json: no "notify" key
    {"notify": {}},
    {"notify": {"targets": {"other": {"channel": "discord_dm", "recipient": "111"}}}},
    {"notify": {"targets": {"default": {"channel": "discord_dm", "recipient": "nobody"}}}},
])
async def test_no_resolvable_target_is_reported_not_logged_as_sent(turn, config):
    integration, channel, dm, _ = _wired(config)
    turn()
    out = await integration._notify_adam("hi")
    assert out["status"] == "not_sent" and "no notify target" in out["message"]
    assert channel.sent == [] and dm.sent == []


async def test_a_target_on_an_unregistered_channel_is_not_sent(turn):
    """The router would hand this to LogNotifier and report success."""
    config = {"notify": {"targets": {"default": {"channel": "discord", "recipient": "111"}}}}
    integration, channel, dm, _ = _wired(config)
    turn()
    out = await integration._notify_adam("hi")
    assert out["status"] == "not_sent" and "not connected" in out["message"]


async def test_refused_when_the_target_is_the_conversation_the_turn_is_in(turn):
    integration, channel, _, _ = _wired()
    turn("stagr", channel="conv-222", server_id="srv")
    out = await integration._notify_adam("hi")
    assert out["status"] == "not_sent" and "already talking to Adam" in out["message"]
    assert channel.sent == []


async def test_rate_limit_is_per_persona(turn):
    integration, channel, dm, _ = _wired()
    turn("stagr")
    assert [(await integration._notify_adam(f"m{i}"))["status"] for i in range(3)] == [
        "sent", "sent", "not_sent"]
    turn("managr")
    assert (await integration._notify_adam("mine"))["status"] == "sent"
    assert len(channel.sent) == 2 and len(dm.sent) == 1


async def test_a_failed_post_is_reported_and_not_stored(turn):
    integration, _, _, persistence = _wired(channel_ok=False)
    turn("stagr")
    out = await integration._notify_adam("hi")
    assert out["status"] == "not_sent" and "post failed" in out["message"]
    assert persistence.rows == []


async def test_outside_a_turn_nothing_is_sent():
    integration, channel, dm, _ = _wired()
    out = await integration._notify_adam("hi")
    assert out["status"] == "not_sent"
    assert channel.sent == [] and dm.sent == []


async def test_registered_and_callable_through_the_tool_manager(turn):
    integration, channel, _, _ = _wired()
    manager = ToolManager()
    integration.register_tools(manager)
    turn("stagr")
    out = await manager.execute_tool("notify_adam", {"notify_adam"}, message="via manager")
    assert out["result"]["status"] == "sent"
    assert channel.sent[0][2] == "stagr: via manager"


def test_shipped_agents_json_has_a_default_target_that_resolves():
    config = json.loads(global_config.DEFAULT_AGENTS_FILE.read_text(encoding="utf-8"))
    integration, _, _, _ = _wired(config)
    channel, recipient = integration._target("any-persona")
    assert channel == "discord_dm" and recipient.isdigit()


def test_notify_adam_never_arms_the_composition_rules():
    """Trusted sink (Adam, 2026-11-13): the model cannot choose where it goes,
    so it must compose with untrusted and PII reads without an override."""
    by_name = {t["function"]["name"]: t for t in ALL_TOOL_DEFINITIONS if t.get("type") == "function"}
    notify = by_name["notify_adam"]
    assert notify["service_binding"] == "notify"
    assert notify["is_write"] is False
    assert notify["capabilities"]["exfil_capable"] is False
    names = ["notify_adam", "web_search", "search_tickets", "get_ticket_details"]
    with_notify = [by_name[n] for n in names]
    policy = ToolPolicy(allow=names)
    assert policy.validate_composition(with_notify) == policy.validate_composition(with_notify[1:])


async def test_post_as_persona_survives_a_raising_notifier():
    class _Boom(Notifier):
        async def send(self, recipient, subject, body):
            raise RuntimeError("discord is down")

    router = NotificationRouter()
    router.register("discord_channel", _Boom())
    persistence = _Persistence()
    assert await post_as_persona(
        router, persistence, persona_name="p", user_identifier="agent:p",
        channel="discord_channel", recipient="1", body="p: x", reply_text="x") is False
    assert persistence.rows == []


def test_hourly_limiter_counts_only_allowed_sends(monkeypatch):
    clock = [0.0]
    monkeypatch.setattr("src.clients.notification.time.monotonic", lambda: clock[0])
    limiter = HourlyLimiter(2)
    assert [limiter.allow(), limiter.allow(), limiter.allow()] == [True, True, False]
    clock[0] = 3601.0
    assert limiter.allow() is True


# -- prevention: a send on a channel name nothing registers ------------------

def _literal(node):
    return node.value if isinstance(node, ast.Constant) and isinstance(node.value, str) else None


def _router_calls(method):
    """(file, line, first-argument literal) for each `<x>.<method>("...", ...)`
    in src whose receiver reads as a notifier or router."""
    for path in sorted(SRC.rglob("*.py")):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                    and node.func.attr == method):
                continue
            receiver = ast.unparse(node.func.value).lower()
            if "notif" not in receiver and "router" not in receiver:
                continue
            first = node.args[0] if node.args else next(
                (k.value for k in node.keywords if k.arg == "channel"), None)
            name = _literal(first) if first is not None else None
            if name is not None:
                yield path.relative_to(SRC).as_posix(), node.lineno, name


def test_every_literal_send_channel_is_one_the_app_registers():
    """`NotificationRouter.send` hands an unknown channel to `LogNotifier`,
    which returns True. fixr's `send_discord` and its orphan digest sent on
    "discord" for months: nothing was posted and both reported success, and
    their tests used a fake that took any name."""
    registered = {name for _, _, name in _router_calls("register")}
    assert {"discord_dm", "discord_channel"} <= registered
    unknown = [(f, line, name) for f, line, name in _router_calls("send") if name not in registered]
    assert not unknown, f"send() on a channel nothing registers: {unknown}"
