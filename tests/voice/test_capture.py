# tests/voice/test_capture.py
"""DiscordVoiceCapture lifecycle (DP-238b).

Regression: ``attach_discord`` launches ``capture.start()`` before the Discord
client's own ``start()`` task has logged in. ``wait_until_ready()`` *raises*
("Client has not been properly initialised") when called pre-login, which killed
the whole voice pipeline. ``start()`` must instead poll ``is_ready()`` (which only
returns False) until the gateway is up.
"""
import sys
import types

import pytest

from src.voice.capture import DiscordVoiceCapture
from src.voice.types import AudioFrame


@pytest.fixture
def fake_voice_recv(monkeypatch):
    """Stub the optional ``discord.ext.voice_recv`` dep (not installed in CI)."""
    mod = types.ModuleType("discord.ext.voice_recv")
    mod.VoiceRecvClient = object  # type: ignore[attr-defined]
    mod.BasicSink = lambda cb: ("sink", cb)  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "discord.ext.voice_recv", mod)
    # Ensure ``from discord.ext import voice_recv`` resolves the attribute too.
    ext = sys.modules.setdefault("discord.ext", types.ModuleType("discord.ext"))
    monkeypatch.setattr(ext, "voice_recv", mod, raising=False)
    return mod


class _FakeVoice:
    def __init__(self):
        self.sink = None

    def listen(self, sink):
        self.sink = sink


class _FakeChannel:
    def __init__(self, voice):
        self._voice = voice

    async def connect(self, cls=None):
        return self._voice


class _FakeClient:
    """is_ready() is False for the first ``ready_after`` polls, then True."""

    def __init__(self, channel, ready_after=2):
        self._channel = channel
        self._ready_after = ready_after
        self._polls = 0
        self.wait_until_ready_called = False

    def is_ready(self):
        self._polls += 1
        return self._polls >= self._ready_after

    async def wait_until_ready(self):
        # The bug was calling this pre-login; assert it's never used.
        self.wait_until_ready_called = True
        raise AssertionError("capture must not call wait_until_ready()")

    def get_channel(self, _cid):
        return self._channel


@pytest.mark.asyncio
async def test_start_polls_is_ready_then_connects(fake_voice_recv, monkeypatch):
    monkeypatch.setattr("asyncio.sleep", _instant_sleep)
    voice = _FakeVoice()
    client = _FakeClient(_FakeChannel(voice), ready_after=3)

    capture = DiscordVoiceCapture(client, channel_id=99, on_frame=_noop_frame)
    await capture.start()

    assert client.wait_until_ready_called is False
    assert client._polls >= 3          # waited for the gateway
    assert voice.sink is not None      # listening was set up


@pytest.mark.asyncio
async def test_start_raises_when_channel_missing(fake_voice_recv, monkeypatch):
    monkeypatch.setattr("asyncio.sleep", _instant_sleep)
    client = _FakeClient(channel=None, ready_after=1)
    # get_channel returns None; fetch_channel should be consulted next.

    async def _fetch(_cid):
        return None

    client.fetch_channel = _fetch  # type: ignore[attr-defined]
    capture = DiscordVoiceCapture(client, channel_id=99, on_frame=_noop_frame)

    with pytest.raises(RuntimeError, match="voice channel 99 not found"):
        await capture.start()


async def _instant_sleep(_seconds):
    return None


async def _noop_frame(_frame: AudioFrame):
    return None
