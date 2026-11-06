"""DP-500: a malformed request must get one 400, not an h11 traceback.

Windows' Proactor loop can deliver a read that was already in flight after
uvicorn closed the transport; stock H11Protocol then tries a second 400 on a
MUST_CLOSE connection and raises LocalProtocolError out of the loop callback.
"""
import asyncio
import logging

import h11
import pytest
import uvicorn
from uvicorn.protocols.http.h11_impl import H11Protocol
from uvicorn.server import ServerState

from src.interfaces.http_protocol import GuardedH11Protocol

GARBAGE = b"\x16\x03\x01\x02\x00\x01\x00\x01\xfc\x03\x03 not http\r\n\r\n"


class _Transport:
    """Proactor-like: close() marks closing but does not stop later reads."""

    def __init__(self) -> None:
        self.written = b""
        self.closing = False

    def write(self, data: bytes) -> None:
        self.written += data

    def close(self) -> None:
        self.closing = True

    def is_closing(self) -> bool:
        return self.closing

    def get_extra_info(self, name: str, default=None):
        return {"peername": ("10.0.0.67", 51334), "sockname": ("10.0.0.69", 5003)}.get(name, default)

    def pause_reading(self) -> None: ...
    def resume_reading(self) -> None: ...


def _protocol(cls):
    async def app(scope, receive, send):  # never reached
        raise AssertionError
    loop = asyncio.new_event_loop()
    proto = cls(config=uvicorn.Config(app), server_state=ServerState(), app_state={}, _loop=loop)
    loop.close()  # only needed for construction; nothing here schedules on it
    transport = _Transport()
    proto.connection_made(transport)
    return proto, transport


def test_stock_protocol_raises_on_read_after_close():
    """Pins the upstream bug this module exists for; if it stops failing, drop the subclass."""
    proto, _ = _protocol(H11Protocol)
    proto.data_received(GARBAGE)
    with pytest.raises(h11.LocalProtocolError):
        proto.data_received(GARBAGE)


def test_guarded_protocol_sends_one_400_and_drops_late_bytes(caplog):
    proto, transport = _protocol(GuardedH11Protocol)
    with caplog.at_level(logging.WARNING, logger="src.interfaces.http_protocol"):
        proto.data_received(GARBAGE)
        proto.data_received(GARBAGE)

    assert transport.closing
    assert transport.written.count(b"HTTP/1.1 400") == 1
    rejections = [r for r in caplog.records if r.name == "src.interfaces.http_protocol"]
    assert len(rejections) == 1
    assert "10.0.0.67:51334" in rejections[0].getMessage()
    assert "\\x16\\x03\\x01" in rejections[0].getMessage()
