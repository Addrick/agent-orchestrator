# src/interfaces/http_protocol.py
#
# uvicorn's h11 protocol, hardened for one failure path.
#
# On a malformed request uvicorn sends a 400 and calls transport.close(). On
# Windows' Proactor loop a read already in flight still delivers its bytes
# after that close, so data_received runs again on a connection whose h11
# state is MUST_CLOSE; uvicorn tries to send a second 400 and h11 raises
# LocalProtocolError, which asyncio logs as an unhandled callback traceback.
# The client already has its 400 — the stray bytes are simply dropped here.
#
# uvicorn's own warning ("Invalid HTTP request received.") names neither the
# sender nor the request, so a rejection also logs the peer and a short
# prefix of the offending bytes: that is what identifies the client to fix.

import logging

from uvicorn.protocols.http.h11_impl import H11Protocol

logger = logging.getLogger(__name__)

_PREVIEW_BYTES = 40


class GuardedH11Protocol(H11Protocol):
    """H11Protocol that ignores reads after closing and logs who sent a bad request."""

    _last_chunk: bytes = b""

    def data_received(self, data: bytes) -> None:
        if self.transport.is_closing():
            return
        self._last_chunk = data
        super().data_received(data)

    def send_400_response(self, msg: str) -> None:
        peer = "%s:%d" % self.client if self.client else "unknown peer"
        logger.warning("Rejected malformed HTTP request from %s (bytes start %r)",
                       peer, self._last_chunk[:_PREVIEW_BYTES])
        super().send_400_response(msg)
