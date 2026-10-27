"""Length-framed binary handoff over the task's authenticated TCP session."""

from __future__ import annotations

import struct
from hashlib import sha256
from typing import TYPE_CHECKING, Final

if TYPE_CHECKING:
    import socket
    from pathlib import Path

MAX_FRAME: Final = 16 * 1024 * 1024


class WireError(RuntimeError):
    """A frame was truncated or failed the same-request checksum boundary."""


def _receive(sock: socket.socket, size: int) -> bytes:
    result = bytearray()
    while len(result) < size:
        block = sock.recv(min(size - len(result), 1024 * 1024))
        if not block:
            msg = "peer closed an incomplete frame"
            raise WireError(msg)
        result.extend(block)
    return bytes(result)


def send_bytes(sock: socket.socket, data: bytes) -> None:
    """Send a bounded exact-size frame; no pickle or executable payload."""
    if len(data) > MAX_FRAME:
        msg = "frame exceeds transfer budget"
        raise WireError(msg)
    sock.sendall(struct.pack("!Q", len(data)) + data)


def receive_bytes(sock: socket.socket) -> bytes:
    """Consume exactly one complete frame, preserving subsequent boundaries."""
    size = struct.unpack("!Q", _receive(sock, 8))[0]
    if size > MAX_FRAME:
        msg = "peer frame exceeds transfer budget"
        raise WireError(msg)
    return _receive(sock, size)


def receive_file(sock: socket.socket, destination: Path, digest: str) -> int:
    """Publish a fresh file only after its SHA256 has been checked."""
    data = receive_bytes(sock)
    if sha256(data).hexdigest() != digest:
        msg = "same-request payload checksum mismatch"
        raise WireError(msg)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("xb") as stream:
        _ = stream.write(data)
    return len(data)
