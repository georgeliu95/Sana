"""Real socket pair regression cases for framed payloads."""

import socket
from concurrent.futures import ThreadPoolExecutor

from . import wire


def test_payload_when_tcp_splits_a_frame() -> None:
    # Given: a stream socket and payload larger than one receive.
    left, right = socket.socketpair()
    with left, right, ThreadPoolExecutor(max_workers=1) as pool:
        payload = b"same-request-latent" * 20000
        # When: the frame travels over the real stream.
        future = pool.submit(wire.send_bytes, left, payload)
        received = wire.receive_bytes(right)
        future.result()
    # Then: every byte and the exact frame boundary survive.
    assert received == payload
