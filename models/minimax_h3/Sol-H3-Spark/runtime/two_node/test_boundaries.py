"""Boundary checks for the real binary handoff and result-tree ownership."""

import socket
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from .protocol import owned_path
from .wire import WireError, receive_bytes, receive_file, send_bytes


def test_checksum_when_payload_is_corrupted(tmp_path: Path) -> None:
    # Given: an advertised digest which cannot match the received bytes.
    left, right = socket.socketpair()
    destination = tmp_path / "latent.pt"
    with left, right:
        send_bytes(left, b"changed-payload")
        # When: receiving the same-request tensor file.
        # Then: it fails without publishing a corrupt file.
        with pytest.raises(WireError, match="checksum"):
            receive_file(right, destination, "0" * 64)
    assert not destination.exists()


def test_path_when_request_escapes_task_root(tmp_path: Path) -> None:
    # Given: one owned experiment tree and a path beyond it.
    # When: a remote operation parses that destination.
    # Then: the caller cannot write beyond the experiment.
    with pytest.raises(WireError, match="outside"):
        owned_path(str(tmp_path / "../outside"), tmp_path)


def test_frames_when_payloads_are_consecutive() -> None:
    # Given: two consecutive frames on one persistent connection.
    left, right = socket.socketpair()
    with left, right, ThreadPoolExecutor(max_workers=1) as pool:

        def send() -> None:
            send_bytes(left, b"metadata")
            send_bytes(left, b"binary-PCM" * 10000)

        # When: consuming the first and next frame.
        future = pool.submit(send)
        first, second = receive_bytes(right), receive_bytes(right)
        future.result()
    # Then: one receive never consumes the next frame.
    assert first == b"metadata"
    assert second == b"binary-PCM" * 10000
