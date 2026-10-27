"""Remote Stage2 Worker with one controller clock through completed MP4 return."""

from __future__ import annotations

import json
import os
import socket
import time
from contextlib import ExitStack
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import TYPE_CHECKING, Final, Literal, final

from runtime.early_handoff import capture_files
from runtime.profiling import scope
from .protocol import Command, receive_response, send_command
from .wire import receive_file, send_bytes

if TYPE_CHECKING:
    from collections.abc import Mapping

    from pydantic import JsonValue


@dataclass(frozen=True, slots=True)
class Pending:
    """A submitted operation can overlap the local H3 draft."""

    op: str
    started_ns: int
    capture_bytes: int = 0
    capture_send_s: float = 0.0


@final
class RemoteWorker:
    """Mutable sequence/channel state belongs to one persistent Stage2 session."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.root.mkdir()
        self.stack = ExitStack()
        self.sock = self.stack.enter_context(
            socket.create_connection(
                (os.environ["SPARK_PP_HOST"], int(os.environ["SPARK_PP_PORT"])),
                timeout=1800,
            ),
        )
        self.sock.settimeout(3600)
        self.token: Final = Path(os.environ["SPARK_PP_TOKEN_FILE"]).read_text().strip()
        self.transfers: list[dict[str, JsonValue]] = []
        send_command(
            self.sock,
            Command(
                op="open",
                token=self.token,
                kwargs={"work_dir": str(root / "model")},
            ),
        )
        self.ready = receive_response(self.sock).model_dump()
        _ = (root / "ready.json").write_text(json.dumps(self.ready, indent=2) + "\n")

    def submit(
        self,
        op: Literal["open", "prepare", "run", "release_idle_cache", "close"] = "run",
        **kwargs: JsonValue,
    ) -> Pending:
        """Send control and same-request tensors before returning the pending handle."""
        start = time.monotonic_ns()
        send_command(self.sock, Command(op=op, token=self.token, kwargs=kwargs))
        count, transfer_s = 0, 0.0
        if op == "run":
            captured = Path(str(kwargs["capture_dir"]))
            t0 = time.perf_counter()
            with scope("Spark/transfer/H3_to_LTX"):
                for name in capture_files(early=kwargs.get("early_handoff") is True):
                    data = (captured / name).read_bytes()
                    count += len(data)
                    send_bytes(self.sock, sha256(data).hexdigest().encode())
                    send_bytes(self.sock, data)
            transfer_s = time.perf_counter() - t0
        return Pending(op, start, count, transfer_s)

    def wait(self, pending: Pending) -> Mapping[str, JsonValue]:
        """Use the local receive endpoint; never subtract clocks from two hosts."""
        response = receive_response(self.sock)
        result = response.result
        if response.artifact is not None:
            t0 = time.perf_counter()
            with scope('Spark/transfer/MP4_return_verify_write'):
                size = receive_file(
                    self.sock,
                    Path(response.artifact.path),
                    response.artifact.sha256,
                )
            endpoint = time.monotonic_ns()
            result["remote_final_mp4_complete_monotonic_ns"] = result[
                "final_mp4_complete_monotonic_ns"
            ]
            result["final_mp4_complete_monotonic_ns"] = endpoint
            transfer: dict[str, JsonValue] = {
                "capture_bytes": pending.capture_bytes,
                "capture_send_s": pending.capture_send_s,
                "mp4_bytes": size,
                "mp4_receive_s": time.perf_counter() - t0,
                "clock_boundary": (
                    "controller request entry to checksum-verified MP4 return"
                ),
            }
            result["two_node_transfer"] = transfer
            self.transfers.append(transfer)
            _ = (self.root / "transfers.json").write_text(
                json.dumps(self.transfers, indent=2) + "\n",
            )
        return result

    def call(
        self,
        op: Literal["open", "prepare", "run", "release_idle_cache", "close"] = "run",
        **kwargs: JsonValue,
    ) -> Mapping[str, JsonValue]:
        """Match the existing local Worker interface."""
        with scope(f'Spark/remote/{op}/{kwargs.get('request_id', '')}'):
            return self.wait(self.submit(op, **kwargs))

    def close(self) -> None:
        """Release only this owned remote session and connection."""
        try:
            send_command(self.sock, Command(op="close", token=self.token))
            _ = receive_response(self.sock)
        finally:
            self.stack.close()
