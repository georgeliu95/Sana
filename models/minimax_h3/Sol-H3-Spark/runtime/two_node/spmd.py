"""Mirror authenticated Stage2 commands and input files to the LTX companion."""
from __future__ import annotations
from collections.abc import Mapping
from pathlib import Path
from typing import assert_never

import torch.distributed as dist
from pydantic import JsonValue

from runtime.early_handoff import capture_files
from runtime.profiling import scope, selected
from runtime.stage2 import Session
from .protocol import Command, owned_path
from .wire import WireError


def broadcast(command: Command, root: Path, group: dist.ProcessGroup) -> None:
    """Distribute only validated task commands and this request's two artifacts."""
    files = []
    if command.op == "run":
        capture = owned_path(str(command.kwargs["capture_dir"]), root)
        files = [(str(capture / name), (capture / name).read_bytes())
                 for name in capture_files(early=command.kwargs.get("early_handoff") is True)]
    with scope("Spark/LTX/communication/control_and_input_broadcast"):
        dist.broadcast_object_list([(command.model_dump_json(), files)], src=0, group=group)


def follow(paths: Mapping[str, str], config: Mapping[str, JsonValue], root: Path,
           group: dist.ProcessGroup) -> None:
    """Execute the same existing Stage2 session methods on LTX rank one."""
    session = None
    try:
        while True:
            message = [None]
            dist.broadcast_object_list(message, src=0, group=group)
            encoded, files = message[0]
            command = Command.model_validate_json(encoded)
            for name, data in files:
                target = owned_path(name, root)
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(data)
            identity = str(command.kwargs.get("request_id", ""))
            with scope(f"Spark/server/{command.op}/{identity}/rank{dist.get_rank()}",
                       capture=command.op == "run" and selected(identity)):
                match command.op:
                    case "open":
                        model = owned_path(str(command.kwargs["work_dir"]), root)
                        session = Session(paths, str(model), config)
                    case "prepare" | "run" | "release_idle_cache":
                        if session is None:
                            raise WireError("Stage2 follower session is not open")
                        getattr(session, command.op)(**command.kwargs)
                    case "close":
                        return
                    case unreachable:
                        assert_never(unreachable)
    finally:
        if session is not None:
            session.close()
