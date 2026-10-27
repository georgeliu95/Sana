#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = ["pydantic>=2"]
# ///
# ─── How to run ───
# Launch via the pinned Stage2 torchrun container; see two-node launcher.
"""Serve one Stage2 session to the authenticated controller peer."""

from __future__ import annotations

import json
import os
import socket
from hashlib import sha256
from hmac import compare_digest
from pathlib import Path
from typing import assert_never

from runtime.early_handoff import capture_files
from runtime.profiling import scope, selected
from .protocol import Artifact, Command, Response, owned_path
from .wire import WireError, receive_bytes, receive_file, send_bytes


def main() -> None:
    """Keep models resident through warmup and formal requests."""
    from runtime.stage2 import Session

    root = Path(os.environ["SPARK_PP_RESULT_ROOT"])
    token = Path(os.environ["SPARK_PP_TOKEN_FILE"]).read_text().strip()
    paths = json.loads(Path(os.environ["SPARK_PP_PATHS"]).read_text())
    config = json.loads(Path(os.environ["SPARK_PP_CONFIG"]).read_text())
    import torch
    import torch.distributed as dist
    from datetime import timedelta
    from . import spmd
    from runtime.stage2_ops.qualify_nccl import qualify
    torch.cuda.set_device(0)
    dist.init_process_group("nccl", timeout=timedelta(minutes=45))
    group = dist.new_group(backend="gloo", timeout=timedelta(minutes=45))
    qualify(root.parent / "ltx-nccl-qualification.json")
    if dist.get_rank() != 0:
        spmd.follow(paths, config, root, group)
        dist.destroy_process_group()
        return
    session = None
    with socket.socket() as listener:
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        listener.bind((os.environ["SPARK_PP_BIND"], int(os.environ["SPARK_PP_PORT"])))
        listener.listen(1)
        peer, address = listener.accept()
        with peer:
            peer.settimeout(3600)
            if address[0] != os.environ["SPARK_PP_PEER"]:
                msg = "unexpected peer"
                raise WireError(msg)
            while True:
                command = Command.model_validate_json(receive_bytes(peer))
                if not compare_digest(command.token, token):
                    msg = "invalid task session token"
                    raise WireError(msg)
                if command.op != "run":
                    spmd.broadcast(command, root, group)
                kwargs = command.kwargs
                identity = str(kwargs.get("request_id", ""))
                with scope(f"Spark/server/{command.op}/{identity}", capture=command.op == "run" and selected(identity)):
                    match command.op:
                        case "open":
                            model = owned_path(str(kwargs["work_dir"]), root)
                            session = Session(paths, str(model), config)
                            result = {"session": session.session_receipt}
                        case "prepare":
                            if session is None:
                                msg = "Stage2 is not open"
                                raise WireError(msg)
                            owned_path(str(kwargs["output_root"]), root)
                            result = session.prepare(**kwargs)
                        case "run":
                            if session is None:
                                msg = "Stage2 is not open"
                                raise WireError(msg)
                            capture = owned_path(str(kwargs["capture_dir"]), root)
                            owned_path(str(kwargs["output_root"]), root)
                            for name in capture_files(early=kwargs.get("early_handoff") is True):
                                digest = receive_bytes(peer).decode()
                                receive_file(peer, capture / name, digest)
                            spmd.broadcast(command, root, group)
                            result = session.run(**kwargs)
                        case "release_idle_cache":
                            if session is None:
                                msg = "Stage2 is not open"
                                raise WireError(msg)
                            result = session.release_idle_cache(**kwargs)
                        case "close":
                            if session is not None:
                                session.close()
                            send_bytes(
                                peer,
                                Response(status="PASS").model_dump_json().encode(),
                            )
                            break
                        case unreachable:
                            assert_never(unreachable)
                    artifact = None
                    if command.op == "run":
                        path = owned_path(str(result["output"]), root)
                        data = path.read_bytes()
                        artifact = Artifact(path=str(path), sha256=sha256(data).hexdigest())
                    response = Response.model_validate_json(
                        json.dumps(
                            {
                                "status": "PASS",
                                "result": result,
                                "artifact": None
                                if artifact is None
                                else artifact.model_dump(),
                            },
                            default=str,
                        ),
                    )
                    send_bytes(peer, response.model_dump_json().encode())
                    if artifact is not None:
                        send_bytes(peer, data)
    if session is not None:
        session.close()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
