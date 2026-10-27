#!/usr/bin/env python3
# /// script
# requires-python = ">=3.12"
# dependencies = ["pydantic>=2"]
# ///
# Run with the pinned Stage1 torchrun launcher on node4 and node1.
"""Use the existing Spark controller with a synchronous native H3 SP2 worker."""
from __future__ import annotations

import json
import os
from functools import partial
from pathlib import Path
from typing import ClassVar, Literal, assert_never
from collections.abc import Mapping
from concurrent.futures import Future, ThreadPoolExecutor

import torch.distributed as dist
from pydantic import BaseModel, ConfigDict, JsonValue

import cp2_executor as cp2
import infer
from runtime import pipeline
from runtime.profiling import scope, selected
from runtime.stage1 import Session
from runtime.two_node.protocol import Command
from runtime.two_node.proxy import RemoteWorker
from runtime.two_node.wire import WireError


class RunArguments(BaseModel):
    """Parse the native Stage1 run contract at the command boundary."""
    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True)
    case: dict[str, JsonValue]
    conditioning_path: str
    outputdir: str


class Stage1Worker:
    """Keep the native SP2 worker resident while rank zero schedules requests."""

    def __init__(self, session: Session) -> None:
        self.session = session
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="h3-request")

    def dispatch(self, command: Command) -> Mapping[str, JsonValue]:
        request = RunArguments.model_validate(command.kwargs)
        directory = Path(request.conditioning_path).parent
        payload = [(str(p.relative_to(directory)), p.read_bytes())
                   for p in directory.rglob("*") if p.is_file()] if dist.get_rank() == 0 else None
        with scope("Spark/H3/communication/conditioning_broadcast"):
            files = cp2.broadcast(payload)
        if dist.get_rank() != 0:
            for relative, data in files:
                target = directory / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(data)
        case = request.case
        from runtime.experiment_arm import apply_arm
        apply_arm(str(case.get("request_id", case["case_id"])))
        with scope(f"Spark/H3/CP{dist.get_world_size()}/request", capture=selected(str(case.get("request_id", case["case_id"])))):
            return self.session.run(request.case, request.conditioning_path, request.outputdir)

    def call(self, op: Literal["run"] = "run", **kwargs: JsonValue) -> Mapping[str, JsonValue]:
        command = Command(op=op, token="", kwargs=kwargs)
        cp2.broadcast(command.model_dump_json())
        return self.dispatch(command)

    def submit(self, **kwargs: JsonValue) -> Future[Mapping[str, JsonValue]]:
        """One H3 request owns all H3 collectives while the controller schedules LTX."""
        return self.executor.submit(self.call, **kwargs)

    def wait(self, pending: Future[Mapping[str, JsonValue]]) -> Mapping[str, JsonValue]:
        return pending.result(timeout=1800)

    def close(self) -> None:
        self.executor.shutdown(wait=True, cancel_futures=True)
        cp2.broadcast(Command(op="close", token="").model_dump_json())
        self.session.close()

    def follow(self) -> None:
        """Run the same tensor program on the companion host for each request."""
        while True:
            command = Command.model_validate_json(cp2.broadcast(None))
            match command.op:
                case "run":
                    self.dispatch(command)
                case "close":
                    self.session.close()
                    return
                case "open" | "prepare" | "release_idle_cache":
                    raise WireError(f"Unsupported H3 follower operation: {command.op}")
                case unreachable:
                    assert_never(unreachable)


def main() -> None:
    """Preserve original warmup, fresh Qwen, sampling, transfer, and E2E clock."""
    cp2.install()
    paths = json.loads(Path(os.environ["SPARK_PP_PATHS"]).read_text())
    config = json.loads(Path(os.environ["SPARK_PP_CONFIG"]).read_text())
    output = Path(os.environ["SPARK_PP_RESULT_ROOT"])
    stage = Stage1Worker(Session(paths, str(output.parent / "h3-resident"), config))
    if dist.get_rank() != 0:
        stage.follow()
        return
    original_environment = pipeline.worker_environment

    def environment(name: str, paths: Mapping[str, str]) -> Mapping[str, str]:
        result = original_environment(name, paths)
        for key in ("RANK", "WORLD_SIZE", "LOCAL_RANK", "LOCAL_WORLD_SIZE", "GROUP_RANK",
                    "ROLE_RANK", "ROLE_WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT",
                    "TORCHELASTIC_RUN_ID", "TORCHELASTIC_ERROR_FILE"):
            result.pop(key, None)
        return result

    def factory(name: Literal["stage1", "stage2", "qwen"], root: Path,
                paths: Mapping[str, str], config: Mapping[str, JsonValue]) -> Stage1Worker | RemoteWorker | pipeline.Worker:
        match name:
            case "stage1":
                root.mkdir()
                return stage
            case "stage2":
                return RemoteWorker(root)
            case "qwen":
                return pipeline.Worker(name, root, paths, config)
            case unreachable:
                assert_never(unreachable)

    native_recipe = pipeline.load_recipe
    def distributed_recipe(task: str, **options: str | None) -> Mapping[str, JsonValue]:
        recipe = native_recipe(task, **options)
        recipe["stage1"]["sequence_parallel_degree"] = dist.get_world_size()
        recipe["stage2"]["sequence_parallel_degree"] = dist.get_world_size()
        recipe["stage2"]["single_gpu_all2all"] = dist.get_world_size() == 1
        recipe["stage2"]["communication"] = "NCCL"
        return recipe
    pipeline.load_recipe = distributed_recipe
    pipeline.worker_environment = environment
    infer.Pipeline = partial(pipeline.Pipeline, worker_factory=factory)
    infer.main()


if __name__ == "__main__":
    main()
