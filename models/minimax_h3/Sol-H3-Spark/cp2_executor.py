"""SPMD executor: torchrun owns one native FastVideo worker on each host."""
from __future__ import annotations

import os
from collections.abc import Callable
from typing import Any

import cloudpickle
import torch.distributed as dist
from fastvideo.worker.executor import Executor
from fastvideo.worker.gpu_worker import Worker

CPU_GROUP: Any = None


class TorchrunExecutor(Executor):
    def _init_executor(self) -> None:
        global CPU_GROUP
        world = int(os.environ["SPARK_CP_WORLD"])
        if int(os.environ["WORLD_SIZE"]) != world or self.fastvideo_args.sp_size != world:
            raise RuntimeError("H3 world size must match the requested CP group")
        self.worker = Worker(self.fastvideo_args, local_rank=0,
                             rank=int(os.environ["RANK"]),
                             distributed_init_method="env://")
        self.worker.init_device()
        CPU_GROUP = dist.new_group(backend="gloo")
        from fastvideo.distributed import get_sp_world_size
        if dist.get_world_size() != world or get_sp_world_size() != world:
            raise RuntimeError("Native H3 context parallel group was not initialized")

    def collective_rpc(self, method: str | bytes | Callable[..., Any],
                       timeout: float | None = None, args: tuple = (),
                       kwargs: dict[str, Any] | None = None) -> list[Any]:
        # Both torchrun ranks execute the same control program. Tensor collectives
        # remain in the unmodified FastVideo worker/model implementation.
        options = kwargs or {}
        if isinstance(method, bytes):
            method = cloudpickle.loads(method)
        if isinstance(method, str):
            return [getattr(self.worker, method)(*args, **options)]
        return [method(self.worker, *args, **options)]

    def execute_forward(self, forward_batch: Any, fastvideo_args: Any) -> Any:
        if os.environ.get("SPARK_CP_ABLATION") == "1":
            import torch._dynamo.config
            # FastVideo starts a fresh thread; Dynamo overrides are ContextVars.
            torch._dynamo.config.recompile_limit = 16
        return self.worker.execute_forward(forward_batch, fastvideo_args)

    def set_lora_adapter(self, lora_nickname: str, lora_path: str | None = None,
                         strength: float = 1.0, accumulate: bool = False) -> None:
        result = self.worker.set_lora_adapter(lora_nickname, lora_path, strength, accumulate)
        if result["status"] != "lora_adapter_set":
            raise RuntimeError(result)

    def merge_lora_weights(self) -> None:
        self.worker.merge_lora_weights()

    def unmerge_lora_weights(self) -> None:
        self.worker.unmerge_lora_weights()

    def set_log_queue(self, log_queue: Any) -> None:
        self._log_queue = log_queue

    def clear_log_queue(self) -> None:
        self._log_queue = None

    def shutdown(self) -> None:
        self.worker.shutdown()


def install() -> None:
    Executor.get_class = staticmethod(lambda _args: TorchrunExecutor)
    if int(os.environ["SPARK_CP_WORLD"]) > 1:
        from resident_weights import install as install_weights
        install_weights()


def broadcast(value: Any) -> Any:
    objects = [value]
    dist.broadcast_object_list(objects, src=0, group=CPU_GROUP)
    return objects[0]


def barrier() -> None:
    dist.barrier(group=CPU_GROUP)


def share_capture(path: Any) -> None:
    """Share the rank-zero output receipt without copying or decoding video."""
    import json
    receipt = broadcast(json.loads(path.read_text()) if dist.get_rank() == 0 else None)
    receipt["output_owner_rank"] = 0
    if dist.get_rank() != 0:
        path.write_text(json.dumps(receipt, indent=2) + "\n")
