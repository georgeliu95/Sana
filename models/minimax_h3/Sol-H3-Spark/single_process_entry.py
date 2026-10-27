#!/usr/bin/env python3
# Qwen, H3 and LTX in one process: single_entry.py (cp4_entry with local workers) with the Qwen
# and Stage2 worker processes replaced by their Sessions built in this process and driven the way runtime/worker.py
# drives them (the same Session(paths, work_dir, config), one profile request and spans file per operation), one
# operation at a time in the calling thread (Stage2's "prepare", submitted next to the H3 request, only creates its
# output directory). The three now share torch's process-global settings, so global-state.jsonl next to the pipeline
# root records them after each session is built.
"""Run the Spark controller, H3, Qwen and LTX in one process."""
from __future__ import annotations

import importlib
import json
import os
import sys
from collections.abc import Callable, Mapping
from concurrent.futures import Future
from functools import partial
from pathlib import Path
from typing import Literal, assert_never

import torch
import torch.distributed as dist
from pydantic import JsonValue

import cp2_executor as cp2
import infer
from runtime import pipeline
from runtime.stage1 import Session
from single_entry import Stage1Worker

ENV_KEYS = ("TORCHINDUCTOR_CACHE_DIR", "TRITON_CACHE_DIR", "SPARK_LTX_V2_MODE", "H3_LTX_SOURCE_FRAMES",
            "H3_LTX_OUTPUT_FRAMES", "SOL_ATTN_STRICT", "FASTVIDEO_VSA_SM100A", "FASTVIDEO_VSA_CUTEDSL",
            "PYTORCH_CUDA_ALLOC_CONF")


def recorder(path: Path) -> Callable[[str], None]:
    """Append torch's process-global numerics, compile and environment settings under a label."""
    def value(call: Callable[[], object]) -> object:
        try:
            return call()
        except Exception as error:  # an API missing in this torch is recorded, not fatal
            return f"unavailable: {type(error).__name__}"

    def record(label: str) -> None:
        backends = torch.backends
        state = {
            "label": label,
            "float32_matmul_precision": torch.get_float32_matmul_precision(),
            "matmul": {"allow_tf32": backends.cuda.matmul.allow_tf32,
                       "fp16_reduced": backends.cuda.matmul.allow_fp16_reduced_precision_reduction,
                       "bf16_reduced": backends.cuda.matmul.allow_bf16_reduced_precision_reduction},
            "cudnn": {"enabled": backends.cudnn.enabled, "benchmark": backends.cudnn.benchmark,
                      "deterministic": backends.cudnn.deterministic, "allow_tf32": backends.cudnn.allow_tf32},
            "sdp": {"flash": value(backends.cuda.flash_sdp_enabled),
                    "mem_efficient": value(backends.cuda.mem_efficient_sdp_enabled),
                    "math": value(backends.cuda.math_sdp_enabled), "cudnn": value(backends.cuda.cudnn_sdp_enabled),
                    "fp16_bf16_reduction_math": value(backends.cuda.fp16_bf16_reduction_math_sdp_allowed)},
            "dynamo": {key: getattr(torch._dynamo.config, key, None)
                       for key in ("recompile_limit", "automatic_dynamic_shapes")},
            "inductor": {key: getattr(torch._inductor.config, key, None)
                         for key in ("max_autotune", "max_autotune_gemm", "coordinate_descent_tuning",
                                     "fx_graph_cache", "freezing", "emulate_precision_casts")},
            "env": {key: os.environ.get(key) for key in ENV_KEYS},
            "sys_path_head": sys.path[:8],
        }
        with path.open("a") as stream:
            stream.write(json.dumps(state) + "\n")

    return record


class LocalWorker:
    """pipeline.Worker's interface over a Session built in this process, driven as runtime/worker.py drives it."""

    def __init__(self, name: str, root: Path, paths: Mapping[str, str], config: Mapping[str, JsonValue],
                 record: Callable[[str], None]) -> None:
        from spark_profile.sol_state import PROFILES
        self.name, self.root, self.sequence = name, Path(root), 0
        self.root.mkdir()
        self.profile = PROFILES[name]
        module = importlib.import_module("runtime." + name)
        self.session = module.Session(paths=paths, work_dir=str(self.root / "model"), config=config)
        record(f"{name} session built ({self.root.name})")

    def call(self, op: str = "run", **kwargs: JsonValue) -> Mapping[str, JsonValue]:
        from spark_profile.exporting import write_records
        if op not in ("run", "prepare", "release_idle_cache"):
            raise ValueError("Unsupported session operation")
        self.sequence += 1
        response = self.root / f"response-{self.sequence:05d}.json"
        with self.profile.request(f"sol/{self.name}/{op}/{response.stem}") as recording:
            result = getattr(self.session, op)(**kwargs)
        write_records(response.with_suffix(".spans.json"), recording.collect())
        return result

    def submit(self, op: str = "run", **kwargs: JsonValue) -> Future[Mapping[str, JsonValue]]:
        pending: Future[Mapping[str, JsonValue]] = Future()
        pending.set_result(self.call(op, **kwargs))
        return pending

    def wait(self, pending: Future[Mapping[str, JsonValue]]) -> Mapping[str, JsonValue]:
        return pending.result()

    def close(self) -> None:
        self.session.close()


def main() -> None:
    """single_entry.main with the Qwen and Stage2 sessions in this process."""
    cp2.install()
    paths = json.loads(Path(os.environ["SPARK_PP_PATHS"]).read_text())
    config = json.loads(Path(os.environ["SPARK_PP_CONFIG"]).read_text())
    output = Path(os.environ["SPARK_PP_RESULT_ROOT"])
    record = recorder(output.parent / "global-state.jsonl")  # the pipeline creates output itself
    record("start")
    stage = Stage1Worker(Session(paths, str(output.parent / "h3-resident"), config))
    record("stage1 session built")
    if dist.get_rank() != 0:
        stage.follow()
        return

    def factory(name: Literal["stage1", "stage2", "qwen"], root: Path,
                paths: Mapping[str, str], config: Mapping[str, JsonValue]) -> Stage1Worker | LocalWorker:
        match name:
            case "stage1":
                root.mkdir()
                return stage
            case "stage2" | "qwen":
                return LocalWorker(name, root, paths, config, record)
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
    infer.Pipeline = partial(pipeline.Pipeline, worker_factory=factory)
    infer.main()
    record("end")


if __name__ == "__main__":
    main()
