"""Compare the original upscaler and distributed original segments on fixed inputs."""
from __future__ import annotations

from collections.abc import Callable
import json
from pathlib import Path
import statistics
import time
from typing import TypedDict

from pydantic import BaseModel, ConfigDict
import torch
import torch.distributed as dist

from .h3_upscale import H3_INPUT, load_upscaler, upscale_comfy_normalized_h3
from .parallel_upscaler import install


class Paths(BaseModel):
    model_config = ConfigDict(frozen=True)
    h3_upscaler_source: Path
    h3_upscaler_checkpoint: Path


class Timing(TypedDict):
    variant: str
    wall_s: list[float]
    mean_s: float
    stdev_s: float


def measured(call: Callable[[], torch.Tensor]) -> float:
    dist.barrier()
    torch.cuda.synchronize()
    start = time.perf_counter()
    call()
    torch.cuda.synchronize()
    elapsed = torch.tensor(time.perf_counter() - start, device="cuda", dtype=torch.float64)
    dist.all_reduce(elapsed, op=dist.ReduceOp.MAX)
    return elapsed.item()


def qualify(paths_file: Path, output: Path) -> None:
    paths = Paths.model_validate_json(paths_file.read_text())
    model, _ = load_upscaler(paths.h3_upscaler_source, paths.h3_upscaler_checkpoint,
                            device=torch.device("cuda", 0), torch_module=torch)
    original = model.forward
    generator = torch.Generator(device="cuda").manual_seed(71)
    inputs = [torch.randn(H3_INPUT, generator=generator, device="cuda", dtype=torch.bfloat16)
              for _ in range(2)]
    with torch.inference_mode():
        expected = [upscale_comfy_normalized_h3(model, value) for value in inputs]
        receipt = install(model)
        distributed = model.forward
        actual = [upscale_comfy_normalized_h3(model, value) for value in inputs]
        for value, reference in zip(actual, expected, strict=True):
            torch.testing.assert_close(value, reference, atol=0, rtol=0)
        times: dict[str, list[float]] = {"original": [], "distributed": []}
        for repeat in range(10):
            order = ("original", "distributed") if repeat % 2 == 0 else ("distributed", "original")
            for name in order:
                model.forward = original if name == "original" else distributed
                times[name].append(measured(lambda: upscale_comfy_normalized_h3(model, inputs[0])))
    rows: list[Timing] = [{"variant": name, "wall_s": values, "mean_s": statistics.mean(values),
                          "stdev_s": statistics.stdev(values)} for name, values in times.items()]
    output.write_text(json.dumps({"status": "PASS", "rank": dist.get_rank(),
        "world": dist.get_world_size(), "input_shape": list(H3_INPUT), "dtype": "bfloat16",
        "fixed_input_cases": 2, "max_abs": 0, "n": 10, "warmup_excluded": True,
        "timings": rows, "receipt": receipt}, indent=2) + "\n")
