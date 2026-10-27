#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = ["torch"]
# ///
# ─── How to run ───
# Use pinned Torch and the runtime PYTHONPATH; torchrun this file on two nodes.
# See README.md for transport environment, shape, repeats and correctness gates.
"""Serial, paired CUDA-event measurements of separate synchronous/async All2All."""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import timedelta
import json
import os
import statistics
import time
from typing import TypeAlias

import torch
import torch.distributed as dist

from runtime.stage2_ops.async_all2all import (
    ExchangeGroup, exchange_separate_async, send_heads_async,
)
from runtime.stage2_ops.nccl_all2all import NCCLAll2All
from .verify_async_all2all import InputShape, make_inputs, verify_outputs, verify_small_cases

Exchange: TypeAlias = Callable[[list[torch.Tensor]], list[torch.Tensor]]


@dataclass(frozen=True, slots=True)
class BenchConfig:
    tokens: int
    warmup: int
    repeats: int
    compile: bool


@dataclass(frozen=True, slots=True)
class Samples:
    label: str
    rank: int
    cuda_ms: list[float]
    wall_ms: list[float]
    cuda_mean_ms: float
    cuda_median_ms: float
    cuda_stdev_ms: float


@dataclass(frozen=True, slots=True)
class PairResult:
    shape: InputShape
    world: int
    payload_bytes_per_input: int
    payload_bytes_per_peer_per_input: int
    remote_payload_bytes_per_rank: int
    compile: bool
    samples: list[Samples]
    paired_peak_allocated_mib_by_rank: list[float]


def summarize(label: str, rank: int, values: torch.Tensor) -> Samples:
    """Keep all observations, including per-iteration maxima across ranks."""
    cuda = values[:, 0].tolist()
    wall = values[:, 1].tolist()
    return Samples(label, rank, cuda, wall, statistics.mean(cuda),
                   statistics.median(cuda), statistics.stdev(cuda))


def measure_pair(exchanges: tuple[Exchange, Exchange], values: list[torch.Tensor],
                 config: BenchConfig) -> tuple[list[Samples], list[float]]:
    """Alternate A/B order, with rank alignment outside each measured interval."""
    for _ in range(config.warmup):
        for exchange in exchanges:
            exchange(values)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    observations: list[list[tuple[float, float]]] = [[], []]
    for repeat in range(config.repeats):
        order = (0, 1) if repeat % 2 == 0 else (1, 0)
        for index in order:
            dist.barrier()
            torch.cuda.synchronize()
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            before = time.perf_counter()
            start.record()
            outputs = exchanges[index](values)
            end.record()
            end.synchronize()
            elapsed = (time.perf_counter() - before) * 1000
            observations[index].append((start.elapsed_time(end), elapsed))
            del outputs
    peak = torch.tensor([torch.cuda.max_memory_allocated() / 2**20], device="cuda")
    all_peaks = [torch.empty_like(peak) for _ in range(dist.get_world_size())]
    dist.all_gather(all_peaks, peak)
    local = torch.tensor(observations, device="cuda", dtype=torch.float64)
    all_samples = [torch.empty_like(local) for _ in range(dist.get_world_size())]
    dist.all_gather(all_samples, local)
    matrix = torch.stack(all_samples).cpu()
    samples: list[Samples] = []
    for index, name in enumerate(("sequential", "async_separate")):
        for rank in range(dist.get_world_size()):
            samples.append(summarize(name, rank, matrix[rank, index]))
        samples.append(summarize(name, -1, matrix[:, index].amax(dim=0)))
    return samples, [item.item() for item in all_peaks]


def benchmark_shape(shape: InputShape, config: BenchConfig) -> PairResult:
    """Verify the frozen public API and custom op before measuring the exact payload."""
    world, rank = dist.get_world_size(), dist.get_rank()
    group = ExchangeGroup(None, world)
    values = make_inputs(shape, rank, torch.device("cuda", torch.cuda.current_device()))
    comm = NCCLAll2All(rank=rank, world_size=world, seqlen=shape.tokens,
                      hidden_dim=shape.heads * shape.head_dim, num_sms=0,
                      tensor_dtype=torch.bfloat16)
    comm.set_rank_tokens([shape.tokens] * world)

    def reference(inputs: list[torch.Tensor]) -> list[torch.Tensor]:
        return [comm.send_recv_heads(value) for value in inputs]

    def candidate(inputs: list[torch.Tensor]) -> list[torch.Tensor]:
        return send_heads_async(inputs, comm.comm_id, world)

    try:
        expected, actual = reference(values), candidate(values)
        verify_outputs(values, actual, group)
        for source, baseline, received in zip(values, expected, actual, strict=True):
            torch.testing.assert_close(received, baseline, atol=0, rtol=0)
            torch.testing.assert_close(comm.gather_heads(received), source, atol=0, rtol=0)
        torch.library.opcheck(send_heads_async, (values, comm.comm_id, world),
                              test_utils=("test_schema", "test_faketensor"))
        del expected, actual
        exchanges = (reference, candidate)
        if config.compile:
            exchanges = (torch.compile(reference, fullgraph=True, dynamic=False),
                         torch.compile(candidate, fullgraph=True, dynamic=False))
            expected, actual = exchanges[0](values), exchanges[1](values)
            for baseline, received in zip(expected, actual, strict=True):
                torch.testing.assert_close(received, baseline, atol=0, rtol=0)
            del expected, actual
        samples, peaks = measure_pair(exchanges, values, config)
        payload = values[0].numel() * values[0].element_size()
        return PairResult(shape, world, payload, payload // world,
                          shape.tensors * payload * (world - 1) // world,
                          config.compile, samples, peaks)
    finally:
        comm.destroy()


def main() -> None:
    """Use torchrun and existing environment; keep setup and verification outside timers."""
    config = BenchConfig(tokens=int(os.environ.get("COMM_TOKENS", "6048")),
                         warmup=int(os.environ.get("COMM_WARMUP", "10")),
                         repeats=int(os.environ.get("COMM_REPEATS", "50")),
                         compile=os.environ.get("COMM_COMPILE", "0") == "1")
    if config.tokens < 1 or config.warmup < 1 or config.repeats < 5:
        raise RuntimeError(config)
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("nccl", timeout=timedelta(seconds=180))
    try:
        group = ExchangeGroup(None, dist.get_world_size())
        cases = verify_small_cases(group, torch.device("cuda", torch.cuda.current_device()),
                                   exchange_separate_async)
        if dist.get_rank() == 0:
            print(json.dumps({"status": "VERIFIED_SMALL", "cases": cases,
                              "torch": torch.__version__, "cuda": torch.version.cuda,
                              "nccl": torch.cuda.nccl.version(), "world": group.world,
                              "config": asdict(config),
                              "transport_env": {key: os.environ.get(key) for key in (
                                  "NCCL_SOCKET_IFNAME", "GLOO_SOCKET_IFNAME", "NCCL_IB_HCA",
                                  "NCCL_DEBUG", "NCCL_DEBUG_SUBSYS", "NCCL_IB_DISABLE",
                                  "NCCL_CROSS_NIC", "NCCL_NET", "NCCL_PROTO", "NCCL_ALGO")}}), flush=True)
        for head_dim in (128, 64):
            for count in (3, 2):
                result = benchmark_shape(InputShape(1, config.tokens, 32, head_dim, count), config)
                if dist.get_rank() == 0:
                    print(json.dumps({"status": "MEASURED", **asdict(result)}), flush=True)
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
