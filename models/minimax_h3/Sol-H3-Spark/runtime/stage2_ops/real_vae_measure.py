"""Exact output comparison and paired rank-aware timing for a loaded real VAE."""
from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import dataclass
import hashlib
import time
from typing import TypedDict

import torch
import torch.distributed as dist

from .real_vae_setup import require

Decode = Callable[[], Iterator[torch.Tensor]]


class ErrorMetrics(TypedDict):
    bitwise_equal: bool
    finite: bool
    max_abs: float | None
    rmse: float | None
    elements: int
    chunk_shapes: list[list[int]]
    rng_equal: bool
    expected_rng_sha256: str
    actual_rng_sha256: str


class RankSample(TypedDict):
    rank: int
    wall_s: float
    start_allocated_bytes: int
    peak_allocated_bytes: int
    start_reserved_bytes: int
    peak_reserved_bytes: int


class Sample(TypedDict):
    max_rank_wall_s: float
    ranks: list[RankSample]


@dataclass(frozen=True, slots=True)
class Captured:
    chunks: tuple[torch.Tensor, ...]
    rng_state: torch.Tensor
    final_offset: int


def capture(decode: Decode, generator: torch.Generator) -> Captured:
    """Clone every original yielded chunk before upstream buffers can be reused."""
    chunks = tuple(chunk.detach().clone() for chunk in decode())
    return Captured(chunks, generator.get_state().clone(), generator.get_offset())


def compare(reference: Captured, actual: Captured) -> ErrorMetrics:
    """Compare BF16 storage bits plus full generator state, including finite checks."""
    require(len(reference.chunks) == len(actual.chunks), "Chunk counts differ")
    bitwise, finite, max_abs, sum_squared, elements = True, True, 0.0, 0.0, 0
    for left, right in zip(reference.chunks, actual.chunks, strict=True):
        require(left.shape == right.shape and left.dtype == right.dtype == torch.bfloat16,
                "Real decoder output shape or BF16 dtype differs")
        bitwise &= torch.equal(left.view(torch.int16), right.view(torch.int16))
        finite &= bool(torch.isfinite(left).all().item() and torch.isfinite(right).all().item())
        difference = left.float() - right.float()
        max_abs = max(max_abs, difference.abs().max().item())
        sum_squared += difference.square().sum(dtype=torch.float64).item()
        elements += left.numel()
    require(elements > 0, "Real decoder produced no output elements")
    return {"bitwise_equal": bitwise, "finite": finite, "max_abs": max_abs if finite else None,
            "rmse": (sum_squared / elements) ** 0.5 if finite else None, "elements": elements,
            "chunk_shapes": [list(chunk.shape) for chunk in actual.chunks],
            "rng_equal": torch.equal(reference.rng_state, actual.rng_state),
            "expected_rng_sha256": hashlib.sha256(reference.rng_state.numpy().tobytes()).hexdigest(),
            "actual_rng_sha256": hashlib.sha256(actual.rng_state.numpy().tobytes()).hexdigest()}


def measured(decode: Decode) -> Sample:
    """Time only iterator consumption plus CUDA completion; gather metrics afterward."""
    dist.barrier()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    start_allocated, start_reserved = torch.cuda.memory_allocated(), torch.cuda.memory_reserved()
    start = time.perf_counter()
    for chunk in decode():
        del chunk
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    values = torch.tensor([elapsed, start_allocated, torch.cuda.max_memory_allocated(),
                           start_reserved, torch.cuda.max_memory_reserved()], device="cuda", dtype=torch.float64)
    gathered = [torch.empty_like(values) for _ in range(dist.get_world_size())]
    dist.all_gather(gathered, values)
    rows: list[RankSample] = []
    for rank, item in enumerate(gathered):
        wall, allocated, peak_allocated, reserved, peak_reserved = item.tolist()
        rows.append({"rank": rank, "wall_s": wall, "start_allocated_bytes": int(allocated),
                     "peak_allocated_bytes": int(peak_allocated), "start_reserved_bytes": int(reserved),
                     "peak_reserved_bytes": int(peak_reserved)})
    return {"max_rank_wall_s": max(row["wall_s"] for row in rows), "ranks": rows}
