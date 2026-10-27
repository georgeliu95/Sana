#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = ["torch"]
# ///
# ─── How to run ───
# Use pinned Torch: python -m torch.distributed.run --standalone
#   --nproc-per-node=2 verify_async_all2all.py
"""Independent exact oracle for token/head redistribution, including strided inputs."""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import timedelta
import json
import os
from typing import TYPE_CHECKING

import torch
import torch.distributed as dist

if TYPE_CHECKING:
    from .async_all2all import ExchangeGroup


@dataclass(frozen=True, slots=True)
class InputShape:
    batch: int
    tokens: int
    heads: int
    head_dim: int
    tensors: int
    strided: bool = False


def make_inputs(shape: InputShape, rank: int, device: torch.device) -> list[torch.Tensor]:
    """Encode rank and tensor identity in exactly representable BF16 integers."""
    stride = 2 if shape.strided else 1
    dims = (shape.batch, shape.tokens, shape.heads, shape.head_dim * stride)
    size = shape.batch * shape.tokens * shape.heads * shape.head_dim * stride
    indices = torch.arange(size, dtype=torch.int64, device=device).reshape(dims)
    return [((indices * 13 + rank * 19 + tensor * 23) % 509 - 254)
            .to(torch.bfloat16)[..., ::stride] for tensor in range(shape.tensors)]


def verify_outputs(values: list[torch.Tensor], received: list[torch.Tensor],
                   group: ExchangeGroup) -> None:
    """Compare to independent global concatenation and reconstruct every input shard."""
    rank = dist.get_rank(group.group)
    for value, actual in zip(values, received, strict=True):
        source = value.contiguous()
        all_sources = [torch.empty_like(source) for _ in range(group.world)]
        dist.all_gather(all_sources, source, group=group.group)
        heads_per_rank = value.shape[2] // group.world
        expected = torch.cat([item[:, :, rank * heads_per_rank:(rank + 1) * heads_per_rank]
                              for item in all_sources], dim=1)
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
        all_outputs = [torch.empty_like(actual) for _ in range(group.world)]
        dist.all_gather(all_outputs, actual, group=group.group)
        tokens = value.shape[1]
        restored = torch.cat([item[:, rank * tokens:(rank + 1) * tokens]
                              for item in all_outputs], dim=2)
        torch.testing.assert_close(restored, value, atol=0, rtol=0)


def verify_small_cases(group: ExchangeGroup, device: torch.device,
                       exchange: Callable[[list[torch.Tensor], ExchangeGroup], list[torch.Tensor]]) -> int:
    """Exercise B=1/B=2, KV/QKV, head_dim64/128, and contiguous/strided layouts."""
    cases = 0
    for batch in (1, 2):
        for count in (2, 3):
            for head_dim in (64, 128):
                for strided in (False, True):
                    shape = InputShape(batch, 7, 8, head_dim, count, strided)
                    values = make_inputs(shape, dist.get_rank(group.group), device)
                    before = [value.clone() for value in values]
                    received = exchange(values, group)
                    verify_outputs(values, received, group)
                    for value, saved in zip(values, before, strict=True):
                        torch.testing.assert_close(value, saved, atol=0, rtol=0)
                    cases += 1
    return cases


def main() -> None:
    """Run with torchrun; CPU Gloo by default, NCCL only when explicitly selected."""
    from .async_all2all import ExchangeGroup, exchange_separate_async

    backend = os.environ.get("VERIFY_BACKEND", "gloo")
    device = torch.device("cpu")
    if backend == "nccl":
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
        device = torch.device("cuda", torch.cuda.current_device())
    dist.init_process_group(backend, timeout=timedelta(seconds=120))
    try:
        cases = verify_small_cases(ExchangeGroup(None, dist.get_world_size()), device,
                                   exchange_separate_async)
        if dist.get_rank() == 0:
            print(json.dumps({"status": "PASS", "cases": cases,
                              "world": dist.get_world_size(), "backend": backend,
                              "torch": torch.__version__, "absolute_error": 0}))
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
