"""Assign the author's existing two temporal segments to separate CP ranks."""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol, TypedDict

import torch
import torch.distributed as dist
from torch.nn import functional as F


class Upscaler(Protocol):
    forward: Callable[..., torch.Tensor]
    _forward_seg: Callable[[torch.Tensor, float | None, tuple[int, ...]], torch.Tensor]


@dataclass(frozen=True, slots=True)
class UpscalerPartitionError(RuntimeError):
    detail: str

    def __str__(self) -> str:
        return self.detail


class Receipt(TypedDict):
    rank: int
    world: int
    requests: int
    local_segments: int
    broadcast_segments: int


def install(model: Upscaler) -> Receipt:
    """Keep the original forward/blend; replace segment calls with distributed results."""
    rank, world = dist.get_rank(), dist.get_world_size()
    forward, segment = model.forward, model._forward_seg
    receipt: Receipt = {"rank": rank, "world": world, "requests": 0,
                        "local_segments": 0, "broadcast_segments": 0}

    def distributed_forward(x: torch.Tensor, scale: float | None = None,
                            target_size: tuple[int, ...] | None = None,
                            enable_chunking: bool = True) -> torch.Tensor:
        if (tuple(x.shape) != (1, 24, 37, 24, 42) or scale != 2.0
                or target_size != (37, 48, 84) or not enable_chunking):
            raise UpscalerPartitionError("Distributed segments require the pinned 37x24x42 draft geometry")
        padded = F.pad(x, (0, 0, 0, 0, 5, 5), mode="replicate")
        boundaries = ((0, 42), (22, 42))
        outputs: dict[int, torch.Tensor] = {}
        for index, (start, stop) in enumerate(boundaries):
            if index % world != rank:
                continue
            with torch.cuda.nvtx.range(f"Spark/bridge/upscaler/local_segment/{index}/owner{rank}"):
                value = segment(padded[:, :, start:stop].contiguous(), scale, (stop - start, 48, 84))
            outputs[index] = value.contiguous()
            receipt["local_segments"] += 1
        index = 0

        def deliver(sample: torch.Tensor, supplied_scale: float | None,
                    size: tuple[int, ...]) -> torch.Tensor:
            nonlocal index
            start, stop = boundaries[index]
            if tuple(sample.shape) != (1, 24, stop - start, 24, 42) or size != (stop - start, 48, 84):
                raise UpscalerPartitionError("Original upscaler segment contract changed")
            owner = index % world
            value = (outputs.pop(index) if rank == owner else
                     torch.empty((1, 24, stop - start, 48, 84), device=x.device, dtype=x.dtype))
            with torch.cuda.nvtx.range(f"Spark/bridge/communication/segment_broadcast/{index}/owner{owner}"):
                dist.broadcast(value, src=owner)
            receipt["broadcast_segments"] += 1
            index += 1
            return value

        model._forward_seg = deliver
        try:
            result = forward(x, scale=scale, target_size=target_size, enable_chunking=enable_chunking)
        finally:
            model._forward_seg = segment
        if index != len(boundaries):
            raise UpscalerPartitionError("Original upscaler must consume exactly two segments")
        receipt["requests"] += 1
        return result

    model.forward = distributed_forward
    return receipt
