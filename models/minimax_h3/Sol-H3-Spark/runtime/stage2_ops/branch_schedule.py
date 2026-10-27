"""Execute independent adapter and audio encoder branches on separate ranks."""
from __future__ import annotations
from collections.abc import Callable
from dataclasses import dataclass
import time
import torch
import torch.distributed as dist
from torch import Tensor


@dataclass(frozen=True, slots=True)
class BranchResult:
    video: Tensor
    audio: Tensor
    wall_seconds: float


def run_branches(adapter: Callable[[], Tensor], audio: Callable[[], Tensor],
                 video_shape: tuple[int, ...], audio_shape: tuple[int, ...],
                 device: torch.device, dtype: torch.dtype) -> BranchResult:
    """Rank 1 owns the complete adapter; rank 0 owns original audio encoding."""
    begin = time.perf_counter()
    rank = dist.get_rank()
    with torch.cuda.nvtx.range("Spark/bridge/parallel_adapter_audio"):
        video = adapter() if rank == 1 else torch.empty(video_shape, device=device, dtype=dtype)
        sound = audio() if rank == 0 else torch.empty(audio_shape, device=device, dtype=dtype)
        with torch.cuda.nvtx.range("Spark/bridge/adapter_latent_broadcast"):
            dist.broadcast(video, src=1)
        with torch.cuda.nvtx.range("Spark/bridge/audio_latent_broadcast"):
            dist.broadcast(sound, src=0)
        torch.cuda.synchronize()
    return BranchResult(video, sound, time.perf_counter() - begin)
