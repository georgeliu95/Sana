"""Keep LTX video state partitioned while preserving global noise and positions."""
from __future__ import annotations

from dataclasses import dataclass, replace

import torch
import torch.distributed as dist
from ltx_core.types import LatentState


@dataclass(frozen=True, slots=True)
class ShardContractError(RuntimeError):
    detail: str

    def __str__(self) -> str:
        return self.detail


def partition_state(state: LatentState, rank: int, world: int) -> LatentState:
    """Slice an already-noised video state without consuming any additional RNG."""
    tokens = state.latent.shape[1]
    if world not in (1, 2, 4) or not 0 <= rank < world or tokens % world:
        raise ShardContractError("Uniform CP1/2/4 video tokens and a valid rank are required")
    if state.generated_keyframe_layout is not None:
        raise ShardContractError("Generated keyframe slots are outside the matched T2V experiment")
    if state.attention_mask is not None:
        raise ShardContractError("This experiment supports the matched unmasked T2V workload")
    width = tokens // world
    start, stop = rank * width, (rank + 1) * width
    return replace(state, latent=state.latent[:, start:stop],
                   clean_latent=state.clean_latent[:, start:stop],
                   denoise_mask=state.denoise_mask[:, start:stop],
                   positions=state.positions[:, :, start:stop],
                   keyframes_mask=(None if state.keyframes_mask is None else
                                   state.keyframes_mask[:, start:stop]))


def restore_state(local: LatentState, original: LatentState) -> LatentState:
    """Restore the original decoder interface with one final video AllGather."""
    world = dist.get_world_size()
    from ltx_core.multigpu.transformer.sequence_parallel import gather_output_tokens
    with torch.cuda.nvtx.range("Spark/LTX/communication/final_latent_AllGather"):
        latent = gather_output_tokens(local.latent, [local.latent.shape[1]] * world)
    return replace(original, latent=latent)
