"""Explicit packed-output protocol for inference-only BF16 tile64 gate merging."""
from __future__ import annotations

from dataclasses import dataclass
import functools
import os
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    import torch


@dataclass(frozen=True, slots=True)
class MergeContractError(RuntimeError):
    detail: str

    def __str__(self) -> str:
        return self.detail


def enabled_from_environment() -> bool:
    flag = os.environ.get("SPARK_CP_H3_MERGE_UNTILE", "0")
    if flag not in ("0", "1"):
        raise MergeContractError(f"SPARK_CP_H3_MERGE_UNTILE must be 0 or 1, got {flag!r}")
    return flag == "1"


class PackedShape(Protocol):
    total_seq_length: int
    untile_combined_index: torch.Tensor


@dataclass(frozen=True, slots=True)
class PackedMetadata:
    untile_combined_index: torch.Tensor
    variable_block_sizes: torch.Tensor
    num_prefix_tiles: int
    num_video_tiles: int
    current_timestep: int
    VSA_sparsity: float = 0.9
    tile_elems: int = 64
    exempt: bool = True
    dense_layers: tuple[int, ...] = ()

    @property
    def total_seq_length(self) -> int:
        return self.untile_combined_index.numel()


def packed_metadata(mapping: torch.Tensor, sizes: torch.Tensor, prefix: int,
                    video: int, step: int) -> PackedMetadata:
    return PackedMetadata(mapping, sizes, prefix, video, step)


def packed_fake(query: torch.Tensor, value: torch.Tensor, mapping: torch.Tensor) -> torch.Tensor:
    return query.new_empty((query.shape[0], mapping.numel(), query.shape[2], value.shape[-1]))


def packed_postprocess(output: torch.Tensor, metadata: PackedShape) -> torch.Tensor:
    """Identity only for the declared packed protocol; never guess from tensor size."""
    if (output.ndim != 4 or output.shape[1] != metadata.total_seq_length
            or metadata.untile_combined_index.numel() != metadata.total_seq_length):
        raise MergeContractError("packed merge returned padded or inconsistent row geometry")
    return output


@functools.lru_cache(maxsize=1)
def _kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def merge(Out, Gate, Compressed, Index, Result,
              ELEMENTS: tl.constexpr, ROWS: tl.constexpr, HEADS: tl.constexpr, DIM: tl.constexpr,
              TILE: tl.constexpr, OB: tl.constexpr, OS: tl.constexpr, OH: tl.constexpr, OD: tl.constexpr,
              GB: tl.constexpr, GS: tl.constexpr, GH: tl.constexpr, GD: tl.constexpr,
              CB: tl.constexpr, CT: tl.constexpr, CH: tl.constexpr, CD: tl.constexpr,
              BLOCK: tl.constexpr):
        index = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        valid = index < ELEMENTS
        dim = index % DIM
        head = index // DIM % HEADS
        row = index // (DIM * HEADS) % ROWS
        batch = index // (DIM * HEADS * ROWS)
        padded = tl.load(Index + row, valid, 0)
        out_index = batch * OB + padded * OS + head * OH + dim * OD
        gate_index = batch * GB + padded * GS + head * GH + dim * GD
        compressed_index = batch * CB + (padded // TILE) * CT + head * CH + dim * CD
        a = tl.load(Out + out_index, valid, 0).to(tl.float32)
        gate = tl.load(Gate + gate_index, valid, 0).to(tl.float32)
        compressed = tl.load(Compressed + compressed_index, valid, 0).to(tl.float32)
        product = (compressed * gate).to(tl.bfloat16).to(tl.float32)
        result = (a + product).to(tl.bfloat16)
        tl.store(Result + index, result, valid)
    return merge


def fused_merge_untile(out: torch.Tensor, gate: torch.Tensor, compressed: torch.Tensor,
                       tile: int, index: torch.Tensor) -> torch.Tensor:
    """Qualified kernel arithmetic; map bounds/uniqueness come from native metadata."""
    import torch
    import triton
    if (torch.is_grad_enabled() or any(x.requires_grad for x in (out, gate, compressed))
            or out.ndim != 4 or out.shape != gate.shape or out.dtype != torch.bfloat16
            or gate.dtype != out.dtype or compressed.dtype != out.dtype or index.dtype != torch.int64
            or index.ndim != 1 or not index.is_contiguous() or index.numel() == 0
            or out.device.type != "cuda" or any(x.device != out.device for x in (gate, compressed, index))
            or tile != 64 or out.shape[1] % tile
            or compressed.shape != (out.shape[0], out.shape[1] // tile, out.shape[2], out.shape[3])):
        raise MergeContractError("packed merge requires inference BF16 CUDA BSHD, tile64 compression, and native int64 map")
    result = torch.empty((out.shape[0], index.numel(), out.shape[2], out.shape[3]),
                         dtype=out.dtype, device=out.device)
    _kernel()[(triton.cdiv(result.numel(), 512),)](
        out, gate, compressed, index, result, result.numel(), index.numel(), out.shape[2], out.shape[3],
        tile, *out.stride(), *gate.stride(), *compressed.stride(), BLOCK=512,
        num_warps=4, enable_fp_fusion=False)
    return result
