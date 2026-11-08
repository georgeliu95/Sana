"""H3 output-to-slot and receive-to-tile fusion, preserving the existing wire layout."""
from __future__ import annotations

from dataclasses import dataclass
import functools
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch
    from triton.runtime.jit import JITFunction


@dataclass(frozen=True, slots=True)
class FusionError(RuntimeError):
    detail: str


@functools.lru_cache(maxsize=1)
def kernels() -> tuple[JITFunction, JITFunction]:
    import triton
    import triton.language as tl

    @triton.jit
    def merge_slots(Out, Gate, Compressed, Index, D0, D1, D2, D3,
                    ROWS: tl.constexpr, SHARD: tl.constexpr, HEADS: tl.constexpr, DIM: tl.constexpr,
                    OS: tl.constexpr, OH: tl.constexpr, OD: tl.constexpr,
                    GS: tl.constexpr, GH: tl.constexpr, GD: tl.constexpr,
                    CT: tl.constexpr, CH: tl.constexpr, CD: tl.constexpr,
                    BLOCK: tl.constexpr):
        owner = tl.program_id(1)
        index = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        valid = index < SHARD * HEADS * DIM
        dim = index % DIM
        head = index // DIM % HEADS
        row = owner * SHARD + index // (HEADS * DIM)
        live = valid & (row < ROWS)
        padded = tl.load(Index + row, live, 0)
        a = tl.load(Out + padded * OS + head * OH + dim * OD, live, 0).to(tl.float32)
        gate = tl.load(Gate + padded * GS + head * GH + dim * GD, live, 0).to(tl.float32)
        compressed = tl.load(Compressed + padded // 64 * CT + head * CH + dim * CD, live, 0).to(tl.float32)
        product = (compressed * gate).to(tl.bfloat16).to(tl.float32)
        result = (a + product).to(tl.bfloat16)
        dest = tl.where(owner == 0, D0, tl.where(owner == 1, D1, tl.where(owner == 2, D2, D3)))
        tl.store(dest + index, tl.where(live, result, 0), valid)

    @triton.jit
    def receive_tiles(S0, S1, S2, S3, Index, Dest,
                      ROWS: tl.constexpr, SHARD: tl.constexpr, PADDED: tl.constexpr,
                      HEADS: tl.constexpr, DIM: tl.constexpr, INPUTS: tl.constexpr,
                      B0: tl.constexpr, B1: tl.constexpr, B2: tl.constexpr, B3: tl.constexpr,
                      BLOCK: tl.constexpr):
        index = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        valid = index < INPUTS * ROWS * HEADS * DIM
        dim = index % DIM
        head = index // DIM % HEADS
        row = index // (HEADS * DIM) % ROWS
        batch = index // (ROWS * HEADS * DIM)
        owner = row // SHARD
        offset = (row % SHARD) * HEADS * DIM + head * DIM + dim
        source = tl.where(owner == 0, S0, tl.where(owner == 1, S1, tl.where(owner == 2, S2, S3)))
        batch_stride = tl.where(owner == 0, B0, tl.where(owner == 1, B1, tl.where(owner == 2, B2, B3)))
        value = tl.load(source + batch * batch_stride + offset, valid, 0)
        padded = tl.load(Index + row, valid, 0)
        tl.store(Dest + ((batch * PADDED + padded) * HEADS + head) * DIM + dim, value, valid)

    return merge_slots, receive_tiles


def merge_into(out: torch.Tensor, gate: torch.Tensor, compressed: torch.Tensor,
               mapping: torch.Tensor, destinations: tuple[torch.Tensor, ...]) -> None:
    """Merge and untile into four preallocated peer slots; clear only transport tail rows."""
    import torch
    import triton
    if (len(destinations) != 4 or out.ndim != 4 or out.shape[0] != 1 or
            out.shape != gate.shape or out.dtype != torch.bfloat16 or gate.dtype != out.dtype or
            compressed.shape != (1, out.shape[1] // 64, out.shape[2], out.shape[3]) or
            compressed.dtype != out.dtype or out.shape[1] % 64 or
            mapping.dtype != torch.int64 or mapping.ndim != 1 or not mapping.is_contiguous()):
        raise FusionError("output fusion requires one BF16 tile64 batch and four peer slots")
    shard, heads, dim = destinations[0].shape
    if (mapping.numel() > 4 * shard or mapping.numel() <= 3 * shard or
            any(x.shape != (shard, heads, dim) or not x.is_contiguous() or
                x.dtype != out.dtype or x.device != out.device for x in destinations) or
            (heads, dim) != out.shape[2:] or gate.device != out.device or
            compressed.device != out.device or mapping.device != out.device):
        raise FusionError("output fusion slot geometry/device mismatch")
    merge, _ = kernels()
    merge[(triton.cdiv(shard * heads * dim, 512), 4)](
        out, gate, compressed, mapping, *destinations,
        mapping.numel(), shard, heads, dim,
        *out.stride()[1:], *gate.stride()[1:], *compressed.stride()[1:],
        BLOCK=512, num_warps=4, enable_fp_fusion=False)


def receive_scatter(sources: tuple[torch.Tensor, ...], mapping: torch.Tensor,
                    destination: torch.Tensor) -> None:
    """Read existing peer buffers once and scatter into a pre-zeroed device tile allocation."""
    import torch
    import triton
    if len(sources) != 4 or destination.ndim != 4 or not destination.is_contiguous():
        raise FusionError("input fusion requires four source buffers and contiguous tiles")
    inputs, padded, heads, dim = destination.shape
    shard = sources[0].shape[1]
    if (destination.dtype != torch.bfloat16 or padded % 64 or
            mapping.ndim != 1 or mapping.dtype != torch.int64 or not mapping.is_contiguous() or
            mapping.device != destination.device or mapping.numel() > 4 * shard or
            any(x.shape != (inputs, shard, heads, dim) or x.dtype != destination.dtype or
                x.device != destination.device or x.stride()[1:] != (heads * dim, dim, 1)
                for x in sources)):
        raise FusionError("input fusion source layout/device mismatch")
    _, scatter = kernels()
    scatter[(triton.cdiv(inputs * mapping.numel() * heads * dim, 1024),)](
        *sources, mapping, destination, mapping.numel(), shard, padded, heads, dim, inputs,
        *(x.stride(0) for x in sources), BLOCK=1024, num_warps=4)
