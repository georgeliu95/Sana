"""Fuse VSA output untile, transport padding and original NCCL send layout."""
from __future__ import annotations
from dataclasses import dataclass
import torch
import torch.distributed as dist
from torch import Tensor
import triton
import triton.language as tl


@dataclass(frozen=True, slots=True)
class TransportContractError(RuntimeError):
    """Reject shapes outside the packed BF16 inference contract."""
    detail: str


@triton.jit
def _pack_tiled_output(X, INDEX, PACKED, N: tl.constexpr, S: tl.constexpr,
                       H: tl.constexpr, D: tl.constexpr, T: tl.constexpr,
                       XS: tl.constexpr, XH: tl.constexpr, XD: tl.constexpr,
                       BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    valid = offsets < N
    d = offsets % D
    seq = offsets // D % S
    head = offsets // (D * S) % H
    peer = offsets // (D * S * H)
    global_seq = peer * S + seq
    tile_seq = tl.load(INDEX + global_seq, valid & (global_seq < T), other=0)
    value = tl.load(X + tile_seq * XS + head * XH + d * XD,
                    valid & (global_seq < T), other=0)
    tl.store(PACKED + offsets, value, valid)


def exchange_tiled(x: Tensor, index: Tensor, world: int,
                   group: dist.ProcessGroup | None) -> Tensor:
    """Match the native [dest,head,local-token,batch=1,dim] wire layout."""
    if x.ndim != 4 or x.shape[0] != 1 or x.dtype != torch.bfloat16 or world not in (2, 4):
        raise TransportContractError("Expected CP2/CP4 batch=1 BF16 VSA output")
    tokens = index.numel()
    shard = (tokens + world - 1) // world
    heads, dim = x.shape[2:]
    packed = torch.empty((world * heads, shard, 1, dim), device=x.device, dtype=x.dtype)
    received = torch.empty_like(packed)
    with torch.cuda.nvtx.range("Spark/H3/communication/untile_pad_pack"):
        _pack_tiled_output[(triton.cdiv(packed.numel(), 1024),)](
            x, index, packed, packed.numel(), shard, heads, dim, tokens,
            x.stride(1), x.stride(2), x.stride(3), 1024)
    with torch.cuda.nvtx.range("Spark/H3/communication/output_AllToAll"):
        dist.all_to_all_single(received, packed, group=group)
    with torch.cuda.nvtx.range("Spark/H3/communication/output_unpack"):
        return received.transpose(0, 2).contiguous()


@torch.library.custom_op("spark_h3_cp::reverse_tiled", mutates_args=())
def reverse_tiled_exchange(x: Tensor, index: Tensor, world: int) -> Tensor:
    """Keep the original SP process group and return independent output storage."""
    from fastvideo.distributed.parallel_state import get_sp_group
    return exchange_tiled(x, index, world, get_sp_group().device_group)


@reverse_tiled_exchange.register_fake
def fake_reverse(x: Tensor, index: Tensor, world: int) -> Tensor:
    """Describe the same output for the existing fullgraph regional compiler."""
    return x.new_empty((1, (index.numel() + world - 1) // world, x.shape[2] * world, x.shape[3]))
