# /// script
# requires-python = ">=3.12"
# dependencies = ["torch"]
# ///
# ─── How to run ───
# Imported by the pinned LTX runtime; see verify_async_all2all.py and README.md.
"""Overlap packing with separate NCCL exchanges; retain every in-flight buffer."""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.distributed as dist


@dataclass(frozen=True, slots=True)
class ExchangeGroup:
    """The already-initialized process group and its uniform partition count."""
    group: dist.ProcessGroup | None
    world: int


@dataclass(frozen=True, slots=True)
class PendingTransfer:
    """Keep send, receive and work alive until all consuming-stream waits are issued."""
    packed: torch.Tensor
    received: torch.Tensor
    work: dist.Work


@dataclass(frozen=True, slots=True)
class AsyncLaunchError(RuntimeError):
    """The distributed backend violated the async-op handle contract."""
    detail: str

    def __str__(self) -> str:
        return self.detail


def exchange_separate_async(values: list[torch.Tensor],
                            group: ExchangeGroup) -> list[torch.Tensor]:
    """Issue the same ordered collectives, allowing later packs before earlier waits."""
    pending: list[PendingTransfer] = []
    for value in values:
        batch, tokens, heads, head_dim = value.shape
        packed = value.reshape(batch, tokens, group.world, heads // group.world, head_dim)
        packed = packed.permute(2, 0, 1, 3, 4).contiguous()
        received = torch.empty_like(packed)
        work = dist.all_to_all_single(received, packed, group=group.group, async_op=True)
        if work is None:
            raise AsyncLaunchError("async_op=True returned no distributed Work handle")
        pending.append(PendingTransfer(packed, received, work))
    outputs: list[torch.Tensor] = []
    for transfer in pending:
        # On CUDA this orders consumption after NCCL, without a device-wide sync.
        transfer.work.wait()
        world, batch, tokens, local_heads, head_dim = transfer.received.shape
        output = transfer.received.permute(1, 0, 2, 3, 4)
        outputs.append(output.reshape(batch, world * tokens, local_heads, head_dim).contiguous())
    return outputs


@torch.library.custom_op("spark_cp::send_heads_async", mutates_args=())
def send_heads_async(values: list[torch.Tensor], comm_id: int, world: int) -> list[torch.Tensor]:
    """Compile-compatible KV/QKV exchange, with the frozen communicator's validation."""
    from runtime.stage2_ops.nccl_all2all import REGISTRY, TransferContractError

    comm = REGISTRY[comm_id]
    if len(values) not in (2, 3) or world != comm.world_size:
        raise TransferContractError("Async exchange requires KV/QKV and the initialized world")
    shape = values[0].shape
    if len(shape) != 4:
        raise TransferContractError("Async exchange requires BTHD tensors")
    for value in values:
        if value.shape != shape:
            raise TransferContractError("Async exchange tensors must share their BTHD shape")
        comm.check(value, shape[1], shape[2])
    with torch.cuda.nvtx.range(f"Spark/LTX/communication/AllToAll/async{len(values)}"):
        return exchange_separate_async(values, ExchangeGroup(comm.group, world))


@send_heads_async.register_fake
def send_heads_async_fake(values: list[torch.Tensor], comm_id: int,
                          world: int) -> list[torch.Tensor]:
    """Each output has all tokens and only this rank's heads, with fresh storage."""
    return [value.new_empty((value.shape[0], value.shape[1] * world,
                            value.shape[2] // world, value.shape[3])) for value in values]
