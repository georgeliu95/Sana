# /// script
# requires-python = ">=3.12"
# dependencies = ["torch"]
# ///
# ─── How to run ───
# Imported by pinned runtime; verify_peer_all2all.py supplies the exact oracle.
"""CP2/B1 redistribution that places the local half directly in the final output."""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.distributed as dist


@dataclass(frozen=True, slots=True)
class PeerContractError(RuntimeError):
    """The candidate only supports the fixed CP2/B1 inference geometry."""
    detail: str

    def __str__(self) -> str:
        return self.detail


@dataclass(frozen=True, slots=True)
class PeerGroup:
    """Local rank and global peer id for the same initialized two-rank group."""
    group: dist.ProcessGroup | None
    rank: int
    global_peer: int

    @classmethod
    def from_initialized(cls, group: dist.ProcessGroup | None) -> PeerGroup:
        if dist.get_world_size(group) != 2:
            raise PeerContractError("Peer exchange requires exactly two ranks")
        rank = dist.get_rank(group)
        peer = 1 - rank if group is None else dist.get_global_rank(group, 1 - rank)
        return cls(group, rank, peer)


def exchange_pair(send: torch.Tensor, receive: torch.Tensor, group: PeerGroup) -> list[dist.Work]:
    """Post matched P2P pairs together; callers retain both tensors through wait."""
    ops = [dist.P2POp(dist.isend, send, group.global_peer, group.group),
           dist.P2POp(dist.irecv, receive, group.global_peer, group.group)]
    if group.rank == 1:
        ops.reverse()
    return dist.batch_isend_irecv(ops)


def send_peer(value: torch.Tensor, group: PeerGroup) -> torch.Tensor:
    """Pack remote heads only; receive directly into the output's remote token range."""
    batch, tokens, heads, head_dim = value.shape
    if batch != 1 or heads % 2:
        raise PeerContractError("Send specialization requires B=1 and even heads")
    local, peer, half = group.rank, 1 - group.rank, heads // 2
    output = value.new_empty((1, tokens * 2, half, head_dim))
    packed = value[:, :, peer * half:(peer + 1) * half].contiguous()
    received = output[:, peer * tokens:(peer + 1) * tokens]
    works = exchange_pair(packed, received, group)
    output[:, local * tokens:(local + 1) * tokens].copy_(
        value[:, :, local * half:(local + 1) * half])
    for work in works:
        work.wait()
    return output


def gather_peer(value: torch.Tensor, group: PeerGroup) -> torch.Tensor:
    """Send remote tokens directly and write each head half into the final layout."""
    batch, total, heads, head_dim = value.shape
    if batch != 1 or total % 2:
        raise PeerContractError("Gather specialization requires B=1 and even tokens")
    local, peer, tokens = group.rank, 1 - group.rank, total // 2
    output = value.new_empty((1, tokens, heads * 2, head_dim))
    packed = value[:, peer * tokens:(peer + 1) * tokens].contiguous()
    received = torch.empty_like(packed)
    works = exchange_pair(packed, received, group)
    output[:, :, local * heads:(local + 1) * heads].copy_(
        value[:, local * tokens:(local + 1) * tokens])
    for work in works:
        work.wait()
    output[:, :, peer * heads:(peer + 1) * heads].copy_(received)
    return output


@torch.library.custom_op("spark_cp::send_heads_peer", mutates_args=())
def send_heads_peer(value: torch.Tensor, comm_id: int, world: int) -> torch.Tensor:
    """Original inference checks plus a CP2/B1-only send specialization."""
    from runtime.stage2_ops.nccl_all2all import REGISTRY

    comm = REGISTRY[comm_id]
    if world != 2 or world != comm.world_size:
        raise PeerContractError("Peer send custom op requires the initialized CP2 group")
    comm.check(value, value.shape[1], value.shape[2])
    with torch.cuda.nvtx.range("Spark/LTX/communication/Peer/send_heads"):
        return send_peer(value, PeerGroup.from_initialized(comm.group))


@send_heads_peer.register_fake
def send_heads_peer_fake(value: torch.Tensor, comm_id: int, world: int) -> torch.Tensor:
    return value.new_empty((value.shape[0], value.shape[1] * world,
                            value.shape[2] // world, value.shape[3]))


@torch.library.custom_op("spark_cp::gather_heads_peer", mutates_args=())
def gather_heads_peer(value: torch.Tensor, comm_id: int, world: int) -> torch.Tensor:
    """Original inference checks plus a CP2/B1-only inverse specialization."""
    from runtime.stage2_ops.nccl_all2all import REGISTRY

    comm = REGISTRY[comm_id]
    if world != 2 or world != comm.world_size:
        raise PeerContractError("Peer gather custom op requires the initialized CP2 group")
    comm.check(value, value.shape[1] // world, value.shape[2] * world)
    with torch.cuda.nvtx.range("Spark/LTX/communication/Peer/gather_heads"):
        return gather_peer(value, PeerGroup.from_initialized(comm.group))


@gather_heads_peer.register_fake
def gather_heads_peer_fake(value: torch.Tensor, comm_id: int, world: int) -> torch.Tensor:
    return value.new_empty((value.shape[0], value.shape[1] // world,
                            value.shape[2] * world, value.shape[3]))
