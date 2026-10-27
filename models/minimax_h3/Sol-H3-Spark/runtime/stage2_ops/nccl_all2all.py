"""Cross-host BTHD redistribution for the original LTX sequence-parallel API."""
from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
import os
from typing import Final, TypedDict

import torch
import torch.distributed as dist

from .async_all2all import send_heads_async
from .peer_all2all import gather_heads_peer, send_heads_peer


@dataclass(frozen=True, slots=True)
class TransferContractError(RuntimeError):
    """An unsupported input must fail before entering a collective."""
    detail: str

    def __str__(self) -> str:
        return self.detail


class Receipt(TypedDict):
    backend: str
    world_size: int
    instances: int
    factory_restored: bool


REGISTRY: Final[dict[int, "NCCLAll2All"]] = {}


@torch.library.custom_op("spark_cp::send_heads", mutates_args=())
def send_heads(x: torch.Tensor, comm_id: int, world: int) -> torch.Tensor:
    """Redistribute local tokens and all heads into all tokens and local heads."""
    comm = REGISTRY[comm_id]
    b, t, h, d = x.shape
    comm.check(x, t, h)
    with torch.cuda.nvtx.range("Spark/LTX/communication/AllToAll/send_heads"):
        packed = x.reshape(b, t, world, h // world, d).permute(2, 0, 1, 3, 4).contiguous()
        received = torch.empty_like(packed)
        dist.all_to_all_single(received, packed, group=comm.group)
        return received.permute(1, 0, 2, 3, 4).reshape(b, world * t, h // world, d).contiguous()


@send_heads.register_fake
def send_heads_fake(x: torch.Tensor, comm_id: int, world: int) -> torch.Tensor:
    return x.new_empty((x.shape[0], x.shape[1] * world, x.shape[2] // world, x.shape[3]))


@torch.library.custom_op("spark_cp::gather_heads", mutates_args=())
def gather_heads(x: torch.Tensor, comm_id: int, world: int) -> torch.Tensor:
    """Invert the token/head exchange without changing values or dtype."""
    comm = REGISTRY[comm_id]
    b, total, h, d = x.shape
    comm.check(x, total // world, h * world)
    with torch.cuda.nvtx.range("Spark/LTX/communication/AllToAll/gather_heads"):
        packed = x.reshape(b, world, total // world, h, d).permute(1, 0, 2, 3, 4).contiguous()
        received = torch.empty_like(packed)
        dist.all_to_all_single(received, packed, group=comm.group)
        return received.permute(1, 2, 0, 3, 4).reshape(b, total // world, world * h, d).contiguous()


@gather_heads.register_fake
def gather_heads_fake(x: torch.Tensor, comm_id: int, world: int) -> torch.Tensor:
    return x.new_empty((x.shape[0], x.shape[1] // world, x.shape[2] * world, x.shape[3]))


class NCCLAll2All:
    """Own mutable sequence metadata; NCCL owns transport and watchdog timeout."""

    def __init__(self, rank: int, world_size: int, seqlen: int, hidden_dim: int,
                 num_sms: int, tensor_dtype: torch.dtype,
                 group: dist.ProcessGroup | None = None,
                 timeout_seconds: float | None = None) -> None:
        if world_size != dist.get_world_size(group) or world_size not in (1, 2, 4):
            raise TransferContractError("All2All requires the initialized CP1, CP2, or CP4 group")
        self.rank, self.world_size = rank, world_size
        self.seqlen, self.hidden_dim = seqlen, hidden_dim
        self.tensor_dtype, self.group = tensor_dtype, group
        self.rank_tokens: int | None = None
        self.comm_id = len(REGISTRY)
        self.timeout_seconds = timeout_seconds
        self.peer_exchange = os.environ.get("SPARK_CP_PEER_EXCHANGE", "0") == "1"
        REGISTRY[self.comm_id] = self

    def set_rank_tokens(self, rank_num_tokens: list[int]) -> None:
        if (len(rank_num_tokens) != self.world_size or len(set(rank_num_tokens)) != 1
                or not 0 < rank_num_tokens[0] <= self.seqlen):
            raise TransferContractError("LTX NCCL transport requires uniform positive token shards")
        self.rank_tokens = rank_num_tokens[0]

    def set_timeout_seconds(self, seconds: float) -> None:
        self.timeout_seconds = seconds

    def check(self, x: torch.Tensor, tokens: int, heads: int) -> None:
        if (self.rank_tokens != tokens or heads % self.world_size
                or x.ndim != 4 or x.dtype != self.tensor_dtype
                or x.device.type != "cuda" or x.requires_grad):
            raise TransferContractError("Input violates the BF16 inference BTHD shard contract")

    def send_recv_heads(self, x: torch.Tensor, *, copy_out: bool = False) -> torch.Tensor:
        if self.world_size == 1:
            self.check(x, x.shape[1], x.shape[2])
            return x.clone() if copy_out else x
        if self.peer_exchange and self.world_size == 2 and x.shape[0] == 1:
            return send_heads_peer(x, self.comm_id, self.world_size)
        return send_heads(x, self.comm_id, self.world_size)

    def gather_heads(self, x: torch.Tensor, *, copy_out: bool = False) -> torch.Tensor:
        if self.world_size == 1:
            self.check(x, x.shape[1], x.shape[2])
            return x.clone() if copy_out else x
        if self.peer_exchange and self.world_size == 2 and x.shape[0] == 1:
            return gather_heads_peer(x, self.comm_id, self.world_size)
        return gather_heads(x, self.comm_id, self.world_size)

    def send_recv_async(self, values: list[torch.Tensor], *, copy_out: bool = False) -> list[torch.Tensor]:
        if self.world_size == 1:
            for value in values:
                self.check(value, value.shape[1], value.shape[2])
            return [value.clone() for value in values] if copy_out else values
        return send_heads_async(values, self.comm_id, self.world_size)

    def destroy(self) -> None:
        del REGISTRY[self.comm_id]


@contextmanager
def nccl_all2all_factory() -> Iterator[Receipt]:
    """Scope constructor substitution; keep the upstream attention and sampling."""
    import ltx_kernels
    original = ltx_kernels.All2All
    before = len(REGISTRY)
    receipt: Receipt = {"backend": "NCCL", "world_size": dist.get_world_size(),
                        "instances": 0, "factory_restored": False}
    ltx_kernels.All2All = NCCLAll2All
    try:
        yield receipt
    finally:
        receipt["instances"] = len(REGISTRY) - before
        ltx_kernels.All2All = original
        receipt["factory_restored"] = True
