#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = ["torch"]
# ///
# ─── How to run ───
# Pinned Torch: torchrun --standalone --nproc-per-node=2 verify_peer_all2all.py
"""Independent send/gather layout oracles for the CP2/B1 peer specialization."""
from __future__ import annotations

from collections.abc import Callable
from datetime import timedelta
import json
import os
from typing import TYPE_CHECKING, TypeAlias

import torch
import torch.distributed as dist

from .verify_async_all2all import InputShape, make_inputs

if TYPE_CHECKING:
    from .peer_all2all import PeerGroup

PeerOperation: TypeAlias = Callable[[torch.Tensor, "PeerGroup"], torch.Tensor]


def expected_send(value: torch.Tensor, group: PeerGroup) -> torch.Tensor:
    """Build global tokens independently, then select this rank's head interval."""
    source = value.contiguous()
    sources = [torch.empty_like(source) for _ in range(2)]
    dist.all_gather(sources, source, group=group.group)
    half = value.shape[2] // 2
    return torch.cat(sources, dim=1)[:, :, group.rank * half:(group.rank + 1) * half].contiguous()


def expected_gather(value: torch.Tensor, group: PeerGroup) -> torch.Tensor:
    """Build all heads independently, then select this rank's token interval."""
    source = value.contiguous()
    sources = [torch.empty_like(source) for _ in range(2)]
    dist.all_gather(sources, source, group=group.group)
    half = value.shape[1] // 2
    return torch.cat(sources, dim=2)[:, group.rank * half:(group.rank + 1) * half].contiguous()


def verify_cases(group: PeerGroup, device: torch.device,
                 operations: tuple[PeerOperation, PeerOperation]) -> int:
    """Use unrelated gather inputs so inverse mistakes cannot cancel in roundtrip."""
    send, gather = operations
    cases = 0
    for head_dim in (64, 128):
        for strided in (False, True):
            for count in (2, 3):
                shape = InputShape(1, 7, 8, head_dim, count, strided)
                values = make_inputs(shape, group.rank, device)
                for value in values:
                    before = value.clone()
                    expected = expected_send(value, group)
                    actual = send(value, group)
                    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
                    torch.testing.assert_close(gather(actual, group), value, atol=0, rtol=0)
                    torch.testing.assert_close(value, before, atol=0, rtol=0)
                inverse = InputShape(1, 14, 4, head_dim, 1, strided)
                value = make_inputs(inverse, group.rank, device)[0]
                before = value.clone()
                torch.testing.assert_close(gather(value, group), expected_gather(value, group),
                                           atol=0, rtol=0)
                torch.testing.assert_close(value, before, atol=0, rtol=0)
                cases += 1
    return cases


def main() -> None:
    """CPU Gloo by default; use VERIFY_BACKEND=nccl only for coordinated GPU work."""
    from .peer_all2all import PeerGroup, gather_peer, send_peer

    backend = os.environ.get("VERIFY_BACKEND", "gloo")
    device = torch.device("cpu")
    if backend == "nccl":
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
        device = torch.device("cuda", torch.cuda.current_device())
    dist.init_process_group(backend, timeout=timedelta(seconds=120))
    try:
        cases = verify_cases(PeerGroup.from_initialized(None), device, (send_peer, gather_peer))
        if dist.get_rank() == 0:
            print(json.dumps({"status": "PASS_PEER", "cases": cases, "backend": backend,
                              "torch": torch.__version__, "absolute_error": 0}), flush=True)
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
