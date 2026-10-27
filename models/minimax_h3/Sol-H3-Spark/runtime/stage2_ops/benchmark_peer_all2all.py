#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = ["torch"]
# ///
# ─── How to run ───
# Pinned two-node torchrun, same argv/env as benchmark_async_all2all.py.
# Copy peer_all2all.py into runtime/stage2_ops/ before running this script.
"""Paired timing of CP2 peer send, inverse gather and attention transfer cycles."""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from datetime import timedelta
from enum import StrEnum
import json
import os
from typing import assert_never

import torch
import torch.distributed as dist

from .benchmark_async_all2all import BenchConfig, Samples, measure_pair
from runtime.stage2_ops.nccl_all2all import NCCLAll2All, gather_heads, send_heads
from runtime.stage2_ops.peer_all2all import (
    PeerGroup, gather_heads_peer, gather_peer, send_heads_peer, send_peer,
)
from .verify_async_all2all import InputShape, make_inputs
from .verify_peer_all2all import verify_cases


class TransferKind(StrEnum):
    SEND = "send"
    GATHER = "gather"
    CYCLE = "qkv_send_plus_one_gather"


@dataclass(frozen=True, slots=True)
class TransferCase:
    kind: TransferKind
    head_dim: int
    tensors: int


@dataclass(frozen=True, slots=True)
class TransferResult:
    kind: TransferKind
    shape: InputShape
    payload_bytes_per_input: int
    outbound_bytes_per_rank: int
    samples: list[Samples]
    paired_peak_allocated_mib_by_rank: list[float]


def measure_case(case: TransferCase, config: BenchConfig) -> TransferResult:
    """Compare original custom ops directly, independent of integration flags."""
    shape = InputShape(1, config.tokens, 32, case.head_dim, case.tensors)
    match case.kind:
        case TransferKind.GATHER:
            shape = replace(shape, tokens=shape.tokens * 2, heads=shape.heads // 2)
        case TransferKind.SEND | TransferKind.CYCLE:
            pass
        case unreachable:
            assert_never(unreachable)
    values = make_inputs(shape, dist.get_rank(), torch.device("cuda", torch.cuda.current_device()))
    comm = NCCLAll2All(rank=dist.get_rank(), world_size=2, seqlen=config.tokens,
                      hidden_dim=32 * case.head_dim, num_sms=0, tensor_dtype=torch.bfloat16)
    comm.set_rank_tokens([config.tokens, config.tokens])

    def reference(inputs: list[torch.Tensor]) -> list[torch.Tensor]:
        match case.kind:
            case TransferKind.GATHER:
                return [gather_heads(value, comm.comm_id, 2) for value in inputs]
            case TransferKind.SEND:
                return [send_heads(value, comm.comm_id, 2) for value in inputs]
            case TransferKind.CYCLE:
                exchanged = [send_heads(value, comm.comm_id, 2) for value in inputs]
                return [gather_heads(exchanged[0], comm.comm_id, 2), *exchanged[1:]]
            case unreachable:
                assert_never(unreachable)

    def candidate(inputs: list[torch.Tensor]) -> list[torch.Tensor]:
        match case.kind:
            case TransferKind.GATHER:
                return [gather_heads_peer(value, comm.comm_id, 2) for value in inputs]
            case TransferKind.SEND:
                return [send_heads_peer(value, comm.comm_id, 2) for value in inputs]
            case TransferKind.CYCLE:
                exchanged = [send_heads_peer(value, comm.comm_id, 2) for value in inputs]
                return [gather_heads_peer(exchanged[0], comm.comm_id, 2), *exchanged[1:]]
            case unreachable:
                assert_never(unreachable)

    try:
        expected, actual = reference(values), candidate(values)
        for baseline, output in zip(expected, actual, strict=True):
            torch.testing.assert_close(output, baseline, atol=0, rtol=0)
        match case.kind:
            case TransferKind.GATHER:
                torch.library.opcheck(gather_heads_peer, (values[0], comm.comm_id, 2),
                                      test_utils=("test_schema", "test_faketensor"))
            case TransferKind.SEND | TransferKind.CYCLE:
                torch.library.opcheck(send_heads_peer, (values[0], comm.comm_id, 2),
                                      test_utils=("test_schema", "test_faketensor"))
                inverse = send_heads_peer(values[0], comm.comm_id, 2)
                restored = gather_heads_peer(inverse, comm.comm_id, 2)
                torch.testing.assert_close(restored, values[0], atol=0, rtol=0)
                del inverse, restored
            case unreachable:
                assert_never(unreachable)
        del expected, actual
        exchanges = (reference, candidate)
        if config.compile:
            exchanges = (torch.compile(reference, fullgraph=True, dynamic=False),
                         torch.compile(candidate, fullgraph=True, dynamic=False))
            expected, actual = exchanges[0](values), exchanges[1](values)
            for baseline, output in zip(expected, actual, strict=True):
                torch.testing.assert_close(output, baseline, atol=0, rtol=0)
            del expected, actual
        samples, peaks = measure_pair(exchanges, values, config)
        renamed = [replace(sample, label="peer_split" if sample.label == "async_separate"
                           else sample.label) for sample in samples]
        payload = values[0].numel() * values[0].element_size()
        calls = case.tensors + int(case.kind == TransferKind.CYCLE)
        return TransferResult(case.kind, shape, payload, calls * payload // 2, renamed, peaks)
    finally:
        comm.destroy()


def main() -> None:
    """Initialize exactly CP2, run exact oracles, then time warmed serial requests."""
    config = BenchConfig(int(os.environ.get("COMM_TOKENS", "6048")),
                         int(os.environ.get("COMM_WARMUP", "10")),
                         int(os.environ.get("COMM_REPEATS", "50")),
                         os.environ.get("COMM_COMPILE", "0") == "1")
    if config.tokens < 1 or config.warmup < 1 or config.repeats < 5:
        raise RuntimeError(config)
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("nccl", timeout=timedelta(seconds=180))
    try:
        group = PeerGroup.from_initialized(None)
        cases = verify_cases(group, torch.device("cuda", torch.cuda.current_device()),
                             (send_peer, gather_peer))
        if dist.get_rank() == 0:
            print(json.dumps({"status": "VERIFIED_PEER_SMALL", "cases": cases, "world": 2,
                              "torch": torch.__version__, "cuda": torch.version.cuda,
                              "nccl": torch.cuda.nccl.version(), "config": asdict(config),
                              "transport_env": {key: os.environ.get(key) for key in (
                                  "NCCL_SOCKET_IFNAME", "GLOO_SOCKET_IFNAME", "NCCL_IB_HCA",
                                  "NCCL_DEBUG", "NCCL_DEBUG_SUBSYS", "NCCL_IB_DISABLE")}}), flush=True)
        for head_dim in (128, 64):
            for case in (TransferCase(TransferKind.SEND, head_dim, 3),
                         TransferCase(TransferKind.SEND, head_dim, 2),
                         TransferCase(TransferKind.GATHER, head_dim, 1),
                         TransferCase(TransferKind.CYCLE, head_dim, 3)):
                result = measure_case(case, config)
                if dist.get_rank() == 0:
                    print(json.dumps({"status": "MEASURED_PEER", **asdict(result)}), flush=True)
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
