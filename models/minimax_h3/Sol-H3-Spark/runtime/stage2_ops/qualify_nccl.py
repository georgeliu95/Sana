"""Numerical qualification of real multi-host NCCL token/head redistribution."""
from __future__ import annotations
import json
from pathlib import Path

import torch
import torch.distributed as dist
from torch.nn.functional import scaled_dot_product_attention

from .nccl_all2all import NCCLAll2All


def qualify(path: Path) -> None:
    """Given global BF16 inputs, prove sharded attention matches the full input."""
    rank, world = dist.get_rank(), dist.get_world_size()
    rows = []
    for dim in (64, 128):
        # Given: all ranks derive exactly the same nontrivial, deterministic input.
        full = torch.arange(world * 7 * 32 * dim, device="cuda", dtype=torch.float32)
        full = (full.mul(0.017).sin()).reshape(1, world * 7, 32, dim).to(torch.bfloat16)
        local = full[:, rank * 7:(rank + 1) * 7].contiguous()
        comm = NCCLAll2All(rank, world, 7, 32 * dim, 1, torch.bfloat16)
        comm.set_rank_tokens([7] * world)
        # When: use the actual transport, then use its inverse.
        sharded = comm.send_recv_heads(local)
        restored = comm.gather_heads(sharded)
        # Then: independently selected global heads and original tokens are exact.
        reference = full[:, :, rank * (32 // world):(rank + 1) * (32 // world)]
        torch.testing.assert_close(sharded, reference, atol=0, rtol=0)
        torch.testing.assert_close(restored, local, atol=0, rtol=0)
        # When: apply the real SDPA to redistributed heads and gather its output.
        view = sharded.transpose(1, 2)
        actual = comm.gather_heads(scaled_dot_product_attention(view, view, view).transpose(1, 2).contiguous())
        global_view = full.transpose(1, 2)
        expected = scaled_dot_product_attention(global_view, global_view, global_view)
        expected = expected.transpose(1, 2)[:, rank * 7:(rank + 1) * 7]
        # Then: compare against an independently evaluated unsharded attention.
        torch.testing.assert_close(actual, expected, atol=0.02, rtol=0.02)
        rows.append({"head_dim": dim, "transport_max_abs": 0.0,
                     "attention_max_abs": (actual.float() - expected.float()).abs().max().item()})
        comm.destroy()
    torch.cuda.synchronize()
    path.write_text(json.dumps({"status": "PASS", "rank": rank, "world_size": world,
                               "dtype": "bfloat16", "cases": rows}, indent=2) + "\n")
