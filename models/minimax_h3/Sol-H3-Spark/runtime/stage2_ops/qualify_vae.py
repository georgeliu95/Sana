#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = ["torch"]
# ///
# ─── How to run ───
# Use the already pinned LTX CUDA environment; do not install another Torch build.
# PYTHONPATH=<campaign-parent>:<ltx-core/src> python -m torch.distributed.run \
#   --standalone --nproc_per_node=2 --module vae-candidate.qualify_gpu
# For two nodes use the benchmark's existing torchrun rendezvous arguments.
# ──────────────────
"""CUDA qualification using real Philox and upstream tile accumulation/blending."""
from __future__ import annotations

import json
import os
from collections.abc import Iterator
from contextlib import closing
from dataclasses import dataclass
from typing import TypedDict

import torch
import torch.distributed as dist
from ltx_core.model.video_vae.conv_video_decoder import ConvVideoDecoder, TilingConfig
from ltx_core.tiling import (
    DEFAULT_SPLIT_OPERATION, Tile, create_tiles, identity_mapping_operation,
    split_by_size,
)
from ltx_core.types import SpatioTemporalScaleFactors

from .parallel_vae import Receipt, install


@dataclass(frozen=True, slots=True)
class QualificationError(RuntimeError):
    detail: str

    def __str__(self) -> str:
        return self.detail


class FixtureDecoder(ConvVideoDecoder):
    """Use upstream tiling/blending with a small fresh-storage stochastic forward."""

    def __init__(self, *, complementary: bool) -> None:
        torch.nn.Module.__init__(self)
        self.video_downscale_factors = SpatioTemporalScaleFactors(1, 1, 1)
        self.complementary = complementary
        self.forward_calls = 0

    def _prepare_tiles(self, latent: torch.Tensor, tiling_config: TilingConfig | None = None) -> list[Tile]:
        splitters = [DEFAULT_SPLIT_OPERATION, DEFAULT_SPLIT_OPERATION,
                     split_by_size(5, 2), split_by_size(4, 1), split_by_size(5, 2)]
        tiles = create_tiles(latent.shape, splitters, [identity_mapping_operation] * 5)
        if self.complementary:
            return tiles
        return [tile._replace(masks_1d=(*tile.masks_1d[:2], tile.masks_1d[2] * 0.75,
                                        *tile.masks_1d[3:])) for tile in tiles]

    def forward(self, sample: torch.Tensor, timestep: torch.Tensor | None = None,
                generator: torch.Generator | None = None) -> torch.Tensor:
        self.forward_calls += 1
        noise = torch.randn(sample.shape, dtype=sample.dtype, device=sample.device, generator=generator)
        scalar = torch.randn((1,), dtype=sample.dtype, device=sample.device, generator=generator)
        return sample * 0.5 + noise * 0.125 + scalar * 0.03125


class CaseResult(TypedDict):
    seed: int
    initial_offset: int
    final_offset: int
    chunks: int
    output_bitwise_equal: bool
    final_rng_equal: bool


class SuiteResult(TypedDict):
    complementary: bool
    cases: list[CaseResult]
    receipt: Receipt


def collect(chunks: Iterator[torch.Tensor]) -> tuple[torch.Tensor, ...]:
    """Clone yielded buffers so later upstream overlap mutation cannot hide errors."""
    return tuple(chunk.clone() for chunk in chunks)


def require(condition: bool, detail: str) -> None:
    if not condition:
        raise QualificationError(detail)


def qualify(complementary: bool) -> SuiteResult:
    # Given: actual upstream blend and a real explicit CUDA Philox generator.
    decoder = FixtureDecoder(complementary=complementary)
    serial = decoder.tiled_decode
    original_forward = decoder.forward
    receipt = install(decoder)
    device = torch.device("cuda", torch.cuda.current_device())
    latent = torch.arange(1 * 3 * 7 * 6 * 7, device=device, dtype=torch.float32)
    latent = latent.reshape(1, 3, 7, 6, 7).to(torch.bfloat16) / 1024
    rng = torch.Generator(device=device).manual_seed(19)
    rng.set_offset(256)
    collect(decoder.tiled_decode(latent, generator=rng))
    require(receipt["calibrated_decodes"] == 1, "Complete warmup failed to commit plan")
    assignments = receipt["assignments"][0]
    owners = torch.tensor(assignments["owners"], device=device, dtype=torch.int64)
    all_owners = [torch.empty_like(owners) for _ in range(dist.get_world_size())]
    dist.all_gather(all_owners, owners)
    require(all(torch.equal(owners, peer) for peer in all_owners), "Rank assignments differ")
    results: list[CaseResult] = []
    for seed, offset in ((0, 0), (17, 123456), (991, 64)):
        # When: new requests change seed, offset, data and storage address.
        current = latent.clone() + seed / 1024
        rng.manual_seed(seed).set_offset(offset)
        expected = collect(serial(current, generator=rng))
        expected_state, expected_offset = rng.get_state(), rng.get_offset()
        rng.manual_seed(seed).set_offset(offset)
        calls_before = decoder.forward_calls
        actual = collect(decoder.tiled_decode(current, generator=rng))
        # Then: output chunks, full generator state and local work match serial.
        same = len(actual) == len(expected) and all(
            torch.equal(left, right) for left, right in zip(actual, expected, strict=True))
        same_rng = torch.equal(rng.get_state(), expected_state)
        require(same, f"Output mismatch for seed={seed}, offset={offset}")
        require(same_rng, f"Final RNG mismatch for seed={seed}, offset={offset}")
        require(decoder.forward_calls - calls_before == assignments["owners"].count(dist.get_rank()),
                "Steady decode executed tiles owned by another rank")
        require(decoder.forward == original_forward, "Forward was not restored after success")
        results.append({"seed": seed, "initial_offset": offset, "final_offset": expected_offset,
                        "chunks": len(actual), "output_bitwise_equal": same, "final_rng_equal": same_rng})
    return {"complementary": complementary, "cases": results, "receipt": receipt}


def qualify_early_close() -> None:
    # Given: three temporal groups, so the first yield precedes full calibration.
    decoder = FixtureDecoder(complementary=True)
    original_forward = decoder.forward
    receipt = install(decoder)
    device = torch.device("cuda", torch.cuda.current_device())
    latent = torch.zeros((1, 3, 9, 6, 7), dtype=torch.bfloat16, device=device)
    rng = torch.Generator(device=device).manual_seed(9)
    # When: a caller abandons the calibration generator after its first chunk.
    with closing(decoder.tiled_decode(latent, generator=rng)) as chunks:
        next(chunks)
    # Then: no partial plan survives; the forward is restored and retry can calibrate.
    require(receipt["calibrated_decodes"] == 0, "Partial calibration was committed")
    require(decoder.forward == original_forward, "Forward was not restored after close")
    rng.manual_seed(11).set_offset(128)
    collect(decoder.tiled_decode(latent, generator=rng))
    require(receipt["calibrated_decodes"] == 1, "Calibration retry failed")
    # Given: the same geometry now has a complete plan.
    # When: every rank closes a steady iterator at the same yield boundary.
    rng.manual_seed(13).set_offset(512)
    with closing(decoder.tiled_decode(latent.clone(), generator=rng)) as chunks:
        next(chunks)
    # Then: restoration happens, no completion is recorded, and retry succeeds.
    require(decoder.forward == original_forward, "Forward was not restored after steady close")
    require(receipt["completed_decodes"] == 0, "Partial steady decode counted as complete")
    require(rng.get_offset() == 512 + receipt["assignments"][0]["rng_total_advance"],
            "Eager steady close must retain the complete RNG advance")
    rng.manual_seed(17).set_offset(1024)
    collect(decoder.tiled_decode(latent.clone(), generator=rng))
    require(receipt["completed_decodes"] == 1, "Steady retry failed after close")


def main() -> None:
    """Run only when the parent scheduler explicitly assigns GPU resources."""
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", "0")))
    dist.init_process_group("nccl")
    try:
        with torch.inference_mode():
            suites = [qualify(True), qualify(False)]
            qualify_early_close()
        torch.cuda.synchronize()
        print(json.dumps({"rank": dist.get_rank(), "world": dist.get_world_size(),
                          "status": "PASS", "suites": suites, "early_close": "PASS"}), flush=True)
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
