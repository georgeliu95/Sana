"""Global original-tile VAE partitioning with calibrated Philox offsets."""
from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from math import prod
from typing import TypedDict

import torch
import torch.distributed as dist
from ltx_core.model.video_vae.conv_video_decoder import ConvVideoDecoder, TilingConfig
from ltx_core.tiling import group_tiles_by_temporal_slice

from .tile_assignment import assign_lpt


@dataclass(frozen=True, slots=True)
class VaePartitionError(RuntimeError):
    detail: str

    def __str__(self) -> str:
        return self.detail


@dataclass(frozen=True, slots=True)
class TileRng:
    offset: int
    advance: int
    shape: tuple[int, ...]
    dtype: torch.dtype


@dataclass(frozen=True, slots=True)
class DecodePlan:
    tiles: tuple[TileRng, ...]
    owners: tuple[int, ...]
    advance: int


class Assignment(TypedDict):
    owners: list[int]
    output_elements: list[int]
    rank_output_elements: list[int]
    rng_offsets: list[int]
    rng_advances: list[int]
    rng_total_advance: int
    output_shapes: list[list[int]]
    input_slices: list[list[list[int | None]]]


class Receipt(TypedDict):
    world: int
    rank: int
    calibrated_decodes: int
    calibration_tiles: int
    completed_decodes: int
    local_tiles: int
    broadcast_tiles: int
    assignments: list[Assignment]


def install(decoder: ConvVideoDecoder) -> Receipt:
    """Wrap tiled_decode; first complete call calibrates, later calls use global LPT.

    This instance is single-consumer. All WORLD ranks must fully consume each
    decode with matching geometry, seed and initial explicit CUDA RNG offset.
    The original forward must return storage that outlives subsequent forwards.
    """
    rank, world = dist.get_rank(), dist.get_world_size()
    forward, tiled_decode = decoder.forward, decoder.tiled_decode
    plans: dict[str, DecodePlan] = {}
    active = False
    receipt: Receipt = {"world": world, "rank": rank, "calibrated_decodes": 0,
                        "calibration_tiles": 0, "completed_decodes": 0,
                        "local_tiles": 0, "broadcast_tiles": 0, "assignments": []}

    def distributed_decode(
        latent: torch.Tensor, tiling_config: TilingConfig | None = None,
        timestep: torch.Tensor | None = None, generator: torch.Generator | None = None,
    ) -> Iterator[torch.Tensor]:
        nonlocal active
        if active:
            raise VaePartitionError("The VAE partition wrapper cannot be re-entered")
        if generator is None or generator.device.type != "cuda":
            raise VaePartitionError("VAE partitioning requires an explicit CUDA generator")
        groups = group_tiles_by_temporal_slice(decoder._prepare_tiles(latent, tiling_config))
        tiles = tuple(tile for group in groups for tile in group)
        slices = tuple(tuple((axis.start, axis.stop, axis.step) for axis in tile.in_coords)
                       for tile in tiles)
        key = repr((tuple(latent.shape), latent.stride(), latent.dtype, slices))
        origin = generator.get_offset()
        plan = plans.get(key)
        active = True
        try:
            if plan is None:
                observed: list[TileRng] = []

                def record(sample: torch.Tensor, step: torch.Tensor | None = None,
                           rng: torch.Generator | None = None) -> torch.Tensor:
                    index = len(observed)
                    if index >= len(tiles) or rng is not generator:
                        raise VaePartitionError("Calibration forward count or generator mismatch")
                    expected = latent[tiles[index].in_coords]
                    if (sample.data_ptr(), sample.shape, sample.stride()) != (
                        expected.data_ptr(), expected.shape, expected.stride()
                    ):
                        raise VaePartitionError("Original decode changed global tile order")
                    before = generator.get_offset()
                    output = forward(sample, step, rng)
                    observed.append(TileRng(before - origin, generator.get_offset() - before,
                                            tuple(output.shape), output.dtype))
                    return output

                decoder.forward = record
                yield from tiled_decode(latent, tiling_config, timestep, generator)
                if len(observed) != len(tiles) or not observed:
                    raise VaePartitionError("Calibration requires exactly one forward per tile")
                advance = generator.get_offset() - origin
                if sum(item.advance for item in observed) != advance:
                    raise VaePartitionError("RNG was consumed outside original tile forwards")
                costs = tuple(prod(item.shape) for item in observed)
                owners = assign_lpt(costs, world)
                plans[key] = DecodePlan(tuple(observed), owners, advance)
                receipt["calibrated_decodes"] += 1
                receipt["calibration_tiles"] += len(tiles)
                receipt["assignments"].append({
                    "owners": list(owners), "output_elements": list(costs),
                    "rank_output_elements": [sum(cost for cost, owner in zip(costs, owners, strict=True)
                                                 if owner == peer) for peer in range(world)],
                    "rng_offsets": [item.offset for item in observed],
                    "rng_advances": [item.advance for item in observed],
                    "rng_total_advance": advance,
                    "output_shapes": [list(item.shape) for item in observed],
                    "input_slices": [[list(axis) for axis in tile] for tile in slices],
                })
                return

            outputs: dict[int, torch.Tensor] = {}
            for index, (tile, rng_plan, owner) in enumerate(zip(tiles, plan.tiles, plan.owners, strict=True)):
                if owner != rank:
                    continue
                generator.set_offset(origin + rng_plan.offset)
                with torch.cuda.nvtx.range(f"Spark/LTX/ConvVAE/global_tile/{index}/owner{rank}"):
                    output = forward(latent[tile.in_coords], timestep, generator)
                    outputs[index] = output.contiguous()
                if generator.get_offset() != origin + rng_plan.offset + rng_plan.advance:
                    raise VaePartitionError("Decoder RNG consumption changed after calibration")
                if tuple(output.shape) != rng_plan.shape or output.dtype != rng_plan.dtype:
                    raise VaePartitionError("Decoder output metadata changed after calibration")
                del output
                receipt["local_tiles"] += 1
            generator.set_offset(origin + plan.advance)
            index = 0

            def deliver(sample: torch.Tensor, step: torch.Tensor | None = None,
                        rng: torch.Generator | None = None) -> torch.Tensor:
                nonlocal index
                if index >= len(tiles) or rng is not generator:
                    raise VaePartitionError("Original decode changed forward count or generator")
                expected = latent[tiles[index].in_coords]
                if (sample.data_ptr(), sample.shape, sample.stride()) != (
                    expected.data_ptr(), expected.shape, expected.stride()
                ):
                    raise VaePartitionError("Original decode changed delivery tile order")
                owner, rng_plan = plan.owners[index], plan.tiles[index]
                output = outputs.pop(index) if owner == rank else torch.empty(
                    rng_plan.shape, device=sample.device, dtype=rng_plan.dtype)
                with torch.cuda.nvtx.range(f"Spark/LTX/communication/VAE_tile_broadcast/{index}/owner{owner}"):
                    dist.broadcast(output, src=owner)
                receipt["broadcast_tiles"] += 1
                index += 1
                return output

            decoder.forward = deliver
            yield from tiled_decode(latent, tiling_config, timestep, generator)
            if index != len(tiles) or outputs:
                raise VaePartitionError("Original decode did not consume all precomputed tiles")
            if generator.get_offset() != origin + plan.advance:
                raise VaePartitionError("RNG was consumed outside original tile forwards")
            receipt["completed_decodes"] += 1
        finally:
            decoder.forward = forward
            active = False

    decoder.tiled_decode = distributed_decode
    return receipt
