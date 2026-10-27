#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = ["torch", "pydantic>=2"]
# ///
# ─── How to run ───
# Use the pinned production environment, overlay mounts and torchrun rendezvous.
# SPARK_LTX_V2_MODE=norm SPARK_PP_PATHS=<paths.json> SPARK_PP_RESULT_ROOT=<run>/pipeline \
#   python -m torch.distributed.run ... --module runtime.stage2_ops.qualify_real_vae
# Or call qualify(paths_file, output_path) from an already initialized qualify_entry.
# ──────────────────
"""Qualify actual original VAE tiles against serial on fixed full-size latents."""
from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
import gc
import hashlib
import json
import os
from pathlib import Path
import statistics
from typing import TYPE_CHECKING, Final, TypedDict

import torch
import torch.distributed as dist

from .real_vae_measure import Decode, ErrorMetrics, Sample, capture, compare, measured
from .real_vae_setup import LATENT_SHAPE, Provenance, loaded_vae, require

if TYPE_CHECKING:
    from .parallel_vae import Receipt


@dataclass(frozen=True, slots=True)
class InputCase:
    latent_seed: int
    decoder_seed: int
    initial_offset: int


CASES: Final = (InputCase(71, 0, 0), InputCase(131, 17, 123456))


class CaseResult(TypedDict):
    case: int
    latent_seed: int
    latent_sha256: str
    decoder_seed: int
    initial_offset: int
    expected_final_offset: int
    actual_final_offset: int
    errors: ErrorMetrics


class Pair(TypedDict):
    case: int
    repeat: int
    order: list[str]
    serial: Sample
    parallel: Sample


class TimingSummary(TypedDict):
    case: int
    variant: str
    n: int
    max_rank_wall_mean_s: float
    max_rank_wall_stdev_s: float
    max_rank_peak_allocated_bytes: int
    max_rank_peak_reserved_bytes: int


class Report(TypedDict):
    status: str
    rank: int
    world: int
    batch: int
    repeats_per_case_per_arm: int
    warmup_excluded: bool
    timing_includes_output_clones: bool
    allocator_cache_retained: bool
    provenance: Provenance
    cases: list[CaseResult]
    pairs: list[Pair]
    timing_summary: list[TimingSummary]
    receipt: Receipt


def qualify(paths_file: Path, output: Path, repeats: int = 5) -> None:
    """Require an initialized WORLD group; load only VAE and write rank-specific JSON."""
    require(dist.is_initialized(), "qualify requires an initialized process group")
    require(repeats >= 2, "Paired timing requires at least two repeats")
    device = torch.device("cuda", torch.cuda.current_device())
    with torch.inference_mode(), loaded_vae(paths_file) as loaded:
        from .parallel_vae import install

        decoder, tiling = loaded.decoder, loaded.tiling
        serial = decoder.tiled_decode
        generator = torch.Generator(device=device)
        inputs: list[torch.Tensor] = []
        for case in CASES:
            input_rng = torch.Generator(device=device).manual_seed(case.latent_seed)
            latent = torch.randn(LATENT_SHAPE, device=device, dtype=torch.bfloat16, generator=input_rng)
            dist.broadcast(latent, src=0)
            inputs.append(latent)
        # Warm the actual kernels, then calibrate the candidate through a complete decode.
        generator.manual_seed(19).set_offset(256)
        for chunk in serial(inputs[0], tiling, generator=generator):
            del chunk
        receipt = install(decoder)
        parallel = decoder.tiled_decode
        generator.manual_seed(19).set_offset(256)
        for chunk in parallel(inputs[0], tiling, generator=generator):
            del chunk
        torch.cuda.synchronize()
        require(receipt["calibrated_decodes"] == 1, "Real VAE warmup did not complete calibration")
        case_results: list[CaseResult] = []
        for index, case in enumerate(CASES):
            def serial_call() -> Iterator[torch.Tensor]:
                return serial(inputs[index], tiling, generator=generator)

            def parallel_call() -> Iterator[torch.Tensor]:
                return parallel(inputs[index], tiling, generator=generator)

            generator.manual_seed(case.decoder_seed).set_offset(case.initial_offset)
            reference = capture(serial_call, generator)
            generator.manual_seed(case.decoder_seed).set_offset(case.initial_offset)
            actual = capture(parallel_call, generator)
            errors = compare(reference, actual)
            require(sum(shape[2] for shape in errors["chunk_shapes"]) == 209 and all(
                shape[:2] == [1, 3] and shape[3:] == [512, 896] for shape in errors["chunk_shapes"]),
                "Decoded chunks do not reconstruct 896x512x209")
            case_results.append({"case": index, "latent_seed": case.latent_seed,
                "latent_sha256": hashlib.sha256(inputs[index].view(torch.int16).cpu().numpy().tobytes()).hexdigest(),
                "decoder_seed": case.decoder_seed, "initial_offset": case.initial_offset,
                "expected_final_offset": reference.final_offset, "actual_final_offset": actual.final_offset,
                "errors": errors})
            del reference, actual
        local_pass = all(row["errors"]["bitwise_equal"] and row["errors"]["finite"]
                         and row["errors"]["rng_equal"] for row in case_results)
        passed = torch.tensor(int(local_pass), device=device, dtype=torch.int32)
        dist.all_reduce(passed, op=dist.ReduceOp.MIN)
        global_pass = bool(passed.item())
        pairs: list[Pair] = []
        summary: list[TimingSummary] = []
        gc.collect()
        if global_pass:
            for index, case in enumerate(CASES):
                def serial_call() -> Iterator[torch.Tensor]:
                    return serial(inputs[index], tiling, generator=generator)

                def parallel_call() -> Iterator[torch.Tensor]:
                    return parallel(inputs[index], tiling, generator=generator)

                calls: dict[str, Decode] = {"serial": serial_call, "parallel": parallel_call}
                samples: dict[str, list[Sample]] = {"serial": [], "parallel": []}
                for repeat in range(repeats):
                    order = ["serial", "parallel"] if repeat % 2 == 0 else ["parallel", "serial"]
                    current: dict[str, Sample] = {}
                    for name in order:
                        generator.manual_seed(case.decoder_seed).set_offset(case.initial_offset)
                        with torch.cuda.nvtx.range(f"Qualification/RealVAE/case{index}/pair{repeat}/{name}"):
                            current[name] = measured(calls[name])
                        samples[name].append(current[name])
                    pairs.append({"case": index, "repeat": repeat, "order": order,
                                  "serial": current["serial"], "parallel": current["parallel"]})
                for name, values in samples.items():
                    times = [value["max_rank_wall_s"] for value in values]
                    ranks = [rank for value in values for rank in value["ranks"]]
                    summary.append({"case": index, "variant": name, "n": repeats,
                        "max_rank_wall_mean_s": statistics.mean(times),
                        "max_rank_wall_stdev_s": statistics.stdev(times),
                        "max_rank_peak_allocated_bytes": max(rank["peak_allocated_bytes"] for rank in ranks),
                        "max_rank_peak_reserved_bytes": max(rank["peak_reserved_bytes"] for rank in ranks)})
        report: Report = {"status": "PASS" if global_pass else "FAIL", "rank": dist.get_rank(),
            "world": dist.get_world_size(), "batch": 1, "repeats_per_case_per_arm": repeats,
            "warmup_excluded": True, "timing_includes_output_clones": False,
            "allocator_cache_retained": True, "provenance": loaded.provenance,
            "cases": case_results, "pairs": pairs, "timing_summary": summary, "receipt": receipt}
        output.parent.mkdir(parents=True, exist_ok=True)
        rank_output = output.with_name(f"{output.stem}.rank{dist.get_rank()}{output.suffix}")
        rank_output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        print(json.dumps({"status": report["status"], "output": str(rank_output),
                          "rank": dist.get_rank(), "timing_summary": summary}), flush=True)
        require(global_pass, "Real VAE output or final RNG differs on at least one rank; timings skipped")


def main() -> None:
    """Own process-group initialization only when the caller has not initialized it."""
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", "0")))
    owns_group = not dist.is_initialized()
    if owns_group:
        dist.init_process_group("nccl")
    try:
        output = Path(os.environ["SPARK_PP_RESULT_ROOT"]).parent / "real-vae-qualification.json"
        qualify(Path(os.environ["SPARK_PP_PATHS"]), output)
    finally:
        if owns_group:
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
