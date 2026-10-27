"""Separate uncaptured request statistics from historical and new Nsys samples."""

from __future__ import annotations

import statistics
from typing import Final

from .report_data import (
    JsonObject,
    Profile,
    ReportInputError,
    Table,
    as_array,
    as_object,
    integer,
    number,
    text,
)

COMPARISON_COLUMNS: Final = (
    "route",
    "provenance",
    "capture_date",
    "timing_samples",
    "profile_samples",
    "e2e_mean_s",
    "e2e_median_s",
    "e2e_min_s",
    "e2e_max_s",
    "e2e_sample_stddev_s",
    "profiled_e2e_s",
    "gpu_kernel_seconds",
    "kernel_count",
    "h3_forward_calls",
    "h3_full_stack_nfe",
    "h3_cache_partial_hits",
    "cache_management_gpu_seconds",
    "speedup_vs_historical_baseline",
    "speedup_vs_historical_fullopt",
    "speedup_vs_historical_spark",
    "computation_change",
    "output_scope",
    "workload",
)


def _requests(results: JsonObject) -> tuple[JsonObject, ...]:
    """Accept complete successful requests with explicit capture and NFE receipts."""
    if text(results.required("status"), "benchmark status") != "PASS":
        raise ReportInputError("benchmark", "only a complete PASS run is reportable")
    if results.get("cache_enabled") is not False:
        raise ReportInputError(
            "FBC receipt", "expected FirstBlockCache installation disabled"
        )
    if as_array(results.required("geometry"), "geometry") != [896, 512, 209, 24]:
        raise ReportInputError("geometry", "matched209 workload drift")
    rows = tuple(
        as_object(row, "request")
        for row in as_array(results.required("requests"), "requests")
    )
    for row in rows:
        if (
            row.get("status") != "PASS"
            or type(row.required("profiled")) is not bool
            or number(row.required("e2e_s"), "e2e_s") <= 0
            or integer(row.required("actual_nfe"), "actual NFE") != 4
            or integer(row.required("cache_hits"), "cache hits") != 0
            or number(row.required("cache_management_overhead_s"), "FBC management")
            != 0
        ):
            raise ReportInputError(
                "request", "invalid successful four-complete-stack receipt"
            )
    if not rows or all(row.required("profiled") is True for row in rows):
        raise ReportInputError("ordinary timing", "no uncaptured measured request")
    return rows


def comparison_table(old: JsonObject, results: JsonObject, profile: Profile) -> Table:
    """Keep historical Nsys n=1 and fresh uncaptured timing samples distinct."""
    historic = tuple(
        as_object(row, "historic route")
        for row in as_array(old.required("routes"), "routes")
    )
    historic = tuple(
        next(row for row in historic if row.get("route") == name)
        for name in ("baseline", "fullopt", "spark")
    )
    denominators = tuple(
        number(row.required("native_e2e_s"), "historic E2E") for row in historic
    )
    rows: list[tuple[str, ...]] = []
    for route, elapsed in zip(historic, denominators, strict=True):
        name = text(route.required("route"), "historic route")
        change = {
            "baseline": "original pruned FP8; fullopt switches off",
            "fullopt": "fusions/Sol/FBC0.08/tile batch4",
            "spark": "low-resolution FastH3 draft→upscaler/adapter→LTX2.5",
        }[name]
        rows.append(
            (
                {"baseline": "Baseline", "fullopt": "Fullopt", "spark": "Sol-H3-Spark"}[
                    name
                ],
                "historical n=1 / Nsys; not fresh A/B",
                text(old.required("capture_date"), "capture date"),
                "1",
                "1",
                str(elapsed),
                str(elapsed),
                str(elapsed),
                str(elapsed),
                "",
                str(elapsed),
                str(integer(route.required("gpu_ns"), "historic GPU") / 1e9),
                str(integer(route.required("kernel_count"), "historic kernels")),
                str(
                    integer(
                        route.required("h3_denoiser_forward_calls"), "historic calls"
                    )
                ),
                str(
                    integer(
                        route.required("h3_full_stack_nfe"), "historic complete NFE"
                    )
                ),
                str(
                    integer(
                        route.required("h3_cached_partial_updates"),
                        "historic cache hits",
                    )
                ),
                str(
                    integer(route.get("cache_gpu_kernel_overhead_ns") or 0, "FBC GPU")
                    / 1e9
                ),
                *(str(value / elapsed) for value in denominators),
                change,
                "finalized muxed MP4",
                "896x512;209frames;24FPS;seed0",
            )
        )
    requests = _requests(results)
    ordinary = tuple(
        number(row.required("e2e_s"), "ordinary E2E")
        for row in requests
        if row.required("profiled") is False
    )
    captured = tuple(
        number(row.required("e2e_s"), "profiled E2E")
        for row in requests
        if row.required("profiled") is True
    )
    if profile.ready and len(captured) != 1:
        raise ReportInputError(
            "profile scope", "expected exactly one additional profiled request"
        )
    mean = statistics.mean(ordinary)
    rows.append(
        (
            "H3Adopted",
            "fresh uncaptured timing; separate extra Nsys sample",
            "2026-10-09",
            str(len(ordinary)),
            str(len(captured)),
            str(mean),
            str(statistics.median(ordinary)),
            str(min(ordinary)),
            str(max(ordinary)),
            str(statistics.stdev(ordinary)) if len(ordinary) > 1 else "",
            ";".join(str(value) for value in captured),
            "" if profile.total_ns is None else str(profile.total_ns / 1e9),
            "" if profile.kernel_count is None else str(profile.kernel_count),
            "4",
            "4",
            "0",
            "0",
            *(str(value / mean) for value in denominators),
            "FastH3 student4; represented scalar-FP8→BF16+LoRA→MXFP8;TAEH3",
            "fresh Qwen request entry→finalized muxed MP4",
            "896x512;209frames;24FPS;seed0",
        )
    )
    return Table(COMPARISON_COLUMNS, tuple(rows))
