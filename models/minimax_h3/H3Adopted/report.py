#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = []
# ///
# ─── How to run ───
# uv run --python 3.12 python -m H3Adopted.report --run RUN --old-report OLD --inventory CSV
# PYTHONPATH must include the directory containing H3Adopted.
# ──────────────────
"""Generate evidence-backed H3 comparison CSV and an offline profile explorer."""

from __future__ import annotations

import csv
import json
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Final

from .report_classify import ProfileTables, profile_tables
from .report_comparison import comparison_table
from .report_data import (
    Evidence,
    JsonObject,
    ReportInputError,
    Table,
    evidence,
    load_json,
    load_profile,
    read_table,
)
from .report_history import historical_sources
from .report_html import render_html

METHOD: Final = (
    "历史Baseline/Fullopt/Spark各n=1，2026-09-16同node3的Nsys采集；不是fresh A/B。",
    "新普通请求不开启采集；额外profile请求单列，既不进入普通均值也不扩充其样本量。",
    "输出均为896×512、209帧、24FPS、seed0；新视频8.708333s，新原PCM8.700s，保留约8.33ms尾差。",
    "新路线更换四步FastH3学生、MXFP8精度和TAEH3 decoder；未证明画质等价。",
    "投影权重为pruned scalar-FP8 represented→BF16（scale亦BF16）→完整student LoRA→MXFP8。",
    "新cache_hits=0仅表示未安装FirstBlockCache/denoiser stack skip；保留请求内RoPE/text-refiner缓存及七行AdaLN lookup。",
    "每次Qwen为fresh请求；不用跨请求prompt/output缓存。启动和预热另列并从热请求时间排除。",
    "decode_and_mux_s是native_generate_s子区间，不含audio VAE；这些分段不能相加。",
    "GPU秒为kernel duration求和，排除memcpy/memset；不是GPU walltime或SM utilization。",
    "module/operator按来源唯一归属：launch→NVTX或官方GPU投影phase窗口；父子inclusive不重复求和，Unknown保留。",
    "kernel P50/P95/P99是同symbol/grid/block/path分桶中的事件分布，不是请求级尾延迟；分位数不相加。",
    "Fullopt16完整stack+33命中仍有49forward；管理GPU0.962488714s已包含在总GPU时间中。",
    "Fullopt CPU stream等待包含已计GPU工作，不作为额外E2E overhead相加；无cache开关消融。",
    "新FBC管理GPU=0仅源于未安装FBC，不表示其他内存管理或residency清理成本为0。",
    "原101项inventory是事前逐项分析；实验只启用本单卡/模型/几何实际适用的组合。",
)


@dataclass(frozen=True, slots=True)
class ReportInputs:
    run: Path
    old_report: Path
    inventory: Path


@dataclass(frozen=True, slots=True)
class ReportSummary:
    schema_version: int
    generated_utc: str
    run: str
    profile_ready: bool
    methodology: tuple[str, ...]
    comparison: Table
    profile_tables: ProfileTables
    inventory: Table
    new_results: JsonObject
    fullopt_cache: JsonObject
    evidence: tuple[Evidence, ...]
    auxiliary_profiles: tuple[JsonObject, ...]


def _write_table(path: Path, table: Table) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(table.columns)
        writer.writerows(table.rows)


def generate_report(inputs: ReportInputs) -> ReportSummary:
    """Produce auditable tables and an offline explorer without executing GPU work."""
    results_path = inputs.run / "raw" / "pipeline" / "results.json"
    old_summary = inputs.old_report / "summary.json"
    cache_path = inputs.old_report / "fullopt_cache.json"
    results = load_json(results_path)
    profile = load_profile(inputs.run)
    tables = profile_tables(inputs.old_report, profile)
    source_paths = (
        results_path,
        old_summary,
        cache_path,
        inputs.inventory,
        *(
            inputs.old_report / f"{name}.csv"
            for name in ("modules", "operators", "kernels")
        ),
        *historical_sources(inputs.old_report),
    )
    report = ReportSummary(
        1,
        datetime.now(timezone.utc).isoformat(),
        str(inputs.run.resolve()),
        profile.ready,
        METHOD,
        comparison_table(load_json(old_summary), results, profile),
        tables,
        read_table(inputs.inventory),
        results,
        load_json(cache_path),
        tuple(evidence(path) for path in source_paths) + profile.evidence,
        profile.auxiliary,
    )
    output = inputs.run / "report"
    output.mkdir(parents=True, exist_ok=True)
    for name, table in (
        ("comparison", report.comparison),
        ("modules", tables.modules),
        ("operators", tables.operators),
        ("kernels", tables.kernels),
    ):
        _write_table(output / f"{name}.csv", table)
    serialized = json.dumps(
        asdict(report), ensure_ascii=False, allow_nan=False, indent=2
    )
    _ = (output / "summary.json").write_text(serialized + "\n", encoding="utf-8")
    _ = (output / "profile-explorer.html").write_text(
        render_html(serialized), encoding="utf-8"
    )
    return report


def main() -> None:
    """Parse the three standard-library CLI path arguments exactly once."""
    args = sys.argv[1:]
    if args == ["--help"]:
        print((__doc__ or "") + "\n--run RUN --old-report OLD --inventory CSV")
        return
    if len(args) != 6 or len(set(args[::2])) != 3:
        raise ReportInputError(
            "arguments", "--run, --old-report, and --inventory are required"
        )
    values = dict(zip(args[::2], args[1::2], strict=True))
    if set(values) != {"--run", "--old-report", "--inventory"}:
        raise ReportInputError("arguments", "unknown or missing path flag")
    report = generate_report(
        ReportInputs(
            Path(values["--run"]),
            Path(values["--old-report"]),
            Path(values["--inventory"]),
        )
    )
    print(
        json.dumps(
            {
                "report": str(Path(report.run) / "report"),
                "profile_ready": report.profile_ready,
            }
        )
    )


if __name__ == "__main__":
    main()
