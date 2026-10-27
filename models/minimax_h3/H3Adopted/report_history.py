"""Restore historical unassigned kernels without inventing launch metadata."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable
from pathlib import Path

from .report_budget import SymbolBudgets, check_budget, residuals, seconds_ns
from .report_data import (
    JsonObject,
    ReportInputError,
    Table,
    as_array,
    as_object,
    envelope_data,
    integer,
    load_json,
    number,
    text,
)


def historical_sources(old_report: Path) -> tuple[Path, ...]:
    """List saved public envelopes needed to restore omitted Unknown rows."""
    parent = old_report.parent
    return tuple(
        parent / f"{route}-{kind}.json"
        for route, kind in (
            ("baseline", "kernels"),
            ("fullopt", "kernels"),
            ("spark", "kernels"),
            ("baseline", "paths"),
            ("fullopt", "paths"),
        )
    )


def _normalize(table: Table) -> Table:
    index = table.columns.index("module")
    return Table(
        table.columns,
        tuple(
            row[:index] + ("Unknown",) + row[index + 1 :]
            if row[index] == "Other / unassigned"
            else row
            for row in table.rows
        ),
    )


def _envelope_rows(path: Path) -> tuple[JsonObject, ...]:
    data = envelope_data(load_json(path), str(path))
    return tuple(
        as_object(row, "historic bucket")
        for row in as_array(data.required("rows"), "rows")
    )


def _unknown_paths(
    rows: tuple[JsonObject, ...], residuals: SymbolBudgets
) -> tuple[JsonObject, ...]:
    selected = tuple(
        row
        for row in rows
        if text(row.required("name"), "symbol") in residuals.values
        and not any(
            token in text(row.required("nvtx_path"), "path").replace("\\", "")
            for token in (
                "module/conditioner/",
                "module/dit/",
                "module/video_decode/",
                "module/audio_decode/",
            )
        )
    )
    budgets: dict[str, tuple[int, int]] = defaultdict(lambda: (0, 0))
    for row in selected:
        symbol = text(row.required("name"), "symbol")
        count, total = budgets[symbol]
        budgets[symbol] = (
            count + integer(row.required("count"), "count"),
            total + integer(row.required("total_ns"), "ns"),
        )
    if budgets != residuals.values:
        raise ReportInputError(
            "historical Unknown",
            "NVTX bucket residual is not independently recoverable",
        )
    return selected


def _unknown_kernel(
    route: str,
    row: JsonObject,
    count: int,
    total: int,
    module_ns: int,
    classifier: Callable[[str, str], tuple[str, str]],
    exact_bucket: bool,
) -> tuple[str, ...]:
    symbol = text(row.required("name"), "symbol")
    path = text(row.required("nvtx_path"), "path") if exact_bucket else ""
    family, proof = classifier(symbol, path)
    stats = (
        tuple(
            str(number(row.required(key), key) / 1e6)
            for key in ("p50_ns", "p95_ns", "p99_ns", "min_ns", "max_ns")
        )
        if exact_bucket
        else ("",) * 5
    )
    origin = (
        "historical Unknown NVTX bucket; shape absent"
        if exact_bucket
        else "historical GPU-window residual; shape/quantiles unavailable"
    )
    return (
        route,
        "Unknown",
        family,
        proof,
        symbol,
        *("",) * 6,
        str(count),
        str(total / 1e9),
        str(total / count / 1e6),
        *stats,
        str(100 * total / module_ns),
        path,
        origin,
        text(row.required("key"), "source key"),
        path,
        text(row.required("key"), "source key"),
        "historical NVTX bucket" if exact_bucket else "historical GPU-window residual",
        origin,
    )


def restore_unknown(
    old_report: Path,
    modules: Table,
    operators: Table,
    kernels: Table,
    classifier: Callable[[str, str], tuple[str, str]],
) -> tuple[Table, Table, Table]:
    """Preserve real Unknown distributions where recoverable; leave others blank."""
    modules, operators, kernels = (
        _normalize(modules),
        _normalize(operators),
        _normalize(kernels),
    )
    extra: list[tuple[str, ...]] = []
    operator_rows: list[tuple[str, ...]] = []
    summaries = as_array(
        load_json(old_report / "summary.json").required("routes"), "routes"
    )
    for summary in summaries:
        receipt = as_object(summary, "historic receipt")
        key = text(receipt.required("route"), "route")
        route = {"baseline": "Baseline", "fullopt": "Fullopt", "spark": "Sol-H3-Spark"}[
            key
        ]
        global_rows = _envelope_rows(old_report.parent / f"{key}-kernels.json")
        remainder = residuals(route, kernels, global_rows)
        module_ns = integer(receipt.required("unassigned_module_ns"), "Unknown ns")
        if key == "spark":
            selected = tuple(
                row
                for row in global_rows
                if text(row.required("name"), "symbol") in remainder.values
            )
        else:
            selected = _unknown_paths(
                _envelope_rows(old_report.parent / f"{key}-paths.json"), remainder
            )
        added = tuple(
            _unknown_kernel(
                route,
                row,
                *(
                    (
                        integer(row.required("count"), "count"),
                        integer(row.required("total_ns"), "ns"),
                    )
                    if key != "spark"
                    else remainder.values[text(row.required("name"), "symbol")]
                ),
                module_ns,
                classifier,
                key != "spark",
            )
            for row in selected
        )
        if sum(seconds_ns(row[12]) for row in added) != module_ns:
            raise ReportInputError(
                route, "Unknown budget differs from archived summary"
            )
        extra.extend(added)
        family_totals: dict[str, tuple[int, int]] = defaultdict(lambda: (0, 0))
        for row in added:
            count, total = family_totals[row[2]]
            family_totals[row[2]] = count + int(row[11]), total + seconds_ns(row[12])
        operator_rows.extend(
            (
                route,
                "Unknown",
                family,
                str(count),
                str(total / 1e9),
                str(100 * total / module_ns),
            )
            for family, (count, total) in family_totals.items()
        )
    operators = Table(operators.columns, operators.rows + tuple(operator_rows))
    kernels = Table(kernels.columns, kernels.rows + tuple(extra))
    for summary in summaries:
        receipt = as_object(summary, "historic receipt")
        key = text(receipt.required("route"), "route")
        route = {"baseline": "Baseline", "fullopt": "Fullopt", "spark": "Sol-H3-Spark"}[
            key
        ]
        count, total = (
            integer(receipt.required("kernel_count"), "count"),
            integer(receipt.required("gpu_ns"), "ns"),
        )
        for table, duration in (
            (modules, "gpu_seconds"),
            (operators, "gpu_seconds"),
            (kernels, "total_gpu_seconds"),
        ):
            check_budget(table, route, duration, count, total)
    return modules, operators, kernels
