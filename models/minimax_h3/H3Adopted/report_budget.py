"""Exact historical kernel budgets and complete per-symbol residuals."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from decimal import Decimal

from .report_data import JsonObject, ReportInputError, Table, integer, text


@dataclass(frozen=True, slots=True)
class SymbolBudgets:
    values: dict[str, tuple[int, int]]


def seconds_ns(value: str) -> int:
    """Convert recorded decimal seconds back to exact integer nanoseconds."""
    result = Decimal(value) * 1_000_000_000
    if result != result.to_integral_value():
        raise ReportInputError("historical CSV", "nonintegral nanosecond budget")
    return int(result)


def cell(table: Table, row: tuple[str, ...], column: str) -> str:
    return row[table.columns.index(column)]


def residuals(
    route: str, original: Table, global_rows: tuple[JsonObject, ...]
) -> SymbolBudgets:
    known: dict[str, tuple[int, int]] = defaultdict(lambda: (0, 0))
    for row in original.rows:
        if cell(original, row, "route") != route:
            continue
        symbol = cell(original, row, "kernel_name")
        count, total = known[symbol]
        known[symbol] = (
            count + int(cell(original, row, "kernel_count")),
            total + seconds_ns(cell(original, row, "total_gpu_seconds")),
        )
    residuals: dict[str, tuple[int, int]] = {}
    for row in global_rows:
        symbol = text(row.required("name"), "symbol")
        count, total = known.pop(symbol, (0, 0))
        count = integer(row.required("count"), "count") - count
        total = integer(row.required("total_ns"), "total_ns") - total
        if count < 0 or total < 0 or (count == 0) != (total == 0):
            raise ReportInputError(route, "inconsistent historical symbol residual")
        if count:
            residuals[symbol] = count, total
    if known:
        raise ReportInputError(
            route, "classified symbols absent from complete kernel envelope"
        )
    return SymbolBudgets(residuals)


def check_budget(
    table: Table, route: str, duration_column: str, count: int, total: int
) -> None:
    selected = tuple(row for row in table.rows if cell(table, row, "route") == route)
    if (
        sum(int(cell(table, row, "kernel_count")) for row in selected) != count
        or sum(seconds_ns(cell(table, row, duration_column)) for row in selected)
        != total
    ):
        raise ReportInputError(
            route, "historical module/operator/kernel budget mismatch"
        )
