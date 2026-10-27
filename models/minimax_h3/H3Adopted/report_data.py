"""Typed input boundaries and kernel buckets for the H3 comparison report."""

from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, TypeAlias, assert_never

JsonValue: TypeAlias = (
    str | int | float | bool | None | list["JsonValue"] | dict[str, "JsonValue"]
)


class _JsonDecoder(Protocol):
    def __call__(self, payload: str, /) -> JsonValue: ...


def _decode_json(payload: str, decoder: _JsonDecoder) -> JsonValue:
    """Narrow the standard-library decoder to its recursive JSON contract."""
    return decoder(payload)


class ReportInputError(ValueError):
    """Typed exception allowing Python to update its traceback during propagation."""

    context: str
    reason: str

    def __init__(self, context: str, reason: str) -> None:
        self.context = context
        self.reason = reason
        super().__init__(f"{context}: {reason}")


@dataclass(frozen=True, slots=True)
class JsonObject:
    """A validated JSON object boundary with typed field access."""

    values: dict[str, JsonValue]

    def get(self, key: str) -> JsonValue:
        return self.values.get(key)

    def required(self, key: str) -> JsonValue:
        if key not in self.values:
            raise ReportInputError(key, "required field absent")
        return self.values[key]


@dataclass(frozen=True, slots=True)
class Table:
    """Rectangular string-valued CSV and browser table."""

    columns: tuple[str, ...]
    rows: tuple[tuple[str, ...], ...]


@dataclass(frozen=True, slots=True)
class Evidence:
    path: str
    sha256: str
    command: JsonValue = None
    trace: JsonValue = None
    meta: JsonValue = None


@dataclass(frozen=True, slots=True)
class KernelBucket:
    key: str
    symbol: str
    path: str
    count: int
    total_ns: int
    avg_ns: float
    p50_ns: float
    p95_ns: float
    p99_ns: float
    min_ns: float
    max_ns: float
    grid: tuple[int, int, int]
    block: tuple[int, int, int]
    original_path: str = ""
    original_key: str = ""
    attribution: str = ""


@dataclass(frozen=True, slots=True)
class Profile:
    ready: bool
    buckets: tuple[KernelBucket, ...]
    kernel_count: int | None
    total_ns: int | None
    evidence: tuple[Evidence, ...]
    auxiliary: tuple[JsonObject, ...]
    attribution_method: str = "CUDA launch → NVTX path"


def as_object(value: JsonValue, context: str) -> JsonObject:
    """Accept JSON objects and reject all other JSON variants."""
    match value:
        case dict() as mapping:
            return JsonObject(mapping)
        case str() | int() | float() | list() | None:
            raise ReportInputError(context, "expected JSON object")
        case unreachable:
            assert_never(unreachable)


def as_array(value: JsonValue, context: str) -> list[JsonValue]:
    """Accept arrays without accepting arbitrary Python sequences."""
    match value:
        case list() as items:
            return items
        case str() | int() | float() | dict() | None:
            raise ReportInputError(context, "expected JSON array")
        case unreachable:
            assert_never(unreachable)


def text(value: JsonValue, context: str) -> str:
    """Accept strings; missing fields remain explicit rather than empty evidence."""
    match value:
        case str() as result:
            return result
        case int() | float() | dict() | list() | None:
            raise ReportInputError(context, "expected string")
        case unreachable:
            assert_never(unreachable)


def number(value: JsonValue, context: str) -> float:
    """Accept numeric JSON data or CSV number strings, excluding booleans."""
    match value:
        case bool() | dict() | list() | None:
            raise ReportInputError(context, "expected number")
        case int() | float() as result:
            return float(result)
        case str() as result:
            try:
                return float(result)
            except ValueError as error:
                raise ReportInputError(context, "invalid numeric string") from error
        case unreachable:
            assert_never(unreachable)


def integer(value: JsonValue, context: str) -> int:
    """Retain nanosecond integers without loss through floating-point conversion."""
    match value:
        case bool() | dict() | list() | None | float():
            raise ReportInputError(context, "expected integer")
        case int() as result:
            return result
        case str() as result:
            try:
                return int(result)
            except ValueError as error:
                raise ReportInputError(context, "invalid integer string") from error
        case unreachable:
            assert_never(unreachable)


def load_json(path: Path) -> JsonObject:
    """Read public results or Veloq JSON envelopes, never private profiler tables."""
    value = _decode_json(path.read_text(encoding="utf-8"), json.loads)
    return as_object(value, str(path))


def read_table(path: Path) -> Table:
    """Preserve original CSV strings and missing historical launch metadata."""
    with path.open(newline="", encoding="utf-8") as stream:
        records = tuple(tuple(row) for row in csv.reader(stream) if row)
    if not records or any(len(row) != len(records[0]) for row in records[1:]):
        raise ReportInputError(str(path), "empty or nonrectangular CSV")
    return Table(records[0], records[1:])


def evidence(path: Path, envelope: JsonObject | None = None) -> Evidence:
    """Bind claims to content hashes and public command/filter metadata."""
    return Evidence(
        str(path.resolve()),
        hashlib.sha256(path.read_bytes()).hexdigest(),
        None if envelope is None else envelope.get("command"),
        None if envelope is None else envelope.get("trace"),
        None if envelope is None else envelope.get("meta"),
    )


def _optional_text(row: JsonObject, key: str, fallback: str) -> str:
    value = row.get(key)
    return fallback if value is None else text(value, key)


def _bucket(row: JsonObject) -> KernelBucket:
    get = row.required
    grid = tuple(integer(get("grid_" + axis), "grid") for axis in ("x", "y", "z"))
    block = tuple(integer(get("block_" + axis), "block") for axis in ("x", "y", "z"))
    gx, gy, gz = grid
    bx, by, bz = block
    return KernelBucket(
        text(get("key"), "kernel key"),
        text(get("name"), "kernel symbol"),
        text(get("nvtx_path"), "NVTX path"),
        integer(get("count"), "kernel count"),
        integer(get("total_ns"), "kernel total_ns"),
        number(get("avg_ns"), "avg_ns"),
        number(get("p50_ns"), "p50_ns"),
        number(get("p95_ns"), "p95_ns"),
        number(get("p99_ns"), "p99_ns"),
        number(get("min_ns"), "min_ns"),
        number(get("max_ns"), "max_ns"),
        (gx, gy, gz),
        (bx, by, bz),
        _optional_text(row, "original_nvtx_path", text(get("nvtx_path"), "path")),
        _optional_text(row, "original_key", text(get("key"), "key")),
        _optional_text(row, "attribution", ""),
    )


def envelope_data(document: JsonObject, context: str) -> JsonObject:
    """Reject Veloq error envelopes explicitly."""
    if document.get("error") is not None:
        raise ReportInputError(context, str(document.get("error")))
    return as_object(document.required("data"), context)


def load_profile(run: Path) -> Profile:
    """Use exclusive launch-path buckets and verify their independent kernel budget."""
    paths = tuple(
        run / "analysis" / f"{name}.json"
        for name in ("kernels", "paths", "nvtx", "runtime", "memcpy")
    )
    if not all(path.is_file() for path in paths):
        return Profile(
            False, (), None, None, tuple(evidence(p) for p in paths if p.is_file()), ()
        )
    documents = tuple(load_json(path) for path in paths)
    data = tuple(
        envelope_data(doc, str(path))
        for doc, path in zip(documents, paths, strict=True)
    )
    rows = as_array(data[1].required("rows"), "paths rows")
    buckets = tuple(_bucket(as_object(row, "path bucket")) for row in rows)
    count = integer(data[0].required("total_events"), "kernel budget count")
    total = integer(data[0].required("total_duration_ns"), "kernel budget nanoseconds")
    matched = integer(data[1].required("total_matched"), "path bucket count")
    keys = [bucket.key for bucket in buckets]
    if (
        count <= 0
        or total <= 0
        or matched != len(buckets)
        or len(keys) != len(set(keys))
        or sum(bucket.count for bucket in buckets) != count
        or sum(bucket.total_ns for bucket in buckets) != total
    ):
        raise ReportInputError(
            "profile reconciliation",
            "truncated, duplicate, or inconsistent path buckets",
        )
    metadata = documents[1].get("meta")
    projected = (
        None
        if metadata is None
        else as_object(metadata, "paths meta").get("projected_module_windows")
    )
    method = (
        "official NVTX GPU-projected phase windows"
        if projected is not None
        else "CUDA launch → NVTX path"
    )
    extra_sources = tuple(
        path for path in sorted((run / "analysis").glob("*.json")) if path not in paths
    )
    return Profile(
        True,
        buckets,
        count,
        total,
        tuple(evidence(p, doc) for p, doc in zip(paths, documents, strict=True))
        + tuple(evidence(path) for path in extra_sources),
        documents[2:],
        method,
    )
