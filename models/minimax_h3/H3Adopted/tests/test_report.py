"""Behavior checks for profile attribution and nonadditive statistics."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def test_module_when_output_scope_is_nested_inside_decoder() -> None:
    # Given: a pixel conversion launch within an inclusive TAE decoder range.
    from H3Adopted.report_classify import classify_module

    path = r"request/H3Adopted\/TAEH3/H3Adopted\/pixels_D2H"
    # When: assigning that launch to one module.
    result = classify_module(path)
    # Then: only the deepest meaningful output scope owns the kernel.
    assert result == "Pixel conversion/output"


def test_module_when_no_recognized_range_exists() -> None:
    # Given: a kernel launched under opaque runtime annotations.
    from H3Adopted.report_classify import classify_module

    # When: assigning its module.
    result = classify_module("request/compiler_graph/replay")
    # Then: missing evidence remains visible.
    assert result == "Unknown"


def test_operator_when_mxfp8_producer_fuses_swiglu() -> None:
    # Given: one fused producer in the actual MXFP8 FC2 scope.
    from H3Adopted.report_classify import classify_operator

    # When: classifying the indivisible kernel.
    family, evidence = classify_operator(
        "swiglu_kernel", r"H3Adopted\/MXFP8\/swiglu_fc2\/transformer_blocks.1.ff.fc_out"
    )
    # Then: activation and quantization are not duplicated into separate rows.
    assert family == "Fused / SwiGLU + MXFP8"
    assert evidence


def test_typed_errors_when_generator_contextmanager_propagates_them() -> None:
    # Given: typed errors whose identity is meaningful to the boundary handler.
    from collections.abc import Iterator
    from contextlib import contextmanager

    from H3Adopted.mxfp8_kernels import QuantizationContractError
    from H3Adopted.report_data import ReportInputError

    @contextmanager
    def scope() -> Iterator[None]:
        yield

    # When: Python assigns traceback during generator-contextmanager propagation.
    for error in (
        QuantizationContractError("mx", "invalid"),
        ReportInputError("json", "invalid"),
    ):
        try:
            with scope():
                raise error
        except (QuantizationContractError, ReportInputError) as propagated:
            # Then: the original exception type and instance reach the caller.
            assert propagated is error
        else:
            raise AssertionError("expected the original typed error")


def _write_profile(run: Path) -> None:
    """Write a tiny synthetic two-shape profile only into the test directory."""
    import json

    directory = run / "analysis"
    directory.mkdir()
    buckets = []
    for index, grid, count in ((0, 1, 1), (1, 2, 2)):
        buckets.append(
            {
                "key": f"shape-{index}",
                "name": "synthetic_gemm",
                "nvtx_path": r"request/H3Adopted\/MXFP8\/linear\/transformer_blocks.1",
                "count": count,
                "total_ns": 100 * count,
                "avg_ns": 100,
                "p50_ns": 91 + index,
                "p95_ns": 100,
                "p99_ns": 101,
                "min_ns": 90,
                "max_ns": 110,
                "grid_x": grid,
                "grid_y": 1,
                "grid_z": 1,
                "block_x": 32,
                "block_y": 1,
                "block_z": 1,
            }
        )
    for name in ("kernels", "paths", "nvtx", "runtime", "memcpy"):
        data = {
            "rows": buckets if name == "paths" else [],
            "total_events": 3,
            "total_duration_ns": 300,
            "total_matched": 2,
        }
        (directory / f"{name}.json").write_text(json.dumps({"data": data}))


def test_profile_when_shape_buckets_share_a_symbol(tmp_path: Path) -> None:
    """Distinct grid shapes retain their own distribution and exclusive budgets."""
    from H3Adopted.report_data import load_profile

    _write_profile(tmp_path)
    result = load_profile(tmp_path)
    assert result.kernel_count == 3
    assert result.total_ns == 300
    assert tuple(bucket.grid[0] for bucket in result.buckets) == (1, 2)
    assert tuple(bucket.p50_ns for bucket in result.buckets) == (91, 92)


def test_profile_when_primary_path_export_is_truncated(tmp_path: Path) -> None:
    """A missing primary bucket cannot silently produce an incomplete GPU table."""
    import json

    import pytest
    from H3Adopted.report_data import ReportInputError, load_profile

    _write_profile(tmp_path)
    path = tmp_path / "analysis" / "paths.json"
    data = json.loads(path.read_text())
    data["data"]["rows"].pop()
    path.write_text(json.dumps(data))
    with pytest.raises(ReportInputError, match="inconsistent path buckets"):
        load_profile(tmp_path)


def test_profile_when_export_has_duplicate_source_keys(tmp_path: Path) -> None:
    """Double-counted source buckets are rejected even if duration still matches."""
    import json

    import pytest
    from H3Adopted.report_data import ReportInputError, load_profile

    _write_profile(tmp_path)
    path = tmp_path / "analysis" / "paths.json"
    data = json.loads(path.read_text())
    data["data"]["rows"][1]["key"] = data["data"]["rows"][0]["key"]
    path.write_text(json.dumps(data))
    with pytest.raises(ReportInputError, match="duplicate"):
        load_profile(tmp_path)


def test_profile_when_capture_is_pending(tmp_path: Path) -> None:
    """No capture means missing GPU values rather than zero-duration work."""
    from H3Adopted.report_data import load_profile

    result = load_profile(tmp_path)
    assert result.ready is False
    assert result.kernel_count is None
    assert result.total_ns is None


def test_html_when_a_kernel_symbol_contains_script_markup() -> None:
    """User-controlled symbols cannot close the inert JSON payload element."""
    from H3Adopted.report_html import render_html

    result = render_html('{"symbol":"</script><script>alert(1)</script>"}')
    assert "</script><script>alert(1)</script>" not in result
    assert r"\u003c/script\u003e" in result


def test_report_when_timing_and_extra_profile_are_both_present(tmp_path: Path) -> None:
    """Profile timing stays outside the ordinary mean and historical Unknown survives."""
    import csv
    import json

    from H3Adopted.report import ReportInputs, generate_report

    workspace = Path(__file__).resolve().parents[4]
    runs = workspace / "runs"
    old_report = runs / "nsys-matched209-recovered-20261008" / "analysis" / "report"
    inventory = (
        runs / "h3-optimizations-adoption-20261009" / "optimization-inventory.csv"
    )
    pipeline = tmp_path / "raw" / "pipeline"
    pipeline.mkdir(parents=True)
    requests = [
        {
            "request_id": str(index),
            "status": "PASS",
            "profiled": index == 3,
            "e2e_s": elapsed,
            "actual_nfe": 4,
            "cache_hits": 0,
            "cache_management_overhead_s": 0,
        }
        for index, elapsed in enumerate((24, 26, 25, 34))
    ]
    (pipeline / "results.json").write_text(
        json.dumps(
            {
                "status": "PASS",
                "geometry": [896, 512, 209, 24],
                "cache_enabled": False,
                "requests": requests,
            }
        )
    )
    result = generate_report(ReportInputs(tmp_path, old_report, inventory))
    comparison = result.comparison
    fresh = comparison.rows[-1]
    assert fresh[comparison.columns.index("timing_samples")] == "3"
    assert float(fresh[comparison.columns.index("e2e_mean_s")]) == 25
    assert fresh[comparison.columns.index("profiled_e2e_s")] == "34.0"
    assert fresh[comparison.columns.index("gpu_kernel_seconds")] == ""
    kernels = result.profile_tables.kernels
    route_index = kernels.columns.index("route")
    count_index = kernels.columns.index("kernel_count")
    assert (
        sum(
            int(row[count_index])
            for row in kernels.rows
            if row[route_index] == "Sol-H3-Spark"
        )
        == 39419
    )
    unknown = [
        row
        for row in kernels.rows
        if row[route_index] == "Sol-H3-Spark"
        and row[kernels.columns.index("module")] == "Unknown"
    ]
    assert sum(int(row[count_index]) for row in unknown) == 95
    assert all(row[kernels.columns.index("p95_ms")] == "" for row in unknown)
    with (tmp_path / "report" / "kernels.csv").open(newline="") as stream:
        exported = tuple(csv.reader(stream))
    assert all(len(row) == len(kernels.columns) for row in exported)
    assert (tmp_path / "report" / "profile-explorer.html").is_file()

    # Once the extra profile export arrives, retain each shape percentile intact.
    _write_profile(tmp_path)
    profiled = generate_report(ReportInputs(tmp_path, old_report, inventory))
    fresh_kernels = profiled.profile_tables.kernels
    new_rows = [row for row in fresh_kernels.rows if row[route_index] == "H3Adopted"]
    assert sum(int(row[count_index]) for row in new_rows) == 3
    assert tuple(
        int(row[fresh_kernels.columns.index("grid_x")]) for row in new_rows
    ) == (1, 2)
    assert tuple(
        float(row[fresh_kernels.columns.index("p50_ms")]) for row in new_rows
    ) == (91 / 1e6, 92 / 1e6)
