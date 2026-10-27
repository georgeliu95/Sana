"""Source-backed operator and official projection attribution regressions."""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def test_operator_when_actual_mxfp8_symbols_lack_a_precision_literal() -> None:
    """CUTLASS block scales and source producers identify the true precision."""
    from H3Adopted.report_classify import classify_module, classify_operator

    scope = r"phase/H3Adopted\\/DiT"
    assert classify_module(scope) == "H3 DiT"
    symbols = (
        (
            "cutlass3x_sm120_gemm_block_scaled_ue8m0xe4m3_ue8m0xe4m3_stream_k",
            "GEMM / MXFP8",
        ),
        ("quant_kernel", "Quantization / MXFP8"),
        ("swiglu_kernel", "Fused / SwiGLU + MXFP8"),
        ("merge", "Attention / coarse gate merge"),
    )
    for symbol, expected in symbols:
        family, proof = classify_operator(symbol, scope)
        assert family == expected
        assert proof
    assert classify_operator("merge", "phase/Qwen")[0] == "Unknown"


def test_profile_when_official_projection_replaces_missing_launch_paths(
    tmp_path: Path,
) -> None:
    """Derived ownership must expose the method and retain original raw evidence."""
    from H3Adopted.report_data import as_array, as_object, load_json, load_profile
    from test_report import _write_profile

    _write_profile(tmp_path)
    path = tmp_path / "analysis" / "paths.json"
    document = load_json(path)
    document.values["meta"] = {
        "projected_module_windows": [
            {
                "scope": "synthetic unit-test single request",
                "device": 0,
                "start_ns": 0,
                "end_ns": 1000,
            }
        ]
    }
    data = as_object(document.required("data"), "test paths")
    for index, value in enumerate(as_array(data.required("rows"), "rows")):
        row = as_object(value, "synthetic bucket")
        row.values["original_key"] = f"raw-{index}"
        row.values["original_nvtx_path"] = "__no_nvtx__"
        row.values["attribution"] = "synthetic official-projection fixture"
    _ = path.write_text(json.dumps(document.values))
    profile = load_profile(tmp_path)
    assert profile.attribution_method == "official NVTX GPU-projected phase windows"
    assert tuple(bucket.original_key for bucket in profile.buckets) == (
        "raw-0",
        "raw-1",
    )
    assert all(bucket.original_path == "__no_nvtx__" for bucket in profile.buckets)
    assert all(bucket.attribution for bucket in profile.buckets)
