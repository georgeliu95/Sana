"""Assign each NVTX-correlated launch to one module and one operator family."""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from .report_data import KernelBucket, Profile, Table, read_table
from .report_history import restore_unknown

_MODULE_RULES: Final = (
    ("pixels_d2h", "Pixel conversion/output"),
    ("video_unpack", "Video unpack"),
    ("taeh3", "TAEH3 decoder/output"),
    ("audio_vae", "H3 audio VAE"),
    ("audio_decod", "H3 audio VAE"),
    ("qwen", "Qwen encoder"),
    ("conditioning_stage", "Conditioner"),
    ("conditioner", "Conditioner"),
    ("video_decode", "H3 video VAE"),
    ("video_decod", "H3 video VAE"),
    ("ltx", "LTX denoise"),
    ("h3adopted/dit", "H3 DiT"),
    ("denoising_stage", "H3 DiT"),
    ("denois", "H3 DiT"),
    ("transformer_blocks", "H3 DiT"),
    ("h3_adopted/mxfp8", "H3 DiT"),
    ("h3adopted/mxfp8", "H3 DiT"),
    ("h3_mxfp8", "H3 DiT"),
    ("video_latent", "H3 DiT"),
)


def nvtx_labels(path: str) -> tuple[str, ...]:
    """Split Veloq path separators while retaining slashes escaped inside labels."""
    return tuple(re.sub(r"\\+/", "/", label) for label in re.split(r"(?<!\\)/", path))


def classify_module(path: str) -> str:
    """Use the deepest recognized NVTX label; unknown children keep their parent."""
    for label in reversed(nvtx_labels(path)):
        text = label.lower()
        for token, module in _MODULE_RULES:
            if token in text:
                return module
    return "Unknown"


def classify_operator(symbol: str, path: str) -> tuple[str, str]:
    """Return an indivisible kernel family and its attribution evidence."""
    name = symbol.lower()
    context = "/".join(nvtx_labels(path)).lower()
    if name == "merge" and classify_module(path) == "H3 DiT":
        return (
            "Attention / coarse gate merge",
            "runtime/stage1_ops/vsa.py:324 → fused_ops.build_forward(fused_merge=True) → fused.py::merge",
        )
    mx = (
        "mxfp8" in context
        or "mxfp8" in name
        or "block_scaled_ue8m0xe4m3_ue8m0xe4m3" in name
        or (
            classify_module(path) == "H3 DiT"
            and name in {"quant_kernel", "swiglu_kernel"}
        )
    )
    if mx and ("swiglu_kernel" in name or "silu_mxfp8" in name):
        return (
            "Fused / SwiGLU + MXFP8",
            "MXFP8 source producer + H3 phase or NVTX scope",
        )
    if mx and ("quant_kernel" in name or "quantize" in name):
        return "Quantization / MXFP8", "MXFP8 source producer + H3 phase or NVTX scope"
    if any(
        token in name
        for token in ("implicit_gemm", "conv", "_fprop_", "_dgrad_", "_wgrad_")
    ):
        return "Convolution / cuDNN and ATen", "convolution symbol family"
    if any(
        token in name
        for token in ("block_sparse", "bsa_fwd", "bsa_attn", "sol_attn_fwd")
    ):
        return "Attention / sparse core", "sparse attention symbol family"
    if any(token in name for token in ("flash_fwd", "fmha", "sdpa", "flash_attention")):
        return "Attention / dense", "dense attention symbol family"
    if any(
        token in name
        for token in ("map_to_index", "blockmask", "topk", "pooling", "pool_kernel")
    ):
        return "Attention / sparse routing", "routing symbol family"
    if any(token in name for token in ("gemm", "_mma_", "matmul", "gemv", "cublas")):
        if mx:
            return "GEMM / MXFP8", "block-scaled ue8m0xe4m3 symbol or MXFP8 scope"
        if any(token in name for token in ("e4m3", "fp8", "qqtst")):
            return "GEMM / FP8", "GEMM scalar-FP8 symbol family"
        if any(token in name for token in ("bf16", "bfloat", "tst_mma")):
            return "GEMM / BF16", "GEMM BF16 symbol family"
        if any(token in name for token in ("_f16", "float16", "::half")):
            return "GEMM / FP16", "GEMM FP16 symbol family"
        return "GEMM / other", "GEMM symbol; precision unresolved"
    if any(
        token in name
        for token in ("rms", "norm", "rope", "rotary", "modulat", "gate_merge")
    ):
        return "Norm / RoPE / modulation", "norm/rotary/modulation symbol family"
    if any(token in name for token in ("swiglu", "silu", "gelu", "activation")):
        return "Activation / MLP", "activation symbol family"
    if any(token in name for token in ("dequant", "act_quant", "quantize")):
        return "Quantization / other", "quantization symbol family"
    if any(token in name for token in ("reduce", "reduction", "amax", "sum_kernel")):
        return "Reduction / statistics", "reduction symbol family"
    if any(
        token in name
        for token in (
            "copy",
            "cast",
            "transpose",
            "permute",
            "concat",
            "gather",
            "index",
        )
    ):
        return "Tensor / layout / copy / cast", "layout/copy/cast symbol family"
    if any(
        token in name
        for token in (
            "elementwise",
            "pointwise",
            "triton_poi_",
            "vectorized",
            "normalize_pixels",
        )
    ):
        return "Pointwise / other fusion", "pointwise symbol family"
    return "Unknown", "insufficient operator evidence"


@dataclass(frozen=True, slots=True)
class ProfileTables:
    modules: Table
    operators: Table
    kernels: Table


def _extend(table: Table) -> Table:
    return Table(
        table.columns
        + (
            "nvtx_path",
            "capture_origin",
            "source_key",
            "original_nvtx_path",
            "original_source_key",
            "attribution_method",
            "bucket_attribution",
        ),
        tuple(
            row
            + (
                "",
                "historical n=1 / Nsys",
                "",
                "",
                "",
                "archived module attribution",
                "",
            )
            for row in table.rows
        ),
    )


def profile_tables(old_report: Path, profile: Profile) -> ProfileTables:
    """Preserve historical buckets and add exclusive new path/shape buckets."""
    old_modules = read_table(old_report / "modules.csv")
    old_operators = read_table(old_report / "operators.csv")
    old_kernels = _extend(read_table(old_report / "kernels.csv"))
    old_modules, old_operators, old_kernels = restore_unknown(
        old_report, old_modules, old_operators, old_kernels, classify_operator
    )
    if not profile.ready:
        return ProfileTables(old_modules, old_operators, old_kernels)
    module_totals: dict[str, tuple[int, int]] = defaultdict(lambda: (0, 0))
    operator_totals: dict[tuple[str, str], tuple[int, int]] = defaultdict(
        lambda: (0, 0)
    )
    buckets: list[tuple[KernelBucket, str, str, str]] = []
    for bucket in profile.buckets:
        module = classify_module(bucket.path)
        family, proof = classify_operator(bucket.symbol, bucket.path)
        count, total = module_totals[module]
        module_totals[module] = count + bucket.count, total + bucket.total_ns
        count, total = operator_totals[module, family]
        operator_totals[module, family] = count + bucket.count, total + bucket.total_ns
        buckets.append((bucket, module, family, proof))
    total_gpu = sum(value[1] for value in module_totals.values())
    modules = tuple(
        (
            "H3Adopted",
            module,
            str(count),
            str(total / 1e9),
            str(100 * total / total_gpu),
            profile.attribution_method + "; exclusive deepest recognized phase",
        )
        for module, (count, total) in sorted(
            module_totals.items(), key=lambda item: -item[1][1]
        )
    )
    operators = tuple(
        (
            "H3Adopted",
            module,
            family,
            str(count),
            str(total / 1e9),
            str(100 * total / module_totals[module][1]),
        )
        for (module, family), (count, total) in sorted(
            operator_totals.items(), key=lambda item: -item[1][1]
        )
    )
    kernels = tuple(
        (
            "H3Adopted",
            module,
            family,
            proof,
            bucket.symbol,
            *(str(value) for value in bucket.grid),
            *(str(value) for value in bucket.block),
            str(bucket.count),
            str(bucket.total_ns / 1e9),
            str(bucket.avg_ns / 1e6),
            str(bucket.p50_ns / 1e6),
            str(bucket.p95_ns / 1e6),
            str(bucket.p99_ns / 1e6),
            str(bucket.min_ns / 1e6),
            str(bucket.max_ns / 1e6),
            str(100 * bucket.total_ns / module_totals[module][1]),
            bucket.path,
            "fresh separate profile request; n=1",
            bucket.key,
            bucket.original_path,
            bucket.original_key,
            profile.attribution_method,
            bucket.attribution,
        )
        for bucket, module, family, proof in buckets
    )
    return ProfileTables(
        Table(old_modules.columns, old_modules.rows + modules),
        Table(old_operators.columns, old_operators.rows + operators),
        Table(old_kernels.columns, old_kernels.rows + kernels),
    )
