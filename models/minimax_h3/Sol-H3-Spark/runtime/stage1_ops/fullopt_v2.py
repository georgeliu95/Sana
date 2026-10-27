"""Optional FC2 MXFP8 using the same operators qualified for Fullopt_v2."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import assert_never

from torch import nn


class Mode(StrEnum):
    """Native preserves the released Spark W8A8 path."""

    NATIVE = "native"
    FC2 = "fc2_mxfp8"
    FC2_SWIGLU = "fc2_mxfp8_swiglu"


@dataclass(frozen=True, slots=True)
class Receipt:
    """Describe the changed projection scope and its represented source weights."""

    mode: Mode
    body_fc2_projections: int
    fused_feedforwards: int
    source_weights: str = "resident BF16 after the original student LoRA merge"
    destination: str = "MXFP8 E4M3/E8M0 BlockWise1x32 SWIZZLE_32_4_4"
    activation_precision_changed: bool = True


def _activate_namespace() -> None:
    """Resolve shared operators before either snapshots or installation."""
    model_root = str(Path(__file__).resolve().parents[3])
    if model_root not in sys.path:
        sys.path.insert(0, model_root)


def install(model: nn.Module, mode: Mode) -> Receipt:
    """Expose only the 50 FC2 linears to the existing FastVideo MXFP8 installer."""
    if mode is Mode.NATIVE:
        return Receipt(
            mode,
            0,
            0,
            destination="native tensorwise W8A8",
            activation_precision_changed=False,
        )
    _activate_namespace()
    from H3Adopted.quantization import install_fastvideo, install_fastvideo_swiglu

    view = nn.Module()
    bodies = nn.ModuleList()
    for index in range(50):
        body, feedforward = nn.Module(), nn.Module()
        feedforward.add_module(
            "fc_out", model.get_submodule(f"transformer_blocks.{index}.ff.fc_out")
        )
        body.add_module("ff", feedforward)
        bodies.append(body)
    view.add_module("transformer_blocks", bodies)
    projections = install_fastvideo(view)
    if len(projections) != 50:
        raise RuntimeError(f"expected 50 FC2 projections, got {len(projections)}")
    match mode:
        case Mode.FC2:
            fused = ()
        case Mode.FC2_SWIGLU:
            fused = install_fastvideo_swiglu(model)
        case Mode.NATIVE:
            raise RuntimeError("native mode must return before MXFP8 installation")
        case unreachable:
            assert_never(unreachable)
    return Receipt(mode, len(projections), len(fused))


@dataclass(frozen=True, slots=True)
class Counts:
    """Accumulate only real custom-op execution, excluding fake tracing."""

    gemms: int = 0
    quantizations: int = 0
    fused_swiglu_quantizations: int = 0


def snapshot(mode: Mode) -> Counts:
    """Return cumulative actual FC2 calls in this dedicated model worker."""
    if mode is Mode.NATIVE:
        return Counts()
    _activate_namespace()
    from H3Adopted.mxfp8_ops import dispatch_counts

    rows = dispatch_counts()
    return Counts(
        sum(row.gemms for row in rows),
        sum(row.quantizations for row in rows),
        sum(row.fused_swiglu_quantizations for row in rows),
    )


def delta(before: Counts, after: Counts) -> Counts:
    """Use per-request call differences rather than cumulative warmup counters."""
    return Counts(
        after.gemms - before.gemms,
        after.quantizations - before.quantizations,
        after.fused_swiglu_quantizations - before.fused_swiglu_quantizations,
    )
