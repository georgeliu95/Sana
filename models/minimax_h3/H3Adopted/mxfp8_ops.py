# SPDX-License-Identifier: Apache-2.0
"""Opaque H3 MXFP8 boundaries and runtime dispatch observations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import torch
from torch import Tensor

from .mxfp8_kernels import (
    Mxfp8Tensors,
    QuantizationContractError,
    quantize_swizzled,
    scaled_mm,
    swiglu_quantize,
)


@dataclass(frozen=True, slots=True)
class DispatchReceipt:
    name: str
    gemms: int
    quantizations: int
    fused_swiglu_quantizations: int


class _Counts:
    """An accumulator mutated ONLY by actual opaque GPU dispatch."""

    __slots__ = ("gemms", "quantizations", "fused")

    def __init__(self) -> None:
        self.gemms = 0
        self.quantizations = 0
        self.fused = 0


_COUNTS: Final[dict[str, _Counts]] = {}
_TAGS: Final[list[str]] = []


def register_dispatch(name: str) -> Tensor:
    if name in _COUNTS:
        raise QuantizationContractError(
            name, "duplicate installation in the same process"
        )
    _COUNTS[name] = _Counts()
    _TAGS.append(name)
    return torch.tensor(len(_TAGS) - 1, device="cpu", dtype=torch.int64)


def _dispatch_name(identity: Tensor) -> str:
    if identity.device.type != "cpu" or identity.ndim != 0:
        raise QuantizationContractError("dispatch identity", "requires a CPU scalar")
    return _TAGS[int(identity.item())]


@torch.library.custom_op("h3_adopted::quantize", mutates_args=())
def quantize_op(x: Tensor, identity: Tensor) -> tuple[Tensor, Tensor]:
    tag = _dispatch_name(identity)
    with torch.cuda.nvtx.range(f"H3Adopted/MXFP8/quantize/{tag}"):
        value = quantize_swizzled(x.reshape(-1, x.shape[-1]).contiguous())
    _COUNTS[tag].quantizations += 1
    return value.payload, value.scales


@quantize_op.register_fake
def _quantize_fake(x: Tensor, identity: Tensor) -> tuple[Tensor, Tensor]:
    from .mxfp8_kernels import scale_numel

    rows = x.numel() // x.shape[-1]
    return (
        x.new_empty((rows, x.shape[-1]), dtype=torch.float8_e4m3fn),
        x.new_empty((scale_numel(rows, x.shape[-1]),), dtype=torch.uint8),
    )


@torch.library.custom_op("h3_adopted::linear", mutates_args=())
def linear_op(
    x: Tensor,
    weight: Tensor,
    scales: Tensor,
    identity: Tensor,
    output_dtype: torch.dtype,
) -> Tensor:
    tag = _dispatch_name(identity)
    with torch.cuda.nvtx.range(f"H3Adopted/MXFP8/linear/{tag}"):
        activation = quantize_swizzled(x.reshape(-1, x.shape[-1]).contiguous())
        result = scaled_mm(
            activation, Mxfp8Tensors(weight, scales), output_dtype=output_dtype
        )
    _COUNTS[tag].gemms += 1
    _COUNTS[tag].quantizations += 1
    return result.reshape(*x.shape[:-1], weight.shape[0])


@linear_op.register_fake
def _linear_fake(
    x: Tensor,
    weight: Tensor,
    scales: Tensor,
    identity: Tensor,
    output_dtype: torch.dtype,
) -> Tensor:
    return x.new_empty((*x.shape[:-1], weight.shape[0]), dtype=output_dtype)


@torch.library.custom_op("h3_adopted::prequantized_linear", mutates_args=())
def prequantized_op(
    payload: Tensor,
    activation_scales: Tensor,
    weight: Tensor,
    weight_scales: Tensor,
    identity: Tensor,
) -> Tensor:
    tag = _dispatch_name(identity)
    with torch.cuda.nvtx.range(f"H3Adopted/MXFP8/prequantized_linear/{tag}"):
        result = scaled_mm(
            Mxfp8Tensors(payload, activation_scales),
            Mxfp8Tensors(weight, weight_scales),
        )
    _COUNTS[tag].gemms += 1
    return result


@prequantized_op.register_fake
def _prequantized_fake(
    payload: Tensor,
    activation_scales: Tensor,
    weight: Tensor,
    weight_scales: Tensor,
    identity: Tensor,
) -> Tensor:
    return payload.new_empty((payload.shape[0], weight.shape[0]), dtype=torch.bfloat16)


@torch.library.custom_op("h3_adopted::swiglu_linear", mutates_args=())
def swiglu_op(
    x: Tensor, weight: Tensor, scales: Tensor, identity: Tensor, round_activation: bool
) -> Tensor:
    tag = _dispatch_name(identity)
    with torch.cuda.nvtx.range(f"H3Adopted/MXFP8/swiglu_fc2/{tag}"):
        activation, _ = swiglu_quantize(
            x.reshape(-1, x.shape[-1]).contiguous(), round_activation=round_activation
        )
        result = scaled_mm(activation, Mxfp8Tensors(weight, scales))
    _COUNTS[tag].gemms += 1
    _COUNTS[tag].fused += 1
    return result.reshape(*x.shape[:-1], weight.shape[0])


@swiglu_op.register_fake
def _swiglu_fake(
    x: Tensor, weight: Tensor, scales: Tensor, identity: Tensor, round_activation: bool
) -> Tensor:
    return x.new_empty((*x.shape[:-1], weight.shape[0]))


def dispatch_counts() -> tuple[DispatchReceipt, ...]:
    """Snapshot all actual custom-op calls; fake compilation adds no counts."""
    return tuple(
        DispatchReceipt(name, count.gemms, count.quantizations, count.fused)
        for name, count in sorted(_COUNTS.items())
    )


def snapshot_dispatch(name: str) -> DispatchReceipt:
    """Return actual opaque-call counts for one installed projection."""
    count = _COUNTS[name]
    return DispatchReceipt(name, count.gemms, count.quantizations, count.fused)
