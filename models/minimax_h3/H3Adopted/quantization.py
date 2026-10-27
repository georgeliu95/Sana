# SPDX-License-Identifier: Apache-2.0
"""Projection adapters for local H3 MXFP8 with actual opaque-op dispatch counts.

The FastVideo installation consumes merged BF16 weights BEFORE native FP8
conversion. Legacy GB10 conversion explicitly requantizes represented weights;
it does not reinterpret the checkpoint's scalar-scaled E4M3 bytes as MXFP8.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MethodType
from typing import Protocol

import torch
from torch import Tensor, nn

from .mxfp8_kernels import Mxfp8Tensors, QuantizationContractError, quantize_swizzled
from .mxfp8_ops import (
    DispatchReceipt as DispatchReceipt,
)
from .mxfp8_ops import (
    dispatch_counts as dispatch_counts,
)
from .mxfp8_ops import (
    linear_op,
    prequantized_op,
    quantize_op,
    register_dispatch,
    snapshot_dispatch,
    swiglu_op,
)


@dataclass(frozen=True, slots=True)
class QuantizationReceipt:
    name: str
    input_features: int
    output_features: int
    source_weight_dtype: str
    source_scheme: str
    destination_scheme: str = "E4M3_E8M0_BlockWise1x32_SWIZZLE_32_4_4"
    activation_precision_changes: bool = True


class Mxfp8Linear(nn.Module):
    """Resident MXFP8 inference Linear; one explicit load-time weight conversion."""

    def __init__(
        self,
        weight: Mxfp8Tensors,
        bias: Tensor | None,
        receipt: QuantizationReceipt,
        output_dtype: torch.dtype,
    ) -> None:
        super().__init__()
        self.register_buffer("weight", weight.payload, persistent=False)
        self.register_buffer("weight_scale", weight.scales, persistent=False)
        self.register_buffer("bias", bias, persistent=False)
        self.receipt = receipt
        self.output_dtype = output_dtype
        self.dispatch_id = register_dispatch(receipt.name)

    @classmethod
    def from_module(
        cls, source: nn.Module, *, name: str, output_dtype: torch.dtype | None = None
    ) -> Mxfp8Linear:
        """Convert nn.Linear or GB10 Fp8Linear, preserving represented bias/weights."""
        parameter = source._parameters.get("weight")
        represented = (
            source.get_buffer("weight") if parameter is None else parameter
        ).detach()
        source_scheme = (
            "GB10_dequantized_W8A16" if parameter is None else "floating_weight"
        )
        original_dtype = str(represented.dtype)
        if represented.dtype == torch.float8_e4m3fn:
            represented = represented.to(torch.bfloat16) * source.get_buffer(
                "weight_scale"
            ).to(torch.bfloat16)
            source_scheme = "GB10_scalar_E4M3_requantization"
        if represented.dtype not in (torch.bfloat16, torch.float16, torch.float32):
            raise QuantizationContractError(
                name, f"unsupported weight dtype {represented.dtype}"
            )
        dtype = represented.dtype if output_dtype is None else output_dtype
        bias = source._parameters.get("bias", source._buffers.get("bias"))
        converted_bias = None if bias is None else bias.detach().to(dtype)
        receipt = QuantizationReceipt(
            name,
            represented.shape[1],
            represented.shape[0],
            original_dtype,
            source_scheme,
        )
        return cls(
            quantize_swizzled(represented.contiguous()), converted_bias, receipt, dtype
        )

    def forward(self, x: Tensor) -> Tensor:
        result = linear_op(
            x, self.weight, self.weight_scale, self.dispatch_id, self.output_dtype
        )
        return result if self.bias is None else result + self.bias

    def dispatch_snapshot(self) -> DispatchReceipt:
        return snapshot_dispatch(self.receipt.name)


class _NativeLinear(Protocol):
    _fp8_weight: Tensor
    _fp8_weight_scale: Tensor
    quant_method: Mxfp8QuantizeMethod
    bias: Tensor | None
    skip_bias_add: bool


class Mxfp8QuantizeMethod:
    """FastVideo apply ABI; native ReplicatedLinear retains tuple/bias handling."""

    def __init__(self, name: str) -> None:
        self.name = name
        self.dispatch_id = register_dispatch(name)

    def quantize_input(self, x: Tensor) -> tuple[Tensor, Tensor, None]:
        payload, scales = quantize_op(x, self.dispatch_id)
        return payload, scales, None

    def wants_prequantized_input(self) -> bool:
        return True

    def apply(
        self,
        layer: _NativeLinear,
        x: Tensor,
        bias: Tensor | None = None,
        pre_quantized: tuple[Tensor, Tensor, None] | None = None,
    ) -> Tensor:
        if pre_quantized is None:
            result = linear_op(
                x,
                layer._fp8_weight,
                layer._fp8_weight_scale,
                self.dispatch_id,
                torch.bfloat16,
            )
        else:
            payload, scales, _ = pre_quantized
            result = prequantized_op(
                payload,
                scales,
                layer._fp8_weight,
                layer._fp8_weight_scale,
                self.dispatch_id,
            )
            result = result.reshape(*x.shape[:-1], layer._fp8_weight.shape[0])
        return result if bias is None else result + bias


def is_fastvideo_target(name: str) -> bool:
    """Select six projections per main block; exclude refiner/AdaLN/gate/head."""
    return name.startswith("transformer_blocks.") and name.endswith(
        (
            ".attn.to_q",
            ".attn.to_k",
            ".attn.to_v",
            ".attn.to_out",
            ".ff.fc_in",
            ".ff.fc_out",
        )
    )


def install_fastvideo(model: nn.Module) -> tuple[QuantizationReceipt, ...]:
    """Mutate loaded single-device main-block linears before tensorwise conversion."""
    from fastvideo.layers.linear import LinearBase, UnquantizedLinearMethod
    from fastvideo.layers.quantization.fp8_config import FP8QuantizeMethod
    from torch.distributed.tensor import DTensor

    receipts: list[QuantizationReceipt] = []
    for name, module in list(model.named_modules()):
        if not is_fastvideo_target(name):
            if isinstance(module, LinearBase) and isinstance(
                module.quant_method, FP8QuantizeMethod
            ):
                module.quant_method = UnquantizedLinearMethod()
            continue
        if not isinstance(module, LinearBase):
            raise QuantizationContractError(name, "target is not native LinearBase")
        local = (
            module.weight.to_local()
            if isinstance(module.weight, DTensor)
            else module.weight
        )
        represented = local.detach()
        if represented.dtype != torch.bfloat16 or not represented.is_cuda:
            raise QuantizationContractError(
                name, "expected merged resident BF16 weight"
            )
        quantized = quantize_swizzled(represented.contiguous())
        module.register_buffer("_fp8_weight", quantized.payload, persistent=False)
        module.register_buffer("_fp8_weight_scale", quantized.scales, persistent=False)
        module.quant_method = Mxfp8QuantizeMethod(name)
        del module._parameters["weight"]
        receipts.append(
            QuantizationReceipt(
                name,
                represented.shape[1],
                represented.shape[0],
                str(represented.dtype),
                "FastVideo_merged_BF16_direct",
            )
        )
    if not receipts:
        raise QuantizationContractError(
            "FastVideo installation", "no main projections selected"
        )
    return tuple(receipts)


class _TupleLinear(Protocol):
    def __call__(self, x: Tensor) -> tuple[Tensor, Tensor | None]: ...


class _FeedForward(Protocol):
    fc_in: _TupleLinear
    fc_out: _NativeLinear


def _fastvideo_ff(self: _FeedForward, x: Tensor) -> Tensor:
    packed, _ = self.fc_in(x)
    identity = self.fc_out.quant_method.dispatch_id
    return swiglu_op(
        packed, self.fc_out._fp8_weight, self.fc_out._fp8_weight_scale, identity, False
    )


def install_fastvideo_swiglu(model: nn.Module) -> tuple[str, ...]:
    """Fuse FastVideo's value-first, final-BF16-store-only SwiGLU into FC2 input."""
    names: list[str] = []
    for name, module in model.named_modules():
        if name.startswith("transformer_blocks.") and name.endswith(".ff"):
            if module.fc_out.bias is not None or not module.fuse_swiglu:
                raise QuantizationContractError(
                    name, "requires bias-free native fused SwiGLU mode"
                )
            module.forward = MethodType(_fastvideo_ff, module)
            names.append(name)
    if not names:
        raise QuantizationContractError(
            "SwiGLU installation", "no native fused feed-forwards"
        )
    return tuple(names)


def install_transformer(model: nn.Module) -> tuple[QuantizationReceipt, ...]:
    """Replace legacy Diffusers projections through the separate checkpoint adapter."""
    from .legacy_quantization import install_transformer as install_legacy

    return install_legacy(model)


def project_qkv(
    q_layer: _NativeLinear,
    k_layer: _NativeLinear,
    v_layer: _NativeLinear,
    x: Tensor,
) -> tuple[Tensor, Tensor, Tensor]:
    """Quantize the shared activation once while preserving native bias handling."""
    prequantized = q_layer.quant_method.quantize_input(x)
    query, key, value = (
        layer.quant_method.apply(
            layer,
            x,
            None if layer.skip_bias_add else layer.bias,
            pre_quantized=prequantized,
        )
        for layer in (q_layer, k_layer, v_layer)
    )
    return query, key, value
