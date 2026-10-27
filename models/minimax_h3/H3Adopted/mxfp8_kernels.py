# SPDX-License-Identifier: Apache-2.0
# Copyright contributors to the vLLM-Omni project.
"""MXFP8 E4M3 payloads with padded SWIZZLE_32_4_4 E8M0 scales.

Adapted from h3-optimizations/vllm-omni/vllm_omni/diffusion/layers/mxfp8.py
at 090f41b83156ae61e6977e7d6069fd36b8fe584c. Quantization arithmetic and
scale layout are retained. SwiGLU follows the selected local target's
value/gate order and BF16 rounding, rather than upstream vLLM's CUDA SiLU.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    import torch
    from torch import Tensor


class QuantizationContractError(ValueError):
    """Typed exception allowing Python to update its traceback during propagation."""

    context: str
    reason: str

    def __init__(self, context: str, reason: str) -> None:
        self.context = context
        self.reason = reason
        super().__init__(f"{context}: {reason}")



@dataclass(frozen=True, slots=True)
class Mxfp8Tensors:
    payload: Tensor
    scales: Tensor


class _Launch(Protocol):
    """Triton launch ABI: resident pointers plus dimensions and compile options."""

    def __call__(self, *args: Tensor | int, **options: int | bool) -> None: ...


class _Kernel(Protocol):
    def __getitem__(self, grid: tuple[int, int]) -> _Launch: ...


def scale_numel(rows: int, k: int) -> int:
    """Return scale bytes, padding rows to128 and scale columns to4."""
    if rows < 0 or k <= 0 or k % 32:
        raise QuantizationContractError("scale layout", f"invalid rows={rows}, k={k}")
    return ((rows + 127) // 128) * 128 * (((k // 32 + 3) // 4) * 4)


@lru_cache(maxsize=1)
def _kernels() -> tuple[_Kernel, _Kernel]:
    """Load Triton only at first GPU dispatch; geometry checks remain CPU-only."""
    import triton
    import triton.language as tl

    @triton.jit
    def exponent(amax):
        bits = amax.to(tl.int32, bitcast=True)
        e0 = ((bits >> 23) & 0xFF) - 135
        threshold = (((e0 + 135) << 23) | 0x600000).to(tl.float32, bitcast=True)
        e = tl.maximum(e0 + (amax > threshold).to(tl.int32), -127)
        return e + 127, ((127 - e) << 23).to(tl.float32, bitcast=True)

    @triton.jit
    def offsets(r, c, n_col_blocks):
        tile = (r // 128) * n_col_blocks + c // 4
        return tile * 512 + (r % 32) * 16 + ((r % 128) // 32) * 4 + c % 4

    @triton.jit
    def bf16_round(x):
        return tl.inline_asm_elementwise(
            "{ .reg .b16 h; cvt.rn.bf16.f32 h, $1; cvt.f32.bf16 $0, h; }",
            constraints="=f,f",
            args=[x],
            dtype=tl.float32,
            is_pure=True,
            pack=1,
        )

    @triton.jit
    def quant_kernel(X, Q, S, rows, k, stride, BR: tl.constexpr, G: tl.constexpr):
        r = tl.program_id(0) * BR + tl.arange(0, BR)
        g = tl.program_id(1) * G + tl.arange(0, G)
        c = tl.program_id(1) * (G * 32) + tl.arange(0, G * 32)
        mask = (r < rows)[:, None] & (c < k)[None, :]
        value = tl.load(X + r[:, None].to(tl.int64) * stride + c[None, :], mask, 0).to(
            tl.float32
        )
        grouped = tl.reshape(value, [BR, G, 32])
        sbyte, inv = exponent(tl.max(tl.abs(grouped), 2))
        q = tl.reshape(grouped * inv[:, :, None], [BR, G * 32])
        tl.store(
            Q + r[:, None].to(tl.int64) * k + c[None, :], q.to(tl.float8e4nv), mask
        )
        tl.store(
            S + offsets(r[:, None], g[None, :], (k // 32 + 3) // 4),
            sbyte.to(tl.uint8),
            (r < rows)[:, None] & (g < k // 32)[None, :],
        )

    @triton.jit
    def swiglu_kernel(
        X,
        Q,
        S,
        Y,
        rows,
        k,
        stride,
        VALUE_FIRST: tl.constexpr,
        ROUND_ACT: tl.constexpr,
        STORE: tl.constexpr,
        BR: tl.constexpr,
        G: tl.constexpr,
    ):
        r = tl.program_id(0) * BR + tl.arange(0, BR)
        g = tl.program_id(1) * G + tl.arange(0, G)
        c = tl.program_id(1) * (G * 32) + tl.arange(0, G * 32)
        mask = (r < rows)[:, None] & (c < k)[None, :]
        base = X + r[:, None].to(tl.int64) * stride
        gate_offset = k if VALUE_FIRST else 0
        value_offset = 0 if VALUE_FIRST else k
        gate = tl.load(base + gate_offset + c[None, :], mask, 0).to(tl.float32)
        value = tl.load(base + value_offset + c[None, :], mask, 0).to(tl.float32)
        activated = gate * tl.sigmoid(gate)
        if ROUND_ACT:
            activated = bf16_round(activated)
        product = bf16_round(value * activated)
        if STORE:
            tl.store(Y + r[:, None].to(tl.int64) * k + c[None, :], product, mask)
        grouped = tl.reshape(product, [BR, G, 32])
        sbyte, inv = exponent(tl.max(tl.abs(grouped), 2))
        q = tl.reshape(grouped * inv[:, :, None], [BR, G * 32])
        tl.store(
            Q + r[:, None].to(tl.int64) * k + c[None, :], q.to(tl.float8e4nv), mask
        )
        tl.store(
            S + offsets(r[:, None], g[None, :], (k // 32 + 3) // 4),
            sbyte.to(tl.uint8),
            (r < rows)[:, None] & (g < k // 32)[None, :],
        )

    return quant_kernel, swiglu_kernel


def _allocate(x: Tensor, k: int) -> Mxfp8Tensors:
    import torch

    return Mxfp8Tensors(
        torch.empty(x.shape[0], k, dtype=torch.float8_e4m3fn, device=x.device),
        torch.zeros(scale_numel(x.shape[0], k), dtype=torch.uint8, device=x.device),
    )


def _resident_matrix(x: Tensor, *, packed: bool) -> int:
    import torch

    width_multiple = 64 if packed else 32
    supported = (
        (torch.bfloat16,) if packed else (torch.bfloat16, torch.float16, torch.float32)
    )
    if (
        x.ndim != 2
        or not x.is_cuda
        or x.dtype not in supported
        or x.stride(-1) != 1
        or x.shape[-1] <= 0
        or x.shape[-1] % width_multiple
    ):
        raise QuantizationContractError(
            "MXFP8 input", f"unsupported {x.shape}, {x.dtype}, {x.device}"
        )
    return x.shape[-1] // 2 if packed else x.shape[-1]


def quantize_swizzled(x: Tensor) -> Mxfp8Tensors:
    """Quantize a resident row-major floating matrix along 32-element K blocks."""
    import torch

    k = _resident_matrix(x, packed=False)
    result = _allocate(x, k)
    if x.shape[0]:
        with torch.cuda.device(x.device):
            _kernels()[0][((x.shape[0] + 31) // 32, (k // 32 + 7) // 8)](
                x,
                result.payload,
                result.scales,
                x.shape[0],
                k,
                x.stride(0),
                BR=32,
                G=8,
                num_warps=4,
            )
    return result


def swiglu_quantize(
    x: Tensor,
    *,
    value_first: bool = True,
    round_activation: bool = True,
    return_bf16: bool = False,
) -> tuple[Mxfp8Tensors, Tensor | None]:
    """Fuse local tl.sigmoid SwiGLU, selected BF16 rounding, and MXFP8 packing.

    round_activation=True matches GB10 fused_swiglu's two BF16 roundings;
    False matches FastVideo minimax_h3_swiglu's final-store-only rounding.
    """
    import torch

    k = _resident_matrix(x, packed=True)
    result = _allocate(x, k)
    rounded = (
        torch.empty(x.shape[0], k, dtype=torch.bfloat16, device=x.device)
        if return_bf16
        else None
    )
    if x.shape[0]:
        with torch.cuda.device(x.device):
            _kernels()[1][((x.shape[0] + 15) // 16, (k // 32 + 7) // 8)](
                x,
                result.payload,
                result.scales,
                result.payload if rounded is None else rounded,
                x.shape[0],
                k,
                x.stride(0),
                VALUE_FIRST=value_first,
                ROUND_ACT=round_activation,
                STORE=return_bf16,
                BR=16,
                G=8,
                num_warps=4,
                enable_fp_fusion=True,
            )
    return result, rounded


def scaled_mm(
    activation: Mxfp8Tensors,
    weight: Mxfp8Tensors,
    *,
    output_dtype: torch.dtype | None = None,
) -> Tensor:
    """Use public PyTorch BlockWise1x32 scaled_mm with E8M0 swizzled scales."""
    import torch
    from torch.nn.functional import ScalingType, SwizzleType

    return torch.nn.functional.scaled_mm(
        activation.payload,
        weight.payload.t(),
        scale_a=activation.scales.view(torch.float8_e8m0fnu),
        scale_b=weight.scales.view(torch.float8_e8m0fnu),
        scale_recipe_a=ScalingType.BlockWise1x32,
        scale_recipe_b=ScalingType.BlockWise1x32,
        swizzle_a=SwizzleType.SWIZZLE_32_4_4,
        swizzle_b=SwizzleType.SWIZZLE_32_4_4,
        output_dtype=torch.bfloat16 if output_dtype is None else output_dtype,
        use_fast_accum=False,
    )
