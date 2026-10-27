"""Fused NVFP4 weight-only linear for the resident Qwen conditioner.

ComfyUI runs the MiniMax Qwen3-VL text encoder with FP32 activations and ``full_precision_mm``: every NVFP4 Linear
dequantizes its packed weight to a full FP32 copy (``comfy_kitchen.dequantize_nvfp4``) and then runs a TF32 GEMM.
At prompt lengths of tens of tokens both steps only stream weights: 4.5 bytes written and read back per weight for
0.56 bytes of NVFP4 storage. This kernel reads the packed E2M1 codes and the cuBLAS-swizzled E4M3 block scales
directly and multiplies in the same TF32 precision class:

* code x block scale has at most 6 significant bits, so the decoded weight is exact in TF32 (the dequantized FP32 copy
  also folds in the per-tensor scale and is then truncated to TF32 by the GEMM); the per-tensor scale is applied to
  the FP32 result instead;
* activations are rounded to nearest TF32, as cuBLAS does;
* split-K partial sums are added in a fixed order (no atomics), so results are deterministic.

The result is not bitwise equal to the ComfyUI path (different accumulation order); against an FP64 product it is
slightly more accurate. Only the resident, LoRA-free, FP32 ``full_precision_mm`` case is routed here; anything else
keeps ComfyUI's forward.
"""
from __future__ import annotations

import os
import types

import torch
import triton
import triton.language as tl

_MAX_TOKENS = 512
_LAYOUT = "TensorCoreNVFP4Layout"


@triton.jit
def _nvfp4_linear_kernel(x_ptr, q_ptr, s_ptr, out_ptr, M, N, K, stride_xm, scale_col_blocks,
                         BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
                         K_PER_SPLIT: tl.constexpr):
    pid_n = tl.program_id(0)
    pid_k = tl.program_id(1)
    offs_m = tl.program_id(2) * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    # cuBLAS block-scale layout: 128 x 4 tiles, row r of a tile at (r % 32) * 16 + (r // 32) * 4.
    row_part = (offs_n // 128) * scale_col_blocks * 512 + (offs_n % 32) * 16 + ((offs_n % 128) // 32) * 4
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k0 in range(pid_k * K_PER_SPLIT, (pid_k + 1) * K_PER_SPLIT, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)
        x = tl.load(x_ptr + offs_m[:, None] * stride_xm + offs_k[None, :], mask=offs_m[:, None] < M, other=0.0)
        # Round to nearest TF32 (as cuBLAS does); the MMA alone would drop the low 13 mantissa bits.
        x = ((x.to(tl.int32, bitcast=True) + 0x1000) & -8192).to(tl.float32, bitcast=True)
        packed = tl.load(q_ptr + offs_n[:, None] * (K // 2) + (k0 // 2 + tl.arange(0, BLOCK_K // 2))[None, :])
        packed = packed.to(tl.int32)
        # High nibble first: element 2j is byte j >> 4, element 2j + 1 is byte j & 15.
        codes = tl.reshape(tl.join(packed >> 4, packed & 15), (BLOCK_N, BLOCK_K))
        exponent = (codes >> 1) & 3
        mantissa = codes & 1
        bits = tl.where(exponent == 0, mantissa * (126 << 23), ((exponent + 126) << 23) | (mantissa << 22))
        value = (bits | ((codes & 8) << 28)).to(tl.float32, bitcast=True)
        blocks = k0 // 16 + tl.arange(0, BLOCK_K // 16)
        scale_index = row_part[:, None] + (blocks // 4)[None, :] * 512 + (blocks % 4)[None, :]
        scale = tl.load(s_ptr + scale_index).to(tl.float32)
        w = tl.reshape(tl.reshape(value, (BLOCK_N, BLOCK_K // 16, 16)) * scale[:, :, None], (BLOCK_N, BLOCK_K))
        acc += tl.dot(x, tl.trans(w), input_precision="tf32")
    out = out_ptr + pid_k * M * N + offs_m[:, None] * N + offs_n[None, :]
    tl.store(out, acc, mask=offs_m[:, None] < M)


def _config(n: int, k: int) -> tuple[int, int, int, int, int] | None:
    """(BLOCK_N, BLOCK_K, split-K, stages, warps), from a sweep over the Qwen3-VL-32B shapes on GB10."""
    if n >= 16384:
        cfg = (64, 128, 1, 2, 4)
    else:
        cfg = (256, 64, 8 if n <= 2048 else 4, 3, 8)
    block_n, block_k, split_k = cfg[:3]
    return cfg if n % block_n == 0 and k % (block_k * split_k) == 0 else None


def nvfp4_linear(x: torch.Tensor, qdata: torch.Tensor, block_scale: torch.Tensor,
                 tensor_scale: torch.Tensor) -> torch.Tensor:
    """``x @ dequantize_nvfp4(qdata, tensor_scale, block_scale).T`` for FP32 ``x`` of shape (M, K)."""
    m, k = x.shape
    n = qdata.shape[0]
    block_n, block_k, split_k, stages, warps = _config(n, k)
    x = x.contiguous()
    partial = torch.empty((split_k, m, n), device=x.device, dtype=torch.float32)
    grid = (n // block_n, split_k, triton.cdiv(m, 64))
    _nvfp4_linear_kernel[grid](x, qdata, block_scale, partial, m, n, k, x.stride(0), block_scale.shape[1] // 4,
                               BLOCK_M=64, BLOCK_N=block_n, BLOCK_K=block_k, K_PER_SPLIT=k // split_k,
                               num_stages=stages, num_warps=warps)
    out = partial[0] if split_k == 1 else partial.sum(0)
    return out * tensor_scale


def _ineligible(self, input, args, kwargs) -> str | None:
    """Why ``input`` must keep ComfyUI's forward, or None when the fused kernel computes the same linear."""
    weight = self.weight
    params = getattr(weight, "_params", None)
    checks = (
        ("extra arguments", not args and not kwargs),
        ("weight layout", getattr(weight, "_layout_cls", None) == _LAYOUT),
        ("input dtype/device", input.dtype == torch.float32 and input.is_cuda),
        # ComfyUI dequantizes to the input dtype when either flag is set (the CLIP sets a forced FP32 cast).
        ("NVFP4 matmul path", bool(getattr(self, "_full_precision_mm", False)
                                   or getattr(self, "comfy_force_cast_weights", False))),
        ("weight/bias function", not self.weight_function and not self.bias_function),
        ("transposed weight", not getattr(params, "transposed", False)),
        ("padded weight", tuple(weight._qdata.shape) == (self.out_features, self.in_features // 2)),
        ("too many tokens", input.numel() // self.in_features <= _MAX_TOKENS),
        ("shape", _config(self.out_features, self.in_features) is not None),
    )
    return next((reason for reason, ok in checks if not ok), None)


def _fused_forward(self, input, *args, **kwargs):
    import comfy.model_management
    import comfy.ops

    if _ineligible(self, input, args, kwargs) is not None:
        return type(self).forward(self, input, *args, **kwargs)
    weight = self.weight
    params = weight._params
    comfy.ops.run_every_op()
    # ModelOpt AWQ-style smoothing, as in ComfyUI's forward.
    pre_quant_scale = getattr(self, "pre_quant_scale", None)
    if pre_quant_scale is not None:
        input = input * comfy.model_management.cast_to_device(pre_quant_scale, input.device, input.dtype)
    out = nvfp4_linear(input.reshape(-1, self.in_features), weight._qdata, params.block_scale, params.scale)
    if self.bias is not None:
        out = out + self.bias.to(device=out.device, dtype=out.dtype)
    return out.view(*input.shape[:-1], self.out_features)


def install(model: torch.nn.Module) -> int:
    """Route every ComfyUI NVFP4 Linear in ``model`` through the fused kernel; returns how many.

    ``SOL_QWEN_NVFP4_FUSED=0`` keeps ComfyUI's dequantize-then-GEMM forward everywhere.
    """
    if os.environ.get("SOL_QWEN_NVFP4_FUSED", "1") == "0":
        return 0
    count = 0
    for module in model.modules():
        if getattr(module, "layout_type", None) == _LAYOUT and hasattr(module, "_full_precision_mm"):
            module.forward = types.MethodType(_fused_forward, module)
            count += 1
    return count
