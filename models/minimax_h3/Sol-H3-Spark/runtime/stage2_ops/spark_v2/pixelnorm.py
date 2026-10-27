"""In-place PixelNorm, optional BF16 affine, and SiLU for VAE workspaces."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice


@dataclass(frozen=True, slots=True)
class WorkspaceContractError(Exception):
    """Reject a layout or dtype outside the qualified workspace contract."""

    shape: tuple[int, ...]
    dtype: str


@triton.jit
def _pixel_norm_silu(
    X: tl.tensor, Scale: tl.tensor, Shift: tl.tensor,
    B: tl.constexpr, C: tl.constexpr, T: tl.constexpr, H: tl.constexpr, W: tl.constexpr,
    X0: tl.constexpr, X1: tl.constexpr, X2: tl.constexpr, X3: tl.constexpr, X4: tl.constexpr,
    A0: tl.constexpr, A1: tl.constexpr, A2: tl.constexpr, A3: tl.constexpr, A4: tl.constexpr,
    D0: tl.constexpr, D1: tl.constexpr, D2: tl.constexpr, D3: tl.constexpr, D4: tl.constexpr,
    EPS: tl.constexpr, AFFINE: tl.constexpr, BC: tl.constexpr, BP: tl.constexpr,
) -> None:
    pixels = tl.program_id(0) * BP + tl.arange(0, BP)
    channel = tl.arange(0, BC)
    w = pixels % W
    h = pixels // W % H
    t = pixels // (W * H) % T
    b = pixels // (W * H * T)
    offset = (b.to(tl.int64) * X0 + t.to(tl.int64) * X2 + h.to(tl.int64) * X3 + w.to(tl.int64) * X4)[:, None] + channel[None, :] * X1
    mask = (pixels[:, None] < B * T * H * W) & (channel[None, :] < C)
    dtype = X.dtype.element_ty
    value = tl.load(X + offset, mask=mask, other=0).to(tl.float32)
    square = (value * value).to(dtype).to(tl.float32)
    if C >= 128 and C % 128 == 0:
        # Captured BF16 MeanOps uses input_vec_size=8, vt0=4.
        LANES: tl.constexpr = min(32, C // 8)
        lanes = tl.arange(0, LANES)[None, :]
        lane_total = tl.full((BP, LANES), 0, tl.float32)
        for component in tl.static_range(8):
            accumulator = tl.full((BP, LANES), 0, tl.float32)
            for group in tl.static_range(C // (LANES * 8)):
                idx = tl.broadcast_to(lanes * 8 + component + group * LANES * 8, (BP, LANES))
                accumulator = accumulator + tl.gather(square, idx, axis=1)
            lane_total = lane_total + accumulator
        for stage in tl.static_range(5 if LANES == 32 else 4):
            shift = LANES >> (stage + 1)
            other = tl.gather(lane_total, tl.broadcast_to(tl.minimum(lanes + shift, LANES - 1), (BP, LANES)), axis=1)
            lane_total = lane_total + other
        total = tl.sum(tl.where(lanes == 0, lane_total, 0), axis=1)
    else:
        total = tl.sum(square, axis=1)
    mean = (total / C).to(dtype).to(tl.float32)
    variance = (mean + EPS).to(dtype).to(tl.float32)
    rms = libdevice.sqrt_rn(variance).to(dtype).to(tl.float32)
    value = libdevice.div_rn(value, rms[:, None]).to(dtype).to(tl.float32)
    if AFFINE:
        s = (b.to(tl.int64) * A0 + t * A2 + h * A3 + w * A4)[:, None] + channel[None, :] * A1
        d = (b.to(tl.int64) * D0 + t * D2 + h * D3 + w * D4)[:, None] + channel[None, :] * D1
        scale = tl.load(Scale + s, mask=mask, other=0).to(tl.float32)
        shift = tl.load(Shift + d, mask=mask, other=0).to(tl.float32)
        multiplier = (scale + 1.0).to(dtype).to(tl.float32)
        value = (value * multiplier).to(dtype).to(tl.float32)
        value = (value + shift).to(dtype).to(tl.float32)
    result = libdevice.div_rn(value, 1.0 + libdevice.exp(-value))
    tl.store(X + offset, result.to(dtype), mask=mask)


def pixel_norm_silu_(
    x: torch.Tensor,
    eps: float = 1e-8,
    affine: tuple[torch.Tensor, torch.Tensor] | None = None,
) -> None:
    """Preserve eager BF16 materialization points and write only the passed view."""
    if x.ndim != 5 or x.dtype != torch.bfloat16 or x.device.type != "cuda":
        raise WorkspaceContractError(tuple(x.shape), str(x.dtype))
    if affine is None:
        scale, shift = x, x
    else:
        scale, shift = (tensor.expand_as(x) for tensor in affine)
    batch, channels, frames, height, width = x.shape
    _pixel_norm_silu[(triton.cdiv(batch * frames * height * width, 8),)](
        x, scale, shift, batch, channels, frames, height, width,
        *x.stride(), *scale.stride(), *shift.stride(),
        eps, affine is not None, triton.next_power_of_2(channels), 8,
        num_warps=4, enable_fp_fusion=False,
    )
