"""LTX receive-to-device and K/V summary fusion for four existing RDMA slots."""
from __future__ import annotations
from dataclasses import dataclass
import functools
import torch
import triton
import triton.language as tl

@dataclass(frozen=True, slots=True)
class PipelineContractError(RuntimeError):
    detail: str

@triton.jit
def receive_summary(S0, S1, S2, S3, Dest, KC, VC,
                    SHARD: tl.constexpr, H: tl.constexpr, N: tl.constexpr,
                    B0: tl.constexpr, B1: tl.constexpr, B2: tl.constexpr, B3: tl.constexpr, ITEM: tl.constexpr):
    block, head = tl.program_id(0), tl.program_id(1)
    item: tl.constexpr = ITEM
    rows = block * 64 + tl.arange(0, 64)
    dims = tl.arange(0, 128)
    owner = rows // SHARD
    src = tl.where(owner == 0, S0, tl.where(owner == 1, S1, tl.where(owner == 2, S2, S3)))
    batch_stride = tl.where(owner == 0, B0, tl.where(owner == 1, B1, tl.where(owner == 2, B2, B3)))
    offset = item * batch_stride + (rows % SHARD * H + head) * 128
    values = tl.load(src[:, None] + offset[:, None] + dims[None, :],
                     (rows < 4 * SHARD)[:, None], 0)
    tl.store(Dest + ((item * 4 * SHARD + rows[:, None]) * H + head) * 128 + dims[None, :],
             values, (rows < 4 * SHARD)[:, None])
    if item == 1:
        q_offset = (rows % SHARD * H + head) * 128
        q = tl.load(src[:, None] + q_offset[:, None] + dims[None, :],
                    (rows < 4 * SHARD)[:, None], 0)
        tl.store(Dest + (rows[:, None] * H + head) * 128 + dims[None, :],
                 q, (rows < 4 * SHARD)[:, None])
        summary = tl.sum(values, axis=0) / tl.minimum(64, 4 * SHARD - block * 64)
        tl.store(KC + (block * H + head) * 128 + dims, summary)
    if item == 2:
        tl.store(VC + (block * H + head) * 128 + dims, tl.sum(values, axis=0))

@functools.lru_cache(maxsize=4)
def native_reduction_warps(tokens: int, heads: int, device: torch.device) -> tuple[int, int]:
    """Calibrate once in warmup: BF16 sum order depends on the native warp layout."""
    from .preprocess import _reduce_kv, _reduce_kc_kernel, _reduce_vc_kernel
    values = torch.zeros((1, tokens, heads, 128), device=device, dtype=torch.bfloat16)
    _reduce_kv(values, values)
    return _reduce_kc_kernel.best_config.num_warps, _reduce_vc_kernel.best_config.num_warps

def receive_prepare(sources: tuple[torch.Tensor, ...]) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
    """Receive every value once, preserving native tile64 summary arithmetic."""
    if len(sources) != 4 or sources[0].ndim != 4:
        raise PipelineContractError("four Q/K/V source slots are required")
    _, shard, heads, dim = sources[0].shape
    if shard <= 0 or dim != 128 or any(
        x.shape != (3, shard, heads, 128) or x.dtype != torch.bfloat16 or not x.is_cuda
        or x.device != sources[0].device or x.stride()[1:] != (heads * 128, 128, 1)
        for x in sources
    ):
        raise PipelineContractError("receive fusion requires BF16 BTHD slots, D128 and batch one")
    gathered = torch.empty((3, 4 * shard, heads, 128), dtype=sources[0].dtype, device=sources[0].device)
    blocks = triton.cdiv(4 * shard, 64)
    kc = torch.empty((1, blocks, heads, 128), dtype=gathered.dtype, device=gathered.device)
    vc = torch.empty_like(kc)
    warps = native_reduction_warps(4 * shard, heads, gathered.device)
    for item, count in enumerate(warps, start=1):
        receive_summary[(blocks, heads)](*sources, gathered, kc, vc, shard, heads, blocks,
                                        *(x.stride(0) for x in sources), ITEM=item,
                                        num_warps=count, num_stages=1)
    return gathered, (kc, vc)

def forward(gathered: torch.Tensor, tau: float, targets: tuple[torch.Tensor, ...] | None = None,
            summaries: tuple[torch.Tensor, torch.Tensor] | None = None) -> torch.Tensor | None:
    """Use native attention arithmetic, optionally reading summaries and writing peer slots."""
    from triton.tools.tensor_descriptor import TensorDescriptor
    from .preprocess import _compute_diag_threshold, _reduce_kv
    from .triton_ref.fwd import _forward_tma

    if (gathered.ndim != 4 or gathered.shape[0] != 3 or gathered.shape[-1] != 128
        or not gathered.is_contiguous() or not gathered.is_cuda or gathered.dtype != torch.bfloat16
        or gathered.shape[1] % 4 or torch.cuda.get_device_capability(gathered.device) != (12, 1)):
        raise PipelineContractError("LTX fusion is qualified for contiguous BF16 QKV on GB10 only")
    _, tokens, heads, dim = gathered.shape
    shard = tokens // 4
    if targets is not None and (len(targets) != 4 or any(
        x.shape != (shard, heads, dim) or x.dtype != gathered.dtype
        or x.device != gathered.device or not x.is_contiguous() for x in targets
    )):
        raise PipelineContractError("output slot geometry differs from the four-rank wire layout")
    q, k, v = gathered.chunk(3, dim=0)
    kc, vc = _reduce_kv(k, v) if summaries is None else summaries
    blocks = triton.cdiv(tokens, 64)
    if any(x.shape != (1, blocks, heads, 128) or x.dtype != gathered.dtype
           or x.device != gathered.device or not x.is_contiguous() for x in (kc, vc)):
        raise PipelineContractError("precomputed summary geometry differs from native tile64")
    threshold = _compute_diag_threshold(q, kc, tau=tau, scale=dim ** -0.5)
    output = torch.empty_like(v) if targets is None else None
    desc_output = q if output is None else output
    slots = (q, q, q, q) if targets is None else targets
    _forward_tma[(1, blocks, heads)](
        *(TensorDescriptor.from_tensor(x, [1, 64, 1, dim]) for x in (q, k, v)),
        *(TensorDescriptor.from_tensor(x, [1, 32, 1, dim]) for x in (kc, vc)),
        threshold, TensorDescriptor.from_tensor(desc_output, [1, 64, 1, dim]),
        dim ** -0.5, tokens, blocks, blocks, False, heads, dim, blocks, dim, 64, 32,
        *slots, shard, targets is not None,
    )
    return output
