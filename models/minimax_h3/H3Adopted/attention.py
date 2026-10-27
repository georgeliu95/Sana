"""Experimental native Sage candidate; default route stays cuDNN until qualified."""
from __future__ import annotations
import torch
from torch import Tensor

from .attention_vendor.bsa_utils.sage_quant_sm120 import quantize_sage_qkv_sm120
from .attention_vendor.sage_provider import bsa_attn_sm120_blk64_sage_fwd


def attention(q: Tensor,k: Tensor,v: Tensor,indices: Tensor,counts: Tensor,
              scale: float,*,block_sizes: Tensor|None=None) -> Tensor:
    """Accept BSHD views and full per-row/partial-tile metadata."""
    if q.dtype!=torch.bfloat16 or q.shape[-1]!=128 or q.ndim!=4:
        raise ValueError("native Sage requires rank4 BF16 BSHD with D128")
    if k.shape!=v.shape or k.shape[:1]+k.shape[2:]!=q.shape[:1]+q.shape[2:]:
        raise ValueError("native Sage candidate supports MHA only")
    with torch.cuda.nvtx.range("H3Adopted/Sage/quantize"):
        operands=quantize_sage_qkv_sm120(q.transpose(1,2),k.transpose(1,2),v.transpose(1,2))
    output=torch.empty_like(q)
    with torch.cuda.nvtx.range("H3Adopted/Sage/attention"):
        bsa_attn_sm120_blk64_sage_fwd(*operands,indices,indices.shape[-1],
                                     block_sizes=block_sizes,q2k_block_nums=counts,
                                     softmax_scale=scale,out=output.transpose(1,2))
    return output


def sage_bsa_forward(q: Tensor,k: Tensor,v: Tensor,indices: Tensor,*,
                     q2k_block_nums: Tensor,block_sizes: Tensor,
                     sparse_block_size: int,softmax_scale: float,layout: str) -> tuple[Tensor,Tensor]:
    """Selected-only native VSA ABI; its caller discards LSE."""
    if sparse_block_size != 64 or layout != "bshd":
        raise ValueError("native Sage integration requires tile64 BSHD")
    out = attention(q,k,v,indices,q2k_block_nums,softmax_scale,block_sizes=block_sizes)
    return out,torch.empty(0,device=q.device,dtype=torch.float32)
