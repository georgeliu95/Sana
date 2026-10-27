"""BF16 projection concatenation within one attention call, without step caching."""

from __future__ import annotations

from dataclasses import dataclass
from typing import assert_never, override

import torch
from torch import Tensor, nn
from torch.distributed.tensor import DTensor
from ltx_core.model.transformer.attention import Attention


@dataclass(frozen=True, slots=True)
class ProjectionContractError(RuntimeError):
    """Reject an attention projection outside the tested resident BF16 contract."""

    role: str


def local(value: Tensor) -> Tensor:
    """Resolve singleton distributed storage without changing its represented values."""
    match value:
        case DTensor():
            return value.to_local()
        case Tensor():
            return value
        case unreachable:
            assert_never(unreachable)


class MergedAttention(nn.Module):
    """Own merged weights and preserve every original attention branch."""

    def __init__(self, original: Attention, *, self_attention: bool) -> None:
        super().__init__()
        self.original = original
        self.self_attention = self_attention
        layers = (original.to_q, original.to_k, original.to_v) if self_attention else (original.to_k, original.to_v)
        weights = tuple(local(layer.weight).detach() for layer in layers)
        biases = tuple(local(layer.bias).detach() for layer in layers if layer.bias is not None)
        if len(biases) != len(layers) or any(w.dtype != torch.bfloat16 or not w.is_cuda for w in weights):
            raise ProjectionContractError(role="attention qkv/kv")
        self.width = weights[0].shape[0]
        self.merged = nn.Linear(weights[0].shape[1], sum(w.shape[0] for w in weights), bias=True, device="meta", dtype=torch.bfloat16)
        self.merged.weight = nn.Parameter(torch.cat(weights), requires_grad=False)
        self.merged.bias = nn.Parameter(torch.cat(biases), requires_grad=False)

    @override
    def forward(self, x: Tensor, context: Tensor | None = None, mask: Tensor | None = None,
                pe: Tensor | None = None, k_pe: Tensor | None = None,
                perturbation_mask: Tensor | None = None, all_perturbed: bool = False) -> Tensor:
        """Merge only projections sharing the same input in the current call."""
        source = self.original
        if all_perturbed:
            values = source.to_v(x if context is None else context)
            output = values
        else:
            if self.self_attention and context is None:
                q, k, values = self.merged(x).split(self.width, dim=-1)
            else:
                context = x if context is None else context
                if self.self_attention:
                    # A nonstandard explicit context keeps the exact original path.
                    q, k, values = source.to_q(x), source.to_k(context), source.to_v(context)
                else:
                    q = source.to_q(x)
                    k, values = self.merged(context).split(self.width, dim=-1)
            values = values.contiguous()
            q, k = source.preattention_function(q, k, source, mask, pe, k_pe)
            if mask is None:
                output = source.attention_function(q, k, values, source.heads)
            else:
                output = source.masked_attention_function(q, k, values, source.heads, mask)
            if perturbation_mask is not None:
                output = output * perturbation_mask + values * (1 - perturbation_mask)
        if source.to_gate_logits is not None:
            output = source.gated_attention_function(x, output, source)
        return source.to_out(output)


def install(core: nn.Module) -> int:
    """Replace six attention modules per block after the original Sol installation."""
    count = 0
    for block in core.get_submodule("transformer_blocks").children():
        while "_orig_mod" in dict(block.named_children()):
            block = block.get_submodule("_orig_mod")
        for name in ("attn1", "audio_attn1", "attn2", "audio_attn2", "audio_to_video_attn", "video_to_audio_attn"):
            source = block.get_submodule(name)
            match source:
                case Attention():
                    block.add_module(name, MergedAttention(source, self_attention=name in ("attn1", "audio_attn1")))
                    count += 1
                case _:
                    raise ProjectionContractError(role=name)
    return count
