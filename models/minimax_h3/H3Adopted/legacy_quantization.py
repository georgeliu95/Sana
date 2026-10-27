# SPDX-License-Identifier: Apache-2.0
"""Legacy Diffusers checkpoint adapter kept separate from the FastVideo runtime."""

from __future__ import annotations

from torch import nn

from .mxfp8_kernels import QuantizationContractError
from .quantization import Mxfp8Linear, QuantizationReceipt


def install_transformer(model: nn.Module) -> tuple[QuantizationReceipt, ...]:
    """Replace only the legacy Diffusers main-block attention and FF projections."""
    suffixes = (
        ".attn.to_qkv",
        ".attn.to_q",
        ".attn.to_k",
        ".attn.to_v",
        ".attn.to_out.0",
        ".ff.net.0.proj",
        ".ff.net.2",
    )
    receipts: list[QuantizationReceipt] = []
    for name, module in list(model.named_modules()):
        if name.startswith("transformer_blocks.") and name.endswith(suffixes):
            converted = Mxfp8Linear.from_module(module, name=name)
            model.set_submodule(name, converted)
            receipts.append(converted.receipt)
    if not receipts:
        raise QuantizationContractError(
            "Diffusers installation", "no main projections selected"
        )
    return tuple(receipts)
