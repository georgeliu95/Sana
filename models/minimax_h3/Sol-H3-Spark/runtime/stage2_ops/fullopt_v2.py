"""Opt-in MXFP8 for the 48 BF16 video FC2 projections in LTX."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Final, assert_never, override

import torch
from torch import Tensor, nn
from torch.distributed.tensor import DTensor

BLOCKS: Final = 48
WEIGHT_SHAPE: Final = (4096, 16384)


@dataclass(frozen=True, slots=True)
class WeightContractError(RuntimeError):
    """Reject weights outside the matched LTX video projection contract."""

    shape: tuple[int, ...]
    dtype: str

    @override
    def __str__(self) -> str:
        """Describe the represented weight that was refused."""
        return f"expected CUDA BF16 {WEIGHT_SHAPE}, got {self.shape}/{self.dtype}"


@dataclass(frozen=True, slots=True)
class Receipt:
    """Limit the precision change to video FC2 after the original LoRA merge."""

    projections: int = BLOCKS
    source: str = "resident merged BF16 LTX video FC2"
    destination: str = "MXFP8 E4M3/E8M0 BlockWise1x32 SWIZZLE_32_4_4"
    audio_fc2_changed: bool = False
    activation_precision_changed: bool = True


def _local(tensor: Tensor) -> Tensor:
    """Resolve an existing singleton distributed tensor without new communication."""
    match tensor:
        case DTensor():
            return tensor.to_local()
        case Tensor():
            return tensor
        case unreachable:
            assert_never(unreachable)


def install(model: nn.Module) -> Receipt:
    """Replace only video FC2; reuse the already qualified generic MXFP8 Linear."""
    root = str(Path(__file__).resolve().parents[3])
    if root not in sys.path:
        sys.path.insert(0, root)
    from H3Adopted.quantization import Mxfp8Linear  # noqa: PLC0415 - Opt-in dependency.

    for index in range(BLOCKS):
        ff = model.get_submodule(f"transformer_blocks.{index}.ff.net")
        source = ff.get_submodule("2")
        weight = _local(source.get_parameter("weight")).detach()
        if (
            tuple(weight.shape) != WEIGHT_SHAPE
            or weight.dtype != torch.bfloat16
            or not weight.is_cuda
        ):
            raise WeightContractError(tuple(weight.shape), str(weight.dtype))
        view = nn.Module()
        view.register_parameter("weight", nn.Parameter(weight, requires_grad=False))
        bias = dict(source.named_parameters(recurse=False)).get("bias")
        view.register_parameter(
            "bias",
            None
            if bias is None
            else nn.Parameter(_local(bias).detach(), requires_grad=False),
        )
        ff.add_module(
            "2", Mxfp8Linear.from_module(view, name=f"Spark/LTX/video_fc2/{index}"),
        )
    return Receipt()
