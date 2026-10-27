"""Whole-model NVTX attribution outside native regional compilation."""
from __future__ import annotations
from typing import TypeVar
import torch
from torch import nn

_Result = TypeVar("_Result")


def _begin(module: nn.Module, inputs: tuple[torch.Tensor, ...]) -> None:
    torch.cuda.nvtx.range_push("H3Adopted/DiT")


def _end(module: nn.Module, inputs: tuple[torch.Tensor, ...], result: _Result) -> None:
    torch.cuda.nvtx.range_pop()


def install_dit_scope(model: nn.Module) -> None:
    model.register_forward_pre_hook(_begin)
    model.register_forward_hook(_end, always_call=True)
