# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Ollin Boer Bohan
"""TAEH3 architecture adapted from madebyollin/taehv revision
62f7591f59dfbb4c3c02b7a621d180a9eeaba26c and the pinned h3-optimizations
reference-runtime implementation. Checkpoint state names are unchanged.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies
of the Software, and to permit persons to whom the Software is furnished to do
so, subject to the following conditions: The above copyright notice and this
permission notice shall be included in all copies or substantial portions of
the Software. THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO
EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES
OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
DEALINGS IN THE SOFTWARE.
"""
from __future__ import annotations

from typing import assert_never

import torch
from torch import nn
from torch.nn import functional as F


class Clamp(nn.Module):
    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return torch.tanh(value / 3.0) * 3.0


class MemBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(channels * 2, channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1),
        )

    def forward(self, value: torch.Tensor, past: torch.Tensor) -> torch.Tensor:
        return F.relu(value + self.conv(torch.cat((value, past), dim=1)))


class TemporalGrow(nn.Module):
    def __init__(self, channels: int, stride: int) -> None:
        super().__init__()
        self.stride = stride
        self.conv = nn.Conv2d(channels, channels * stride, 1, bias=False)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        nt, channels, height, width = value.shape
        return self.conv(value).reshape(nt * self.stride, channels, height, width)


class DecoderModel(nn.Sequential):
    """Checkpoint-compatible decoder retaining one previous frame per MemBlock."""

    def __init__(self) -> None:
        super().__init__(
            Clamp(), nn.Conv2d(24, 256, 3, padding=1), nn.ReLU(inplace=True),
            MemBlock(256), MemBlock(256), MemBlock(256),
            nn.Upsample(scale_factor=2, mode="nearest"), TemporalGrow(256, 1),
            nn.Conv2d(256, 128, 3, padding=1, bias=False),
            MemBlock(128), MemBlock(128), MemBlock(128),
            nn.Upsample(scale_factor=2, mode="nearest"), TemporalGrow(128, 2),
            nn.Conv2d(128, 64, 3, padding=1, bias=False),
            MemBlock(64), MemBlock(64), MemBlock(64),
            nn.Upsample(scale_factor=2, mode="nearest"), TemporalGrow(64, 2),
            nn.Conv2d(64, 64, 3, padding=1, bias=False), nn.ReLU(inplace=True),
            nn.Conv2d(64, 12, 3, padding=1),
        )

    def decode_chunk(
        self, value: torch.Tensor, memory: dict[int, torch.Tensor]
    ) -> torch.Tensor:
        """Run NTCHW chunk while carrying causal features across chunk boundaries."""
        batch, time, channels, height, width = value.shape
        flat = value.reshape(batch * time, channels, height, width)
        for index, layer in enumerate(self):
            match layer:
                case MemBlock():
                    _, current_channels, current_height, current_width = flat.shape
                    sequence = flat.reshape(batch, -1, current_channels, current_height, current_width)
                    previous = memory.get(index)
                    if previous is None:
                        previous = torch.zeros_like(sequence[:, :1])
                    past = torch.cat((previous, sequence[:, :-1]), dim=1).reshape_as(flat)
                    memory[index] = sequence[:, -1:].clone(memory_format=torch.contiguous_format)
                    flat = layer(flat, past)
                case nn.Module():
                    flat = layer(flat)
                case unreachable:
                    assert_never(unreachable)
        pixels = F.pixel_shuffle(flat, 2).clamp_(0.0, 1.0)
        return pixels.reshape(batch, -1, pixels.shape[1], pixels.shape[2], pixels.shape[3])
