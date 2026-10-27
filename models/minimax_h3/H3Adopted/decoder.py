# SPDX-License-Identifier: Apache-2.0
"""Pinned, approximate TAEH3 decoding of normalized H3 diffusion latents.

TAEH3 directly consumes normalized diffusion latents. Full-H3 VAE mean/std,
pixel inverse normalization, and window blending must not be applied.
"""
from __future__ import annotations

import hashlib
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING, Final, assert_never

if TYPE_CHECKING:
    import torch

TAEH3_REVISION: Final = "62f7591f59dfbb4c3c02b7a621d180a9eeaba26c"
TAEH3_CHECKPOINT_SHA256: Final = "4fd022bfcab08772fe0536b17ea1a3bbb5625be11e397868d1c5d891863d4c13"
TAEH3_CHECKPOINT_SIZE_BYTES: Final = 22_709_752
TAEH3_CHECKPOINT_URL: Final = (
    f"https://raw.githubusercontent.com/madebyollin/taehv/{TAEH3_REVISION}/safetensors/taeh3.safetensors"
)


class DecoderContractError(ValueError):
    """Typed diagnostics with mutable Python exception traceback state."""

    shape: tuple[int, ...]
    reason: str

    def __init__(self, shape: tuple[int, ...], reason: str) -> None:
        super().__init__(shape, reason)
        self.shape = shape
        self.reason = reason

    def __str__(self) -> str:
        return f"TAEH3 latent contract: {self.reason}; shape={self.shape}"


class CheckpointMismatch(RuntimeError):
    """Artifact diagnostics allowing context managers to update traceback state."""

    path: Path
    actual_bytes: int
    actual_sha256: str

    def __init__(self, path: Path, actual_bytes: int, actual_sha256: str) -> None:
        # Existing exception API preserves all three artifact identity diagnostics.
        super().__init__(path, actual_bytes, actual_sha256)
        self.path = path
        self.actual_bytes = actual_bytes
        self.actual_sha256 = actual_sha256

    def __str__(self) -> str:
        return (
            f"TAEH3 checkpoint identity mismatch: {self.path}; "
            f"bytes={self.actual_bytes}/{TAEH3_CHECKPOINT_SIZE_BYTES}; "
            f"sha256={self.actual_sha256}/{TAEH3_CHECKPOINT_SHA256}"
        )


class ComputeDtype(StrEnum):
    FP16 = "fp16"
    FP32 = "fp32"


@dataclass(frozen=True, slots=True)
class DecoderConfig:
    device: str = "cuda:0"
    dtype: ComputeDtype = ComputeDtype.FP16
    chunk_size: int = 5

    def __post_init__(self) -> None:
        if self.chunk_size < 1:
            raise DecoderContractError((), "chunk_size must be positive")


def decoded_shape(shape: tuple[int, ...]) -> tuple[int, int, int, int, int]:
    """Parse native BCTHW geometry into the exact requested RGB geometry."""
    if len(shape) != 5:
        raise DecoderContractError(shape, "expected rank-five BCTHW")
    batch, channels, time, height, width = shape
    if batch < 1 or channels != 24 or min(time, height, width) < 1:
        raise DecoderContractError(shape, "requires nonempty 24-channel latent")
    if time % 5 != 2:
        raise DecoderContractError(shape, "latent time must be 5*k+2")
    return batch, 3, (time // 5) * 17 + 5, height * 16, width * 16


def temporal_indices(latent_start: int, raw_frames: int) -> list[int]:
    """Exclude three warmup frames per raw 20-frame H3 temporal window."""
    return [index for index in range(raw_frames) if (latent_start * 4 + index) % 20 >= 3]


def verify_checkpoint(path: Path) -> bytes:
    """Verify the local artifact before loading any checkpoint tensors."""
    payload = path.read_bytes()
    digest = hashlib.sha256(payload).hexdigest()
    if len(payload) != TAEH3_CHECKPOINT_SIZE_BYTES or digest != TAEH3_CHECKPOINT_SHA256:
        raise CheckpointMismatch(path, len(payload), digest)
    return payload


class TAEH3Decoder:
    """Resident decoder; each request owns fresh causal temporal memory."""

    def __init__(self, checkpoint: Path, config: DecoderConfig = DecoderConfig()) -> None:
        import torch
        from safetensors.torch import load
        from .tae_model import DecoderModel

        payload = verify_checkpoint(checkpoint)
        state = load(payload)
        self.model = DecoderModel()
        decoder_state = {
            name.removeprefix("decoder."): tensor
            for name, tensor in state.items()
            if name.startswith("decoder.")
        }
        self.model.load_state_dict(decoder_state, strict=True)
        match config.dtype:
            case ComputeDtype.FP16:
                self.compute_dtype = torch.float16
            case ComputeDtype.FP32:
                self.compute_dtype = torch.float32
            case unreachable:
                assert_never(unreachable)
        self.config = config
        self.model.eval().requires_grad_(False).to(
            device=config.device, dtype=self.compute_dtype
        )
        self.checkpoint = checkpoint.resolve()

    def iter_normalized(self, latent: torch.Tensor) -> Iterator[torch.Tensor]:
        """Yield finalized RGB BCTHW chunks in [0,1], without changing latents."""
        import torch

        expected = decoded_shape(tuple(latent.shape))
        if not latent.dtype.is_floating_point:
            raise DecoderContractError(tuple(latent.shape), "requires floating diffusion latents")
        with torch.inference_mode():
            value = latent.permute(0, 2, 1, 3, 4).to(
                device=self.config.device, dtype=self.compute_dtype
            )
            memory: dict[int, torch.Tensor] = {}
            written = 0
            for start in range(0, value.shape[1], self.config.chunk_size):
                raw = self.model.decode_chunk(
                    value[:, start : start + self.config.chunk_size], memory
                )
                kept = raw[:, temporal_indices(start, raw.shape[1])]
                written += kept.shape[1]
                yield kept.permute(0, 2, 1, 3, 4)
            if written != expected[2]:
                raise DecoderContractError(tuple(latent.shape), f"wrote {written}/{expected[2]} frames")

    def decode_normalized(
        self,
        latent: torch.Tensor,
        callback: Callable[[torch.Tensor, int, int], None],
    ) -> None:
        """Stream every generated frame through the caller-owned output sink."""
        total = decoded_shape(tuple(latent.shape))[2]
        written = 0
        for chunk in self.iter_normalized(latent):
            callback(chunk, written, total)
            written += chunk.shape[2]

    def decode_tensor(self, latent: torch.Tensor) -> torch.Tensor:
        """Materialize BCTHW pixels only for small parity checks."""
        import torch

        return torch.cat(list(self.iter_normalized(latent)), dim=2)
