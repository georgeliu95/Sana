# /// script
# requires-python = ">=3.12"
# dependencies = []
# ///
# ─── How to run ───
# Enabled only during the separate numerical qualification run.
"""Incremental exact-byte evidence without retaining full RGB tensor dumps."""
from __future__ import annotations

from hashlib import sha256
from typing import TYPE_CHECKING, Final

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path

    from torch import Tensor


FRAME_COUNT: Final = 209


def tensor_digest(tensor: Tensor) -> str:
    """Hash canonical tensor bytes, preserving BF16 representation."""
    import torch  # noqa: PLC0415 — qualification is optional.

    return sha256(tensor.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()


def frame_digest(chunks: Iterator[Tensor], path: Path) -> Iterator[Tensor]:
    """Consume the exact encoder input order and publish only after exhaustion."""
    import torch  # noqa: PLC0415 — qualification is optional.

    digest = sha256()
    frames = 0
    for chunk in chunks:
        digest.update(chunk.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
        frames += int(chunk.shape[0])
        yield chunk
    if frames != FRAME_COUNT:
        msg = f"qualification frame count drift: {frames}"
        raise RuntimeError(msg)
    with path.open("x") as stream:
        _ = stream.write(digest.hexdigest() + "\n")
