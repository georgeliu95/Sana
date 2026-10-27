"""Load only the pinned production Conv VAE with the exact Spark optimizations."""
from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
import gc
import hashlib
import os
from pathlib import Path
from typing import TYPE_CHECKING, Final, TypedDict

from pydantic import BaseModel, ConfigDict
import torch

if TYPE_CHECKING:
    from ltx_core.model.video_vae.conv_video_decoder import ConvVideoDecoder, TilingConfig

CHECKPOINT_SHA256: Final = "685b06ee3d9b2039647698fc4ea33175112462fc374e2777312c907897dfce8d"
CHECKPOINT_BYTES: Final = 1452269922
OVERLAY_SHA256: Final = "d8098a2770063a0d2b235de77060b86e8f351f29843db5c2f082c037bcb41f4b"
LATENT_SHAPE: Final = (1, 128, 27, 16, 28)


@dataclass(frozen=True, slots=True)
class RealVaeQualificationError(RuntimeError):
    detail: str

    def __str__(self) -> str:
        return self.detail


class Paths(BaseModel):
    model_config = ConfigDict(frozen=True)
    ltx_root: Path
    output_video_vae: Path


class Provenance(TypedDict):
    checkpoint: str
    checkpoint_sha256: str
    checkpoint_bytes: int
    decoder_source: str
    decoder_source_sha256: str
    memory_efficient_source: str
    memory_efficient_sha256: str
    norm_fusion: str
    layout: str
    decoder_class: str
    latent_shape: list[int]
    pixel_shape: list[int]
    tiles: dict[str, dict[str, int]]
    five_dimensional_weights: int
    all_five_dimensional_weights_channels_last: bool


@dataclass(frozen=True, slots=True)
class LoadedVae:
    decoder: ConvVideoDecoder
    tiling: TilingConfig
    provenance: Provenance


def require(condition: bool, detail: str) -> None:
    if not condition:
        raise RealVaeQualificationError(detail)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


@contextmanager
def loaded_vae(paths_file: Path) -> Iterator[LoadedVae]:
    """Mirror models.construct and Session's direct-NHWC context, loading no DiT."""
    paths = Paths.model_validate_json(paths_file.read_text())
    checkpoint = paths.output_video_vae.expanduser().resolve(strict=True)
    require(os.environ.get("SPARK_LTX_V2_MODE") == "norm", "Require the production SPARK_LTX_V2_MODE=norm")
    require(checkpoint.stat().st_size == CHECKPOINT_BYTES, "Pinned Conv VAE checkpoint size differs")
    checkpoint_hash = file_sha256(checkpoint)
    require(checkpoint_hash == CHECKPOINT_SHA256, "Pinned Conv VAE checkpoint hash differs")
    from runtime.stage2_ops.models import TILES, public_refiner_modules
    from runtime.stage2_ops.conv_direct_nhwc import installed_direct_upsample_nhwc
    from runtime.profiling import observed

    base, compat = public_refiner_modules(paths.ltx_root)
    from ltx_core.model.video_vae import memory_efficient_decode
    from ltx_core.model.video_vae import conv_video_decoder

    overlay_path = Path(memory_efficient_decode.__file__).resolve(strict=True)
    overlay_hash = file_sha256(overlay_path)
    require(overlay_hash == OVERLAY_SHA256, "Mounted VAE overlay differs from frozen deploy-v5")
    device = torch.device("cuda", torch.cuda.current_device())
    registry = base.ModelRegistry()
    decoder_block = compat.VideoDecoder(str(checkpoint), torch.bfloat16, device,
        registry=registry, alloc_trim_strategy=base.AllocatorTrimStrategy.DEFER)
    tiling = compat.TileSizeConfig(**{
        name: compat.DimensionSizeConfig(**values) for name, values in TILES.items()})
    pixel_shape = base.VideoPixelShape(batch=1, frames=209, height=512, width=896, fps=24)
    tiling.validate(compat.tiling_scale_factors_for_vae(str(checkpoint)), pixel_shape)
    gc.collect()
    torch.cuda.empty_cache()
    decoder = decoder_block._decoder_builder.build(device=device, dtype=torch.bfloat16).eval().requires_grad_(False)
    require(type(decoder).__name__ == "ConvVideoDecoder", "Builder returned a different decoder")
    require(hasattr(decoder, "_original_forward"), "Memory-efficient decoder module op was not applied")
    weights = [parameter for parameter in decoder.parameters() if parameter.ndim == 5]
    channels_last = bool(weights) and all(parameter.is_contiguous(memory_format=torch.channels_last_3d)
                                          for parameter in weights)
    require(channels_last, "Decoder builder did not load original channels-last weights")
    decoder.forward = observed("Spark/LTX/ConvVAE/forward", decoder.forward)
    source_path = Path(conv_video_decoder.__file__).resolve(strict=True)
    provenance: Provenance = {
        "checkpoint": str(checkpoint), "checkpoint_sha256": checkpoint_hash,
        "checkpoint_bytes": checkpoint.stat().st_size,
        "decoder_source": str(source_path), "decoder_source_sha256": file_sha256(source_path),
        "memory_efficient_source": str(overlay_path), "memory_efficient_sha256": overlay_hash,
        "norm_fusion": "original_BF16_PixelNorm_SiLU", "layout": "production_direct_upsample_NHWC",
        "decoder_class": type(decoder).__name__, "latent_shape": list(LATENT_SHAPE),
        "pixel_shape": [1, 3, 209, 512, 896], "tiles": TILES,
        "five_dimensional_weights": len(weights), "all_five_dimensional_weights_channels_last": channels_last,
    }
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    with installed_direct_upsample_nhwc(torch_module=torch):
        yield LoadedVae(decoder, tiling, provenance)
