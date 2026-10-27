# /// script
# requires-python = ">=3.12"
# dependencies = ["pydantic>=2"]
# ///
# ─── How to run ───
# Imported by the pinned H3 and LTX runtimes; tests are in the campaign/tests.
"""Explicit early-video handoff; original PCM remains a required late dependency."""
from __future__ import annotations

import os
import time
from hashlib import file_digest
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar, Literal

from pydantic import AliasChoices, BaseModel, ConfigDict, Field, JsonValue


def sha256(path: Path) -> str:
    """Stream one task artifact through the standard-library SHA-256 helper."""
    with path.open("rb") as stream:
        return file_digest(stream, "sha256").hexdigest()

if TYPE_CHECKING:
    from collections.abc import Mapping
    from concurrent.futures import Future

    from torch import Tensor


class HandoffError(RuntimeError):
    """An early handoff violated identity, ordering, or tensor contracts."""


class Identity(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True)
    request_id: str = Field(validation_alias=AliasChoices("request_id", "case_id"))


class AudioRelease(Identity):
    kind: Literal["upscaler_complete"] = "upscaler_complete"


class EarlyVideo(Identity):
    kind: Literal["early_video"] = "early_video"
    case_id: str
    prompt: str
    seed: int
    source_index: int
    video_sha256: str


def enabled(request_id: str) -> bool:
    """Warmup uses the original path; paired arms choose only this schedule."""
    if request_id == "warmup":
        return False
    if os.environ.get("SPARK_CP_AUDIO_PAIRED") == "1":
        return request_id.endswith("_overlap")
    return os.environ.get("SPARK_CP_FULL_AUDIO") == "1"


def wait_ready(
    path: Path,
    request_id: str,
    *,
    producer: Future[Mapping[str, JsonValue]] | Future[None] | None = None,
    timeout: float = 300.0,
) -> Path:
    """Wait for an atomic same-request receipt, propagating producer failures."""
    deadline = time.monotonic() + timeout
    while not path.is_file():
        if producer is not None and producer.done():
            _ = producer.result()
            msg = f"producer completed without ready receipt: {path}"
            raise HandoffError(msg)
        if time.monotonic() >= deadline:
            msg = f"handoff timeout: {path}"
            raise HandoffError(msg)
        time.sleep(0.001)
    if Identity.model_validate_json(path.read_text()).request_id != request_id:
        msg = f"handoff identity mismatch: {path}"
        raise HandoffError(msg)
    return path


def _publish(path: Path, value: BaseModel) -> None:
    if path.exists():
        raise FileExistsError(path)
    temporary = path.with_suffix(".json.pending")
    with temporary.open("x") as stream:
        _ = stream.write(value.model_dump_json(indent=2) + "\n")
    _ = temporary.rename(path)


def release_audio(directory: Path, request_id: str) -> None:
    """Release H3 audio only after both upscaler segments have been joined."""
    _publish(directory / "audio-release.json", AudioRelease(request_id=request_id))


def publish_video(directory: Path, native: Tensor, identity: Mapping[str, JsonValue]) -> None:
    """Keep the original FP32-to-BF16 cast and publish its immutable CPU tensor."""
    import torch  # noqa: PLC0415 — control-plane users do not load the GPU runtime.

    video = native.to(torch.bfloat16).contiguous()
    path = directory / "early_video.pt"
    with path.open("xb") as stream:
        torch.save({"h3_normalized": video}, stream)
    metadata = EarlyVideo.model_validate({
        **identity,
        "request_id": identity.get("request_id", identity["case_id"]),
        "video_sha256": sha256(path),
    })
    _publish(directory / "early_capture.json", metadata)


def load_video(directory: Path, request_id: str) -> tuple[EarlyVideo, Tensor]:
    """Validate early video separately from the complete H3/PCM capture ABI."""
    import torch  # noqa: PLC0415 — control-plane users do not load the GPU runtime.

    metadata = EarlyVideo.model_validate_json((directory / "early_capture.json").read_text())
    path = directory / "early_video.pt"
    if metadata.request_id != request_id or sha256(path) != metadata.video_sha256:
        msg = "early video identity or digest mismatch"
        raise HandoffError(msg)
    class VideoTensor(BaseModel):
        model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True, arbitrary_types_allowed=True)
        h3_normalized: torch.Tensor

    tensor = VideoTensor.model_validate(torch.load(path, map_location="cpu", weights_only=True)).h3_normalized
    if (tensor.shape != (1, 24, 37, 24, 42)
            or tensor.dtype != torch.bfloat16 or not bool(torch.isfinite(tensor).all())):
        msg = "early video tensor ABI mismatch"
        raise HandoffError(msg)
    return metadata, tensor.contiguous()


def capture_files(*, early: bool) -> tuple[str, str]:
    """Explicit file ABI for complete versus early handoffs."""
    return ("early_capture.json", "early_video.pt") if early else (
        "capture.json", "stage1_direct_tensors.pt",
    )


def local_capture(capture_dir: Path) -> Path:
    """Map only this run's worker capture path to the shared local H3 output."""
    root = Path(os.environ["SPARK_PP_RESULT_ROOT"]).resolve()
    relative = capture_dir.resolve().relative_to(root)
    h3_root = Path(os.environ["SPARK_CP_LOCAL_HANDOFF_ROOT"]).resolve()
    target = (h3_root / relative).resolve()
    _ = target.relative_to(h3_root)
    return target
