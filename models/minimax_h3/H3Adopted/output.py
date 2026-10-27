"""Bounded streaming RGB-to-MP4 output with the original H3 PCM.

Two pinned host slots overlap GPU decode/D2H with one persistent PyAV encoder.
The sink never assembles a full floating-point clip.
"""
from __future__ import annotations

from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import ExitStack
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from types import TracebackType
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    import torch
    from av.audio.stream import AudioStream
    from av.container import OutputContainer
    from av.video.stream import VideoStream


class OutputContractError(ValueError):
    """Output diagnostics with mutable Python exception traceback state."""

    reason: str

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason

    def __str__(self) -> str:
        return f"H3 MP4 output contract: {self.reason}"


@dataclass(frozen=True, slots=True)
class VideoSpec:
    height: int
    width: int
    frames: int
    fps: int = 24

    def __post_init__(self) -> None:
        if min(self.height, self.width, self.frames, self.fps) < 1:
            raise OutputContractError("video dimensions, frame count, and FPS must be positive")
        if self.height % 2 or self.width % 2:
            raise OutputContractError("H264 yuv420p requires even pixel dimensions")


@dataclass(frozen=True, slots=True)
class AudioPCM:
    """Original finite stereo PCM in CT or BCT format; never trimmed by the sink."""

    samples: torch.Tensor
    sample_rate: int = 32000


@dataclass(frozen=True, slots=True)
class OutputStats:
    frames: int
    chunks: int
    peak_pending: int
    raw_rgb_bytes: int


@dataclass(frozen=True, slots=True)
class _EncodeItem:
    host: torch.Tensor
    ready: torch.cuda.Event | None
    gpu_owner: torch.Tensor | None
    first_frame: int


def exact_rgb_u8(frames: torch.Tensor, spec: VideoSpec) -> torch.Tensor:
    """Own the FP32 work buffer and preserve baseline round-to-nearest-even."""
    import torch

    shape = tuple(frames.shape)
    if len(shape) != 5 or shape[0] != 1 or shape[1] != 3:
        raise OutputContractError(f"requires one BCTHW RGB chunk, received {shape}")
    if shape[-2] < spec.height or shape[-1] < spec.width:
        raise OutputContractError(f"chunk smaller than requested crop: {shape}")
    if not frames.dtype.is_floating_point:
        raise OutputContractError("RGB input must be floating point in [0,1]")
    pixels = frames[0, :, :, :spec.height, :spec.width].permute(1, 2, 3, 0)
    work = pixels.to(dtype=torch.float32, copy=True, memory_format=torch.contiguous_format)
    return work.clamp_(0.0, 1.0).mul_(255.0).round_().to(torch.uint8)


class Mp4Sink:
    """Mutable encoder session owning bounded slots, futures, and packet counters."""

    def __init__(self, path: Path, spec: VideoSpec, audio: AudioPCM) -> None:
        self.path = path
        self.spec = spec
        self.audio = audio
        self._stack = ExitStack()
        self._pending: deque[Future[None]] = deque()
        self._slots: dict[int, torch.Tensor] = {}
        self._copy_stream: torch.cuda.Stream | None = None
        self._next_frame = 0
        self._chunks = 0
        self._peak_pending = 0
        self._finished = False
        self._container: OutputContainer
        self._video: VideoStream
        self._audio_stream: AudioStream
        self._executor: ThreadPoolExecutor

    def __enter__(self) -> Mp4Sink:
        import av

        with ExitStack() as setup:
            self._container = setup.enter_context(av.open(str(self.path), mode="w", format="mp4"))
            self._executor = setup.enter_context(
                ThreadPoolExecutor(max_workers=1, thread_name_prefix="h3-mp4")
            )
            self._video = self._container.add_stream("h264", rate=self.spec.fps)
            self._video.width = self.spec.width
            self._video.height = self.spec.height
            self._video.pix_fmt = "yuv420p"
            self._video.options = {"crf": "18", "preset": "ultrafast", "threads": "0"}
            self._audio_stream = self._container.add_stream("aac", rate=self.audio.sample_rate)
            self._audio_stream.layout = "stereo"
            self._stack = setup.pop_all()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> Literal[False]:
        # Existing interface requires all three exception parameters.
        try:
            if exc is None:
                self.finish()
        finally:
            self._stack.__exit__(exc_type, exc, traceback)
        return False

    def submit(self, frames: torch.Tensor, first_frame: int, total_frames: int) -> None:
        """Queue one finalized chunk; callback matches TAEH3 and full-VAE sinks."""
        # Existing decoder callback requires independent offset and total arguments.
        import torch

        if self._finished or first_frame != self._next_frame or total_frames != self.spec.frames:
            raise OutputContractError(f"nonsequential chunk at {first_frame}/{total_frames}")
        if len(self._pending) == 2:
            self._pending.popleft().result()
        rgb = exact_rgb_u8(frames, self.spec)
        count = rgb.shape[0]
        if first_frame + count > total_frames:
            raise OutputContractError("chunk exceeds advertised video length")
        slot_index = self._chunks % 2
        host = self._slots.get(slot_index)
        if host is None or host.shape[0] < count:
            host = torch.empty(
                rgb.shape, dtype=torch.uint8, device="cpu", pin_memory=rgb.is_cuda
            )
            self._slots[slot_index] = host
        ready: torch.cuda.Event | None = None
        owner: torch.Tensor | None = None
        if rgb.is_cuda:
            if self._copy_stream is None:
                self._copy_stream = torch.cuda.Stream(device=rgb.device)
            self._copy_stream.wait_stream(torch.cuda.current_stream(rgb.device))
            ready = torch.cuda.Event()
            with torch.cuda.stream(self._copy_stream):
                host[:count].copy_(rgb, non_blocking=True)
                rgb.record_stream(self._copy_stream)
                ready.record(self._copy_stream)
            owner = rgb
        else:
            host[:count].copy_(rgb)
        item = _EncodeItem(host[:count], ready, owner, first_frame)
        self._pending.append(self._executor.submit(self._encode, item))
        self._peak_pending = max(self._peak_pending, len(self._pending))
        self._next_frame += count
        self._chunks += 1

    def _encode(self, item: _EncodeItem) -> None:
        import av

        if item.ready is not None:
            item.ready.synchronize()
        for offset, pixels in enumerate(item.host.numpy()):
            frame = av.VideoFrame.from_ndarray(pixels, format="rgb24")
            frame.pts = item.first_frame + offset
            frame.time_base = Fraction(1, self.spec.fps)
            for packet in self._video.encode(frame):
                self._container.mux(packet)

    def finish(self) -> OutputStats:
        """Drain all D2H/encoder work and mux unmodified original PCM as AAC."""
        import av
        import numpy as np
        import torch

        if self._finished:
            return self.stats()
        while self._pending:
            self._pending.popleft().result()
        if self._next_frame != self.spec.frames:
            raise OutputContractError(f"wrote {self._next_frame}/{self.spec.frames} frames")
        for packet in self._video.encode():
            self._container.mux(packet)
        samples = self.audio.samples
        if samples.ndim == 3 and samples.shape[0] == 1:
            samples = samples[0]
        if samples.ndim != 2 or samples.shape[0] != 2 or self.audio.sample_rate < 1:
            raise OutputContractError("original PCM must be stereo CT or BCT")
        pcm = np.ascontiguousarray(samples.detach().to(device="cpu", dtype=torch.float32).numpy())
        audio_frame = av.AudioFrame.from_ndarray(pcm, format="fltp", layout="stereo")
        audio_frame.sample_rate = self.audio.sample_rate
        audio_frame.pts = 0
        audio_frame.time_base = Fraction(1, self.audio.sample_rate)
        for packet in self._audio_stream.encode(audio_frame):
            self._container.mux(packet)
        for packet in self._audio_stream.encode():
            self._container.mux(packet)
        self._finished = True
        return self.stats()

    def stats(self) -> OutputStats:
        return OutputStats(
            self._next_frame, self._chunks, self._peak_pending,
            self._next_frame * self.spec.height * self.spec.width * 3,
        )
