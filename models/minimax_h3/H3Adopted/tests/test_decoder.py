"""Geometry contracts for the substituted streaming video decoder."""
from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from H3Adopted import decoder
from H3Adopted.output import OutputContractError


def test_decoded_shape_when_matched_209_geometry() -> None:
    # Given: the native normalized latent for 896x512 and 209 frames.
    latent_shape = (1, 24, 62, 32, 56)
    # When: applying the pinned decoder temporal and spatial mapping.
    shape = decoder.decoded_shape(latent_shape)
    # Then: every requested frame and pixel dimension is retained.
    assert shape == (1, 3, 209, 512, 896)


@pytest.mark.parametrize("latent_time,frames", [(2, 5), (7, 22), (37, 124), (107, 362)])
def test_decoded_shape_when_valid_h3_window(latent_time: int, frames: int) -> None:
    # Given: supported H3 temporal windows.
    shape = (1, 24, latent_time, 2, 3)
    # When: mapping to output pixels.
    result = decoder.decoded_shape(shape)
    # Then: first-window warmup frames are excluded exactly.
    assert result == (1, 3, frames, 32, 48)


def test_temporal_indices_when_chunk_crosses_window_boundary() -> None:
    # Given: a raw output chunk starting at global raw frame 16.
    # When: keeping finalized decoder frames across the next 20-frame window.
    indices = decoder.temporal_indices(latent_start=4, raw_frames=12)
    # Then: global frames 20, 21, 22 are excluded.
    assert indices == [0, 1, 2, 3, 7, 8, 9, 10, 11]


@pytest.mark.parametrize("shape", [(1, 24, 61, 32, 56), (1, 23, 62, 32, 56), (1, 24, 62, 0, 56)])
def test_decoded_shape_when_contract_is_invalid(shape: tuple[int, ...]) -> None:
    # Given: a latent violating temporal, channel, or spatial constraints.
    # When: parsing its shape.
    with pytest.raises(decoder.DecoderContractError):
        decoder.decoded_shape(shape)
    # Then: decoding never silently crops or repeats requested output.


def test_checkpoint_identity_when_size_is_wrong(tmp_path: Path) -> None:
    # Given: a local safetensors filename containing an incomplete artifact.
    checkpoint = tmp_path / "taeh3.safetensors"
    checkpoint.write_bytes(b"incomplete")
    # When: checking the pinned artifact before allocating model weights.
    with pytest.raises(decoder.CheckpointMismatch) as error:
        decoder.verify_checkpoint(checkpoint)
    # Then: the mismatch reports the actual byte count.
    assert error.value.actual_bytes == 10

def test_rgb_u8_when_halfway_values_are_produced() -> None:
    # Given: halfway boundaries and an owned RGB source tensor.
    torch = pytest.importorskip("torch")
    from H3Adopted.output import VideoSpec, exact_rgb_u8

    values = torch.tensor([0.5 / 255, 1.5 / 255, 2.5 / 255]).reshape(1, 3, 1, 1, 1).expand(1, 3, 1, 2, 2)
    original = values.clone()
    # When: materializing the only required uint8 output buffer.
    rgb = exact_rgb_u8(values, VideoSpec(height=2, width=2, frames=1))
    # Then: ties round to even and the caller-owned pixels remain unchanged.
    assert rgb[0, 0, 0].tolist() == [0, 2, 2]
    assert torch.equal(values, original)


def test_mp4_when_finalized_chunks_and_original_pcm_are_supplied(tmp_path: Path) -> None:
    # Given: three RGB frames and matching original stereo PCM.
    torch = pytest.importorskip("torch")
    av = pytest.importorskip("av")
    from H3Adopted.output import AudioPCM, Mp4Sink, VideoSpec

    path = tmp_path / "out.mp4"
    spec = VideoSpec(height=16, width=16, frames=3)
    pcm = torch.zeros(1, 2, 4000)
    # When: submitting one frame followed by two frames to a bounded session.
    with Mp4Sink(path, spec, AudioPCM(pcm)) as sink:
        sink.submit(torch.zeros(1, 3, 1, 16, 16), 0, 3)
        sink.submit(torch.ones(1, 3, 2, 16, 16), 1, 3)
    # Then: the observable MP4 has every requested frame and a stereo audio track.
    with av.open(str(path)) as container:
        assert len(list(container.decode(video=0))) == 3
        assert container.streams.audio[0].codec_context.layout.name == "stereo"
    assert sink.stats().peak_pending <= 2

@pytest.mark.parametrize("error", [
    decoder.DecoderContractError((1, 24, 61, 32, 56), "wrong temporal window"),
    decoder.CheckpointMismatch(Path("taeh3.safetensors"), 10, "incomplete"),
    OutputContractError("nonsequential frame offset"),
])
def test_contract_error_when_generator_context_unwinds(
    error: ValueError | RuntimeError,
) -> None:
    # Given: each typed exception inside a generator context manager.
    expected_type = type(error)
    expected_message = str(error)

    @contextmanager
    def protected() -> Iterator[None]:
        yield

    # When: propagating the original exception through the context.
    with pytest.raises(expected_type) as observed:
        with protected():
            raise error
    # Then: identity, diagnostic type, and message survive traceback assignment.
    assert observed.value is error
    assert str(observed.value) == expected_message
