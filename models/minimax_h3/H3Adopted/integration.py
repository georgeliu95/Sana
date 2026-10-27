"""Worker-local hooks for the full-resolution FastH3/MXFP8/TAEH3 arm."""
from __future__ import annotations

from dataclasses import asdict
import json
import os
from pathlib import Path
import time

import torch
from torch import nn

from .decoder import DecoderConfig, TAEH3Decoder
from .output import AudioPCM, Mp4Sink, VideoSpec
from .quantization import dispatch_counts, install_fastvideo, install_fastvideo_swiglu

HEIGHT = 512
WIDTH = 896
FRAMES = 209
FPS = 24
SAMPLES = round(FRAMES * 32000 / FPS)
LATENT_SHAPE = (1, 24, 62, 32, 56)


def convert_model(model: nn.Module) -> None:
    """Consume BF16 LoRA-merged weights at the existing conversion seam."""
    receipts = install_fastvideo(model)
    fused = install_fastvideo_swiglu(model)
    if len(receipts) != 300 or len(fused) != 50:
        raise RuntimeError(f"unexpected MXFP8 scope: {len(receipts)}/{len(fused)}")
    from .instrument import install_dit_scope
    install_dit_scope(model)
    from .projection import install_shared_qkv
    model._h3_adopted_shared_qkv = install_shared_qkv(model)
    model._h3_adopted_quantization = tuple(asdict(item) for item in receipts)


def install_capture(worker):
    """Existing FastVideo worker ABI, with direct normalized latent decoding."""
    from fastvideo.pipelines.basic.minimax_h3.packing import (
        h3_dit_patch_size, unpatchify_video_tokens,
    )
    from fastvideo.pipelines.basic.minimax_h3.stages.minimax_h3_latent_preparation import (
        MINIMAX_H3_LAYOUT_KEY,
    )
    from runtime.stage1_ops.native import atomic_json

    state = {"target": None, "captures": 0, "rank0": True, "video_decode_calls": 0}
    worker._sol_h3_capture = state
    decoder = TAEH3Decoder(
        Path(os.environ["H3_ADOPTED_TAE_CHECKPOINT"]),
        DecoderConfig(chunk_size=int(os.environ.get("H3_ADOPTED_TAE_CHUNK", "5"))),
    )
    video_stage = worker.pipeline._stage_name_mapping["video_decoding_stage"]
    audio_stage = worker.pipeline._stage_name_mapping["audio_decoding_stage"]
    original_audio = audio_stage.forward

    def video_capture(batch, args):
        with torch.cuda.nvtx.range("H3Adopted/video_latent"):
            target = state["target"]
            layout = batch.extra[MINIMAX_H3_LAYOUT_KEY]
            if target is None or batch.save_video or batch.return_frames:
                raise RuntimeError("direct decode requires an armed rank-zero native request")
            if layout.num_condition_video_rows or layout.num_condition_audio_rows:
                raise RuntimeError("this arm accepts T2VA only")
            if tuple(batch.raw_latent_shape) != LATENT_SHAPE:
                raise RuntimeError(f"full H3 latent geometry drift: {batch.raw_latent_shape}")
            _, channels, frames, height, width = LATENT_SHAPE
            native = unpatchify_video_tokens(
                batch.latents, frames, height, width, channels, h3_dit_patch_size(args)
            ).detach()
            if native.dtype != torch.float32 or not native.is_cuda:
                raise RuntimeError("expected native normalized FP32 CUDA latent")
            if not bool(torch.isfinite(native).all()):
                raise RuntimeError("nonfinite generated H3 latent")
            state["native_normalized"] = native
            batch.output = native
            return batch

    def audio_capture(batch, args):
        with torch.cuda.nvtx.range("H3Adopted/audio_vae"):
            result = original_audio(batch, args)
        pcm = result.extra["audio"]
        if pcm.ndim != 2 or pcm.shape[1] != 2 or pcm.shape[0] < 1:
            raise RuntimeError(f"original stereo PCM geometry drift: {pcm.shape}")
        if pcm.dtype != torch.float32 or result.extra["audio_sample_rate"] != 32000:
            raise RuntimeError("original PCM precision or sample rate drift")
        native = state.pop("native_normalized")
        target = state["target"]
        directory = Path(target["directory"])
        output = directory / "adopted.mp4"
        if output.exists():
            raise FileExistsError(output)
        audio_samples = min(int(pcm.shape[0]), SAMPLES)
        audio = pcm[:audio_samples].transpose(0, 1).contiguous()
        if not bool(torch.isfinite(audio).all()):
            raise RuntimeError("nonfinite original PCM")
        started = time.monotonic_ns()
        with Mp4Sink(output, VideoSpec(HEIGHT, WIDTH, FRAMES, FPS), AudioPCM(audio)) as sink:
            def submit(frames, start, total):
                with torch.cuda.nvtx.range("H3Adopted/pixels_D2H"):
                    sink.submit(frames, start, total)
            with torch.cuda.nvtx.range("H3Adopted/TAEH3"):
                decoder.decode_normalized(native, submit)
        torch.cuda.synchronize()
        ended = time.monotonic_ns()
        receipt = {
            "status": "PASS", "same_request": True, "external_anchor_used": False,
            "latent_only_transfer": False, "video_decode_calls": 0, "h3_decoder_calls": 0,
            "tae_decoder_calls": 1, "generated_first_frame_path": None,
            "native_normalized_shape": list(native.shape), "native_normalized_dtype": str(native.dtype),
            "tae_checkpoint": str(decoder.checkpoint), "tae_chunk_size": decoder.config.chunk_size,
            "output": str(output), "output_stats": asdict(sink.stats()),
            "audio_samples": audio_samples, "audio_sample_rate": 32000,
            "native_pcm_samples": int(pcm.shape[0]),
            "decode_and_mux_s": (ended - started) / 1e9,
            "final_mp4_complete_monotonic_ns": ended,
        }
        atomic_json(directory / "capture.json", receipt)
        state["captures"] += 1
        state["target"] = None
        return result

    video_stage.forward = video_capture
    audio_stage.forward = audio_capture
    return {"rank0": True, "decoder": "TAEH3", "geometry": list(LATENT_SHAPE)}


def finish_worker_request(worker):
    """Observe completed CUDA work without clearing resident allocator caches."""
    torch.cuda.synchronize()
    state = worker._sol_h3_stage1
    counts = tuple(asdict(item) for item in dispatch_counts())
    return {
        **{key: value for key, value in state.items() if key != "case"},
        "actual_mxfp8": {
            "projection_modules": len(counts),
            "gemms": sum(item["gemms"] for item in counts),
            "activation_quantizations": sum(item["quantizations"] for item in counts),
            "fused_swiglu_quantizations": sum(item["fused_swiglu_quantizations"] for item in counts),
            "module_counts": counts,
        },
        "allocator_cache_retained": True,
    }

def dispatch_snapshot(worker):
    """Existing collective RPC passes its worker even for a counter read."""
    counts = tuple(asdict(item) for item in dispatch_counts())
    return {
        "projection_modules": len(counts),
        "gemms": sum(item["gemms"] for item in counts),
        "activation_quantizations": sum(item["quantizations"] for item in counts),
        "fused_swiglu_quantizations": sum(item["fused_swiglu_quantizations"] for item in counts),
        "module_counts": counts,
    }


def dispatch_delta(before, after):
    previous = {item["name"]: item for item in before["module_counts"]}
    result = {
        **after,
        **{key: after[key] - before[key] for key in
           ("gemms", "activation_quantizations", "fused_swiglu_quantizations")},
        "module_counts": tuple({
            **item,
            **{key: item[key] - previous.get(item["name"], {}).get(key, 0) for key in
               ("gemms", "quantizations", "fused_swiglu_quantizations")}
        } for item in after["module_counts"]),
    }
    expected = (1200, 600, 200)
    actual = tuple(result[key] for key in
                   ("gemms", "activation_quantizations", "fused_swiglu_quantizations"))
    if actual != expected or result["projection_modules"] != 300:
        raise RuntimeError(f"actual four-step MXFP8 dispatch mismatch: {actual}")
    return result
