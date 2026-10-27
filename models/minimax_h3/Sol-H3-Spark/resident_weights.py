"""Broadcast original H3 tensors once at startup; keep steady inference unchanged."""
from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.distributed as dist

import cp2_executor


@dataclass(frozen=True, slots=True)
class TensorHeader:
    name: str
    shape: tuple[int, ...]
    dtype: torch.dtype


def shared(values: Iterator[tuple[str, torch.Tensor]], *, to_cpu: bool) -> Iterator[tuple[str, torch.Tensor]]:
    """Transmit each original tensor without casting, merging, or quantization."""
    count = 0
    transferred = 0
    while True:
        item = next(values, None) if dist.get_rank() == 0 else None
        header = TensorHeader(item[0], tuple(item[1].shape), item[1].dtype) if item is not None else None
        headers = [header]
        dist.broadcast_object_list(headers, src=0, group=cp2_executor.CPU_GROUP)
        received = headers[0]
        if received is None:
            break
        tensor = item[1].to('cuda').contiguous() if item is not None else torch.empty(received.shape, dtype=received.dtype, device='cuda')
        dist.broadcast(tensor, src=0)
        count += 1
        transferred += tensor.numel() * tensor.element_size()
        yield received.name, tensor.cpu() if to_cpu else tensor
    print(f'{{"event":"H3_original_weights_broadcast","rank":{dist.get_rank()},"tensors":{count},"bytes":{transferred}}}', flush=True)


def install() -> None:
    """Scope discovery and loading to this process's original H3 checkpoint."""
    from fastvideo.models.loader import component_loader, fsdp_load

    original_glob = component_loader.glob
    original_iterator = fsdp_load.safetensors_weights_iterator
    original_load = component_loader.safetensors_load_file

    class CheckpointGlob:
        @staticmethod
        def glob(pattern: str) -> list[str]:
            if '/native-model/' not in pattern or Path(pattern).parent.name not in ('transformer', 'audio_vae'):
                return original_glob.glob(pattern)
            paths: list[list[str] | None] = [original_glob.glob(pattern) if dist.get_rank() == 0 else None]
            dist.broadcast_object_list(paths, src=0, group=cp2_executor.CPU_GROUP)
            result = paths[0]
            if result is None:
                raise FileNotFoundError(pattern)
            return result

    def iterator(files: list[str], to_cpu: bool = False, broadcast: bool = True, async_broadcast: bool = True) -> Iterator[tuple[str, torch.Tensor]]:
        if not files or '/native-model/' not in files[0]:
            yield from original_iterator(files, to_cpu=to_cpu, broadcast=broadcast, async_broadcast=async_broadcast)
            return
        values = original_iterator(files, to_cpu=False, broadcast=False) if dist.get_rank() == 0 else iter(())
        yield from shared(values, to_cpu=to_cpu)

    def load_file(filename: str) -> dict[str, torch.Tensor]:
        if '/native-model/' not in filename:
            return original_load(filename)
        values = iter(original_load(filename).items()) if dist.get_rank() == 0 else iter(())
        return dict(shared(values, to_cpu=True))

    component_loader.glob = CheckpointGlob
    fsdp_load.safetensors_weights_iterator = iterator
    component_loader.safetensors_load_file = load_file
