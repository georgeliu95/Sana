"""GPU integration test for CP token/head exchange and sharded attention."""
from __future__ import annotations

import os
from pathlib import Path

import torch
import torch.distributed as dist

from .qualify_nccl import qualify


def main() -> None:
    variant = os.environ.get("SPARK_CP_VARIANT")
    if variant == "peer":
        from .benchmark_peer_all2all import main as qualify_peer
        qualify_peer()
        return
    if variant == "communication":
        from .benchmark_async_all2all import main as qualify_communication
        qualify_communication()
        return
    if variant == "vae":
        from .qualify_real_vae import main as qualify_vae
        qualify_vae()
        return
    torch.cuda.set_device(0)
    dist.init_process_group('nccl')
    try:
        output = Path(os.environ['SPARK_PP_RESULT_ROOT']).parent
        if os.environ.get("SPARK_CP_VARIANT") == "bridge":
            from .qualify_bridge import qualify as qualify_bridge
            qualify_bridge(Path(os.environ["SPARK_PP_PATHS"]), output / "bridge-qualification.json")
        else:
            qualify(output / 'qualification.json')
    finally:
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
