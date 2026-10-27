"""Profile compiled GEMMs by resident weight identity without disabling compilation."""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, assert_never

import nvtx
import torch
from torch import Tensor, nn
from torch.distributed.tensor import DTensor

from runtime.profiling import observed


class Addmm(Protocol):
    """The compiled external addmm boundary used by this frozen PyTorch build."""

    def __call__(self, bias: Tensor, x: Tensor, weight: Tensor, *, alpha: float = 1,
                 beta: float = 1, out: Tensor) -> Tensor: ...


@dataclass(frozen=True, slots=True)
class LinearCall:
    """Concrete weight role and dimensions seen at the external GEMM boundary."""

    role: str
    m: int
    n: int
    k: int


def local_tensor(value: Tensor) -> Tensor:
    """Resolve singleton distributed weights to their existing local storage."""
    match value:
        case DTensor():
            return value.to_local()
        case Tensor():
            return value
        case unreachable:
            assert_never(unreachable)


@contextmanager
def detailed_trace(core: nn.Module, destination: Path) -> Iterator[None]:
    """Annotate an already warmed request; collect actual Linear dispatch counts."""
    from torch._inductor.select_algorithm import extern_kernels
    from ltx_core.model.video_vae import memory_efficient_decode as vae

    weights: dict[int, str] = {}
    inventory: list[dict[str, str | list[int]]] = []
    for name, parameter in core.named_parameters():
        if name.endswith(".weight") and parameter.ndim == 2:
            name = name.removesuffix(".weight").replace("._orig_mod", "")
            weight = local_tensor(parameter)
            weights[weight.data_ptr()] = name
            inventory.append({"role": name, "shape": list(weight.shape), "dtype": str(weight.dtype)})
    counts: Counter[LinearCall] = Counter()
    original: Addmm = extern_kernels.addmm

    def addmm(bias: Tensor, x: Tensor, weight: Tensor, *, alpha: float = 1,
              beta: float = 1, out: Tensor) -> Tensor:
        """Keep the original GEMM, adding CPU NVTX and a shape/role receipt."""
        call = LinearCall(weights.get(weight.data_ptr(), "unattributed"), x.shape[0],
                          weight.shape[1], x.shape[1])
        counts[call] += 1
        with nvtx.annotate(f"Spark/LTX/Linear/{call.role}/M{call.m}_N{call.n}_K{call.k}",
                           domain="spark_h3"):
            return original(bias, x, weight, alpha=alpha, beta=beta, out=out)

    original_mm = extern_kernels.mm
    def mm(x: Tensor, weight: Tensor, *, out: Tensor) -> Tensor:
        """Include bias-free video FC1 and FC2 in the same dispatch registry."""
        call = LinearCall(weights.get(weight.data_ptr(), "unattributed"), x.shape[0], weight.shape[1], x.shape[1])
        counts[call] += 1
        with nvtx.annotate(f"Spark/LTX/Linear/{call.role}/M{call.m}_N{call.n}_K{call.k}", domain="spark_h3"):
            return original_mm(x, weight, out=out)
    extern_kernels.mm = mm
    extern_kernels.addmm = addmm
    restored: list[Callable[[], None]] = []

    def wrap(module: nn.Module, label: str) -> None:
        """Wrap the compiled module from outside its compiled forward function."""
        forward = module.forward
        module.forward = observed(label, forward)
        restored.append(lambda: setattr(module, "forward", forward))

    for name, module in core.named_modules():
        if name and not name.startswith("transformer_blocks") and "weight" in dict(module.named_parameters(recurse=False)):
            if module.get_parameter("weight").ndim == 2:
                wrap(module, f"Spark/LTX/Linear/{name}")
    blocks = core.get_submodule("transformer_blocks")
    for index, block in enumerate(blocks.children()):
        wrap(block, f"Spark/LTX/block/{index:02d}")
    step = 0
    forward = core.forward

    def step_forward(*args: Tensor, **kwargs: Tensor) -> Tensor:
        """Record the three real denoising calls without modifying Sol state."""
        nonlocal step
        current = step
        step += 1
        with nvtx.annotate(f"Spark/LTX/step/{current}", domain="spark_h3"):
            return forward(*args, **kwargs)

    core.forward = step_forward
    restored.append(lambda: setattr(core, "forward", forward))
    for name in ("inplace_conv3d_temporal_chunked", "_causal_pad_free_and_conv", "_pixel_norm_inplace", "_spark_silu_inplace", "_spark_norm_silu"):
        function = getattr(vae, name, None)
        if function is not None:
            setattr(vae, name, observed("Spark/LTX/VAE/" + name, function))
            restored.append(lambda name=name, function=function: setattr(vae, name, function))
    try:
        yield
    finally:
        extern_kernels.mm = original_mm
        extern_kernels.addmm = original
        for restore in reversed(restored):
            restore()
        destination.write_text(json.dumps({
            "boundary": "already-warmed compiled extern_kernels.addmm; weight.data_ptr identity",
            "steps": step, "inventory": inventory,
            "calls": [{"role": call.role, "m": call.m, "n": call.n, "k": call.k, "count": count}
                      for call, count in counts.items()],
        }, indent=2) + "\n", encoding="utf-8")
