"""Scoped loading of the existing pruned FP8 base and faithful student lookup."""
from __future__ import annotations

import glob
import hashlib
import importlib.util
import os
from pathlib import Path

import torch
from torch import nn

PRUNED = "/work/checkpoints/Comfy-MiniMax-H3/diffusion_models/minimax_h3_fl2va_pruned_fp8_scaled.safetensors"
TRANSFORMER = "/work/checkpoints/MiniMax-H3/transformer"
AUDIO = "/work/checkpoints/MiniMax-H3/audio_vae"
ADAPTER = "/work/checkpoints/FastH3-VSA/vsa-datafree/adapter_model.safetensors"


class FixedProjection(nn.Module):
    def __init__(self, outputs: torch.Tensor) -> None:
        super().__init__()
        self.register_buffer("outputs", outputs.to("cuda", torch.bfloat16))
        self.register_buffer("weight", torch.empty(0,device="cuda",dtype=torch.float32),persistent=False)

    def forward(self, indices: torch.Tensor) -> tuple[torch.Tensor, None]:
        return self.outputs.index_select(0,indices[:,0].to(torch.long)),None


class FixedSchedule(nn.Module):
    def __init__(self, plan, record) -> None:
        super().__init__()
        self.plan = plan
        self.record = record
        offset = 0
        for step, row in enumerate(plan):
            self.register_buffer(f"indices_{step}",torch.arange(offset,offset+len(row),
                                 device="cuda",dtype=torch.float32).reshape(-1,1),persistent=False)
            offset += len(row)

    def forward(self, timestep: torch.Tensor) -> torch.Tensor:
        if torch.compiler.is_compiling():
            raise RuntimeError("timestep validation must remain eager")
        step = self.record["calls"] % 4
        if timestep.dtype != torch.float32 or timestep.detach().cpu().tolist() != self.plan[step]:
            raise RuntimeError(f"native timestep plan drift at step {step}")
        self.record["calls"] += 1
        return getattr(self,f"indices_{step}")


def lookup_record(model, pipe, task="t2va"):
    if task != "t2va":
        raise ValueError("the recovered fixed lookup is T2VA only")
    return model._h3_adopted_lookup


def _absorbed(name: str) -> bool:
    return ".adaln_proj.linear." in name or name.startswith(("norm_out.linear.","time_embedder."))


def install_loader(worker) -> None:
    from fastvideo.models.loader import component_loader as component, fsdp_load
    from fastvideo.models.loader.lora_patch import DenseLoRAPatch
    from fastvideo.pipelines import lora_pipeline
    import runtime.stage1 as stage1

    lookup_path = Path(os.environ["H3_ADOPTED_LOOKUP"])
    payload = torch.load(lookup_path,map_location="cpu",weights_only=True)
    outputs,plan = payload["outputs"],payload["plan"]
    source_hash = hashlib.sha256(lookup_path.read_bytes()).hexdigest()
    original_model_load = component.maybe_load_fsdp_model
    original_iterator = fsdp_load.safetensors_weights_iterator
    original_dense = DenseLoRAPatch.from_adapter
    original_lora_file = lora_pipeline.load_file
    original_lora_layer = lora_pipeline.get_lora_layer

    class ScopedGlob:
        def glob(self, pattern, *args, **kwargs):
            if pattern == TRANSFORMER + "/*.safetensors":
                return [PRUNED]
            if pattern == AUDIO + "/*.safetensors":
                return [os.environ["H3_ADOPTED_AUDIO_CHECKPOINT"]]
            return glob.glob(pattern,*args,**kwargs)

    component.glob = ScopedGlob()
    stage1.install_exact_t2va_lookup = lookup_record

    def model_load(*args,**kwargs):
        original_class = kwargs["model_cls"]
        if kwargs["weight_dir_list"] != [PRUNED]:
            return original_model_load(*args,**kwargs)
        class LookupH3(original_class):
            def __init__(self,*model_args,**model_kwargs):
                super().__init__(*model_args,**model_kwargs)
                record = {
                    "mode":"fullrank_student_7row_lookup_from_pinned_teacher_ranges",
                    "native_timestep_plan":plan,"calls":0,"source_sha256":source_hash,
                    "student_time_deltas_included":True,"student_adaln_deltas_included":True,
                    "base_rank8_projection_used":False,"projections":len(outputs),
                }
                self._h3_adopted_lookup = record
                for name,value in outputs.items():
                    parent,_,leaf = name.rpartition(".")
                    target = self.get_submodule(parent)
                    setattr(target,leaf,FixedProjection(value))
                    target.apply_silu = False
                self.time_proj = FixedSchedule(plan,record)
                self.time_embedder = nn.Identity()
                self.time_embedder.fc_in = nn.Module()
                self.time_embedder.fc_in.register_buffer("weight",torch.empty(
                    0,device="cuda",dtype=torch.float32),persistent=False)
                self.adaln_basis = None
        kwargs["model_cls"] = LookupH3
        return original_model_load(*args,**kwargs)

    def iterator(files,*args,**kwargs):
        if files != [PRUNED]:
            yield from original_iterator(files,*args,**kwargs)
            return
        spec = importlib.util.spec_from_file_location("h3_adopted_relayout",
             "/work/runtime/gb10-matched-209-v1/relayout.py")
        if spec is None or spec.loader is None:
            raise ImportError("pinned GB10 relayout")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        source = module.read_pruned_fp8_checkpoint(PRUNED,56,128)
        for name,value in source["tensors"].items():
            if _absorbed(name):
                continue
            if value.dtype == torch.float8_e4m3fn:
                scale = source["quant"][name.removesuffix(".weight")]["weight_scale"]
                represented = value.to("cuda",torch.bfloat16) * scale.to("cuda",torch.bfloat16)
            else:
                represented = value.to("cuda")
            yield name,represented
        for name,value in outputs.items():
            yield name + ".outputs",value.to("cuda",torch.bfloat16)

    def dense(cls,path,*args,**kwargs):
        patch = original_dense(path,*args,**kwargs)
        if path == ADAPTER and patch is not None:
            patch._additive = {name:value for name,value in patch._additive.items() if not _absorbed(name)}
        return patch

    def lora_file(path,*args,**kwargs):
        values = original_lora_file(path,*args,**kwargs)
        if str(path) == ADAPTER:
            values = {name:value for name,value in values.items() if not _absorbed(name)}
        return values

    def lora_layer(*args,**kwargs):
        if not kwargs.get("training_mode",False):
            kwargs["retain_cpu_weight"] = False
        return original_lora_layer(*args,**kwargs)

    component.maybe_load_fsdp_model = model_load
    fsdp_load.safetensors_weights_iterator = iterator
    DenseLoRAPatch.from_adapter = classmethod(dense)
    lora_pipeline.load_file = lora_file
    lora_pipeline.get_lora_layer = lora_layer
