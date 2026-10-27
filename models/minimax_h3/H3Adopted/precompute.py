# /// script
# requires-python = ">=3.12"
# dependencies = ["torch", "safetensors"]
# ///
"""Recover only timestep/AdaLN byte ranges and retain seven merged output rows.

Usage: python precompute.py --index ORIGINAL_INDEX --adapter VSA_ADAPTER --output NEW_PT
This is startup preparation; no BF16 weight shards are materialized on disk.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import struct
import time
import urllib.request

import torch
import torch.nn.functional as F
from safetensors import safe_open

REPO = "MiniMaxAI/MiniMax-H3"
REVISION = "9bfb6693f2cf6de171db46d1aa586f67d773a1da"


@dataclass(frozen=True, slots=True)
class RangeRecord:
    dtype: str
    shape: tuple[int, ...]
    begin: int
    end: int


class TensorRanges:
    def __init__(self, weight_map: dict[str, str]) -> None:
        self.weight_map = weight_map
        self.headers: dict[str, tuple[int, dict[str, RangeRecord]]] = {}
        self.receipts: list[dict[str, str | int | list[int]]] = []
        self.total_bytes = 0

    def read(self, shard: str, begin: int, end: int) -> bytes:
        url = f"https://huggingface.co/{REPO}/resolve/{REVISION}/transformer/{shard}?h3range={begin}-{end-1}"
        request = urllib.request.Request(url, headers={"Range": f"bytes={begin}-{end-1}"})
        with urllib.request.urlopen(request, timeout=90) as response:
            expected = f"bytes {begin}-{end-1}/"
            if response.status != 206 or not response.headers.get("Content-Range", "").startswith(expected):
                raise RuntimeError("server did not honor exact byte range")
            payload = response.read(end - begin + 1)
        if len(payload) != end - begin:
            raise RuntimeError(f"range size mismatch: {len(payload)} vs {end-begin}")
        self.total_bytes += len(payload)
        return payload

    def tensor(self, key: str) -> torch.Tensor:
        shard = self.weight_map[key]
        if shard not in self.headers:
            count = struct.unpack("<Q", self.read(shard, 0, 8))[0]
            if not 0 < count < 8_000_000:
                raise ValueError("unexpected safetensors header size")
            header = json.loads(self.read(shard, 8, 8 + count))
            records = {
                name: RangeRecord(value["dtype"], tuple(value["shape"]),
                                  value["data_offsets"][0], value["data_offsets"][1])
                for name, value in header.items() if name != "__metadata__"
            }
            self.headers[shard] = (8 + count, records)
        base, records = self.headers[shard]
        record = records[key]
        data = self.read(shard, base + record.begin, base + record.end)
        self.receipts.append({
            "key": key, "shard": shard, "bytes": len(data), "shape": list(record.shape),
            "sha256": hashlib.sha256(data).hexdigest(),
        })
        dtypes = {"BF16": torch.bfloat16, "F16": torch.float16, "F32": torch.float32}
        return torch.frombuffer(bytearray(data), dtype=dtypes[record.dtype]).reshape(record.shape).to("cuda")


@torch.inference_mode()
def precompute(index: Path, adapter: Path, output: Path) -> None:
    from fastvideo.layers.visual_embedding import Timesteps
    from fastvideo.models.schedulers.scheduling_minimax_h3 import MiniMaxH3Scheduler

    if output.exists():
        raise FileExistsError(output)
    started = time.monotonic()
    reader = TensorRanges(json.loads(index.read_text())["weight_map"])
    schedules = []
    for shift in (12.0, 3.0):
        scheduler = MiniMaxH3Scheduler(shift=shift)
        scheduler.set_timesteps(5, device="cpu")
        schedules.append(scheduler.timesteps.tolist())
    plan = [sorted(set((v, a))) for v, a in zip(*schedules)]
    if [len(row) for row in plan] != [1,2,2,2]:
        raise RuntimeError(f"four-update plan drift: {plan}")
    print("PLAN", plan, flush=True)
    embed = Timesteps(256, True, 0)
    with safe_open(str(adapter), framework="pt", device="cpu") as student:
        time_weights = {}
        for layer in ("linear_1", "linear_2"):
            for suffix, delta in (("weight", "diff"), ("bias", "diff_b")):
                name = f"time_embedder.{layer}"
                original = reader.tensor(name + "." + suffix).float()
                time_weights[name + "." + suffix] = original + student.get_tensor(name + "." + delta).to("cuda",torch.float32)
        embeddings = []
        for row in plan:
            value = embed(torch.tensor(row,device="cuda",dtype=torch.float32))
            value = F.linear(value,time_weights["time_embedder.linear_1.weight"],time_weights["time_embedder.linear_1.bias"])
            value = F.linear(F.silu(value),time_weights["time_embedder.linear_2.weight"],time_weights["time_embedder.linear_2.bias"])
            embeddings.append(F.silu(value).to(torch.bfloat16))
        del time_weights
        outputs = {}
        targets = [f"transformer_blocks.{i}.adaln_proj.linear" for i in range(50)] + ["norm_out.linear"]
        for number, name in enumerate(targets):
            print(f"RECOVER {number+1}/51 {name}",flush=True)
            weight = reader.tensor(name + ".weight").to(torch.bfloat16)
            bias = reader.tensor(name + ".bias").float()
            if name.startswith("transformer_blocks."):
                left = student.get_tensor(name + ".lora_B.weight").to("cuda",torch.bfloat16)
                right = student.get_tensor(name + ".lora_A.weight").to("cuda",torch.bfloat16)
                delta = left @ right
                weight.add_(delta)
                del left,right,delta
                bias = (bias + student.get_tensor(name + ".diff_b").to("cuda",torch.float32)).to(torch.bfloat16)
            else:
                weight = (weight.float() + student.get_tensor(name + ".diff").to("cuda",torch.float32)).to(torch.bfloat16)
                bias = (bias + student.get_tensor(name + ".diff_b").to("cuda",torch.float32)).to(torch.bfloat16)
            projected = torch.cat([F.linear(value,weight,bias) for value in embeddings])
            if not bool(torch.isfinite(projected).all()):
                raise RuntimeError(f"nonfinite student modulation: {name}")
            outputs[name] = projected.cpu()
            del projected,weight,bias
            print(f"DONE {number+1}/51 bytes={reader.total_bytes} elapsed={time.monotonic()-started:.2f}s",flush=True)
    torch.save({"outputs":outputs,"plan":plan,"schedules":schedules},output)
    receipt = {
        "status":"PASS","repo":REPO,"revision":REVISION,
        "mode":"native_FP32_student_time_and_BF16_merged_fullrank_AdaLN_7row_lookup",
        "student_delta_omitted":False,"student_strength":1.0,
        "student_adapter":str(adapter),"native_timestep_plan":plan,
        "tensor_ranges":reader.receipts,"total_received_bytes":reader.total_bytes,
        "outputs_sha256":hashlib.sha256(output.read_bytes()).hexdigest(),
        "output_bytes":output.stat().st_size,"preparation_s":time.monotonic()-started,
        "whole_shard_sha256_verified":False,
    }
    output.with_suffix(".json").write_text(json.dumps(receipt,indent=2)+"\n")
    print(json.dumps({k:v for k,v in receipt.items() if k!="tensor_ranges"}),flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index",type=Path,required=True)
    parser.add_argument("--adapter",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    precompute(args.index,args.adapter,args.output)


if __name__=="__main__":
    main()
