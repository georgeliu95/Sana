# /// script
# requires-python = ">=3.12"
# dependencies = []
# ///
"""Run full-resolution H3 adoption on the pre-existing, pinned node3 runtimes.

Usage: python -m H3Adopted.run --runtime PORT --paths PATHS_JSON --prompt PROMPT_JSON
       --output NEW_DIRECTORY [--measure 3] [--profile-final]
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--paths", type=Path, required=True)
    parser.add_argument("--prompt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--measure", type=int, default=3)
    parser.add_argument("--profile-final", action="store_true")
    args = parser.parse_args()
    sys.path.insert(0, str(args.runtime.resolve()))
    from runtime import pipeline
    from runtime.config import load_recipe
    from runtime.worker import write_json

    original_environment = pipeline.worker_environment
    def worker_environment(name, paths):
        environment = original_environment(name, paths)
        environment["PYTHONPATH"] = str(Path(__file__).resolve().parent.parent) + os.pathsep + environment["PYTHONPATH"]
        if environment.get("SPARK_PROFILE_ROOT"):
            environment["PYTHONPATH"] += os.pathsep + environment["SPARK_PROFILE_ROOT"]
        environment["FASTVIDEO_MINIMAX_H3_FUSIONS"] = "all"
        return environment
    pipeline.worker_environment = worker_environment
    paths = json.loads(args.paths.read_text())
    prompt = json.loads(args.prompt.read_text())
    config = load_recipe("t2va")
    config["stage1"].update(width=896, height=512, frames=209)
    config["output"].update(width=896, height=512, frames=209, fps=24)
    case = {"case_id": prompt["name"], "prompt": prompt["prompt"], "seed": 0,
            "source_index": 0, "task": "t2va"}
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=False)
    report = {"status": "STARTING", "config": config, "requests": [],
              "geometry": [896,512,209,24], "warmup_excluded": True,
              "model_changed": True, "decoder_changed": True,
              "baseline_denominator": "archived matched209 same-node profile",
              "timing_scope": "fresh Qwen request entry to finalized muxed MP4",
              "cache_enabled": False, "cache_hits": 0}
    write_json(root / "results.json", report)
    stage1 = qwen = temporary = None
    startup = time.monotonic_ns()
    try:
        stage1 = pipeline.Worker("stage1", root / "stage1-worker", paths, config)
        temporary = pipeline.Worker("qwen", root / "warmup-qwen-worker", paths, config)
        warmup_case = dict(case, case_id="warmup", request_id="warmup", seed=999)
        warmup = root / "warmup"
        warmup.mkdir()
        cond = temporary.call(case=warmup_case, output_root=str(warmup / "qwen"))
        temporary.close()
        temporary = None
        s1 = stage1.call(case=warmup_case, conditioning_path=cond["conditioning_path"],
                         outputdir=str(warmup / "stage1"))
        qwen = pipeline.Worker("qwen", root / "qwen-worker", paths, config)
        report.update(status="READY", startup_and_warmup_s=(time.monotonic_ns()-startup)/1e9,
                      warmup=s1)
        write_json(root / "results.json", report)
        count = args.measure + int(args.profile_final)
        for index in range(count):
            profiled = bool(args.profile_final and index == args.measure)
            request_id = ("profile" if profiled else "measure") + f"-{index:02d}"
            request = root / request_id
            request.mkdir()
            request_case = dict(case, case_id=request_id, request_id=request_id)
            if profiled:
                import torch
                torch.cuda.profiler.start()
                torch.cuda.nvtx.range_push("H3Adopted/request")
            started = time.monotonic_ns()
            cond = qwen.call(case=request_case, output_root=str(request / "qwen"))
            native = stage1.call(case=request_case, conditioning_path=cond["conditioning_path"],
                                outputdir=str(request / "stage1"))
            ended = native["capture"]["final_mp4_complete_monotonic_ns"]
            wall = (ended-started)/1e9
            if not started < ended <= time.monotonic_ns():
                raise RuntimeError("same-host MP4 completion clock drift")
            row = {
                "request_id": request_id, "seed": 0, "status": "PASS",
                "profiled": profiled, "e2e_s": wall, "qwen_s": cond["request_wall_s"],
                "native_generate_s": native["request_wall_s"],
                "decode_and_mux_s": native["capture"]["decode_and_mux_s"],
                "output": native["capture"]["output"],
                "actual_nfe": native["request_transformer_forwards"],
                "cache_hits": 0, "cache_management_overhead_s": 0,
                "mxfp8": native["actual"]["actual_mxfp8"],
                "attention": native["official_vsa_actual"],
                "started_monotonic_ns": started,
                "completed_monotonic_ns": ended,
            }
            report["requests"].append(row)
            write_json(root / "results.json", report)
            print(json.dumps({k:v for k,v in row.items() if k not in ("mxfp8","attention")}), flush=True)
            if profiled:
                torch.cuda.nvtx.range_pop()
                torch.cuda.profiler.stop()
        ordinary = [row["e2e_s"] for row in report["requests"] if not row["profiled"]]
        report.update(status="PASS", mean_e2e_s=sum(ordinary)/len(ordinary),
                      min_e2e_s=min(ordinary), max_e2e_s=max(ordinary))
        write_json(root / "results.json", report)
    except BaseException as error:
        report.update(status="FAIL", error=f"{type(error).__name__}: {error}")
        write_json(root / "results.json", report)
        raise
    finally:
        for worker in (temporary, qwen, stage1):
            if worker is not None:
                worker.close()


if __name__ == "__main__":
    main()
