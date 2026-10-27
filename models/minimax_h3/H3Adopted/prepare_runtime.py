"""Create an isolated, source-audited port of the known-working H3 runtime."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import shutil


def replace_once(text: str, before: str, after: str) -> str:
    if text.count(before) != 1:
        raise ValueError(f"source seam is not unique: {before!r}")
    return text.replace(before, after, 1)


def prepare(source: Path, destination: Path) -> None:
    if destination.exists():
        raise FileExistsError(destination)
    destination.mkdir(parents=True)
    for entry in ("runtime", "configs"):
        shutil.copytree(source / entry, destination / entry,
                        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    target = destination / "runtime/stage1.py"
    original = target.read_text()
    changed = original
    # Support the recovered matched209 source, with explicit expected seams.
    for option, value in (("height", "512"), ("width", "896"), ("num-frames", "209")):
        import re
        pattern = f'("--{option}", ")[0-9]+(")'
        changed, count = re.subn(pattern, lambda m: m[1] + value + m[2], changed)
        if count != 1:
            raise ValueError(f"dimension seam drift: {option}: {count}")
    changed = replace_once(changed, "    pipe = worker.pipeline\n    state =",
                           "    from H3Adopted.checkpoint import install_loader\n"
                           "    install_loader(worker)\n    pipe = worker.pipeline\n    state =")
    changed = replace_once(changed, "        convert_model_to_fp8(model)",
                           "        from H3Adopted.integration import convert_model\n        convert_model(model)")
    changed = replace_once(changed, "        capture.install_capture(worker)",
                           "        from H3Adopted.integration import install_capture\n        install_capture(worker)")
    changed = replace_once(changed, "            actual = self.rpc(finish_worker_request)[0]",
                           "            from H3Adopted.integration import finish_worker_request as adopted_finish\n"
                           "            actual = self.rpc(adopted_finish)[0]")
    changed = replace_once(changed, "            attention_before = self.rpc(native.read_fa4_audit)",
                           "            attention_before = self.rpc(native.read_fa4_audit)\n"
                           "            from H3Adopted.integration import dispatch_snapshot, dispatch_delta\n"
                           "            quant_before = self.rpc(dispatch_snapshot)[0]")
    changed = replace_once(changed, "            actual = self.rpc(adopted_finish)[0]",
                           "            actual = self.rpc(adopted_finish)[0]\n"
                           '            actual["actual_mxfp8"] = dispatch_delta(quant_before, actual["actual_mxfp8"])')
    changed = replace_once(changed, 'capture.get("latent_only_transfer") is not True',
                           'capture.get("tae_decoder_calls") != 1')
    changed = replace_once(changed, '"latent_only_transfer": True,\n                "started_monotonic_ns"',
                           '"latent_only_transfer": False,\n                "started_monotonic_ns"')
    changed = changed.replace('native W8A8 after BF16 LoRA merge; exact BF16 AdaLN lookup',
                              'MXFP8 main300 after BF16 LoRA merge; BF16refiner; exact BF16 AdaLN lookup')
    target.write_text(changed)
    from .attention_port import port_attention
    vsa_original, vsa_changed = port_attention(destination / "runtime/stage1_ops/vsa.py")
    regional = destination / "runtime/stage1_ops/regional.py"
    regional_text = regional.read_text()
    regional_text = replace_once(regional_text, 'or backend != "cudnn_bsa"',
        'or backend not in ("cudnn_bsa", "native_sage")')
    regional_text = regional_text.replace('"sparse_math_changed": False',
        '"sparse_math_changed": backend == "native_sage"')
    regional.write_text(regional_text)
    manifest = {
        "source_runtime": str(source),
        "stage1_original_sha256": hashlib.sha256(original.encode()).hexdigest(),
        "stage1_adopted_sha256": hashlib.sha256(changed.encode()).hexdigest(),
        "geometry": [896, 512, 209, 24],
        "updates": 4,
        "video_decoder": "TAEH3_FP16",
        "precision": "main300_MXFP8_BF16_output; refiner12_BF16; Qwen_NVFP4_AWQ",
        "default_attention": "cuDNN_BSA_SM121_VSA0.9",
        "available_attention": ["cudnn_bsa", "native_sage"],
        "vsa_original_sha256": hashlib.sha256(vsa_original.encode()).hexdigest(),
        "vsa_adopted_sha256": hashlib.sha256(vsa_changed.encode()).hexdigest(),
        "native_fusions": ["modulate", "qknorm_rope", "swiglu"],
        "new_fusion": "swiglu_MXFP8_FC2",
        "model_changed": True, "decoder_changed": True,
    }
    (destination / "adoption-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    args = parser.parse_args()
    prepare(args.source, args.destination)


if __name__ == "__main__":
    main()
