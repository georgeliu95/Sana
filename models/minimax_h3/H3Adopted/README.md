# H3Adopted

Full-resolution, four-NFE FastH3 VSA DataFree with main300 MXFP8, shared QKV activation quantization, fused SwiGLU quantization, resident FP16 TAEH3 and original H3 PCM to finalized MP4.

Matched209 experiment and complete Nsys comparison: [report](../../../runs/h3-adopted-matched209-20261009/README.md). Actual measured version is archived under that run's `metadata/measured-code-v10` and `measured-runtime-v10`; later report/error handling changes are separate.

This package ports only the selected GB10-compatible combination. The [101-item adoption manifest](../../../runs/h3-adopted-matched209-20261009/optimization-adoption.csv) distinguishes actual enabled/equivalent/partial/rejected/inapplicable optimizations. Native Sage is experimental and failed the actual model numerical gate; cuDNN BSA remains the measured default.

The runner uses a pinned, isolated existing node3 FastVideo/runtime environment; it is not a standalone model downloader. `prepare_runtime` checks exact source seams before creating a separate runtime. Reproduction uses the archived v10 snapshot and existing verified assets; see the experiment's replay script.

Lookup assets support only the recorded T2VA/four-step/student-strength1 schedule. Rebuild them for a changed task, flow shift, adapter or schedule. Main projection values originate from the represented pruned scalar-FP8 weights, followed by BF16 student merge and MXFP8; they are not original full-BF16 teacher projection weights.
