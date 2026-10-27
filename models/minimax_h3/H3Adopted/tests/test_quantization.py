"""Behavior checks for adopted MXFP8 producers and projection scope.

GPU functions can also be invoked directly on node3 when pytest is unavailable.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from fastvideo.layers.linear import ReplicatedLinear

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def test_scale_allocation_when_rows_have_tail() -> None:
    # Given: 129 rows and three 32-element groups per row.
    from H3Adopted.mxfp8_kernels import scale_numel

    # When: allocating the cuBLASLt swizzled E8M0 buffer.
    result = scale_numel(129, 96)
    # Then: both row tiles and the four-column scale tile are padded.
    assert result == 1024


def test_constant_matrix_when_block_scales_are_required() -> None:
    # Given: E4M3 data with every activation and weight equal to one.
    import torch
    from H3Adopted.mxfp8_kernels import quantize_swizzled, scaled_mm

    x = torch.ones(129, 128, device="cuda", dtype=torch.bfloat16)
    weight = torch.ones(64, 128, device="cuda", dtype=torch.bfloat16)
    # When: using block-wise 1x32 scales on both GEMM operands.
    result = scaled_mm(quantize_swizzled(x), quantize_swizzled(weight))
    # Then: the padded rows/scales do not corrupt the exact dot product.
    assert torch.equal(result, torch.full_like(result, 128))


def test_fused_swiglu_when_diffusers_value_gate_order_is_used() -> None:
    # Given: Diffusers FC1 stores value before gate.
    import torch
    from fusions import fused_swiglu
    from H3Adopted.mxfp8_kernels import quantize_swizzled, swiglu_quantize

    generator = torch.Generator(device="cuda").manual_seed(31)
    x = torch.randn(129, 256, generator=generator, device="cuda", dtype=torch.bfloat16)
    reference = quantize_swizzled(fused_swiglu(x))
    # When: fusing the SAME tl.sigmoid/two-BF16-rounding activation into MXFP8.
    result, rounded = swiglu_quantize(x, value_first=True, return_bf16=True)
    # Then: product, E4M3 payload, and all swizzled scale bytes match.
    assert torch.equal(rounded, fused_swiglu(x))
    assert torch.equal(
        result.payload.view(torch.uint8), reference.payload.view(torch.uint8)
    )
    assert torch.equal(result.scales, reference.scales)


def test_linear_dispatch_when_weight_and_bias_are_constant() -> None:
    # Given: a real CUDA Linear with a nonzero bias.
    import torch
    from H3Adopted.quantization import Mxfp8Linear

    source = torch.nn.Linear(128, 64, device="cuda", dtype=torch.bfloat16)
    with torch.no_grad():
        source.weight.fill_(1)
        source.bias.fill_(3)
    x = torch.ones(1, 129, 128, device="cuda", dtype=torch.bfloat16)
    # When: converting and invoking the quantized adapter once.
    adapted = Mxfp8Linear.from_module(source, name="constant")
    result = adapted(x)
    # Then: output shape/bias are honored and actual dispatch is recorded.
    assert torch.equal(
        result, torch.full((1, 129, 64), 131, device="cuda", dtype=torch.bfloat16)
    )
    assert adapted.dispatch_snapshot().gemms == 1


def test_fastvideo_selection_when_refiners_and_adaln_exist() -> None:
    # Given: native FastVideo main projections plus unmodified refiner and AdaLN.
    from H3Adopted.quantization import is_fastvideo_target

    included = (
        "transformer_blocks.3.attn.to_q",
        "transformer_blocks.3.attn.to_out",
        "transformer_blocks.3.ff.fc_in",
        "transformer_blocks.3.ff.fc_out",
    )
    excluded = (
        "token_refiner.blocks.0.attn.to_q",
        "transformer_blocks.3.adaln_proj.linear",
        "transformer_blocks.3.attn.compression_gate",
        "norm_out.linear",
    )
    # When: selecting projection names for MXFP8 installation.
    decisions = [is_fastvideo_target(name) for name in included + excluded]
    # Then: only main-stack projections are admitted.
    assert decisions == [True] * 4 + [False] * 4


def test_fused_swiglu_when_native_fastvideo_final_rounding_is_used() -> None:
    # Given: the actual native fused kernel with final-store-only BF16 rounding.
    import torch
    from fastvideo.models.dits.minimax_h3_fusions.swiglu import minimax_h3_swiglu
    from H3Adopted.mxfp8_kernels import quantize_swizzled, swiglu_quantize

    generator = torch.Generator(device="cuda").manual_seed(37)
    x = torch.randn(129, 256, generator=generator, device="cuda", dtype=torch.bfloat16)
    expected = minimax_h3_swiglu(x)
    reference = quantize_swizzled(expected)
    # When: fusing the same native arithmetic and MXFP8 scale packing.
    result, rounded = swiglu_quantize(x, round_activation=False, return_bf16=True)
    # Then: both intermediate target semantics and MXFP8 bytes match.
    assert torch.equal(rounded, expected)
    assert torch.equal(
        result.payload.view(torch.uint8), reference.payload.view(torch.uint8)
    )
    assert torch.equal(result.scales, reference.scales)


def test_dispatch_when_linear_is_regionally_compiled() -> None:
    # Given: a constant matrix projection and the actual opaque custom op.
    import torch
    from H3Adopted.quantization import Mxfp8Linear

    source = torch.nn.Linear(128, 64, bias=False, device="cuda", dtype=torch.bfloat16)
    with torch.no_grad():
        source.weight.fill_(1)
    adapted = Mxfp8Linear.from_module(source, name="compiled_constant")
    compiled = torch.compile(adapted, fullgraph=True, backend="inductor")
    x = torch.ones(1, 129, 128, device="cuda", dtype=torch.bfloat16)
    # When: executing the compiled projection twice.
    first, second = compiled(x), compiled(x)
    # Then: fake tracing added no counts, and both real executions are correct.
    assert torch.equal(first, torch.full_like(first, 128))
    assert torch.equal(second, torch.full_like(second, 128))
    assert adapted.dispatch_snapshot().gemms == 2


def _qkv_layers(
    block_index: int,
) -> tuple[ReplicatedLinear, ReplicatedLinear, ReplicatedLinear]:
    """Construct real native ReplicatedLinear objects with distinct constant weights."""
    import torch
    from fastvideo.layers.linear import ReplicatedLinear
    from H3Adopted.quantization import install_fastvideo

    model = torch.nn.Module()
    blocks = torch.nn.ModuleList([torch.nn.Module() for _ in range(block_index + 1)])
    model.add_module("transformer_blocks", blocks)
    attention = torch.nn.Module()
    blocks[block_index].add_module("attn", attention)
    layers: list[ReplicatedLinear] = []
    for leaf, factor, bias_value in (("to_q", 1, 3), ("to_k", 2, 5), ("to_v", 3, 7)):
        layer = ReplicatedLinear(
            128,
            64,
            bias=True,
            params_dtype=torch.bfloat16,
            prefix=f"transformer_blocks.{block_index}.attn.{leaf}",
        ).cuda()
        with torch.no_grad():
            layer.weight.fill_(factor)
            layer.bias.fill_(bias_value)
        layer.skip_bias_add = leaf == "to_k"
        attention.add_module(leaf, layer)
        layers.append(layer)
    install_fastvideo(model)
    query, key, value = layers
    return query, key, value


def test_shared_qkv_when_native_bias_and_scale_reuse_are_required() -> None:
    # Given: real native linears, with distinct weights and key's deferred bias.
    import torch
    from H3Adopted.quantization import dispatch_counts, project_qkv

    query, key, value = _qkv_layers(0)
    x = torch.ones(1, 129, 128, device="cuda", dtype=torch.bfloat16)
    # When: projecting the same activation through Q, K, and V.
    outputs = project_qkv(query, key, value, x)
    # Then: native output/bias semantics are preserved, with one quantization.
    for output, expected in zip(outputs, (131, 256, 391), strict=True):
        assert torch.equal(output, torch.full_like(output, expected))
    selected = [
        row
        for row in dispatch_counts()
        if row.name.startswith("transformer_blocks.0.attn.")
    ]
    assert sum(row.quantizations for row in selected) == 1
    assert [row.gemms for row in selected] == [1, 1, 1]


def test_shared_qkv_when_native_projections_are_fullgraph_compiled() -> None:
    # Given: the same native ABI and opaque prequantized GEMM operators.
    import torch
    from H3Adopted.quantization import dispatch_counts, project_qkv

    query, key, value = _qkv_layers(1)
    compiled = torch.compile(project_qkv, fullgraph=True, backend="inductor")
    x = torch.ones(1, 129, 128, device="cuda", dtype=torch.bfloat16)
    # When: executing the shared projection twice through regional compilation.
    first, second = compiled(query, key, value, x), compiled(query, key, value, x)
    # Then: fake tracing creates no count and both real invocations share scales.
    for outputs in (first, second):
        for output, expected in zip(outputs, (131, 256, 391), strict=True):
            assert torch.equal(output, torch.full_like(output, expected))
    selected = [
        row
        for row in dispatch_counts()
        if row.name.startswith("transformer_blocks.1.attn.")
    ]
    assert sum(row.quantizations for row in selected) == 2
    assert [row.gemms for row in selected] == [2, 2, 2]
