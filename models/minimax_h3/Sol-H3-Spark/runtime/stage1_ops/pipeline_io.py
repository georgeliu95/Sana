"""Opt-in H3 integration for two layout fusions without changing RDMA wire buffers."""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
import os
from typing import TYPE_CHECKING, Protocol, TypedDict

from .pipeline_fusion_kernels import FusionError, merge_into, receive_scatter

if TYPE_CHECKING:
    import torch
    from ulysses_head_pipeline.mesh import HeadGroupPipeline


@dataclass(frozen=True, slots=True)
class FusionSettings:
    output: bool
    input: bool

    @classmethod
    def from_environment(cls) -> FusionSettings:
        values = tuple(os.environ.get(key, "0") for key in
                       ("SPARK_CP_H3_FUSE_OUTPUT", "SPARK_CP_H3_FUSE_INPUT"))
        if any(value not in ("0", "1") for value in values):
            raise FusionError("fusion options must be 0 or 1")
        return cls(values[0] == "1", values[1] == "1")


class FusionAudit(TypedDict):
    output_enabled: bool
    input_enabled: bool
    input_calls: int
    output_calls: int
    pipeline_calls: int


class TileHolder(Protocol):
    buffer: torch.Tensor | None
    untile_geometry: torch.Tensor | None


class Metadata(Protocol):
    total_seq_length: int
    variable_block_sizes: torch.Tensor
    untile_combined_index: torch.Tensor
    tile_buf_holder: TileHolder


class Backend(Protocol):
    def preprocess_qkv(self, inputs: torch.Tensor, metadata: Metadata) -> torch.Tensor: ...
    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                gate: torch.Tensor, metadata: Metadata) -> torch.Tensor: ...
    def forward_into(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                     gate: torch.Tensor, metadata: Metadata,
                     destinations: list[torch.Tensor]) -> None: ...
    def postprocess_output(self, output: torch.Tensor, metadata: Metadata) -> torch.Tensor: ...


@dataclass(frozen=True, slots=True)
class PipelineFusion:
    backend: Backend
    settings: FusionSettings
    audit: FusionAudit

    def run(self, pipeline: HeadGroupPipeline, inputs: Sequence[torch.Tensor],
            metadata: Metadata) -> torch.Tensor:
        """Use the original metadata holder and caller stream for tile-buffer lifetime."""
        import torch
        from runtime.profiling import scope
        from ulysses_head_pipeline.fused import AttentionIO, run_fused

        def receive(sources: tuple[torch.Tensor, ...]) -> torch.Tensor:
            source = sources[0]
            shape = (len(inputs), metadata.variable_block_sizes.numel() * 64, source.shape[2], source.shape[3])
            holder = metadata.tile_buf_holder
            if holder.buffer is None or holder.buffer.shape != shape:
                holder.buffer = torch.zeros(shape, device=source.device, dtype=source.dtype)
            elif holder.untile_geometry is not metadata.untile_combined_index:
                holder.buffer.zero_()
            with scope("Spark/H3/pipeline/fused_receive_scatter"):
                receive_scatter(sources, metadata.untile_combined_index, holder.buffer)
            holder.untile_geometry = metadata.untile_combined_index
            self.audit["input_calls"] += 1
            return holder.buffer

        def forward(gathered: torch.Tensor, targets: tuple[torch.Tensor, ...] | None) -> torch.Tensor | None:
            tiled = gathered if self.settings.input else self.backend.preprocess_qkv(gathered, metadata)
            q, k, v, gate = tiled.chunk(4, dim=0)
            if targets is not None:
                with scope("Spark/H3/pipeline/attention_output_into"):
                    self.backend.forward_into(q, k, v, gate, metadata, list(targets))
                self.audit["output_calls"] += 1
                return None
            output = self.backend.forward(q, k, v, gate, metadata)
            return self.backend.postprocess_output(output, metadata)

        if len(inputs) != 4:
            raise FusionError("H3 fusion requires Q/K/V and the published compression gate")
        result = run_fused(pipeline, inputs, AttentionIO(forward, receive if self.settings.input else None,
                                                       self.settings.output), metadata.total_seq_length)
        self.audit["pipeline_calls"] += 1
        return result


def merge_untile_into(out: torch.Tensor, gate: torch.Tensor, compressed: torch.Tensor,
                      tile: int, mapping: torch.Tensor, destinations: list[torch.Tensor]) -> None:
    """Keep the native forward seam's argument order and BF16 rounding."""
    if tile != 64:
        raise FusionError("output fusion requires tile64")
    from runtime.profiling import scope
    with scope("Spark/H3/pipeline/fused_merge_into"):
        merge_into(out, gate, compressed, mapping, tuple(destinations))
