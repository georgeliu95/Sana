"""LTX Sol-Attn producer/consumer fusion on the unchanged four-rank RDMA schedule."""
from __future__ import annotations
from collections.abc import Sequence
from dataclasses import dataclass
import os
from typing import TYPE_CHECKING, Protocol, TypedDict
from runtime.stage1_ops.pipeline_fusion_kernels import FusionError
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
                       ("SPARK_CP_LTX_FUSE_OUTPUT", "SPARK_CP_LTX_FUSE_INPUT"))
        if any(value not in ("0", "1") for value in values):
            raise FusionError("LTX fusion flags must be 0 or 1")
        return cls(values[0] == "1", values[1] == "1")

class FusionAudit(TypedDict):
    pipeline_calls: int
    input_calls: int
    output_calls: int

class SolAttention(Protocol):
    def __call__(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, heads: int, *,
                 part: int, parts: int, pipeline_gathered: torch.Tensor,
                 pipeline_targets: tuple[torch.Tensor, ...] | None,
                 pipeline_summaries: tuple[torch.Tensor, torch.Tensor] | None) -> torch.Tensor | None: ...

@dataclass(frozen=True, slots=True)
class PipelineFusion:
    attention: SolAttention
    settings: FusionSettings
    audit: FusionAudit

    def reset(self) -> None:
        self.audit.update(pipeline_calls=0, input_calls=0, output_calls=0)

    def run(self, pipeline: HeadGroupPipeline, inputs: Sequence[torch.Tensor]) -> torch.Tensor:
        """Keep shared summaries on the caller stream until the attention consumes them."""
        from techniques.sparse_backends.sol_attn_backend import _load_sol_attn
        from runtime.profiling import scope
        from ulysses_head_pipeline.fused import AttentionIO, run_fused
        _load_sol_attn()
        from sol_attn.pipeline import receive_prepare

        summaries: tuple[torch.Tensor, torch.Tensor] | None = None
        group = 0

        def receive(sources: tuple[torch.Tensor, ...]) -> torch.Tensor:
            nonlocal summaries
            with scope("Spark/LTX/pipeline/fused_receive_summary"):
                gathered, summaries = receive_prepare(sources)
            self.audit["input_calls"] += 1
            return gathered

        def forward(gathered: torch.Tensor, targets: tuple[torch.Tensor, ...] | None) -> torch.Tensor | None:
            nonlocal group
            _, tokens, heads, dim = gathered.shape
            q, k, v = (x.view(1, tokens, heads * dim) for x in gathered)
            with scope("Spark/LTX/pipeline/attention_output_into" if targets is not None
                       else "Spark/LTX/pipeline/attention_prepared"):
                out = self.attention(q, k, v, heads, part=group, parts=len(pipeline.sizes(inputs[0].shape[1])),
                                     pipeline_gathered=gathered, pipeline_targets=targets,
                                     pipeline_summaries=summaries)
            group += 1
            if targets is not None:
                self.audit["output_calls"] += 1
            return None if out is None else out.reshape(tokens, heads, dim)

        result = run_fused(pipeline, inputs, AttentionIO(forward, receive if self.settings.input else None,
                                                       self.settings.output), 4 * inputs[0].shape[0])
        self.audit["pipeline_calls"] += 1
        return result
