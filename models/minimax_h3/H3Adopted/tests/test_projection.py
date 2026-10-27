"""Scope and semantic regressions for native projection source adaptation."""
from __future__ import annotations
import torch
from torch import nn
from H3Adopted import projection


class TupleLinear(nn.Module):
    def __init__(self, multiplier: float) -> None:
        super().__init__()
        self.multiplier = multiplier

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, None]:
        return x * self.multiplier, None


class NativeAttention(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.to_q, self.to_k, self.to_v = TupleLinear(1), TupleLinear(2), TupleLinear(3)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        query, _ = self.to_q(hidden_states)
        key, _ = self.to_k(hidden_states)
        value, _ = self.to_v(hidden_states)
        return query + key + value


class Block(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.attn = NativeAttention()


class Model(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.transformer_blocks = nn.ModuleList([Block() for _ in range(50)])
        self.token_refiner = nn.ModuleList([Block() for _ in range(2)])


def test_native_scope_preserves_refiner_and_forward_semantics(monkeypatch) -> None:
    model = Model()
    calls = []
    def shared(q, k, v, x):
        calls.append(1)
        return q(x)[0], k(x)[0], v(x)[0]
    monkeypatch.setattr(projection, "project_qkv", shared)
    original_refiner = model.token_refiner[0].attn.forward.__func__
    x = torch.tensor([1.0, 2.0])
    expected = model.transformer_blocks[0].attn(x)
    receipt = projection.install_shared_qkv(model)
    assert receipt["main_attentions"] == 50
    assert torch.equal(model.transformer_blocks[0].attn(x), expected)
    assert model.token_refiner[0].attn.forward.__func__ is original_refiner
    assert torch.equal(model.token_refiner[0].attn(x), expected)
    assert len(calls) == 1
    compiled = torch.compile(model.transformer_blocks[0].attn, fullgraph=True, backend="eager")
    assert torch.equal(compiled(x), expected)
    assert len(calls) == 2
