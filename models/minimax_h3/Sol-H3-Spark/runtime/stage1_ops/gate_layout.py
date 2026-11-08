"""Opt-in BF16 compression-gate storage layout, resolved before compilation."""
from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
import os
from typing import TYPE_CHECKING, Protocol, TypedDict

if TYPE_CHECKING:
    import torch


@dataclass(frozen=True, slots=True)
class GateLayoutError(RuntimeError):
    detail: str

    def __str__(self) -> str:
        return self.detail


class GateModel(Protocol):
    def named_parameters(self) -> Iterator[tuple[str, torch.nn.Parameter]]: ...


class Receipt(TypedDict):
    enabled: bool
    parameter_count: int
    converted_count: int
    already_column_major_count: int
    dtype: str
    shapes: list[list[int]]
    before_strides: list[list[int]]
    after_strides: list[list[int]]
    retained_storage_bytes_before: int
    retained_storage_bytes_after: int
    modeled_peak_extra_bytes: int


def install(model: GateModel) -> Receipt:
    """Replace one gate's storage at a time, retaining Parameter/autograd identity.

    Must run after loading/auditing all 50 published gates, before any compiled
    forward or saved backward graph. No per-request state or persistent copies.
    """
    flag = os.environ.get("SPARK_CP_H3_GATE_COLUMN_MAJOR", "0")
    if flag not in ("0", "1"):
        raise GateLayoutError(f"SPARK_CP_H3_GATE_COLUMN_MAJOR must be 0 or 1, got {flag!r}")
    receipt: Receipt = {"enabled": flag == "1", "parameter_count": 0, "converted_count": 0,
        "already_column_major_count": 0, "dtype": "torch.bfloat16", "shapes": [],
        "before_strides": [], "after_strides": [], "retained_storage_bytes_before": 0,
        "retained_storage_bytes_after": 0, "modeled_peak_extra_bytes": 0}
    if flag == "0":
        return receipt
    import torch

    parameters = [(name, parameter) for name, parameter in model.named_parameters()
                  if name.endswith(".to_gate_compress.weight")]
    if len(parameters) != 50:
        raise GateLayoutError(f"expected 50 audited compression gate parameters, got {len(parameters)}")
    # Complete preflight before mutating the first parameter. Tensor references add no storage.
    pointers: set[int] = set()
    for name, parameter in parameters:
        if (parameter.layout != torch.strided or parameter.ndim != 2
                or min(parameter.shape) <= 1 or parameter.dtype != torch.bfloat16
                or parameter.storage_offset() != 0
                or parameter.stride() not in ((parameter.shape[1], 1), (1, parameter.shape[0]))
                or parameter.untyped_storage().nbytes() != parameter.numel() * parameter.element_size()):
            raise GateLayoutError(f"unsupported BF16 gate storage: {name}")
        pointer = parameter.untyped_storage().data_ptr()
        if pointer in pointers:
            raise GateLayoutError(f"shared compression gate storage: {name}")
        pointers.add(pointer)
    receipt["parameter_count"] = len(parameters)
    receipt["shapes"] = [list(shape) for shape in sorted({tuple(p.shape) for _, p in parameters})]
    receipt["before_strides"] = [list(stride) for stride in sorted({tuple(p.stride()) for _, p in parameters})]
    receipt["retained_storage_bytes_before"] = sum(p.untyped_storage().nbytes() for _, p in parameters)
    with torch.no_grad():
        for name, parameter in parameters:
            if parameter.stride() == (1, parameter.shape[0]):
                receipt["already_column_major_count"] += 1
                continue
            packed = parameter.t().contiguous().t()
            # Same-width integer views compare raw BF16 bits without another dense copy.
            if not torch.equal(parameter.view(torch.int16), packed.view(torch.int16)):
                raise GateLayoutError(f"layout conversion changed published gate bits: {name}")
            receipt["modeled_peak_extra_bytes"] = max(receipt["modeled_peak_extra_bytes"],
                                                       packed.untyped_storage().nbytes())
            parameter.set_(packed)
            receipt["converted_count"] += 1
            del packed
    receipt["after_strides"] = [list(stride) for stride in sorted({tuple(p.stride()) for _, p in parameters})]
    receipt["retained_storage_bytes_after"] = sum(p.untyped_storage().nbytes() for _, p in parameters)
    if receipt["retained_storage_bytes_before"] != receipt["retained_storage_bytes_after"]:
        raise GateLayoutError("gate storage size changed during layout conversion")
    return receipt
