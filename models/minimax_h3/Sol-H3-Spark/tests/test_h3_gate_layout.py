"""Gate-layout policy tests and real CPU-Torch storage/identity regression tests."""
import importlib.util
from pathlib import Path
import sys

import pytest

SOURCE = Path(__file__).parents[1] / "runtime/stage1_ops/gate_layout.py"


def helper():
    spec = importlib.util.spec_from_file_location("gate_layout_tested", SOURCE)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_default_flag_leaves_model_uninspected(monkeypatch):
    # Given no opt-in; when installing; then even parameter discovery is untouched.
    monkeypatch.delenv("SPARK_CP_H3_GATE_COLUMN_MAJOR", raising=False)
    class Untouched:
        def named_parameters(self):
            pytest.fail("disabled layout conversion must not inspect parameters")
    result = helper().install(Untouched())
    assert not result["enabled"]
    assert result["converted_count"] == 0


@pytest.mark.parametrize("value", ["yes", "2", "", "true"])
def test_invalid_flag_rejects_before_weight_access(monkeypatch, value):
    # Given invalid opt-in syntax; when parsing; then fail explicitly.
    monkeypatch.setenv("SPARK_CP_H3_GATE_COLUMN_MAJOR", value)
    module = helper()
    with pytest.raises(module.GateLayoutError):
        module.install(None)


def toy_model(torch):
    model = torch.nn.Module()
    model.blocks = torch.nn.ModuleList()
    for index in range(50):
        block = torch.nn.Module()
        block.to_gate_compress = torch.nn.Linear(4, 6, bias=False, dtype=torch.bfloat16)
        with torch.no_grad():
            block.to_gate_compress.weight.copy_(torch.arange(24).reshape(6, 4) / 32 + index)
        model.blocks.append(block)
    return model


def test_real_parameters_preserve_bits_identity_grad_and_idempotence(monkeypatch):
    # Given 50 independent real BF16 CPU parameters, including existing grad and hooks.
    torch = pytest.importorskip("torch")
    monkeypatch.setenv("SPARK_CP_H3_GATE_COLUMN_MAJOR", "1")
    module, model = helper(), toy_model(torch)
    parameters = list(model.parameters())
    snapshots = [parameter.detach().clone() for parameter in parameters]
    for parameter in parameters:
        parameter.grad = torch.ones_like(parameter)
    grads = [parameter.grad for parameter in parameters]
    hook = parameters[0].register_hook(lambda grad: grad)
    hooks = parameters[0]._backward_hooks
    # When converting and then reinstalling.
    first = module.install(model)
    pointers = [parameter.data_ptr() for parameter in parameters]
    second = module.install(model)
    # Then values, Parameter/autograd metadata, and retained bytes survive; second call allocates nothing.
    assert list(model.parameters()) == parameters
    for parameter, snapshot, grad in zip(parameters, snapshots, grads):
        assert torch.equal(parameter.view(torch.int16), snapshot.view(torch.int16))
        assert parameter.shape == (6, 4) and parameter.stride() == (1, 6)
        assert parameter.requires_grad and parameter.grad is grad
    assert parameters[0]._backward_hooks is hooks
    assert pointers == [parameter.data_ptr() for parameter in parameters]
    assert first["converted_count"] == 50 and second["converted_count"] == 0
    assert second["already_column_major_count"] == 50
    assert first["retained_storage_bytes_before"] == first["retained_storage_bytes_after"] == 50 * 48
    assert first["modeled_peak_extra_bytes"] == 48
    assert second["modeled_peak_extra_bytes"] == 0
    hook.remove()


def test_bad_parameter_rejects_before_any_conversion(monkeypatch):
    # Given an unsupported final gate dtype; when installing; then no earlier gate is mutated.
    torch = pytest.importorskip("torch")
    monkeypatch.setenv("SPARK_CP_H3_GATE_COLUMN_MAJOR", "1")
    module, model = helper(), toy_model(torch)
    model.blocks[-1].to_gate_compress.weight = torch.nn.Parameter(torch.zeros(6, 4))
    original = model.blocks[0].to_gate_compress.weight.data_ptr()
    with pytest.raises(module.GateLayoutError):
        module.install(model)
    assert model.blocks[0].to_gate_compress.weight.data_ptr() == original
