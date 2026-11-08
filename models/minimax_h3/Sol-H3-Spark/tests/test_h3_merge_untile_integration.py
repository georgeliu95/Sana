"""Packed-output protocol tests; GPU arithmetic was qualified separately."""
import ast
import os
from contextlib import nullcontext
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).parents[1] / "runtime/stage1_ops"


def load(name):
    spec = importlib.util.spec_from_file_location(f"test_{name}", ROOT / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("flag,expected", [(None, False), ("0", False), ("1", True)])
def test_opt_in_defaults_off(monkeypatch, flag, expected):
    # Given an installation flag; when parsing; then select one explicit output protocol.
    if flag is None:
        monkeypatch.delenv("SPARK_CP_H3_MERGE_UNTILE", raising=False)
    else:
        monkeypatch.setenv("SPARK_CP_H3_MERGE_UNTILE", flag)
    assert load("merge_untile").enabled_from_environment() is expected


def test_invalid_flag_fails_explicitly(monkeypatch):
    # Given invalid syntax; when installing; then reject rather than silently falling back.
    monkeypatch.setenv("SPARK_CP_H3_MERGE_UNTILE", "yes")
    module = load("merge_untile")
    with pytest.raises(module.MergeContractError):
        module.enabled_from_environment()


@pytest.mark.parametrize("rows", [9793, 12032])
def test_packed_postprocess_never_indexes_again(rows):
    # Given packed output or a mistaken padded output; when postprocessing; then return identity or reject.
    module = load("merge_untile")
    class Output:
        ndim = 4
        shape = (1, rows, 7, 128)
        def __getitem__(self, index):
            pytest.fail("packed output must never be gathered a second time")
    output = Output()
    metadata = SimpleNamespace(total_seq_length=9793, untile_combined_index=SimpleNamespace(numel=lambda: 9793))
    if rows == 9793:
        assert module.packed_postprocess(output, metadata) is output
    else:
        with pytest.raises(module.MergeContractError):
            module.packed_postprocess(output, metadata)


@pytest.mark.parametrize("packed", [False, True])
def test_native_ast_passes_mapping_only_to_explicit_packed_merge(packed):
    # Given the actual native forward; when rewriting; then its selected output protocol gets the proper ABI.
    checkout = os.environ.get("FASTVIDEO_ROOT")
    if checkout is None:
        pytest.skip("set FASTVIDEO_ROOT to the matching FastVideo checkout")
    native_path = Path(checkout) / "fastvideo/attention/backends/video_sparse_attn_h3.py"
    native = ast.parse(native_path.read_text())
    impl = next(node for node in native.body if isinstance(node, ast.ClassDef) and node.name == "MiniMaxH3VSAImpl")
    forward = next(node for node in impl.body if isinstance(node, ast.FunctionDef) and node.name == "forward")
    source = ast.unparse(forward)
    tree = load("fused").optimized_tree(source, bshd=True, fused_merge=True, packed_output=packed)
    calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)
             and isinstance(node.func, ast.Name) and node.func.id == "_optimized_merge"]
    assert len(calls) == 1
    assert len(calls[0].args) == (5 if packed else 4)
    if packed:
        assert ast.unparse(calls[0].args[-1]) == "attn_metadata.untile_combined_index"


def test_packed_fake_and_metadata_preserve_map_without_gpu_scalar_reads():
    # Given symbolic dimensions; when constructing packed metadata/fake; then packed rows come from map shape.
    module = load("merge_untile")
    mapping = SimpleNamespace(numel=lambda: 9793)
    query = SimpleNamespace(shape=(1, 12032, 7, 128), new_empty=lambda shape: shape)
    metadata = module.packed_metadata(mapping, SimpleNamespace(), 8, 180, 2)
    assert metadata.total_seq_length == 9793 and metadata.untile_combined_index is mapping
    assert module.packed_fake(query, query, mapping) == (1, 9793, 7, 128)


def test_actual_regional_packed_branch_passes_mapping_and_matches_fake_shape():
    # Given the production nested regional functions with only the custom-op registration seam stubbed.
    helper = load("merge_untile")
    tree = ast.parse((ROOT / "regional.py").read_text())
    install = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "install")
    packed = next(node for node in install.body if isinstance(node, ast.If)
                  and ast.unparse(node.test) == "packed_output")
    packed.body = [node for node in packed.body if not isinstance(node, ast.ImportFrom)]
    forward = next(node for node in install.body if isinstance(node, ast.FunctionDef) and node.name == "body_forward")
    fakes, seen = {}, []
    class Tensor:
        def __init__(self, shape):
            self.shape, self.ndim = shape, len(shape)
        def new_empty(self, shape):
            return Tensor(shape)
        def contiguous(self):
            return self
        def __getitem__(self, index):
            pytest.fail("packed regional output must not be gathered again")
    def custom_op(name, **kwargs):
        def register(function):
            function.register_fake = lambda fake: fakes.setdefault(name, fake)
            return function
        return register
    mapping = SimpleNamespace(numel=lambda: 9793)
    def original(q, k, v, gate, metadata):
        seen.append(metadata)
        return Tensor((1, metadata.total_seq_length, 7, 128))
    receipt = {"body_runtime_calls": 0}
    namespace = {"packed_output": True, "torch": SimpleNamespace(Tensor=Tensor,
        library=SimpleNamespace(custom_op=custom_op)), "packed_fake": helper.packed_fake,
        "packed_metadata": helper.packed_metadata, "packed_postprocess": helper.packed_postprocess,
        "_step_number": lambda step: step, "_layer_number": lambda layer, count: 0,
        "scope": lambda name: nullcontext(), "body_originals": [original], "receipt": receipt,
        "body_op": lambda *args: pytest.fail("packed mode reached padded-body operator")}
    exec(compile(ast.Module(body=[packed, forward], type_ignores=[]), "regional.py", "exec"), namespace)
    metadata = SimpleNamespace(VSA_sparsity=0.9, tile_elems=64, exempt=True, dense_layers=(),
        variable_block_sizes=SimpleNamespace(), untile_combined_index=mapping,
        num_prefix_tiles=8, num_video_tiles=180, _sol_h3_cpu_step=2)
    query = Tensor((1, 12032, 7, 128))
    # When executing the actual forwarding/custom-op functions and their registered fake.
    result = namespace["body_forward"](SimpleNamespace(_sol_h3_cpu_layer=0), query, query, query, query, metadata)
    fake = fakes["sol_h3_spark_vsa::body_packed"](query, query, query, query, None, mapping, 0, 8, 180, 2)
    # Then mapping survives reconstructed metadata, runtime/fake agree, and postprocess remains identity.
    assert seen[0].untile_combined_index is mapping
    assert result.shape == fake.shape == (1, 9793, 7, 128)
    assert receipt["body_runtime_calls"] == 1
    assert helper.packed_postprocess(result, seen[0]) is result
