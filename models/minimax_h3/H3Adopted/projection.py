"""Audited projection-only port of native H3 attention."""
from __future__ import annotations

import hashlib
import inspect
import textwrap
from types import MethodType

from torch import nn

from .quantization import project_qkv


def install_shared_qkv(model: nn.Module) -> dict[str, str | int]:
    """Replace exactly three native calls on the fifty quantized main attentions."""
    targets = [module for name, module in model.named_modules()
               if name.startswith("transformer_blocks.") and name.endswith(".attn")]
    if len(targets) != 50:
        raise RuntimeError(f"native H3 main attention scope drift: {len(targets)}")
    original = targets[0].forward.__func__
    source = textwrap.dedent(inspect.getsource(original))
    before = (
        "    query, _ = self.to_q(hidden_states)\n"
        "    key, _ = self.to_k(hidden_states)\n"
        "    value, _ = self.to_v(hidden_states)\n"
    )
    if source.count(before) != 1:
        raise RuntimeError("native QKV source seam drift")
    changed = source.replace(before,
        "    query, key, value = project_qkv(\n"
        "        self.to_q, self.to_k, self.to_v, hidden_states)\n", 1)
    # Dynamo resolves LOAD_GLOBAL guards through the native module namespace.
    original.__globals__["project_qkv"] = project_qkv
    namespace = dict(original.__globals__)
    namespace["project_qkv"] = project_qkv
    exec(compile(changed, "<H3Adopted-shared-qkv>", "exec"), namespace)
    forward = namespace["forward"]
    for target in targets:
        if target.forward.__func__ is not original:
            raise RuntimeError("inconsistent native attention implementation")
        target.forward = MethodType(forward, target)
    return {
        "main_attentions": len(targets),
        "native_forward_sha256": hashlib.sha256(source.encode()).hexdigest(),
        "ported_forward_sha256": hashlib.sha256(changed.encode()).hexdigest(),
        "ordinary_quantizations_per_request": 600,
    }
