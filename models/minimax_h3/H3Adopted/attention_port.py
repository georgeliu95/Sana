"""Isolated sparse-provider port; keep selected masks and native VSA policy."""
from __future__ import annotations
from pathlib import Path


def port_attention(target: Path) -> tuple[str, str]:
    original = target.read_text()
    text = original.replace('backend != "cudnn_bsa"',
        'backend not in ("cudnn_bsa", "native_sage")').replace(
        'backend == "cudnn_bsa"', 'backend in ("cudnn_bsa", "native_sage")')
    text = text.replace('cudnn_bsa_calls', 'selected_sparse_calls')
    for seam in (
        '    if backend not in ("cudnn_bsa", "native_sage"):\n',
    ):
        if text.count(seam) != 2:
            raise RuntimeError("attention selector source drift")
        text = text.replace(seam,
            '    backend = os.environ.get("H3_ADOPTED_ATTENTION", backend)\n'+seam)
    before = '        bsa_forward = BSA.block_sparse_attention_forward'
    if text.count(before) != 1:
        raise RuntimeError("selected BSA source seam drift")
    text = text.replace(before,before+
        '\n        if backend == "native_sage":\n'
        '            from H3Adopted.attention import sage_bsa_forward\n'
        '            bsa_forward = sage_bsa_forward')
    before = 'and torch.allclose(actual_f32, ref_f32, atol=BSA_ATOL, rtol=BSA_RTOL)'
    if text.count(before) != 1:
        raise RuntimeError("selected validation source seam drift")
    text = text.replace(before,
        'and (relative_l2 <= 0.08 if backend == "native_sage" else\n'
        '                               torch.allclose(actual_f32, ref_f32, atol=BSA_ATOL, rtol=BSA_RTOL))')
    before = '    state["official_vsa"] = record'
    if text.count(before) != 1:
        raise RuntimeError("selected provider record seam drift")
    text = text.replace(before,
        '    if backend == "native_sage":\n'
        '        from H3Adopted import attention as provider\n'
        '        p = Path(provider.__file__).resolve()\n'
        '        record.update(kernel_entry="H3Adopted.attention.sage_bsa_forward",\n'
        '            kernel_source=str(p), kernel_source_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),\n'
        '            lse_api="empty; native VSA discards selected-attention LSE",\n'
        '            approximation="QK INT8/PV FP8, mean-centered K")\n'
        '        record["warmup_selected_validation"].update(atol=None,rtol=None,\n'
        '            criterion="finite and relative_l2 <= 0.08 for each of 200 actual calls",\n'
        '            relative_l2_limit=0.08,tolerance_source="declared approximate Sage qualification gate")\n'+before)
    target.write_text(text)
    return original,text
