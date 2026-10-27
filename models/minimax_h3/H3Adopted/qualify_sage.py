"""Bounded numerical qualification for the optional native SM12 Sage candidate."""
from __future__ import annotations
import json
import math
from pathlib import Path
import torch
from .attention import attention


@torch.inference_mode()
def qualify(output: Path) -> None:
    if output.exists():
        raise FileExistsError(output)
    torch.manual_seed(42)
    receipts = []
    for sq, sk in ((128,192),(129,191),(256,256)):
        heads = 2
        q = torch.randn(1,sq,heads,128,device="cuda",dtype=torch.bfloat16)
        k = torch.randn(1,sk,heads,128,device="cuda",dtype=torch.bfloat16)
        v = torch.randn_like(k)
        nq,nk = math.ceil(sq/64),math.ceil(sk/64)
        indices = torch.arange(nk,device="cuda",dtype=torch.int32).expand(1,heads,nq,nk).clone()
        counts = torch.full((1,heads,nq),nk,device="cuda",dtype=torch.int32)
        counts[:,:,::2] = max(1,nk-1)
        sizes = torch.full((nk,),64,device="cuda",dtype=torch.int32)
        sizes[-1] = sk-(nk-1)*64
        sizes[0] = 47
        # BSA traverses the active prefix backwards; match ascending reference.
        positions = torch.arange(nk,device="cuda").expand_as(indices)
        selected = torch.where(positions<counts[...,None],counts[...,None]-1-positions,positions)
        reversed_indices = indices.gather(-1,selected).contiguous()
        actual = attention(q,k,v,reversed_indices,counts,128**-0.5,block_sizes=sizes)
        qh,kh,vh = (x.transpose(1,2).float() for x in (q,k,v))
        token_index = torch.arange(sk,device="cuda")
        qblock = torch.arange(sq,device="cuda")//64
        allowed = (token_index//64)[None,None,None,:] < counts[:,:,qblock,None]
        allowed = allowed & (token_index%64 < sizes[token_index//64])[None,None,None,:]
        logits = (qh @ kh.transpose(-1,-2))*128**-0.5
        reference = (logits.masked_fill(~allowed,float("-inf")).softmax(-1) @ vh).transpose(1,2)
        diff = actual.float()-reference
        rms = float(diff.norm()/reference.norm().clamp_min(1e-20))
        finite = bool(torch.isfinite(actual).all())
        row = {"q":sq,"k":sk,"heads":heads,"relative_l2":rms,
               "max_abs":float(diff.abs().max()),"finite":finite,"pass":finite and rms<0.08}
        receipts.append(row)
        print(json.dumps(row),flush=True)
        if not row["pass"]:
            raise RuntimeError("native Sage numerical qualification failed")
    output.write_text(json.dumps({"status":"PASS","gpu":torch.cuda.get_device_name(),
        "capability":list(torch.cuda.get_device_capability()),"cases":receipts,
        "boundary":"tiny synthetic only; model-shape speed and model warmup remain required"},indent=2)+"\n")


if __name__ == "__main__":
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,required=True)
    qualify(parser.parse_args().output)
