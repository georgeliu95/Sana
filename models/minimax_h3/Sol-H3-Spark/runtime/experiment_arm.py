"""Campaign-only switches for interleaved same-process full inference ablations."""
from enum import StrEnum
import os

class Arm(StrEnum):
    BASE = "base"
    GC = "gc"
    GC_COPY = "gcopy"
    PACK = "pack"
    COPY = "copy"
    BRANCH = "branch"
    COMBO = "combo"

def apply_arm(request_id: str) -> None:
    """Preserve normal launches; experimental request IDs select explicit arms."""
    if os.environ.get("SPARK_CP_ABLATION") != "1":
        return
    arm = Arm.BASE if request_id == "warmup" else Arm(request_id.rsplit("_",1)[-1])
    flags = {"SPARK_CP_SKIP_IDLE_GC": arm in (Arm.GC,Arm.COMBO,Arm.GC_COPY),
             "SPARK_CP_H3_PACK": arm in (Arm.PACK,Arm.COMBO),
             "SPARK_CP_VAE_DIRECT_COPY": arm in (Arm.COPY,Arm.COMBO,Arm.GC_COPY),
             "SPARK_CP_BRANCH_SCHEDULE": arm == Arm.BRANCH}
    os.environ.update({name:str(int(enabled)) for name,enabled in flags.items()})
