"""spark_rdma: two-Spark exchanges over RDMA from host-pinned buffers (DGX Spark has no GPUDirect RDMA)."""
from .transport import RdmaTransport, build, factory

__all__ = ["RdmaTransport", "build", "factory"]
