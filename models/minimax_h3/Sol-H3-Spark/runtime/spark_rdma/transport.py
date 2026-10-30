"""Two-Spark exchanges over RDMA from host-pinned buffers, ordered on CUDA streams (spark_rdma.c).

DGX Spark (GB10) has no GPUDirect RDMA: cudaMalloc memory cannot be registered with the NIC (ibv_reg_mr EFAULT), and
NCCL stages every chunk through its own host FIFO with a 4-CTA copy kernel and skips registering host memory. The DGX
Spark porting guide's fallback is cudaHostAlloc buffers registered with ibv_reg_mr; this transport does exactly that:
one send and one receive buffer split into slots (send_off, recv_off, bytes). Per slot, post(slot, stream) marks the
send region ready (cuStreamWriteValue32) and wait(slot, stream) makes the stream wait until the exchange is done
(cuStreamWaitValue32): no kernel, no SM, no staging copy; a progress thread in the library moves the data (RDMA WRITE
WITH IMMEDIATE over both RoCE ports) with per-slot flow control. Post a slot only after the stream's consumers of the
slot's previous exchange and the producer of its send region; wait before reading the receive region or rewriting the
send region. Kernels should only stream through these buffers (pack / unpack): the GPU does not keep host-pinned
memory in L2 (NCU: 0 % L2 hits), so GEMMs and attention must read device copies.
"""
from __future__ import annotations

import ctypes
import fcntl
import hashlib
import os
from pathlib import Path
import socket
import struct
import subprocess
import tempfile
import threading

import torch
import torch.distributed as dist
from cuda.bindings import driver

_SOURCE = Path(__file__).with_name("spark_rdma.c")


def build(directory: str | os.PathLike | None = None) -> Path:
    """spark_rdma.c compiled into a shared library (gcc, libibverbs / librdmacm), cached by the source's hash under
    SPARK_RDMA_BUILD_DIR (default: <tmp>/spark_rdma)."""
    digest = hashlib.sha256(_SOURCE.read_bytes()).hexdigest()[:16]
    root = Path(directory or os.environ.get("SPARK_RDMA_BUILD_DIR") or Path(tempfile.gettempdir()) / "spark_rdma")
    lib = root / f"libspark_rdma-{digest}.so"
    if not lib.is_file():
        root.mkdir(parents=True, exist_ok=True)
        partial = root / f".{lib.name}.{os.getpid()}"
        subprocess.run([os.environ.get("CC", "gcc"), "-O2", "-Wall", "-fPIC", "-shared", str(_SOURCE), "-o",
                        str(partial), "-l:libibverbs.so.1", "-l:librdmacm.so.1", "-lpthread"],
                       check=True, capture_output=True, text=True)
        os.replace(partial, lib)
    return lib


def _ok(result):
    err = result[0] if isinstance(result, tuple) else result
    if err != driver.CUresult.CUDA_SUCCESS:
        raise RuntimeError(f"CUDA driver call failed: {err}")
    return result[1] if isinstance(result, tuple) and len(result) == 2 else None


def host_pinned(nbytes: int) -> torch.Tensor:
    """cudaHostAlloc'd bytes as a CUDA uint8 tensor (no copy; never freed: transports live as long as the process)."""
    ptr = int(_ok(driver.cuMemHostAlloc(nbytes, 0)))

    class View:
        __cuda_array_interface__ = {"shape": (nbytes,), "typestr": "|u1", "data": (ptr, False), "version": 3}

    t = torch.as_tensor(View(), device="cuda")
    if t.data_ptr() != ptr or not t.is_cuda:
        raise RuntimeError("host-pinned buffer is not usable as a CUDA tensor")
    return t


def rdma_devices() -> list[str]:
    """SPARK_RDMA_DEVS, else NCCL_IB_HCA (a plain comma list, ':port' suffixes dropped), else every RoCE device whose
    port 1 is ACTIVE. The order must match on both ranks: device i talks to the peer's device i."""
    spec = os.environ.get("SPARK_RDMA_DEVS") or os.environ.get("NCCL_IB_HCA", "")
    if spec and not spec.startswith(("^", "=")):
        return [d.split(":")[0] for d in spec.split(",") if d]
    root = Path("/sys/class/infiniband")
    return sorted(d.name for d in root.iterdir()
                  if (d / "ports/1/state").read_text().split()[-1] == "ACTIVE"
                  and (d / "ports/1/link_layer").read_text().strip() == "Ethernet")


def roce_ips(devs: list[str]) -> str:
    """IPv4 of each RoCE device's netdev, comma-separated (containers share the host network)."""
    ips = []
    for dev in devs:
        netdev = sorted(os.listdir(f"/sys/class/infiniband/{dev}/device/net"))[0]
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            ifreq = fcntl.ioctl(sock.fileno(), 0x8915, struct.pack("256s", netdev[:15].encode()))  # SIOCGIFADDR
        finally:
            sock.close()
        ips.append(socket.inet_ntoa(ifreq[20:24]))
    return ",".join(ips)


class RdmaTransport:
    """Collective constructor on a 2-rank group; slots = [(send_off, recv_off, bytes)], the same on both ranks."""

    def __init__(self, group, nbytes: int, slots: list[tuple[int, int, int]], *, qps: int = 4, chunk: int = 1 << 20,
                 port: int = 20411, connect_timeout_s: float = 60.0):
        if dist.get_world_size(group) != 2:
            raise ValueError("spark_rdma connects two ranks")
        self.handle = None
        rank = dist.get_rank(group)
        error = None
        try:
            _ok(driver.cuInit(0))
            lib = ctypes.CDLL(str(build()))
            devs = rdma_devices()
            local = roce_ips(devs)
        except Exception as exc:  # noqa: BLE001 -- reported to the peer before raising
            error, local = exc, None
        ips = [None, None]
        dist.all_gather_object(ips, local, group=group)
        if error is not None or None in ips:
            raise RuntimeError(f"spark_rdma unavailable on rank {ips.index(None)}") from error
        lib.sr_open.restype = ctypes.c_void_p
        lib.sr_open.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_char_p, ctypes.c_int, ctypes.c_int,
                                ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_size_t, ctypes.c_void_p]
        lib.sr_slot.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_size_t, ctypes.c_size_t, ctypes.c_size_t]
        lib.sr_early.restype = ctypes.c_uint
        for fn in ("sr_connect", "sr_error", "sr_close", "sr_early"):
            getattr(lib, fn).argtypes = [ctypes.c_void_p]
        self.lib, self.ips, self.devs = lib, ips, devs
        self.send = host_pinned(nbytes)
        self.recv = host_pinned(nbytes)
        self.flags = host_pinned(8 * len(slots)).view(torch.int32)
        self.flags.zero_()
        torch.cuda.synchronize()
        handle = lib.sr_open(rank, ips[rank].encode(), ips[1 - rank].encode(), port, qps, self.send.data_ptr(),
                             self.recv.data_ptr(), nbytes, chunk, self.flags.data_ptr())
        opened = bool(handle) and all(lib.sr_slot(handle, i, *slot) == 0 for i, slot in enumerate(slots))
        both = [None, None]
        dist.all_gather_object(both, opened, group=group)  # rank 0 listens from here on
        if not all(both):
            if handle:
                lib.sr_close(handle)
            raise RuntimeError(f"spark_rdma: sr_open / sr_slot failed ({both})")
        # rank 0 blocks until rank 1 connected: bound the wait (a stuck connect leaves a parked thread behind)
        result = []
        thread = threading.Thread(target=lambda: result.append(lib.sr_connect(handle)), daemon=True)
        thread.start()
        thread.join(connect_timeout_s)
        connected = [None, None]
        dist.all_gather_object(connected, bool(result) and result[0] == 0, group=group)
        if not all(connected):
            raise RuntimeError(f"spark_rdma: connection failed or timed out ({connected})")
        self.handle = handle
        self.seq = [0] * len(slots)
        self._flags_ptr = self.flags.data_ptr()
        self._wait_flags = driver.CUstreamWaitValue_flags.CU_STREAM_WAIT_VALUE_GEQ

    def view(self, which: str, offset: int, shape: tuple[int, ...], dtype: torch.dtype) -> torch.Tensor:
        buf = self.send if which == "send" else self.recv
        nbytes = torch.Size(shape).numel() * torch.empty((), dtype=dtype).element_size()
        return buf[offset:offset + nbytes].view(dtype).view(shape)

    def post(self, slot: int, stream: torch.cuda.Stream | None = None) -> None:
        self.seq[slot] += 1
        s = driver.CUstream((stream or torch.cuda.current_stream()).cuda_stream)
        _ok(driver.cuStreamWriteValue32(s, driver.CUdeviceptr(self._flags_ptr + 8 * slot), self.seq[slot], 0))

    def wait(self, slot: int, stream: torch.cuda.Stream | None = None) -> None:
        s = driver.CUstream((stream or torch.cuda.current_stream()).cuda_stream)
        _ok(driver.cuStreamWaitValue32(s, driver.CUdeviceptr(self._flags_ptr + 8 * slot + 4), self.seq[slot],
                                       self._wait_flags))

    def error(self) -> int:
        return self.lib.sr_error(self.handle) if self.handle else 1

    def close(self) -> None:
        if self.handle:
            self.lib.sr_close(self.handle)
            self.handle = None


def factory(**options):
    """For LTX-2's AttentionManager.pipeline_transport_factory: (group, nbytes, slots) -> RdmaTransport."""
    return lambda group, nbytes, slots: RdmaTransport(group, nbytes, slots, **options)
