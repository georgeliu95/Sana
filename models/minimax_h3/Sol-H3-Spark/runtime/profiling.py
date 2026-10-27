"""NVTX scopes for CPU scheduling, real GPU work, and cross-node request identity."""
from __future__ import annotations
from collections.abc import Callable, Iterator
from contextlib import contextmanager
import os
import nvtx

@contextmanager
def scope(label: str, *, capture: bool = False) -> Iterator[None]:
    """Balance ranges through failures; capture labels are emitted only when enabled."""
    if capture:
        with nvtx.annotate("SPARK_CAPTURE", domain="spark_h3"):
            with nvtx.annotate(label, domain="spark_h3"):
                yield
    else:
        with nvtx.annotate(label, domain="spark_h3"):
            yield

def observed[**P, R](label: str, function: Callable[P, R]) -> Callable[P, R]:
    """Observe an eager or opaque call without changing its arguments or result."""
    def wrapped(*args: P.args, **kwargs: P.kwargs) -> R:
        with scope(label):
            return function(*args, **kwargs)
    return wrapped

def selected(request_id: str) -> bool:
    """One explicit request identifier selects capture on both hosts."""
    return request_id == os.environ.get("SPARK_NSYS_CASE")
