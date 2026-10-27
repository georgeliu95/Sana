#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = ["pydantic>=2"]
# ///
# ─── How to run ───
# Run in the pinned controller container with the two-node launcher.
"""Select remote Stage2 while preserving the measured matched209 Spark recipe."""
from __future__ import annotations
from functools import partial
from pathlib import Path
import infer
from runtime.pipeline import Pipeline, Worker
from runtime.two_node.proxy import RemoteWorker

def factory(name: str, root: Path, paths, config):
    """Keep Qwen and H3 on the controller; delegate Stage2 to its peer."""
    if name == "stage2":
        return RemoteWorker(root)
    return Worker(name, root, paths, config)

def main() -> None:
    """Invoke the original CLI and measured pipeline through its worker seam."""
    infer.Pipeline = partial(Pipeline, worker_factory=factory)
    infer.main()

if __name__ == "__main__":
    main()
