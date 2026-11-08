"""Translate owned role paths when both distributed stages share one container."""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import os
from pathlib import Path
from typing import Final

from pydantic import JsonValue, TypeAdapter

from .protocol import owned_path

PATH_FIELDS: Final = ("work_dir", "capture_dir", "output_root")
TEXT: Final = TypeAdapter(str)


@dataclass(frozen=True, slots=True)
class HandoffPaths:
    """Keep producer and consumer result trees distinct in one filesystem."""

    local: Path
    remote: Path

    @classmethod
    def from_environment(cls) -> HandoffPaths:
        """Preserve identical logical roots unless a remote role root is explicit."""
        local = Path(os.environ["SPARK_PP_RESULT_ROOT"]).resolve()
        remote = Path(os.environ.get("SPARK_PP_REMOTE_ROOT", str(local))).resolve()
        return cls(local, remote)

    def to_remote(self, value: str) -> str:
        """Translate a controller-owned path into the LTX result tree."""
        relative = owned_path(value, self.local).relative_to(self.local.resolve())
        return str(owned_path(str(self.remote / relative), self.remote))

    def to_local(self, value: str) -> str:
        """Translate a verified LTX artifact back to the controller result tree."""
        relative = owned_path(value, self.remote).relative_to(self.remote.resolve())
        return str(owned_path(str(self.local / relative), self.local))

    def command_kwargs(self, values: Mapping[str, JsonValue]) -> Mapping[str, JsonValue]:
        """Map only protocol path fields, preserving request and recipe values."""
        result = dict(values)
        for field in PATH_FIELDS:
            if field in result:
                result[field] = self.to_remote(TEXT.validate_python(result[field], strict=True))
        return result
