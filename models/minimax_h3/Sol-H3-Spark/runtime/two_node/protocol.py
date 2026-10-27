"""Typed control packets for a single owned remote Stage2 session."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, ClassVar, Literal

from pydantic import BaseModel, ConfigDict, Field, JsonValue

from .wire import WireError, receive_bytes, send_bytes

if TYPE_CHECKING:
    import socket


class Command(BaseModel):
    """Only the listed session methods can cross the network boundary."""

    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True)
    op: Literal["open", "prepare", "run", "release_idle_cache", "close"]
    token: str
    kwargs: dict[str, JsonValue] = Field(default_factory=dict)


class Artifact(BaseModel):
    """A completed MP4 returned from the remote worker."""

    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True)
    path: str
    sha256: str


class Response(BaseModel):
    """Parsed worker receipt with an optional binary artifact."""

    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True)
    status: Literal["READY", "PASS"]
    result: dict[str, JsonValue] = Field(default_factory=dict)
    artifact: Artifact | None = None


def send_command(sock: socket.socket, command: Command) -> None:
    """Serialize only validated command data."""
    send_bytes(sock, command.model_dump_json().encode())


def receive_response(sock: socket.socket) -> Response:
    """Parse untrusted JSON once at the control boundary."""
    return Response.model_validate_json(receive_bytes(sock))


def owned_path(value: str, root: Path) -> Path:
    """Reject paths outside the task result tree, including symlink escapes."""
    path = Path(value).resolve()
    try:
        _ = path.relative_to(root.resolve())
    except ValueError as error:
        msg = "path is outside the task result tree"
        raise WireError(msg) from error
    return path
