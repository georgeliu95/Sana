"""Exercise actual framed handoff between distinct role roots in one filesystem."""
from hashlib import sha256
from pathlib import Path
import socket

import pytest

from runtime.two_node.paths import HandoffPaths
from runtime.two_node.wire import receive_file, send_bytes
from runtime.two_node.protocol import owned_path
from runtime.two_node.wire import WireError


def test_handoff_preserves_producer_file_in_shared_container(tmp_path: Path) -> None:
    # Given: H3 already owns its completed capture in the shared filesystem.
    local, remote = tmp_path / "h3", tmp_path / "ltx"
    capture = local / "run02/stage1/capture.json"
    capture.parent.mkdir(parents=True)
    capture.write_bytes(b"verified-capture")
    mapping = HandoffPaths(local, remote)
    original_inode = capture.stat().st_ino
    target = Path(mapping.to_remote(str(capture)))
    # When: the existing wire receiver writes to the translated LTX root.
    left, right = socket.socketpair()
    with left, right:
        send_bytes(left, capture.read_bytes())
        receive_file(right, target, sha256(capture.read_bytes()).hexdigest())
    # Then: the source is intact and the independent LTX copy is valid.
    assert capture.stat().st_ino == original_inode
    assert target == remote / "run02/stage1/capture.json"
    assert target.read_bytes() == capture.read_bytes()


def test_artifact_returns_to_controller_root(tmp_path: Path) -> None:
    # Given: LTX has completed its MP4 in the shared container.
    mapping = HandoffPaths(tmp_path / "h3", tmp_path / "ltx")
    remote_artifact = tmp_path / "ltx/run02/stage2/clip.mp4"
    # When: the controller resolves the return destination.
    local_artifact = mapping.to_local(str(remote_artifact))
    # Then: it will publish under the controller's own request tree.
    assert local_artifact == str(tmp_path / "h3/run02/stage2/clip.mp4")


def test_command_maps_paths_and_preserves_other_fields(tmp_path: Path) -> None:
    # Given: a command carries both owned paths and non-path settings.
    mapping = HandoffPaths(tmp_path / "h3", tmp_path / "ltx")
    # When: preparing the remote command.
    result = mapping.command_kwargs({"capture_dir":str(tmp_path / "h3/run02/stage1"),
                                    "output_root":str(tmp_path / "h3/run02/stage2"),
                                    "request_id":"run02","early_handoff":True})
    # Then: only the path-bearing fields change.
    assert result == {"capture_dir":str(tmp_path / "ltx/run02/stage1"),
                      "output_root":str(tmp_path / "ltx/run02/stage2"),
                      "request_id":"run02","early_handoff":True}


def test_mapping_rejects_escape(tmp_path: Path) -> None:
    # Given: a request attempts to leave its result tree.
    mapping = HandoffPaths(tmp_path / "h3", tmp_path / "ltx")
    # When/Then: translating fails at the existing ownership boundary.
    with pytest.raises(WireError, match="outside"):
        mapping.to_remote(str(tmp_path / "elsewhere"))


def test_default_mapping_preserves_two_container_contract(tmp_path: Path) -> None:
    # Given: existing dual-container deployments use identical logical roots.
    mapping = HandoffPaths(tmp_path, tmp_path)
    expected = str(owned_path(str(tmp_path / "run02"), tmp_path))
    # When/Then: both directions preserve that path.
    assert mapping.to_remote(expected) == expected
    assert mapping.to_local(expected) == expected
