import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pytest
from deepfake_detector.media_storage import temporary_video_file


def test_temporary_upload_is_removed_after_success():
    path = None
    with temporary_video_file("clip.mp4", b"video bytes") as temp_path:
        path = temp_path
        assert path.is_file() and path.read_bytes() == b"video bytes"
    assert path is not None and not path.exists()


def test_temporary_upload_is_removed_after_failure():
    path = None
    with pytest.raises(RuntimeError):
        with temporary_video_file("clip.mov", b"video bytes") as temp_path:
            path = temp_path
            raise RuntimeError("simulated inference failure")
    assert path is not None and not path.exists()


def test_temporary_upload_rejects_invalid_extension():
    with pytest.raises(ValueError, match="Unsupported"):
        with temporary_video_file("clip.exe", b"bad"):
            pass
