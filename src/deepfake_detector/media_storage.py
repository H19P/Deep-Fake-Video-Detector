"""Random temporary upload storage with unconditional cleanup."""
from contextlib import contextmanager
from pathlib import Path
import tempfile
from typing import Iterator

from .config import SUPPORTED_EXTENSIONS


@contextmanager
def temporary_video_file(filename: str, content, max_size_mb: int = 500) -> Iterator[Path]:
    """Write upload bytes to a randomized temporary path and always remove it."""
    suffix = Path(filename).suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        raise ValueError("Unsupported video format")
    size = len(content)
    if size <= 0:
        raise ValueError("Video file is empty")
    if size > max_size_mb * 1024 * 1024:
        raise ValueError(f"Video exceeds the {max_size_mb} MB upload limit")
    handle = tempfile.NamedTemporaryFile(prefix="deepfake-upload-", suffix=suffix, delete=False)
    path = Path(handle.name)
    try:
        with handle:
            handle.write(content)
            handle.flush()
        # Close before yielding: Windows will not allow VideoCapture to open or
        # remove a NamedTemporaryFile while the creating handle remains open.
        yield path
    finally:
        path.unlink(missing_ok=True)
