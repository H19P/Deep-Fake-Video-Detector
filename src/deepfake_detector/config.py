"""Central configuration and safe upload validation."""
from dataclasses import dataclass
from pathlib import Path

SUPPORTED_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}
CLASS_TO_INDEX = {"real": 0, "fake": 1}

@dataclass(frozen=True)
class InferenceConfig:
    sample_interval_seconds: float = 1.0
    max_frames: int = 16
    max_duration_seconds: float = 600.0
    max_file_size_mb: int = 500
    min_predictions: int = 3
    aggregation: str = "median"
    low_threshold: float = 0.30
    high_threshold: float = 0.70
    image_size: int = 224
    max_width: int = 3840
    max_height: int = 2160

    def __post_init__(self):
        if self.sample_interval_seconds <= 0 or not 1 <= self.max_frames <= 120:
            raise ValueError("Sampling interval and frame limit must be positive")
        if self.max_duration_seconds <= 0 or self.max_file_size_mb <= 0:
            raise ValueError("File size and duration limits must be positive")
        if not 0 <= self.low_threshold < self.high_threshold <= 1:
            raise ValueError("Thresholds must satisfy 0 <= low < high <= 1")
        if self.aggregation not in {"median", "mean"}:
            raise ValueError("aggregation must be median or mean")

def validate_video_path(path: str | Path, max_size_mb: int = 500) -> Path:
    """Validate basic path, extension, and file-size constraints."""
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError("Video file does not exist")
    if p.suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise ValueError("Unsupported video format")
    if p.stat().st_size == 0:
        raise ValueError("Video file is empty")
    if p.stat().st_size > max_size_mb * 1024 * 1024:
        raise ValueError(f"Video exceeds the {max_size_mb} MB upload limit")
    return p
