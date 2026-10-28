"""Deterministic, bounded-memory video sampling and face crop preparation."""
from dataclasses import dataclass
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
from .config import InferenceConfig, validate_video_path

@dataclass
class FaceSample:
    timestamp_seconds: float
    frame_index: int
    face_index: int
    bbox: tuple[int, int, int, int]
    image: Image.Image

@dataclass
class VideoMetadata:
    fps: float
    frame_count: int
    duration_seconds: float
    width: int
    height: int
    sampled_frames: int = 0
    detected_faces: int = 0

def inspect_video(path: str | Path, config: InferenceConfig | None = None) -> VideoMetadata:
    """Open and validate decoder metadata without reading the full video."""
    config = config or InferenceConfig()
    p = validate_video_path(path, config.max_file_size_mb)
    cap = cv2.VideoCapture(str(p))
    try:
        if not cap.isOpened():
            raise ValueError("Video decoder could not open the file")
        fps = float(cap.get(cv2.CAP_PROP_FPS)); count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if fps <= 0 or count <= 0 or width <= 0 or height <= 0:
            raise ValueError("Video has invalid decoder metadata")
        duration = count / fps
        if duration > config.max_duration_seconds:
            raise ValueError("Video exceeds the configured duration limit")
        if width > config.max_width or height > config.max_height:
            raise ValueError("Video resolution exceeds the configured limit")
        if not cap.read()[0]:
            raise ValueError("Video contains no readable frames")
        return VideoMetadata(fps, count, duration, width, height)
    finally:
        cap.release()

def extract_face_samples(path: str | Path, face_detector, config: InferenceConfig | None = None,
                         padding: float = .15, multiple_faces: bool = False):
    """Sample frames at fixed intervals and return padded RGB face crops."""
    config = config or InferenceConfig()
    meta = inspect_video(path, config)
    cap = cv2.VideoCapture(str(path)); samples=[]; sampled=0; detected=0
    requested = max(1, int(meta.duration_seconds / config.sample_interval_seconds) + 1)
    target_count = min(config.max_frames, requested)
    target_indices = np.unique(np.linspace(0, max(0, meta.frame_count - 1), target_count).round().astype(int))
    try:
        for index in target_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(index))
            ok, frame = cap.read()
            if not ok:
                continue
            sampled += 1
            faces = sorted(face_detector.detect(frame), key=lambda f: f[2]*f[3], reverse=True)
            detected += len(faces)
            if not multiple_faces: faces = faces[:1]
            h,w=frame.shape[:2]
            for fi,(x,y,bw,bh,_) in enumerate(faces):
                px=int(bw*padding); py=int(bh*padding)
                crop=frame[max(0,y-py):min(h,y+bh+py), max(0,x-px):min(w,x+bw+px)]
                if crop.size:
                    rgb=cv2.cvtColor(cv2.resize(crop,(config.image_size,config.image_size)),cv2.COLOR_BGR2RGB)
                    samples.append(FaceSample(int(index)/meta.fps,int(index),fi,(x,y,bw,bh),Image.fromarray(rgb)))
        meta.sampled_frames=sampled; meta.detected_faces=detected
        return samples, meta
    finally:
        cap.release()
