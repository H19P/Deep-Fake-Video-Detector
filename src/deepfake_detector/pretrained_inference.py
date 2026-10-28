"""Local video inference using the pinned pretrained adapter."""
from __future__ import annotations

from pathlib import Path
from torchvision import transforms

from .aggregation import aggregate_scores
from .config import InferenceConfig
from .face_detector import FaceDetector
from .pretrained_adapter import INPUT_SIZE, Oo01PretrainedAdapter
from .video_processor import extract_face_samples


class PretrainedInferenceService:
    """Run local face sampling and inference using the released pretrained model."""
    def __init__(self, adapter: Oo01PretrainedAdapter, config: InferenceConfig | None = None,
                 face_detector=None):
        self.adapter = adapter
        self.config = config or InferenceConfig()
        self.face_detector = face_detector or FaceDetector()
        self.transform = transforms.Compose([
            transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize([.485, .456, .406], [.229, .224, .225]),
        ])

    def analyze(self, path: str | Path) -> dict:
        """Return timestamps, real model scores, and configured video aggregation."""
        samples, metadata = extract_face_samples(
            path, self.face_detector, self.config, padding=.30, multiple_faces=False
        )
        tensors = [self.transform(sample.image) for sample in samples]
        scores, native_video_score = self.adapter.score_images(tensors)
        frames = [{
            "timestamp_seconds": sample.timestamp_seconds,
            "frame_index": sample.frame_index,
            "face_index": sample.face_index,
            "bbox": list(sample.bbox),
            "fake_score": float(score),
            "face_image": sample.image,
        } for sample, score in zip(samples, scores)]
        aggregation = aggregate_scores(
            scores, self.config.aggregation, self.config.min_predictions,
            self.config.low_threshold, self.config.high_threshold,
        )
        aggregation["native_pooled_video_score"] = native_video_score if scores else None
        return {
            "metadata": metadata,
            "frames": frames,
            "aggregation": aggregation,
            "model": self.adapter.model_info,
            "device": str(self.adapter.device),
            "sampled_frames": metadata.sampled_frames,
            "detected_faces": metadata.detected_faces,
            "valid_predictions": len(frames),
        }
