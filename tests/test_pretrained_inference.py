import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import cv2
import numpy as np
import torch
from deepfake_detector.config import InferenceConfig
from deepfake_detector.pretrained_inference import PretrainedInferenceService


class OneFace:
    def detect(self, _frame): return [(10, 10, 40, 40, 1.0)]


class FakeAdapter:
    model_info = {"name": "test fake adapter", "architecture": "test", "version": "test"}
    device = torch.device("cpu")
    def score_images(self, images):
        assert all(tuple(image.shape) == (3, 224, 224) for image in images)
        return [.8] * len(images), .81


class NoFaces:
    def detect(self, _frame): return []


def test_video_inference_structure_with_fake_adapter(tmp_path):
    video = tmp_path / "fixture.avi"
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"MJPG"), 5, (64, 64))
    if not writer.isOpened():
        return
    for _ in range(15): writer.write(np.zeros((64, 64, 3), np.uint8))
    writer.release()
    service = PretrainedInferenceService(FakeAdapter(), InferenceConfig(max_frames=5), OneFace())
    result = service.analyze(video)
    assert result["valid_predictions"] >= 3
    assert result["aggregation"]["score"] == .8
    assert result["aggregation"]["native_pooled_video_score"] == .81
    assert "timestamp_seconds" in result["frames"][0]


def test_no_faces_is_inconclusive(tmp_path):
    video = tmp_path / "blank.avi"
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"MJPG"), 5, (64, 64))
    if not writer.isOpened():
        return
    for _ in range(15): writer.write(np.zeros((64, 64, 3), np.uint8))
    writer.release()
    result = PretrainedInferenceService(FakeAdapter(), InferenceConfig(max_frames=5), NoFaces()).analyze(video)
    assert result["aggregation"]["status"] == "inconclusive"
    assert result["aggregation"]["score"] is None
    assert result["valid_predictions"] == 0
