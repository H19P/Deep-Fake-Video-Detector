"""Generated silent video fixture; verifies no-face behavior without model weights."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
import cv2
import numpy as np
from deepfake_detector.video_processor import extract_face_samples
from deepfake_detector.config import InferenceConfig

class NoFaces:
    def detect(self, frame): return []

def test_generated_video_has_no_face_samples(tmp_path):
    path=tmp_path/"fixture.avi"
    writer=cv2.VideoWriter(str(path),cv2.VideoWriter_fourcc(*"MJPG"),5,(64,64))
    if not writer.isOpened(): return
    for _ in range(10): writer.write(np.zeros((64,64,3),dtype=np.uint8))
    writer.release()
    samples,meta=extract_face_samples(path,NoFaces(),InferenceConfig(max_frames=5))
    assert samples==[] and meta.sampled_frames==3 and meta.detected_faces==0
