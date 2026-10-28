import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
import pytest
from deepfake_detector.video_processor import inspect_video

def test_missing_video():
    with pytest.raises(FileNotFoundError): inspect_video("missing.mp4")
def test_corrupt_video(tmp_path):
    p=tmp_path/"broken.mp4";p.write_bytes(b"not a video")
    with pytest.raises(ValueError): inspect_video(p)
