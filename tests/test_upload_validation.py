import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
import pytest
from deepfake_detector.config import validate_video_path

def test_missing_video():
    with pytest.raises(FileNotFoundError): validate_video_path("missing.mp4")
def test_extension_and_size(tmp_path):
    p=tmp_path/"bad.txt";p.write_text("x")
    with pytest.raises(ValueError): validate_video_path(p)
    q=tmp_path/"big.mp4";q.write_bytes(b"12345")
    with pytest.raises(ValueError): validate_video_path(q,max_size_mb=0)
