import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
import pytest
from deepfake_detector.inference import InferenceService

def test_missing_checkpoint():
    with pytest.raises(FileNotFoundError): InferenceService("missing.pt")
