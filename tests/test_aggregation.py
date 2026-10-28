import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
import pytest
from deepfake_detector.aggregation import aggregate_scores

def test_median_and_status():
    r=aggregate_scores([.8,.9,.7],min_predictions=3)
    assert r["score"]==pytest.approx(.8) and r["status"]=="high_fake_score"
def test_insufficient_is_inconclusive(): assert aggregate_scores([.99]) ["status"]=="inconclusive"
def test_invalid_score():
    with pytest.raises(ValueError): aggregate_scores([float("nan")])
