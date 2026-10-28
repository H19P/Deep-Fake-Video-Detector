import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pytest
from deepfake_detector.interpretation import explain_fake_score


def test_high_band_is_plain_and_cautious():
    title, text = explain_fake_score(.82)
    assert title == "High fake-class score"
    assert "not a probability" in text and "does not prove" in text


def test_low_and_uncertain_bands():
    assert explain_fake_score(.2)[0] == "Low fake-class score"
    assert explain_fake_score(.5)[0] == "Inconclusive"
    assert explain_fake_score(None)[0] == "Inconclusive"


def test_out_of_range_score_rejected():
    with pytest.raises(ValueError): explain_fake_score(1.1)
