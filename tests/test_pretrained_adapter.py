import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pytest
import torch
from torch import nn
from deepfake_detector.pretrained_adapter import (
    MODEL_REVISION, Oo01PretrainedAdapter, PretrainedModelError,
)


class TinyDetector(nn.Module):
    def __init__(self):
        super().__init__()
        self.head = nn.Sequential(nn.LayerNorm(1280), nn.Dropout(.4), nn.Linear(1280, 256),
                                  nn.GELU(), nn.Dropout(.2), nn.Linear(256, 1))
    def extract_features(self, x):
        return torch.ones((len(x), 1280), device=x.device)
    def frame_logits(self, features): return self.head(features).squeeze(-1)


def test_safe_checkpoint_loading_and_score_mapping(tmp_path, monkeypatch):
    checkpoint = tmp_path / "weights.pth"
    model = TinyDetector()
    model.head[-1].weight.data.zero_()
    model.head[-1].bias.data.fill_(2.0)
    torch.save({"model_state_dict": model.state_dict()}, checkpoint)
    monkeypatch.setattr("deepfake_detector.pretrained_adapter._download_checkpoint", lambda *_a, **_k: checkpoint)
    monkeypatch.setattr("deepfake_detector.pretrained_adapter.build_oo01_architecture", TinyDetector)
    adapter = Oo01PretrainedAdapter.from_pretrained(device="cpu")
    scores, pooled = adapter.score_images([torch.zeros(3, 224, 224), torch.ones(3, 224, 224)])
    assert len(scores) == 2
    assert scores == pytest.approx([1.0 - torch.sigmoid(torch.tensor(2.0)).item()] * 2)
    assert pooled == pytest.approx(scores[0])
    assert adapter.model_info["version"] == MODEL_REVISION[:7]


def test_missing_weights_fail_clearly(monkeypatch):
    def missing(*_args, **_kwargs):
        raise PretrainedModelError("offline and no cached weights")
    monkeypatch.setattr("deepfake_detector.pretrained_adapter._download_checkpoint", missing)
    with pytest.raises(PretrainedModelError, match="offline"):
        Oo01PretrainedAdapter.from_pretrained(device="cpu")


def test_huggingface_download_errors_are_reported(monkeypatch, tmp_path):
    import huggingface_hub
    monkeypatch.setattr(
        huggingface_hub, "hf_hub_download",
        lambda **_kwargs: (_ for _ in ()).throw(RuntimeError("hub request failed")),
    )
    from deepfake_detector.pretrained_adapter import _download_checkpoint
    with pytest.raises(PretrainedModelError, match="could not be downloaded"):
        _download_checkpoint(tmp_path)
