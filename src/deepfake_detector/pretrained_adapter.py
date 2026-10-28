"""Safe adapter for the pinned oo01 EfficientNet-B0 deepfake video model.

The upstream Python implementation was inspected and re-created here. The
application downloads only the checkpoint from a pinned Hub commit and never
executes remote Python code.
"""
from __future__ import annotations

from pathlib import Path
from typing import Sequence

import torch
from torch import nn

MODEL_REPO = "oo01/deepfake-video-classifier"
MODEL_REVISION = "3b237c2c820fab51cca13b80d726e4be5876de23"
CHECKPOINT_NAME = "ffpp_efficientnet_best.pth"
MODEL_NAME = "oo01 EfficientNet-B0 Deepfake Video Classifier"
MODEL_VERSION = MODEL_REVISION[:7]
INPUT_SIZE = 224


class PretrainedModelError(RuntimeError):
    """Raised when the documented pretrained checkpoint is unavailable/invalid."""


def build_oo01_architecture() -> nn.Module:
    """Recreate the published timm EfficientNet-B0 temporal pooling model."""
    try:
        import timm
    except ImportError as exc:
        raise PretrainedModelError("Missing dependency 'timm'; install requirements.txt") from exc

    class EfficientNetB0VideoDetector(nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = timm.create_model(
                "efficientnet_b0", pretrained=False, num_classes=0, global_pool="avg"
            )
            self.head = nn.Sequential(
                nn.LayerNorm(1280),
                nn.Dropout(0.4),
                nn.Linear(1280, 256),
                nn.GELU(),
                nn.Dropout(0.2),
                nn.Linear(256, 1),
            )

        def extract_features(self, x: torch.Tensor) -> torch.Tensor:
            return self.backbone(x)

        def frame_logits(self, features: torch.Tensor) -> torch.Tensor:
            return self.head(features).squeeze(-1)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            # Upstream shape is (B, T, C, H, W); temporal mean is in feature space.
            if x.ndim != 5:
                raise ValueError("Expected video tensor with shape (B,T,C,H,W)")
            b, t, c, h, w = x.shape
            features = self.extract_features(x.reshape(b * t, c, h, w))
            features = features.reshape(b, t, -1).mean(dim=1)
            return self.head(features).squeeze(-1)

    return EfficientNetB0VideoDetector()


def _download_checkpoint(cache_dir: str | Path | None = None) -> Path:
    """Fetch only the known checkpoint from a pinned model revision."""
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise PretrainedModelError(
            "Missing dependency 'huggingface-hub'; install requirements.txt"
        ) from exc

    local_cache = Path(cache_dir or Path(__file__).resolve().parents[2] / "models" / "pretrained-cache")
    local_cache.mkdir(parents=True, exist_ok=True)
    try:
        return Path(hf_hub_download(
            repo_id=MODEL_REPO,
            filename=CHECKPOINT_NAME,
            revision=MODEL_REVISION,
            cache_dir=str(local_cache),
        ))
    except Exception as exc:
        raise PretrainedModelError(
            "The pretrained checkpoint could not be downloaded or found in the local cache. "
            "Connect to Hugging Face for the first model load and confirm that the pinned "
            "repository revision is reachable, then retry."
        ) from exc


class Oo01PretrainedAdapter:
    """Load the MIT-tagged, Celeb-DF-v2-trained checkpoint and score face crops."""
    def __init__(self, model: nn.Module, device: str | torch.device, checkpoint_path: Path):
        self.model = model.to(device).eval()
        self.device = torch.device(device)
        self.checkpoint_path = checkpoint_path
        self.model_info = {
            "name": MODEL_NAME,
            "architecture": "timm EfficientNet-B0; feature mean temporal pooling; LayerNorm/MLP head",
            "version": MODEL_VERSION,
            "source": f"https://huggingface.co/{MODEL_REPO}/tree/{MODEL_REVISION}",
            "license": "MIT (as declared by model repository)",
            "training_dataset": "Celeb-DF v2 (model card declaration)",
            "class_mapping": {"real": "sigmoid(logit)", "fake": "1 - sigmoid(logit)"},
            "input": "224x224 RGB face crops, ImageNet normalization, up to 16 sampled frames",
        }

    @classmethod
    def from_pretrained(cls, device: str | torch.device | None = None,
                        cache_dir: str | Path | None = None) -> "Oo01PretrainedAdapter":
        """Download/cache pinned weights and load with PyTorch's restricted unpickler."""
        target = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        checkpoint_path = _download_checkpoint(cache_dir)
        try:
            import codecs
            import numpy as np
            # The pinned upstream file's Hugging Face scan lists these serialization
            # globals. They are restricted to NumPy numeric metadata and byte encoding;
            # no model code or arbitrary classes are allowlisted.
            safe_globals = [codecs.encode, np._core.multiarray.scalar, np.dtype]
            for dtype_name in ("Float32DType", "Float64DType", "Int32DType", "Int64DType"):
                dtype_type = getattr(np.dtypes, dtype_name, None)
                if dtype_type is not None:
                    safe_globals.append(dtype_type)
            with torch.serialization.safe_globals(safe_globals):
                checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        except Exception as exc:
            raise PretrainedModelError(
                "The published checkpoint could not be loaded safely with weights_only=True; "
                "the application will not fall back to unrestricted pickle loading."
            ) from exc

        if not isinstance(checkpoint, dict):
            raise PretrainedModelError("Checkpoint has an unsupported structure")
        state = checkpoint.get("model_state_dict", checkpoint)
        if not isinstance(state, dict) or not state:
            raise PretrainedModelError("Checkpoint has no model state dictionary")
        if not all(isinstance(k, str) and isinstance(v, torch.Tensor) for k, v in state.items()):
            raise PretrainedModelError("Checkpoint contains unexpected non-tensor state entries")

        model = build_oo01_architecture()
        try:
            model.load_state_dict(state, strict=True)
        except Exception as exc:
            raise PretrainedModelError(
                "Checkpoint tensors do not match the documented EfficientNet-B0 architecture"
            ) from exc
        return cls(model, target, checkpoint_path)

    @torch.inference_mode()
    def score_images(self, images: Sequence[torch.Tensor]) -> tuple[list[float], float]:
        """Return per-frame fake scores and native pooled-video fake score.

        Per-frame scores use the checkpoint's trained head on each frame feature.
        The native video score applies the documented temporal feature mean before
        the head. Both are softmax-like sigmoid outputs, not calibrated probabilities.
        """
        if not images:
            return [], float("nan")
        features = []
        for image in images:
            if image.ndim != 3 or tuple(image.shape) != (3, INPUT_SIZE, INPUT_SIZE):
                raise ValueError("Each input must be a normalized 3x224x224 RGB face tensor")
            features.append(self.model.extract_features(image.unsqueeze(0).to(self.device)))
        stacked = torch.cat(features, dim=0)
        frame_real = torch.sigmoid(self.model.frame_logits(stacked))
        pooled_real = torch.sigmoid(self.model.head(stacked.mean(dim=0, keepdim=True))).item()
        return (1.0 - frame_real).detach().cpu().tolist(), float(1.0 - pooled_real)
