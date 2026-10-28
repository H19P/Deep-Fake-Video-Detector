"""Modular EfficientNet-B0 binary classifier and safe checkpoint helpers."""
from pathlib import Path
import torch
from torch import nn
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

def build_model(architecture: str = "efficientnet_b0", pretrained: bool = False) -> nn.Module:
    """Create supported classifier architecture with a two-logit head."""
    if architecture != "efficientnet_b0":
        raise ValueError(f"Unsupported architecture: {architecture}")
    model = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT if pretrained else None)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, 2)
    return model

def load_checkpoint(path: str | Path, device: str | torch.device = "cpu"):
    """Load a trusted state-dict checkpoint; never accept user-uploaded checkpoint files."""
    p=Path(path)
    if not p.is_file(): raise FileNotFoundError(f"Model checkpoint not found: {p}")
    try: ckpt=torch.load(p,map_location=device,weights_only=True)
    except Exception as exc: raise ValueError("Checkpoint is invalid or unsafe") from exc
    if not isinstance(ckpt,dict) or "state_dict" not in ckpt or "metadata" not in ckpt:
        raise ValueError("Checkpoint is missing required metadata")
    md=ckpt["metadata"]
    if md.get("class_to_index") != {"real":0,"fake":1} or md.get("architecture") != "efficientnet_b0":
        raise ValueError("Checkpoint class mapping or architecture is incompatible")
    model=build_model(md["architecture"])
    try: model.load_state_dict(ckpt["state_dict"],strict=True)
    except Exception as exc: raise ValueError("Checkpoint weights are incompatible") from exc
    return model.to(device).eval(), md
