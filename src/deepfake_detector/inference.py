"""End-to-end inference service with explicit no-checkpoint failure."""
from pathlib import Path
import hashlib
import torch
from torchvision import transforms
from .aggregation import aggregate_scores
from .config import InferenceConfig
from .face_detector import FaceDetector
from .model import load_checkpoint
from .video_processor import extract_face_samples

class InferenceService:
    def __init__(self, checkpoint: str | Path, config: InferenceConfig | None = None, device: str | None = None, face_detector=None):
        self.config=config or InferenceConfig()
        self.device=torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model,self.model_metadata=load_checkpoint(checkpoint,self.device)
        self.face_detector=face_detector or FaceDetector()
        self.transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize([.485,.456,.406],[.229,.224,.225])])

    @torch.inference_mode()
    def analyze(self,path: str | Path) -> dict:
        samples,meta=extract_face_samples(path,self.face_detector,self.config)
        rows=[]
        for sample in samples:
            x=self.transform(sample.image).unsqueeze(0).to(self.device)
            score=float(torch.softmax(self.model(x),dim=1)[0,1].item())
            rows.append({"timestamp_seconds":sample.timestamp_seconds,"frame_index":sample.frame_index,
                         "face_index":sample.face_index,"bbox":list(sample.bbox),"fake_score":score,"face_image":sample.image})
        agg=aggregate_scores([r["fake_score"] for r in rows],self.config.aggregation,self.config.min_predictions,
                             self.config.low_threshold,self.config.high_threshold)
        return {"metadata":meta,"frames":rows,"aggregation":agg,"model":self.model_metadata,
                "sampled_frames":meta.sampled_frames,"detected_faces":meta.detected_faces,"valid_predictions":len(rows)}
