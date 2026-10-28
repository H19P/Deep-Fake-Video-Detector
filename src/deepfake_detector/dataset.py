"""Manifest-backed face dataset preserving video identities."""
import csv
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

class FaceManifestDataset(Dataset):
    """Read image rows from CSV with image_path,label,video_id,split columns."""
    def __init__(self, manifest, split, training=False, image_size=224):
        self.root=Path(manifest).resolve().parent; self.rows=[]
        with Path(manifest).open(newline="",encoding="utf-8-sig") as f:
            for row in csv.DictReader(f):
                if row.get("split") != split: continue
                label=row.get("label","").lower()
                if label not in {"real","fake"}: raise ValueError(f"Invalid label in manifest: {label}")
                p=(self.root/row["image_path"]).resolve()
                if not p.is_file(): raise FileNotFoundError(f"Missing image referenced by manifest: {p}")
                row["_path"]=p; row["_label"]=0 if label=="real" else 1; self.rows.append(row)
        aug=[transforms.Resize((image_size,image_size))]
        if training: aug += [transforms.RandomHorizontalFlip(),transforms.ColorJitter(brightness=.15,contrast=.15,saturation=.1)]
        aug += [transforms.ToTensor(),transforms.Normalize([.485,.456,.406],[.229,.224,.225])]
        self.transform=transforms.Compose(aug)
    def __len__(self): return len(self.rows)
    def __getitem__(self,index):
        row=self.rows[index]
        with Image.open(row["_path"]) as im: image=self.transform(im.convert("RGB"))
        return image,row["_label"],row.get("video_id","")
