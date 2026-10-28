"""Split source videos before face extraction and emit a grouped image manifest."""
import argparse,csv,hashlib,random,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
import cv2
from deepfake_detector.config import InferenceConfig
from deepfake_detector.face_detector import FaceDetector
from deepfake_detector.video_processor import extract_face_samples

def prepare(input_manifest,output_manifest,output_dir,seed=42):
    rows=[]
    with Path(input_manifest).open(newline="",encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            p=Path(r["video_path"]).resolve(); label=r["label"].lower()
            if label not in {"real","fake"}: raise ValueError(f"Invalid class label: {label}")
            if not p.is_file(): raise FileNotFoundError(f"Video does not exist: {p}")
            source=r.get("source_id") or r.get("pair_id") or hashlib.sha256(str(p).encode()).hexdigest()
            rows.append({**r,"video_path":str(p),"label":label,"group":source})
    if not rows: raise ValueError("Input manifest is empty")
    groups={}
    for row in rows: groups.setdefault(row["group"],[]).append(row)
    keys=list(groups); random.Random(seed).shuffle(keys); n=len(keys)
    assignments={k:("train" if i<int(.70*n) else "val" if i<int(.85*n) else "test") for i,k in enumerate(keys)}
    out=Path(output_dir); detector=FaceDetector(); cfg=InferenceConfig(); output_rows=[]; failures=[]
    for row in rows:
        split=assignments[row["group"]]
        try:
            samples,_=extract_face_samples(row["video_path"],detector,cfg)
            if not samples: failures.append({"video":row["video_path"],"reason":"no faces detected"})
            for i,s in enumerate(samples):
                target=out/split/row["label"]/f"{hashlib.sha256(row['video_path'].encode()).hexdigest()[:16]}_{s.frame_index}_{i}.jpg"
                target.parent.mkdir(parents=True,exist_ok=True); s.image.save(target,quality=95)
                output_rows.append({"image_path":str(target.resolve()),"label":row["label"],"video_id":row.get("video_id") or row["group"],"source_id":row["group"],"split":split,"timestamp_seconds":s.timestamp_seconds,"provenance":row.get("provenance","")})
        except Exception as exc: failures.append({"video":row["video_path"],"reason":str(exc)})
    Path(output_manifest).parent.mkdir(parents=True,exist_ok=True)
    with Path(output_manifest).open("w",newline="",encoding="utf8") as f:
        w=csv.DictWriter(f,fieldnames=["image_path","label","video_id","source_id","split","timestamp_seconds","provenance"]);w.writeheader();w.writerows(output_rows)
    print(f"Processed {len(rows)} videos, wrote {len(output_rows)} face crops; failures={len(failures)}")
    for e in failures: print(f"FAILED {e['video']}: {e['reason']}")
def main():
    p=argparse.ArgumentParser();p.add_argument("--input-manifest",required=True);p.add_argument("--output-manifest",default="data/manifests/faces.csv");p.add_argument("--output-dir",default="data/processed");p.add_argument("--seed",type=int,default=42);a=p.parse_args();prepare(a.input_manifest,a.output_manifest,a.output_dir,a.seed)
if __name__=="__main__":main()
