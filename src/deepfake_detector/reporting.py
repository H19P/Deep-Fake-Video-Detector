"""Forensic-style JSON and CSV reporting without embedding face imagery."""
import csv, hashlib, json, uuid
from datetime import datetime, timezone
from pathlib import Path

def sha256_file(path: str | Path) -> str:
    digest=hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda:f.read(1024*1024),b""): digest.update(block)
    return digest.hexdigest()

def build_report(video_path, result: dict, config: dict | None = None, original_filename: str | None = None) -> dict:
    p=Path(video_path); m=result["metadata"]; a=result["aggregation"]
    return {"analysis_id":str(uuid.uuid4()),"analysis_timestamp_utc":datetime.now(timezone.utc).isoformat(),
            "original_filename":original_filename or p.name,"sha256":sha256_file(p),"file_size_bytes":p.stat().st_size,
            "media":{"fps":m.fps,"frame_count":m.frame_count,"duration_seconds":m.duration_seconds,"width":m.width,"height":m.height},
            "model":{"name":result["model"].get("name"),"architecture":result["model"].get("architecture"),
                     "checkpoint_version":result["model"].get("version",result["model"].get("model_version")),
                     "license":result["model"].get("license"),"source":result["model"].get("source")},
            "sampling":config or {},"sampled_frames":result["sampled_frames"],"detected_faces":result["detected_faces"],
            "valid_predictions":result["valid_predictions"],"frame_scores":[{k:v for k,v in row.items() if k!="face_image"} for row in result["frames"]],
            "aggregation":{"score":a["score"],"q1":a["q1"],"q3":a["q3"],"iqr":a["iqr"],
                            "native_pooled_video_score":a.get("native_pooled_video_score"),
                            "method":config.get("aggregation","median") if config else "median"},
            "thresholds":{"low":config.get("low_threshold",.30) if config else .30,"high":config.get("high_threshold",.70) if config else .70},
            "assessment_status":a["status"],"warnings":["Scores are model sigmoid outputs, not calibrated video probabilities.",
            "Frame observations may be correlated; frame count is not an effective independent sample count.",
            "Thresholds are provisional and require held-out validation calibration.",
            "A file hash identifies analyzed bytes; it does not establish provenance or authenticity."]}

def report_json(report: dict) -> bytes: return json.dumps(report,indent=2,allow_nan=False).encode("utf-8")

def report_csv(result: dict) -> bytes:
    from io import StringIO
    f=StringIO(); writer=csv.DictWriter(f,fieldnames=["timestamp_seconds","frame_index","face_index","bbox","fake_score"]); writer.writeheader()
    for row in result["frames"]: writer.writerow({**{k:v for k,v in row.items() if k!="face_image"},"bbox":json.dumps(row["bbox"])})
    return f.getvalue().encode("utf-8")
