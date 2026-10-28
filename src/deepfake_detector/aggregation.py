"""Video-level score aggregation and cautious decision labels."""
from typing import Sequence
import numpy as np

def aggregate_scores(scores: Sequence[float], method: str = "median", min_predictions: int = 3,
                     low_threshold: float = .30, high_threshold: float = .70) -> dict:
    """Aggregate crop scores; adjacent frames are not independent evidence."""
    values = np.asarray(scores, dtype=float)
    if values.ndim != 1 or not np.isfinite(values).all() or ((values < 0) | (values > 1)).any():
        raise ValueError("Scores must be finite values in [0, 1]")
    if method not in {"median", "mean"}:
        raise ValueError("method must be median or mean")
    if not 0 <= low_threshold < high_threshold <= 1:
        raise ValueError("Invalid thresholds")
    if len(values) == 0 or len(values) < min_predictions:
        return {"score": None, "q1": None, "q3": None, "iqr": None,
                "status": "inconclusive", "count": int(len(values))}
    score = float(np.median(values) if method == "median" else np.mean(values))
    status = "high_fake_score" if score >= high_threshold else "low_fake_score" if score <= low_threshold else "inconclusive"
    q1, q3 = np.quantile(values, [.25, .75])
    return {"score": score, "q1": float(q1), "q3": float(q3), "iqr": float(q3-q1),
            "status": status, "count": int(len(values))}
