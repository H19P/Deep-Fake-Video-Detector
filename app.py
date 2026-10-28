"""Local Streamlit UI for pretrained, model-assisted video screening."""
from __future__ import annotations

import logging
import sys
from pathlib import Path
import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from deepfake_detector.config import InferenceConfig, validate_video_path
from deepfake_detector.pretrained_adapter import (
    MODEL_NAME, Oo01PretrainedAdapter, PretrainedModelError,
)
from deepfake_detector.pretrained_inference import PretrainedInferenceService
from deepfake_detector.media_storage import temporary_video_file
from deepfake_detector.reporting import build_report, report_csv, report_json
from deepfake_detector.interpretation import explain_fake_score

logging.basicConfig(level=logging.INFO)

@st.cache_resource(show_spinner=False)
def load_adapter(device_choice: str) -> Oo01PretrainedAdapter:
    """Cache model weights and model instance, never uploaded media."""
    return Oo01PretrainedAdapter.from_pretrained(device=device_choice)

st.set_page_config(page_title="Local Deepfake Video Analysis", page_icon="🔎", layout="wide")
st.title("Local Deepfake Video Analysis")
st.caption("Screening aid only. A model score cannot establish that a video is authentic or manipulated.")
st.info("**Privacy:** video decoding and inference run on this computer. Videos are written to a randomized temporary file and removed after analysis. On first analysis only, the pretrained model weights are fetched from Hugging Face and cached locally; video content is never sent to that service.")

with st.expander("About this detector and its limits", expanded=False):
    st.markdown(
        f"**Selected model:** {MODEL_NAME}. It uses EfficientNet-B0 on face crops and was trained on Celeb-DF v2, "
        "according to its model card. It may miss other manipulation types, recent generators, or heavily "
        "compressed clips. The project reports model scores, not calibrated probabilities. The published MIT "
        "license applies as declared by the model repository; verify any underlying dataset obligations before "
        "commercial use."
    )

uploaded = st.file_uploader("Choose a video", type=["mp4", "mov", "avi", "mkv", "webm"])
with st.expander("Analysis settings"):
    col1, col2 = st.columns(2)
    interval = col1.number_input("Sampling interval (seconds)", min_value=0.25, max_value=30.0, value=1.0)
    max_frames = col2.number_input("Maximum sampled frames", min_value=3, max_value=16, value=16)

if uploaded is not None:
    st.video(uploaded)
    if uploaded.size > 500 * 1024 * 1024:
        st.error("The video exceeds the 500 MB upload limit.")
    elif st.button("Analyze video locally", type="primary"):
        try:
            config = InferenceConfig(
                sample_interval_seconds=float(interval), max_frames=int(max_frames),
                max_file_size_mb=500, max_duration_seconds=600,
            )
            with temporary_video_file(uploaded.name, uploaded.getbuffer(), config.max_file_size_mb) as temp_path:
                validate_video_path(temp_path, config.max_file_size_mb)
                with st.status("Loading the local detector and analyzing sampled frames…", expanded=True) as status:
                    adapter = load_adapter("cuda" if __import__("torch").cuda.is_available() else "cpu")
                    service = PretrainedInferenceService(adapter, config)
                    result = service.analyze(temp_path)
                    status.update(label="Analysis complete", state="complete", expanded=False)

                report = build_report(
                    temp_path, result,
                    {"sample_interval_seconds": config.sample_interval_seconds,
                     "max_frames": config.max_frames, "aggregation": config.aggregation,
                     "low_threshold": config.low_threshold, "high_threshold": config.high_threshold,
                     "face_detector": "OpenCV Haar cascade; largest face; 30% crop margin"},
                    original_filename=uploaded.name,
                )
            score = result["aggregation"]["score"]
            status_label, explanation = explain_fake_score(
                score, config.low_threshold, config.high_threshold
            )
            st.subheader(status_label)
            if score is None:
                st.info(explanation)
            else:
                st.metric("Median fake-class model score", f"{score:.2f} / 1.00")
                st.write(explanation)
                st.caption(f"Frame-score variability (IQR): {result['aggregation']['iqr']:.3f}.")
            st.caption(
                f"Score guide: ≤ {config.low_threshold:.2f} = low fake-class score; "
                f"> {config.low_threshold:.2f} and < {config.high_threshold:.2f} = inconclusive; "
                f"≥ {config.high_threshold:.2f} = high fake-class score. "
                "These are provisional bands, not validated cutoffs."
            )
            with st.expander("Technical score details"):
                st.write(
                    "The median combines per-frame fake-class scores. The pooled-video score is the model's "
                    "separate native output after averaging frame features. Both are model outputs, not calibrated "
                    "probabilities, and they can differ."
                )
                pooled_score = result["aggregation"]["native_pooled_video_score"]
                st.write("Native pooled-video fake-class score:", "Unavailable" if pooled_score is None else f"{pooled_score:.3f} / 1.000")
            a, b, c, d = st.columns(4)
            a.metric("Execution device", result["device"])
            b.metric("Sampled frames", result["sampled_frames"])
            c.metric("Detected faces", result["detected_faces"])
            d.metric("Valid face predictions", result["valid_predictions"])
            st.caption(f"Model: {adapter.model_info['name']} · version {adapter.model_info['version']}")
            st.warning("Requires independent verification. Adjacent frames are correlated; frame count is not an effective independent sample count.")

            if result["frames"]:
                st.subheader("Frame scores")
                chart_rows = [{"timestamp": row["timestamp_seconds"], "Fake-class score": row["fake_score"]} for row in result["frames"]]
                st.line_chart(chart_rows, x="timestamp", y="Fake-class score", x_label="Timestamp (seconds)", y_label="Model score")
                st.dataframe([
                    {k: v for k, v in row.items() if k != "face_image"}
                    for row in result["frames"]
                ], use_container_width=True)
                st.subheader("Analyzed face crops")
                columns = st.columns(min(4, len(result["frames"])))
                for index, row in enumerate(result["frames"]):
                    columns[index % len(columns)].image(
                        row["face_image"],
                        caption=f"{row['timestamp_seconds']:.2f}s · fake score {row['fake_score']:.3f}",
                    )

            st.download_button("Download JSON report", report_json(report), "analysis_report.json", "application/json")
            st.download_button("Download frame-level CSV", report_csv(result), "frame_scores.csv", "text/csv")
        except PretrainedModelError as exc:
            logging.exception("Pretrained detector initialization/inference failed")
            st.error(str(exc))
        except (ValueError, FileNotFoundError) as exc:
            st.error(str(exc))
        except Exception:
            logging.exception("Local video analysis failed")
            st.error("Analysis failed. Check the video decoder and local model setup, then try again.")

if st.button("Reset / new analysis"):
    st.rerun()
