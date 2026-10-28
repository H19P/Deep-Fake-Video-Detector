# Local Deepfake Video Screening

A local Streamlit application for user-uploaded videos. The default inference path uses one pinned, publicly hosted, already-trained EfficientNet-B0 deepfake checkpoint; **you do not need to collect a dataset or train before trying inference**. The existing EfficientNet-B0 training pipeline remains available as a separate future experiment.

Sharing this project with someone on Windows? Follow the step-by-step [Windows quick start](docs/QUICKSTART_WINDOWS.md).

The selected model, checkpoint provenance, architecture, preprocessing, license declaration, candidate comparison, and limitations are documented in [docs/pretrained-model.md](docs/pretrained-model.md). On first analysis, the app downloads the single checkpoint (about 52 MB) from a pinned Hugging Face revision and caches it under `models/pretrained-cache/`. Videos are processed locally and are not sent to Hugging Face or an external inference API.

The model was trained on Celeb-DF v2 according to its card. It may generalize poorly to other manipulation methods or newer generators. The displayed values are model scores, not calibrated probabilities. High/low scores are not proof of manipulation or authenticity; use independent verification.

## Windows setup (PowerShell)

The project supports Python 3.11–3.14. In PowerShell:

```powershell
cd D:\DeepfakeDetector
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements-dev.txt
python -m pip install -e .
```

If you have not created `.venv` yet:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements-dev.txt
python -m pip install -e .
```

Install GPU-enabled PyTorch separately if needed. For the version pair currently pinned and the environment's CUDA 13.3 driver, use the official CUDA 13.2 wheel index:

```powershell
python -m pip uninstall -y torch torchvision
python -m pip install torch==2.14.1 torchvision==0.29.1 --index-url https://download.pytorch.org/whl/cu132
```

Use the [official PyTorch selector](https://pytorch.org/get-started/locally/) if your driver differs. Check the environment:

```powershell
python scripts\check_environment.py
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU fallback')"
```

## Run the pretrained application

```powershell
streamlit run app.py
```

Open the local URL printed by Streamlit. Select MP4, MOV, AVI, MKV, or WebM and click **Analyze video locally**. The first analysis requires network access for the pinned model weights; later runs use the local cache. If the first download is blocked, the app explains that weights are unavailable and produces no prediction. Media stays on the local machine; the pretrained checkpoint download is the only network transfer.

The application limits uploads to 500 MB, duration to 10 minutes, resolution to 3840×2160, and samples at most 16 frames. It uses the model's documented 224×224 RGB, ImageNet-normalized face crop path with OpenCV Haar detection. Videos with no detected face or fewer than three usable crops return an inconclusive result. The report includes model scores, timestamps, aggregation, detector and warnings. Temporary video files are deleted after success or failure.

## Tests

```powershell
pytest
```

The tests use mocked adapters and tiny generated videos. They do not download model weights and therefore are not a real-model accuracy or video inference validation.

## Optional: train the project's separate EfficientNet-B0 baseline

This is not required for the pretrained app. To experiment with your own model later, obtain and use datasets only under the dataset owners' terms. Create a video manifest with columns `video_path,label,video_id,source_id,provenance`, then prepare splits before extracting face crops:

```powershell
python scripts\prepare_dataset.py --input-manifest data\manifests\videos.csv --output-manifest data\manifests\faces.csv --output-dir data\processed
python -m deepfake_detector.train --manifest data\manifests\faces.csv --output models\efficientnet_b0.pt --epochs 5 --batch-size 8 --freeze-features
python -m deepfake_detector.evaluate --checkpoint models\efficientnet_b0.pt --manifest data\manifests\faces.csv
```

Frame-level random splitting is not a valid held-out evaluation strategy. Keep related original/manipulated videos in one group. The app's selected pretrained model is a different architecture/checkpoint from this trainable baseline.

## Privacy and limitations

The app binds to `127.0.0.1` by default. It does not retain uploaded media or send media to third parties. Reports omit video bytes and face images. A file hash identifies the bytes analyzed but does not establish provenance or authenticity. Do not expose this first version to a multi-user network without authentication, isolated workers, per-user quotas, sandboxed media decoding, encryption, retention controls, and monitoring.

No held-out evaluation has been run by this repository. Scores are not scientific cutoffs and frame count is not an effective independent sample count because nearby frames are correlated. Review the selected model's MIT declaration and Celeb-DF data terms before redistribution or commercial deployment.
