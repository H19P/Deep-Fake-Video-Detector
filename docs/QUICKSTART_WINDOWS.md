# Run the project on another Windows computer

This guide is for Windows 10/11 and PowerShell. The application runs with CPU inference; an NVIDIA GPU is optional. No dataset or training run is required to try the app.

## 1. Copy the project

Share the project folder or a ZIP of the source. Do not copy `.venv`, `__pycache__`, `.pytest_cache`, or uploaded videos. The first analysis downloads the pinned model checkpoint (about 52 MB) and caches it in `models/pretrained-cache/`.

If using a ZIP, extract it to a normal local folder, for example `D:\DeepfakeDetector`. Avoid running it directly from inside the ZIP or from a synchronized/network drive.

## 2. Install Python

Install 64-bit Python 3.11 from [python.org](https://www.python.org/downloads/). In the installer, enable **Add Python to PATH**. Open a new PowerShell window and confirm:

```powershell
py -3.11 --version
```

## 3. Set up the environment

Change to the folder where the project was extracted. Use that actual path if it differs from the example:

```powershell
cd D:\DeepfakeDetector
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements-dev.txt
python -m pip install -e .
```

If PowerShell says script execution is disabled, allow activation for this window only, then activate again:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

The policy change ends when that PowerShell window closes. Alternatively, use Command Prompt and activate with `.venv\Scripts\activate.bat`.

## 4. Check the installation

With `.venv` activated, run:

```powershell
python scripts\check_environment.py
```

PyTorch should be installed and the other packages should say `installed`. `CUDA available: False` is okay; the app will use CPU. For GPU support, install a PyTorch CUDA build that matches the computer's NVIDIA driver using the [official PyTorch install selector](https://pytorch.org/get-started/locally/), then run the environment check again. Do not install CUDA-enabled PyTorch just because the machine has an NVIDIA GPU; the driver and wheel must be compatible.

## 5. Start the application

```powershell
streamlit run app.py
```

Open the local address printed in the terminal (usually `http://localhost:8501`). Keep that terminal open while using the app. To stop it, focus the terminal and press **Ctrl+C**.

On the first analysis, the app needs internet access to download the pinned model checkpoint. Later analyses use the local cache. If the download fails, the app should report that model weights are unavailable and will not invent a result. Videos themselves are processed on the local computer and are not uploaded to the model host.

## 6. Analyze a video

1. Choose a supported video file: MP4, MOV, AVI, MKV, or WebM (decoder support can vary).
2. Check the displayed media details and settings.
3. Click **Analyze video locally** and wait for processing to finish.
4. Review the assessment, score spread, frame timestamps, and detected face crops.
5. Download the JSON report or frame-level CSV if needed.

The upload limit is 500 MB; maximum duration is 10 minutes; maximum resolution is 3840 × 2160. At most 16 frames are sampled. No-face videos or videos with fewer than three usable face crops are inconclusive.

## 7. Optional: run the automated tests

```powershell
pytest
```

Tests use small generated videos and mocks; they do not download the model or demonstrate detection accuracy.

## Common issues

- **`py` or Python not found:** install 64-bit Python 3.11, enable PATH, then open a new terminal.
- **`streamlit` not found:** activate `.venv` and repeat the dependency installation commands.
- **Model download error:** confirm internet access and retry analysis. No prediction is produced without weights.
- **CUDA unavailable:** CPU inference is supported. For GPU use, install a compatible CUDA PyTorch wheel from the official selector.
- **Video cannot be read:** try a standard MP4/H.264 copy. Supported extensions still depend on codecs in the installed OpenCV build.
- **Port 8501 is busy:** start on another local port with `streamlit run app.py --server.port 8502`.
- **Video has no usable face crops:** try a clearer face-visible video. The correct output in that case is inconclusive.

## Important result limitation

The shown value is a model score, not a calibrated probability and not proof that a video is real or fake. The configured score bands are provisional. This repository has not run a held-out evaluation, so it makes no benchmark accuracy claim. Use results only as a screening aid and seek independent verification.
