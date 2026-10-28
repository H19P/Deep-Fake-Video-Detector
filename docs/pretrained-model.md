# Pretrained detector selection and use

## Selected baseline

The local app uses **oo01/deepfake-video-classifier**, checkpoint `ffpp_efficientnet_best.pth`, pinned to Hugging Face commit `3b237c2c820fab51cca13b80d726e4be5876de23`. The checkpoint is listed as 52.1 MB (the repository reports 52.2 MB for the LFS object). The card declares MIT and says the model was trained on Celeb-DF v2. The source exposes an EfficientNet-B0 feature backbone with 1280 features and a LayerNorm/dropout/Linear/GELU/dropout/Linear scalar head. Its video forward path accepts `(B,T,3,224,224)`, averages frame features in time, then applies the head. Its documented video helper uses 16 frames, ImageNet normalization, and a largest-face OpenCV Haar crop with 30% margin; the application follows those crop and normalization settings, but returns inconclusive when no face is detected instead of classifying a center crop.

The checkpoint's sigmoid means **probability-like score for REAL** (training label 1 = real, label 0 = fake). The app inverts it to report a fake-class model score. It also reports the model's native pooled-video fake score. The UI assessment uses the configured median/mean of per-frame fake scores; this is an application aggregation and is not claimed to reproduce the model card's reported test metrics. Neither output is a calibrated probability that the entire video is fake.

The inspected upstream inference class uses `torch.load(..., weights_only=False)`. This application does not use that loader or execute upstream Python. It recreates the published architecture, downloads one checkpoint from the pinned HTTPS Hugging Face revision, and loads it with `weights_only=True`; an incompatible/unsafe checkpoint is rejected. The checkpoint is cached under `models/pretrained-cache/`. Video files are not sent to Hugging Face or any other service.

The Hugging Face card declares MIT for the model repository. Attribute the model to the repository author and review the underlying Celeb-DF v2 dataset terms before redistribution or commercial deployment; the MIT model-card declaration alone should not be treated as a legal opinion about dataset-derived weights. The model card's benchmark numbers are publisher-reported and have not been reproduced by this project. The training domain is Celeb-DF v2; generalization to other manipulations, generators, identities, and compression is unknown.

## Candidate review

| Candidate | Verified details | Decision |
|---|---|---|
| [oo01/deepfake-video-classifier](https://huggingface.co/oo01/deepfake-video-classifier) | 52.1 MB checkpoint present; inspected model source; timm EfficientNet-B0; 224 RGB; 16-frame video input; ImageNet normalization; Haar largest-face crop and 30% margin; sigmoid is P(real); MIT declared; no inference API or remote code needed. | Selected: smallest fit for the existing PyTorch stack and 6 GB GPU. Its upstream loader is unsafe (`weights_only=False`), so this project uses its own architecture and restricted loading. |
| [honi05/deepfake-detection](https://huggingface.co/honi05/deepfake-detection) | 72.8 MB `best_model.pt` present; torchvision EfficientNet-B4 with a 1792→256→1 head; 224 face crop; MTCNN with 20-pixel margin; ImageNet normalization; sigmoid is fake score; MIT declared; local PyTorch inference. | Not selected: larger B4 and requires an additional MTCNN model/runtime path. Its model card and repository provide different checkpoint wrapper examples, adding avoidable integration ambiguity. |
| [mertkayacs/xdfdet](https://github.com/mertkayacs/xdfdet) | Eight EfficientNet-B4 Keras checkpoints; 12 crops per video, 224 pixels, MTCNN; outputs P(real); local inference and MIT source code. Requires Python 3.10–3.12. Weights are CC BY-NC 4.0. | Not selected: Python range excludes this project's Python 3.14 environment and the weights are non-commercial only. |

## Runtime and limitations

- The first inference needs internet access to fetch one approximately 52 MB checkpoint. Later runs use the local cache; model files are ignored by Git.
- CUDA is used when available; otherwise the adapter uses CPU. The selected B0 model is small enough for the target RTX 3050 6 GB GPU.
- The output is an experimental screening signal trained on a single dataset. It can miss unseen manipulation methods and does not prove authenticity or manipulation.
- This project did not evaluate the model on a held-out set and makes no benchmark accuracy claim.
