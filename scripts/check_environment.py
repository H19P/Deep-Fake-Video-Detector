"""Report local Python, package, and CUDA availability."""
import importlib.util,platform,sys
print(f"Python: {sys.version}"); print(f"Platform: {platform.platform()}")
for name in ("torch","torchvision","cv2","streamlit","pytest","sklearn","PIL"):
    print(f"{name}: {'installed' if importlib.util.find_spec(name) else 'missing'}")
try:
 import torch
 print(f"Torch: {torch.__version__}; CUDA available: {torch.cuda.is_available()}")
 if torch.cuda.is_available(): print(f"GPU: {torch.cuda.get_device_name(0)}")
except Exception as exc: print(f"Torch check failed: {exc}")
