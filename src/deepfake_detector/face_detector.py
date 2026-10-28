"""Face detection using OpenCV YuNet when locally installed, else Haar cascade."""
from pathlib import Path
import cv2

class FaceDetector:
    """Detect face rectangles; YuNet requires a separately obtained ONNX model."""
    def __init__(self, yunet_model: str | Path | None = None, score_threshold: float = .8):
        self.detector = None
        self.backend = "opencv_haar"
        if yunet_model and Path(yunet_model).is_file() and hasattr(cv2, "FaceDetectorYN"):
            self.model_path = str(yunet_model)
            self.score_threshold = score_threshold
            self.backend = "yunet"
        else:
            cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            self.detector = cv2.CascadeClassifier(cascade_path)
            if self.detector.empty():
                raise RuntimeError("OpenCV Haar face detector could not be loaded")

    def detect(self, frame):
        """Return face rectangles as (x, y, width, height, score)."""
        h, w = frame.shape[:2]
        if self.backend == "yunet":
            detector = cv2.FaceDetectorYN.create(self.model_path, "", (w, h), self.score_threshold)
            _, faces = detector.detect(frame)
            # YuNet rows contain box coordinates, five landmarks, then confidence.
            return [] if faces is None else [(int(face[0]), int(face[1]), int(face[2]), int(face[3]), float(face[-1])) for face in faces]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        found = self.detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40))
        return [(int(x), int(y), int(bw), int(bh), 1.0) for x,y,bw,bh in found]
