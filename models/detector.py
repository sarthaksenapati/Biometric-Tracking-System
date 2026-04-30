# models/detector.py

import torch
from ultralytics import YOLO
from utils.config import MODEL_PATH, CONF_THRESHOLD


class PersonDetector:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = YOLO(MODEL_PATH)
        self.model.to(self.device)
        print(f"[Detector] YOLO running on {self.device}")

    def detect(self, frame):
        results = self.model.track(frame, persist=True, verbose=False, device=self.device)

        detections = []

        for r in results:
            if r.boxes is None:
                continue

            for box in r.boxes:
                cls = int(box.cls[0])
                if cls != 0:
                    continue

                conf = float(box.conf[0])
                if conf < CONF_THRESHOLD:
                    continue

                x1, y1, x2, y2 = map(int, box.xyxy[0])
                track_id = int(box.id[0]) if box.id is not None else None

                detections.append(
                    {"bbox": (x1, y1, x2, y2), "confidence": conf, "class": "person", "track_id": track_id}
                )

        return detections
