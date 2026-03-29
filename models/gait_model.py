# models/gait_model.py — GPU-accelerated GEI (keeps your approach, adds GPU)

import cv2
import torch
import numpy as np


class GaitModel:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[Gait] ✅  GEI gait model on {self.device}")

    def get_silhouette(self, frame: np.ndarray) -> np.ndarray | None:
        if frame is None or frame.size == 0:
            return None
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        _, thresh = cv2.threshold(blur, 50, 255, cv2.THRESH_BINARY)
        return cv2.resize(thresh, (64, 128))

    def get_embedding(self, frame_sequence: list) -> np.ndarray | None:
        silhouettes = []
        for frame in frame_sequence:
            sil = self.get_silhouette(frame)
            if sil is not None:
                silhouettes.append(sil)

        if not silhouettes:
            return None

        # Stack → GPU tensor → mean (GEI) → normalize
        arr    = np.stack(silhouettes).astype(np.float32) / 255.0   # [N, 128, 64]
        tensor = torch.from_numpy(arr).to(self.device)              # GPU
        gei    = tensor.mean(dim=0).flatten()                        # [8192]  GPU mean
        norm   = torch.norm(gei)
        if norm < 1e-8:
            return None
        return (gei / norm).cpu().numpy().astype(np.float32)        # back to numpy