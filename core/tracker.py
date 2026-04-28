# core/tracker.py

import cv2
import time
import numpy as np
import threading
from collections import deque

from models.detector import PersonDetector
from models.face_model import FaceRecognizer
from models.reid_model import ReIDModel
from models.gait_model import GaitModel
from core.matcher import Matcher
from monitoring.metrics import (
    update_fps,
    record_detection,
    record_identification,
    record_frame_processed,
    observe_matching_latency,
    update_camera_status,
)
from monitoring.alerts import trigger_alert, get_alert_manager


class LiveTracker:
    def __init__(self, cam_id=0, source=0):
        self.cam_id = str(cam_id)
        self.source = source
        self.detector  = PersonDetector()
        self.face_model = FaceRecognizer()
        self.reid_model = ReIDModel()
        self.gait_model = GaitModel()
        self.matcher   = Matcher()

        # Per-track buffers keyed by track_id
        self.gait_buffers = {}   # track_id → deque of crops
        self.pred_buffers = {}   # track_id → deque of name strings

        self.GAIT_BUFFER_SIZE = 10
        self.PRED_BUFFER_SIZE = 7

        # FPS tracking
        self._fps_counter = 0
        self._fps_time = time.time()
        self._current_fps = 0.0

        # Camera health
        self._camera_offline = False
        update_camera_status(self.cam_id, True)

    # ── Buffer helpers ──────────────────────────────────────────────────────
    def _gait_buf(self, tid):
        if tid not in self.gait_buffers:
            self.gait_buffers[tid] = deque(maxlen=self.GAIT_BUFFER_SIZE)
        return self.gait_buffers[tid]

    def _pred_buf(self, tid):
        if tid not in self.pred_buffers:
            self.pred_buffers[tid] = deque(maxlen=self.PRED_BUFFER_SIZE)
        return self.pred_buffers[tid]

    # ── FPS calculation ────────────────────────────────────────────────────
    def _update_fps(self):
        self._fps_counter += 1
        now = time.time()
        elapsed = now - self._fps_time
        if elapsed >= 1.0:
            self._current_fps = self._fps_counter / elapsed
            self._fps_counter = 0
            self._fps_time = now
            update_fps(self.cam_id, self._current_fps)

    # ── Main ────────────────────────────────────────────────────────────────
    def process_frame(self, frame):
        self._update_fps()
        record_frame_processed(self.cam_id)

        detections = self.detector.detect(frame)
        results    = []

        for idx, det in enumerate(detections):
            track_id = det.get("track_id") or idx
            record_detection(self.cam_id)

            x1, y1, x2, y2 = det["bbox"]
            h, w = frame.shape[:2]

            # Pad bounding box
            pad = 10
            x1 = max(0, x1 - pad)
            y1 = max(0, y1 - pad)
            x2 = min(w, x2 + pad)
            y2 = min(h, y2 + pad)

            person_crop = frame[y1:y2, x1:x2]
            if person_crop.size == 0:
                continue

            crop_h, crop_w = person_crop.shape[:2]

            # ── Face ─────────────────────────────────────────────────────
            face_emb = None
            try:
                face_emb = self.face_model.get_embedding(person_crop)
            except Exception as e:
                print(f"[TRACKER] face_model error track {track_id}: {e}")

            # ── Body ─────────────────────────────────────────────────────
            body_emb = None
            try:
                body_emb = self.reid_model.get_embedding(person_crop)
            except Exception as e:
                print(f"[TRACKER] reid_model error track {track_id}: {e}")

            # ── Gait ─────────────────────────────────────────────────────
            gait_buf = self._gait_buf(track_id)
            gait_buf.append(person_crop)

            gait_emb = None
            if len(gait_buf) >= 5:
                try:
                    gait_emb = self.gait_model.get_embedding(list(gait_buf))
                except Exception as e:
                    print(f"[TRACKER] gait_model error track {track_id}: {e}")

            # ── Match ─────────────────────────────────────────────────────
            name, score = self.matcher.identify(
                face_emb=face_emb,
                body_emb=body_emb,
                gait_emb=gait_emb
            )

            if name != "Unknown":
                record_identification(self.cam_id)

            # ── Smooth predictions ────────────────────────────────────────
            pred_buf = self._pred_buf(track_id)
            pred_buf.append(name)
            final_name = max(set(pred_buf), key=pred_buf.count)

            results.append({
                "bbox":     (x1, y1, x2, y2),
                "name":     final_name,
                "score":    score,
                "track_id": track_id
            })

        return results

    # ── Drawing ─────────────────────────────────────────────────────────────
    def draw_results(self, frame, results):
        for res in results:
            x1, y1, x2, y2 = res["bbox"]
            name  = res["name"]
            score = res["score"]
            tid   = res["track_id"]

            color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
            label = f"[{tid}] {name} ({score:.2f})"

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        return frame

    # ── Run loop ─────────────────────────────────────────────────────────────
    def run(self, source=None):
        src = source if source is not None else self.source
        cap = cv2.VideoCapture(src)

        if not cap.isOpened():
            print(f"❌ Could not open camera/source {src}")
            update_camera_status(self.cam_id, False)
            trigger_alert(
                "camera_offline",
                f"Camera {self.cam_id} could not be opened",
                severity="error",
                camera_id=self.cam_id,
                details={"source": str(src)}
            )
            return

        print(f"✅ Tracker running on camera {self.cam_id} — press ESC to quit\n")
        update_camera_status(self.cam_id, True)

        while True:
            ret, frame = cap.read()
            if not ret:
                if not self._camera_offline:
                    print(f"⚠️  Camera {self.cam_id} went offline")
                    self._camera_offline = True
                    update_camera_status(self.cam_id, False)
                    trigger_alert(
                        "camera_offline",
                        f"Camera {self.cam_id} stopped sending frames",
                        severity="error",
                        camera_id=self.cam_id
                    )
                time.sleep(1)
                cap = cv2.VideoCapture(src)
                if cap.isOpened():
                    print(f"✅ Camera {self.cam_id} reconnected")
                    self._camera_offline = False
                    update_camera_status(self.cam_id, True)
                    trigger_alert(
                        "camera_online",
                        f"Camera {self.cam_id} is back online",
                        severity="info",
                        camera_id=self.cam_id
                    )
                continue

            if self._camera_offline:
                print(f"✅ Camera {self.cam_id} reconnected")
                self._camera_offline = False
                update_camera_status(self.cam_id, True)
                trigger_alert(
                    "camera_online",
                    f"Camera {self.cam_id} is back online",
                    severity="info",
                    camera_id=self.cam_id
                )

            results = self.process_frame(frame)
            frame   = self.draw_results(frame, results)

            cv2.imshow("Live Biometric Tracker", frame)

            if cv2.waitKey(1) & 0xFF == 27:
                break

        update_camera_status(self.cam_id, False)
        cap.release()
        cv2.destroyAllWindows()