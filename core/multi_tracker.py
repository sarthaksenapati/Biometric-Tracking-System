# core/multi_tracker.py

import cv2
import time
import numpy as np
from collections import deque

from models.detector   import PersonDetector
from models.face_model import FaceRecognizer
from models.reid_model import ReIDModel
from models.gait_model import GaitModel
from core.matcher      import Matcher


# ── Global Identity Manager ──────────────────────────────────────────────────
class GlobalIdentityManager:
    """
    Tracks where each known person was last seen across all cameras.
    Enables cross-camera handoff logging and transition detection.
    """
    def __init__(self):
        # person_name → {cam_id, last_seen, track_id}
        self.registry = {}

    def update(self, person_name, cam_id, track_id):
        if person_name == "Unknown":
            return

        prev = self.registry.get(person_name)

        if prev and prev["cam_id"] != cam_id:
            elapsed = time.time() - prev["last_seen"]
            # Only log handoff if person was genuinely absent for >3 seconds.
            # Prevents false handoffs when both cameras see the same person
            # simultaneously (which produces 0.1s elapsed — not a real handoff).
            if elapsed > 1.5:
                print(
                    f"[HANDOFF] {person_name} moved "
                    f"Cam{prev['cam_id']} → Cam{cam_id} "
                    f"({elapsed:.1f}s since last seen)"
                )

        # Always update registry every frame regardless of the check above.
        # This keeps status_str() and last_seen() accurate at all times.
        self.registry[person_name] = {
            "cam_id":    cam_id,
            "track_id":  track_id,
            "last_seen": time.time()
        }

    def last_seen(self, person_name):
        return self.registry.get(person_name)

    def status_str(self):
        """One-line summary of where everyone currently is."""
        if not self.registry:
            return "No identities tracked yet"
        parts = []
        for name, info in self.registry.items():
            age = time.time() - info["last_seen"]
            parts.append(f"{name}@Cam{info['cam_id']} ({age:.0f}s ago)")
        return " | ".join(parts)


# ── Multi-Camera Tracker ─────────────────────────────────────────────────────
class MultiCameraTracker:
    """
    Runs 2 cameras with:
    - Shared model set (one detector, face, body, gait, matcher)
    - Per-camera gait and prediction smoothing buffers
    - GlobalIdentityManager for cross-camera handoff
    - Single combined display window
    """

    GAIT_BUFFER_SIZE = 10
    PRED_BUFFER_SIZE = 7

    def __init__(self, cam_sources: dict):
        """
        cam_sources: {cam_id: source}
        e.g. {0: 0, 1: "http://192.168.0.183:4747/video"}
        """
        print("\n[MULTI] Loading shared models...")
        self.detector   = PersonDetector()
        self.face_model = FaceRecognizer()
        self.reid_model = ReIDModel()
        self.gait_model = GaitModel()
        self.matcher    = Matcher()
        self.identity_manager = GlobalIdentityManager()
        print("[MULTI] Models ready.\n")

        # Open cameras
        self.cameras = {}
        for cam_id, source in cam_sources.items():
            cap = cv2.VideoCapture(source)
            if not cap.isOpened():
                print(f"[MULTI] ❌ Could not open Cam{cam_id} (source={source})")
            else:
                print(f"[MULTI] ✅ Cam{cam_id} opened (source={source})")
            self.cameras[cam_id] = cap

        # Per-(cam_id, track_id) buffers
        self.gait_buffers = {}   # (cam_id, track_id) → deque of crops
        self.pred_buffers = {}   # (cam_id, track_id) → deque of name strings

    # ── Buffer helpers ───────────────────────────────────────────────────────
    def _gait_buf(self, cam_id, tid):
        key = (cam_id, tid)
        if key not in self.gait_buffers:
            self.gait_buffers[key] = deque(maxlen=self.GAIT_BUFFER_SIZE)
        return self.gait_buffers[key]

    def _pred_buf(self, cam_id, tid):
        key = (cam_id, tid)
        if key not in self.pred_buffers:
            self.pred_buffers[key] = deque(maxlen=self.PRED_BUFFER_SIZE)
        return self.pred_buffers[key]

    # ── Process one frame from one camera ────────────────────────────────────
    def process_frame(self, frame, cam_id):
        detections = self.detector.detect(frame)
        results    = []

        for idx, det in enumerate(detections):
            track_id = det.get("track_id") or idx

            x1, y1, x2, y2 = det["bbox"]
            h, w = frame.shape[:2]

            pad = 10
            x1 = max(0, x1 - pad)
            y1 = max(0, y1 - pad)
            x2 = min(w, x2 + pad)
            y2 = min(h, y2 + pad)

            person_crop = frame[y1:y2, x1:x2]
            if person_crop.size == 0:
                continue

            # ── Face ─────────────────────────────────────────────────────
            face_emb = None
            try:
                face_emb = self.face_model.get_embedding(person_crop)
            except Exception as e:
                print(f"[Cam{cam_id}] face error track {track_id}: {e}")

            # ── Body ─────────────────────────────────────────────────────
            body_emb = None
            try:
                body_emb = self.reid_model.get_embedding(person_crop)
            except Exception as e:
                print(f"[Cam{cam_id}] body error track {track_id}: {e}")

            # ── Gait ─────────────────────────────────────────────────────
            gait_buf = self._gait_buf(cam_id, track_id)
            gait_buf.append(person_crop)
            gait_emb = None
            if len(gait_buf) >= 5:
                try:
                    gait_emb = self.gait_model.get_embedding(list(gait_buf))
                except Exception as e:
                    print(f"[Cam{cam_id}] gait error track {track_id}: {e}")

            crop_h, crop_w = person_crop.shape[:2]
            print(
                f"[Cam{cam_id}] track={track_id} crop=({crop_w}x{crop_h}) | "
                f"face={'✅' if face_emb is not None else '❌'} | "
                f"body={'✅' if body_emb is not None else '❌'} | "
                f"gait={'✅' if gait_emb is not None else f'buffering ({len(gait_buf)}/5)'}"
            )

            # ── Match ─────────────────────────────────────────────────────
            name, score = self.matcher.identify(
                face_emb=face_emb,
                body_emb=body_emb,
                gait_emb=gait_emb
            )

            # ── Smooth predictions ────────────────────────────────────────
            pred_buf = self._pred_buf(cam_id, track_id)
            pred_buf.append(name)
            final_name = max(set(pred_buf), key=pred_buf.count)

            # ── Cross-camera handoff ──────────────────────────────────────
            self.identity_manager.update(final_name, cam_id, track_id)

            results.append({
                "bbox":     (x1, y1, x2, y2),
                "name":     final_name,
                "score":    score,
                "track_id": track_id,
                "cam_id":   cam_id
            })

        return results

    # ── Draw results on a frame ──────────────────────────────────────────────
    def draw_results(self, frame, results, cam_id):
        # Camera label top-left
        cv2.putText(frame, f"CAM {cam_id}", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

        for res in results:
            x1, y1, x2, y2 = res["bbox"]
            name  = res["name"]
            score = res["score"]
            tid   = res["track_id"]

            color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
            label = f"[{tid}] {name} ({score:.2f})"

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, max(y1 - 10, 15)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

        return frame

    # ── Main run loop ────────────────────────────────────────────────────────
    def run(self):
        print("✅ Multi-camera tracker running — press ESC to quit\n")

        DISPLAY_W = 640   # width per camera panel in combined window
        DISPLAY_H = 480

        while True:
            frames      = {}
            all_results = {}

            # Read and process each camera sequentially
            for cam_id, cap in self.cameras.items():
                ret, frame = cap.read()
                if not ret or frame is None:
                    # Show black panel if camera feed lost
                    frame = np.zeros((DISPLAY_H, DISPLAY_W, 3), dtype=np.uint8)
                    cv2.putText(frame, f"CAM {cam_id} — NO SIGNAL",
                                (30, DISPLAY_H // 2),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                    frames[cam_id]      = frame
                    all_results[cam_id] = []
                    continue

                results = self.process_frame(frame, cam_id)
                frame   = self.draw_results(frame, results, cam_id)

                frames[cam_id]      = frame
                all_results[cam_id] = results

            # ── Combined display ─────────────────────────────────────────
            panels = []
            for cam_id in sorted(frames.keys()):
                panel = cv2.resize(frames[cam_id], (DISPLAY_W, DISPLAY_H))
                panels.append(panel)

            # Vertical divider between cameras
            divider = np.zeros((DISPLAY_H, 4, 3), dtype=np.uint8)
            divider[:] = (255, 255, 0)   # yellow line

            combined = np.hstack([panels[0], divider] + panels[1:]) \
                       if len(panels) > 1 else panels[0]

            # Cross-camera status bar at the bottom
            status = self.identity_manager.status_str()
            bar_h  = 30
            bar    = np.zeros((bar_h, combined.shape[1], 3), dtype=np.uint8)
            cv2.putText(bar, status, (8, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            combined = np.vstack([combined, bar])

            cv2.imshow("Multi-Camera Biometric Tracker", combined)

            if cv2.waitKey(1) & 0xFF == 27:
                break

        # Cleanup
        for cap in self.cameras.values():
            cap.release()
        cv2.destroyAllWindows()