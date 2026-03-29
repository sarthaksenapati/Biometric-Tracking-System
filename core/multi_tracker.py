# core/multi_tracker.py

import cv2
import time
import threading
import numpy as np
from collections import deque
from queue import Queue, Empty

from models.detector   import PersonDetector
from models.face_model import FaceRecognizer
from models.reid_model import ReIDModel
from models.gait_model import GaitModel
from core.matcher      import Matcher


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    a = a.flatten().astype(np.float32)
    b = b.flatten().astype(np.float32)
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / denom) if denom > 1e-8 else 0.0


def _clothing_histogram(crop: np.ndarray, bins: int = 32) -> np.ndarray | None:
    if crop is None or crop.size == 0:
        return None
    h, w = crop.shape[:2]
    if h < 40 or w < 20:
        return None
    torso = crop[int(h * 0.40) : int(h * 0.80), :]
    hsv   = cv2.cvtColor(torso, cv2.COLOR_BGR2HSV)
    hists = []
    for ch in range(3):
        hist = cv2.calcHist([hsv], [ch], None, [bins], [0, 256])
        cv2.normalize(hist, hist)
        hists.append(hist.flatten())
    return np.concatenate(hists).astype(np.float32)


def _hist_sim(a, b) -> float | None:
    if a is None or b is None:
        return None
    return _cosine_sim(a, b)


# ─────────────────────────────────────────────────────────────────────────────
# Shared Identity Cache
# ─────────────────────────────────────────────────────────────────────────────

class SharedIdentityCache:
    THRESHOLD_WITH_FACE = 0.55
    THRESHOLD_NO_FACE   = 0.50
    FRESH_TTL           = 45.0
    _W = {"face": 0.45, "body": 0.30, "gait": 0.15, "cloth": 0.10}

    def __init__(self):
        self._store: dict = {}
        self._lock = threading.Lock()

    def deposit(self, name, cam_id, location, score,
                face_emb, body_emb, gait_emb, cloth_hist):
        if name == "Unknown":
            return
        with self._lock:
            existing = self._store.get(name)
            if existing is None or score > existing["score"]:
                self._store[name] = {
                    "face_emb":   face_emb,
                    "body_emb":   body_emb,
                    "gait_emb":   gait_emb,
                    "cloth_hist": cloth_hist,
                    "score":      score,
                    "cam_id":     cam_id,
                    "location":   location,
                    "timestamp":  time.time(),
                }
                stored = [m for m, v in [("face", face_emb), ("body", body_emb),
                                          ("gait", gait_emb), ("cloth", cloth_hist)]
                          if v is not None]
                print(f"[CACHE] 💾  '{name}' updated from {location} "
                      f"(score={score:.3f}, stored={stored})")

    def query(self, face_emb, body_emb, gait_emb, cloth_hist):
        with self._lock:
            store_copy = dict(self._store)

        if not store_copy:
            return None, 0.0

        has_face  = face_emb is not None
        threshold = self.THRESHOLD_WITH_FACE if has_face else self.THRESHOLD_NO_FACE
        best_name = None
        best_sim  = 0.0
        now       = time.time()

        for name, entry in store_copy.items():
            sims, weights = [], []
            if face_emb is not None and entry["face_emb"] is not None:
                sims.append(_cosine_sim(face_emb, entry["face_emb"]))
                weights.append(self._W["face"])
            if body_emb is not None and entry["body_emb"] is not None:
                sims.append(_cosine_sim(body_emb, entry["body_emb"]))
                weights.append(self._W["body"])
            if gait_emb is not None and entry["gait_emb"] is not None:
                sims.append(_cosine_sim(gait_emb, entry["gait_emb"]))
                weights.append(self._W["gait"])
            h_sim = _hist_sim(cloth_hist, entry["cloth_hist"])
            if h_sim is not None:
                sims.append(h_sim)
                weights.append(self._W["cloth"])
            if not sims:
                continue
            combined = sum(s * w for s, w in zip(sims, weights)) / sum(weights)
            age = now - entry["timestamp"]
            if age > self.FRESH_TTL:
                combined *= max(0.70, 1.0 - 0.004 * (age - self.FRESH_TTL))
            if combined > best_sim:
                best_sim  = combined
                best_name = name

        if best_sim >= threshold:
            return best_name, best_sim
        return None, best_sim

    def summary(self) -> str:
        with self._lock:
            store_copy = dict(self._store)
        if not store_copy:
            return "cache empty"
        now   = time.time()
        parts = []
        for name, e in store_copy.items():
            age = now - e["timestamp"]
            tag = "".join([
                "F" if e["face_emb"]   is not None else "_",
                "B" if e["body_emb"]   is not None else "_",
                "G" if e["gait_emb"]   is not None else "_",
                "C" if e["cloth_hist"] is not None else "_",
            ])
            parts.append(f"{name}[{tag}]({e['location']},{age:.0f}s)")
        return " | ".join(parts)


# ─────────────────────────────────────────────────────────────────────────────
# Global Identity Manager
# ─────────────────────────────────────────────────────────────────────────────

class GlobalIdentityManager:
    RETENTION_SECONDS   = 300
    SIGHTING_DEBOUNCE_S = 2.0

    def __init__(self, cam_locations: dict):
        self.cam_locations   = cam_locations
        self.registry:  dict = {}
        self.event_log: list = []
        self._last_sighting: dict = {}
        self._lock = threading.Lock()

    def _loc(self, cam_id) -> str:
        return self.cam_locations.get(cam_id, f"Cam{cam_id}")

    def update(self, person_name, cam_id, track_id, confidence: float = 0.0):
        if person_name == "Unknown":
            return
        location = self._loc(cam_id)
        now      = time.time()
        with self._lock:
            prev = self.registry.get(person_name)
            if prev and prev["cam_id"] != cam_id:
                elapsed = now - prev["last_seen"]
                if elapsed > 1.5:
                    prev_loc = self._loc(prev["cam_id"])
                    print(f"[HANDOFF] {person_name}  "
                          f"{prev_loc} → {location}  ({elapsed:.1f}s gap)")
                    self.event_log.append({
                        "type":      "handoff",
                        "person":    person_name,
                        "from_loc":  prev_loc,
                        "to_loc":    location,
                        "from_cam":  prev["cam_id"],
                        "to_cam":    cam_id,
                        "elapsed_s": round(elapsed, 1),
                        "timestamp": now,
                    })
            self.registry[person_name] = {
                "cam_id":     cam_id,
                "location":   location,
                "track_id":   track_id,
                "last_seen":  now,
                "confidence": round(confidence, 3),
            }
            last = self._last_sighting.get(person_name, 0)
            if now - last >= self.SIGHTING_DEBOUNCE_S:
                self._last_sighting[person_name] = now
                self.event_log.append({
                    "type":       "sighting",
                    "person":     person_name,
                    "location":   location,
                    "cam_id":     cam_id,
                    "confidence": round(confidence, 3),
                    "timestamp":  now,
                })
            cutoff = now - self.RETENTION_SECONDS
            self.event_log = [e for e in self.event_log if e["timestamp"] >= cutoff]

    def evict_stale_people(self):
        cutoff = time.time() - self.RETENTION_SECONDS
        with self._lock:
            stale = [n for n, info in self.registry.items()
                     if info["last_seen"] < cutoff]
            for n in stale:
                print(f"[IDENTITY] 🗑️  '{n}' evicted")
                del self.registry[n]

    def status_str(self) -> str:
        with self._lock:
            if not self.registry:
                return "No identities tracked yet"
            parts = []
            for name, info in self.registry.items():
                age = time.time() - info["last_seen"]
                parts.append(f"{name} @ {info['location']} ({age:.0f}s ago)")
            return " | ".join(parts)

    def recent_events(self, n: int = 20) -> list:
        cutoff = time.time() - self.RETENTION_SECONDS
        with self._lock:
            valid = [e for e in self.event_log if e["timestamp"] >= cutoff]
        return valid[-n:]

    def active_people(self) -> list:
        cutoff = time.time() - self.RETENTION_SECONDS
        with self._lock:
            return [
                {"name": name, **info}
                for name, info in self.registry.items()
                if info["last_seen"] >= cutoff
            ]


# ─────────────────────────────────────────────────────────────────────────────
# Per-camera frame capture thread
# ─────────────────────────────────────────────────────────────────────────────

class CameraReader(threading.Thread):
    """
    Dedicated thread per camera.

    Reads frames as fast as the camera produces them and always keeps only
    the LATEST frame in a 1-slot buffer. The inference thread picks up the
    latest frame whenever it's ready — old frames are never queued up.

    This is the key fix for lag: capture never waits for inference.
    """

    def __init__(self, cam_id, cap, loc_name):
        super().__init__(daemon=True, name=f"CamReader-{cam_id}")
        self.cam_id   = cam_id
        self.cap      = cap
        self.loc_name = loc_name

        self._frame: np.ndarray | None = None
        self._lock   = threading.Lock()
        self._stop   = threading.Event()
        self.online  = cap.isOpened()

    def run(self):
        while not self._stop.is_set():
            if not self.cap or not self.cap.isOpened():
                self.online = False
                time.sleep(0.1)
                continue

            ret, frame = self.cap.read()
            if not ret or frame is None:
                self.online = False
                time.sleep(0.05)
                continue

            self.online = True
            with self._lock:
                self._frame = frame   # always overwrite — no queue buildup

    def get_latest(self) -> np.ndarray | None:
        """Returns the most recent frame (or None). Never blocks."""
        with self._lock:
            return self._frame.copy() if self._frame is not None else None

    def replace_cap(self, new_cap):
        """Hot-swap the VideoCapture object after a reconnect."""
        with self._lock:
            self.cap    = new_cap
            self._frame = None
        self.online = new_cap.isOpened()

    def stop(self):
        self._stop.set()


# ─────────────────────────────────────────────────────────────────────────────
# Multi-Camera Tracker
# ─────────────────────────────────────────────────────────────────────────────

class MultiCameraTracker:
    """
    N-camera biometric tracker.

    Architecture (fixes the lag)
    ────────────────────────────
    One CameraReader thread per camera captures frames continuously and
    keeps only the LATEST frame in memory.  The main inference loop picks
    up that latest frame and runs models on it.  If models take 200ms,
    the capture thread has already moved on — no frame queue buildup,
    no lag spiral.

    PROCESS_EVERY_N_FRAMES   — run heavy models (face/body/gait) only on
    every Nth frame. Between those frames, the last known result is reused
    for drawing. Keeps display smooth at full camera FPS while inference
    runs at a sustainable rate.

    Typical settings:
        1 cam,  fast GPU  → PROCESS_EVERY_N_FRAMES = 1  (every frame)
        2 cams, mid GPU   → PROCESS_EVERY_N_FRAMES = 2
        3 cams, any GPU   → PROCESS_EVERY_N_FRAMES = 3
    """

    GAIT_BUFFER_SIZE        = 10
    PRED_BUFFER_SIZE        = 7
    RETRY_INTERVAL          = 3.0
    CACHE_DEPOSIT_THRESHOLD = 0.55
    PROCESS_EVERY_N_FRAMES  = 2   # ← tune this: higher = smoother display, less accuracy

    def __init__(self, cam_sources: dict, cam_locations: dict = None):
        print("\n[MULTI] Loading shared models...")
        self.detector   = PersonDetector()
        self.face_model = FaceRecognizer()
        self.reid_model = ReIDModel()
        self.gait_model = GaitModel()
        self.matcher    = Matcher()
        print("[MULTI] Models ready.\n")

        self.cam_sources   = cam_sources
        self.cam_locations = cam_locations or {
            cid: f"Cam{cid}" for cid in cam_sources
        }

        print("[MULTI] Camera → Location mapping:")
        for cid, loc in self.cam_locations.items():
            src = cam_sources.get(cid, "?")
            print(f"         Cam{cid} → '{loc}'  (source={src})")
        print()

        self.identity_manager = GlobalIdentityManager(self.cam_locations)
        self.id_cache         = SharedIdentityCache()

        self._last_retry:   dict = {}
        self.cameras:       dict = {}   # cam_id → VideoCapture
        self._readers:      dict = {}   # cam_id → CameraReader thread
        self.gait_buffers:  dict = {}
        self.pred_buffers:  dict = {}

        # Last known results per cam — reused on skipped frames
        self._last_results: dict = {}   # cam_id → list of result dicts
        self._frame_counters: dict = {} # cam_id → int

        # Latest annotated frame per cam_id for dashboard
        self._frame_buffer:      dict = {}
        self._frame_buffer_lock  = threading.Lock()

        self._stop_event  = threading.Event()
        self._thread: threading.Thread | None = None

        # Open cameras and start reader threads
        for cam_id, source in cam_sources.items():
            cap = self._open_camera(cam_id, source)
            self.cameras[cam_id]       = cap
            self._readers[cam_id]      = CameraReader(cam_id, cap, self._loc(cam_id))
            self._last_results[cam_id] = []
            self._frame_counters[cam_id] = 0
            self._readers[cam_id].start()

    # ── Location helper ───────────────────────────────────────────────────────
    def _loc(self, cam_id) -> str:
        return self.cam_locations.get(cam_id, f"Cam{cam_id}")

    # ── Camera open / retry ───────────────────────────────────────────────────
    def _open_camera(self, cam_id, source) -> cv2.VideoCapture:
        loc = self._loc(cam_id)
        try:
            if isinstance(source, str):
                cap = cv2.VideoCapture(source, cv2.CAP_FFMPEG)
                cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 3000)
                cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, 3000)
            else:
                cap = cv2.VideoCapture(source)

            if isinstance(source, str) and not cap.isOpened():
                time.sleep(0.5)
                cap = cv2.VideoCapture(source, cv2.CAP_FFMPEG)

            if cap.isOpened():
                print(f"[MULTI] ✅ '{loc}' opened  (source={source})")
            else:
                print(f"[MULTI] ❌ '{loc}' unavailable — will retry every "
                      f"{self.RETRY_INTERVAL:.0f}s")
        except Exception as e:
            print(f"[MULTI] ❌ '{loc}' error: {e}")
            cap = cv2.VideoCapture()
        return cap

    def _maybe_retry(self, cam_id) -> bool:
        now = time.time()
        if now - self._last_retry.get(cam_id, 0) < self.RETRY_INTERVAL:
            return False
        self._last_retry[cam_id] = now
        source = self.cam_sources[cam_id]
        print(f"[MULTI] 🔄 Retrying '{self._loc(cam_id)}'...")
        try:
            old = self.cameras.get(cam_id)
            if old:
                old.release()
        except Exception:
            pass
        cap = self._open_camera(cam_id, source)
        self.cameras[cam_id] = cap
        # Hot-swap into the reader thread — no restart needed
        if cam_id in self._readers:
            self._readers[cam_id].replace_cap(cap)
        return cap.isOpened()

    # ── Buffer helpers ────────────────────────────────────────────────────────
    def _gait_buf(self, cam_id, tid) -> deque:
        key = (cam_id, tid)
        if key not in self.gait_buffers:
            self.gait_buffers[key] = deque(maxlen=self.GAIT_BUFFER_SIZE)
        return self.gait_buffers[key]

    def _pred_buf(self, cam_id, tid) -> deque:
        key = (cam_id, tid)
        if key not in self.pred_buffers:
            self.pred_buffers[key] = deque(maxlen=self.PRED_BUFFER_SIZE)
        return self.pred_buffers[key]

    # ── Process one frame ─────────────────────────────────────────────────────
    def process_frame(self, frame, cam_id) -> list:
        detections = self.detector.detect(frame)
        results    = []
        loc        = self._loc(cam_id)

        for idx, det in enumerate(detections):
            track_id        = det.get("track_id") or idx
            x1, y1, x2, y2 = det["bbox"]
            h, w            = frame.shape[:2]
            pad             = 10
            x1 = max(0, x1 - pad);  y1 = max(0, y1 - pad)
            x2 = min(w, x2 + pad);  y2 = min(h, y2 + pad)

            person_crop = frame[y1:y2, x1:x2]
            if person_crop.size == 0:
                continue

            face_emb = None
            try:
                face_emb = self.face_model.get_embedding(person_crop)
            except Exception as e:
                print(f"[{loc}] face error: {e}")

            body_emb = None
            try:
                body_emb = self.reid_model.get_embedding(person_crop)
            except Exception as e:
                print(f"[{loc}] body error: {e}")

            gait_buf = self._gait_buf(cam_id, track_id)
            gait_buf.append(person_crop)
            gait_emb = None
            if len(gait_buf) >= 5:
                try:
                    gait_emb = self.gait_model.get_embedding(list(gait_buf))
                except Exception as e:
                    print(f"[{loc}] gait error: {e}")

            cloth_hist   = _clothing_histogram(person_crop)
            face_visible = face_emb is not None

            print(
                f"[{loc}] track={track_id} | "
                f"face={'✅' if face_visible else '❌'} | "
                f"body={'✅' if body_emb  is not None else '❌'} | "
                f"gait={'✅' if gait_emb  is not None else f'buf({len(gait_buf)}/5)'} | "
                f"cloth={'✅' if cloth_hist is not None else '❌'}"
            )

            cache_name, cache_sim = self.id_cache.query(
                face_emb, body_emb, gait_emb, cloth_hist
            )
            matcher_name, matcher_score = self.matcher.identify(
                face_emb=face_emb,
                body_emb=body_emb,
                gait_emb=gait_emb,
            )

            if matcher_name != "Unknown":
                name, score, source_tag = matcher_name, matcher_score, "matcher"
            elif cache_name is not None:
                name, score, source_tag = cache_name, cache_sim, "cache"
                view_note = " (no face)" if not face_visible else ""
                print(f"[CACHE] 🎯 {loc} track={track_id} → '{name}' "
                      f"(sim={cache_sim:.3f}){view_note}")
            else:
                name, score, source_tag = "Unknown", max(matcher_score, cache_sim), "none"

            if name != "Unknown" and score >= self.CACHE_DEPOSIT_THRESHOLD:
                self.id_cache.deposit(
                    name, cam_id, loc, score,
                    face_emb, body_emb, gait_emb, cloth_hist
                )

            pred_buf = self._pred_buf(cam_id, track_id)
            pred_buf.append(name)
            final_name = max(set(pred_buf), key=pred_buf.count)

            self.identity_manager.update(
                final_name, cam_id, track_id, confidence=score
            )

            results.append({
                "bbox":         (x1, y1, x2, y2),
                "name":         final_name,
                "score":        score,
                "track_id":     track_id,
                "cam_id":       cam_id,
                "location":     loc,
                "source_tag":   source_tag,
                "face_visible": face_visible,
            })

        return results

    # ── Draw results ──────────────────────────────────────────────────────────
    def draw_results(self, frame, results, cam_id) -> np.ndarray:
        loc = self._loc(cam_id)
        cv2.putText(frame, loc, (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 0), 2)

        for res in results:
            x1, y1, x2, y2 = res["bbox"]
            name    = res["name"]
            score   = res["score"]
            tid     = res["track_id"]
            src     = res.get("source_tag", "")
            no_face = not res.get("face_visible", True)

            if name == "Unknown":
                color = (0, 0, 255)
            elif src == "cache":
                color = (0, 140, 255)
            else:
                color = (0, 255, 0)

            label = f"[{tid}] {name} ({score:.2f})"
            if src == "cache":
                label += " [C]"
            if no_face and name != "Unknown":
                label += " [rear/side]"

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, max(y1 - 10, 15)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.52, color, 2)

        return frame

    # ── No-signal placeholder ─────────────────────────────────────────────────
    def _no_signal_panel(self, cam_id, w, h) -> np.ndarray:
        loc   = self._loc(cam_id)
        panel = np.zeros((h, w, 3), dtype=np.uint8)
        for i in range(0, h, 40):
            cv2.line(panel, (0, i), (w, i), (30, 30, 30), 1)
        for j in range(0, w, 40):
            cv2.line(panel, (j, 0), (j, h), (30, 30, 30), 1)
        cv2.putText(panel, loc,
                    (w // 2 - 80, h // 2 - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (80, 80, 80), 2)
        cv2.putText(panel, "NO SIGNAL — retrying...",
                    (w // 2 - 145, h // 2 + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 80, 200), 2)
        return panel

    # ── Dashboard data API ────────────────────────────────────────────────────
    def get_structured_data(self) -> dict:
        def _safe_opened(cap) -> bool:
            try:
                return cap is not None and cap.isOpened()
            except Exception:
                return False

        return {
            "active_people": self.identity_manager.active_people(),
            "events":        self.identity_manager.recent_events(50),
            "cameras": [
                {
                    "cam_id":   cid,
                    "location": self._loc(cid),
                    "online":   self._readers[cid].online
                               if cid in self._readers
                               else _safe_opened(self.cameras.get(cid)),
                }
                for cid in sorted(self.cam_sources.keys())
            ],
        }

    def get_frame(self, cam_id: int) -> bytes | None:
        try:
            with self._frame_buffer_lock:
                frame = self._frame_buffer.get(cam_id)
            if frame is None or frame.size == 0:
                return None
            ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
            return buf.tobytes() if ok else None
        except Exception:
            return None

    def get_all_frames(self) -> dict:
        result = {}
        for cid in sorted(self.cam_sources.keys()):
            data = self.get_frame(cid)
            if data is not None:
                result[cid] = data
        return result

    # ── Background thread API ─────────────────────────────────────────────────
    def start(self):
        if self._thread and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run_loop,
            args=(False,),
            name="MultiTrackerThread",
            daemon=True,
        )
        self._thread.start()
        print("[MULTI] 🚀 Background thread started.")

    def stop(self):
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=5.0)
        for reader in self._readers.values():
            reader.stop()
        print("[MULTI] 🛑 Stopped.")

    @property
    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    # ── Core loop ─────────────────────────────────────────────────────────────
    def _run_loop(self, display: bool = False):
        """
        Inference loop — completely decoupled from capture.

        Each iteration:
        1. Ask each CameraReader for its LATEST frame (non-blocking).
        2. Every PROCESS_EVERY_N_FRAMES, run models on that frame.
        3. On skipped frames, reuse the last result for drawing.
        4. Display / store annotated frames.

        Because capture runs in separate threads, this loop never stalls
        waiting for the camera — it always gets the freshest available frame.
        """
        DISPLAY_W     = 640
        DISPLAY_H     = 480
        evict_counter = 0

        while not self._stop_event.is_set():
            frames:      dict = {}
            all_results: dict = {}

            for cam_id in sorted(self.cameras.keys()):
                reader = self._readers.get(cam_id)

                # ── Reconnect dead cameras ────────────────────────────────
                if reader and not reader.online:
                    self._maybe_retry(cam_id)

                # ── Get latest frame from reader thread ───────────────────
                frame = reader.get_latest() if reader else None

                if frame is None:
                    frames[cam_id]      = self._no_signal_panel(cam_id, DISPLAY_W, DISPLAY_H)
                    all_results[cam_id] = self._last_results.get(cam_id, [])
                    continue

                # ── Frame skip logic ──────────────────────────────────────
                self._frame_counters[cam_id] = (
                    self._frame_counters.get(cam_id, 0) + 1
                )
                run_inference = (
                    self._frame_counters[cam_id] % self.PROCESS_EVERY_N_FRAMES == 0
                )

                if run_inference:
                    results = self.process_frame(frame, cam_id)
                    self._last_results[cam_id] = results
                else:
                    # Reuse last results — still draw on fresh frame
                    results = self._last_results.get(cam_id, [])

                frame               = self.draw_results(frame.copy(), results, cam_id)
                frames[cam_id]      = frame
                all_results[cam_id] = results

            # ── Store latest annotated frames for dashboard ───────────────
            with self._frame_buffer_lock:
                for cid, frm in frames.items():
                    self._frame_buffer[cid] = frm.copy()

            # ── Evict stale people every ~5 s ─────────────────────────────
            evict_counter += 1
            if evict_counter >= 150:
                self.identity_manager.evict_stale_people()
                evict_counter = 0

            if not display:
                continue

            # ── Combined OpenCV display ───────────────────────────────────
            panels = [
                cv2.resize(frames[cid], (DISPLAY_W, DISPLAY_H))
                for cid in sorted(frames.keys())
            ]
            if len(panels) > 1:
                div    = np.zeros((DISPLAY_H, 4, 3), dtype=np.uint8)
                div[:] = (255, 255, 0)
                combined = np.hstack([panels[0], div] + panels[1:])
            else:
                combined = panels[0]

            bar_h  = 44
            bar    = np.zeros((bar_h, combined.shape[1], 3), dtype=np.uint8)
            cv2.putText(bar, self.identity_manager.status_str(),
                        (8, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 200, 200), 1)
            cv2.putText(bar,
                        f"Cache [F=face B=body G=gait C=cloth]: "
                        f"{self.id_cache.summary()}",
                        (8, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (80, 200, 255), 1)
            combined = np.vstack([combined, bar])

            cv2.imshow("Multi-Camera Biometric Tracker", combined)
            if cv2.waitKey(1) & 0xFF == 27:
                break

    # ── Public blocking run ───────────────────────────────────────────────────
    def run(self):
        print("✅ Multi-camera tracker running — press ESC to quit\n")
        self._stop_event.clear()
        try:
            self._run_loop(display=True)
        finally:
            self._cleanup()

    # ── Cleanup ───────────────────────────────────────────────────────────────
    def _cleanup(self):
        print("\n[MULTI] Shutting down...")
        for reader in self._readers.values():
            reader.stop()
        for cam_id, cap in self.cameras.items():
            try:
                cap.release()
                print(f"[MULTI] '{self._loc(cam_id)}' released")
            except Exception:
                pass
        cv2.destroyAllWindows()
        print("[MULTI] Done.")