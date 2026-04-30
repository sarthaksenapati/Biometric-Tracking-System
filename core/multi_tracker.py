# core/multi_tracker.py

import cv2
import json
import os
import time
import threading
import numpy as np
from collections import deque

from models.detector import PersonDetector
from models.face_model import FaceRecognizer
from models.reid_model import ReIDModel
from models.gait_model import GaitModel
from core.matcher import Matcher
from utils.embeddings import save_embedding

# Database and queue support (optional)
USE_DATABASE = os.getenv("USE_DATABASE", "true").lower() == "true"
if USE_DATABASE:
    try:
        from db.models import Event as DBEvent, Camera as DBCamera, Detection as DBDetection
        from cache import get_cache

        print("[MULTI] ✅ Database modules loaded")
    except ImportError as e:
        print(f"[MULTI] ⚠️  Database import failed: {e}")
        USE_DATABASE = False

USE_QUEUE = os.getenv("USE_QUEUE", "false").lower() == "true"
if USE_QUEUE:
    try:
        from task_queue import get_queue

        print("[MULTI] ✅ Queue module loaded")
    except ImportError as e:
        print(f"[MULTI] ⚠️  Queue import failed: {e}")
        USE_QUEUE = False

ADMIN_CONTROL_FILE = "tracker_admin.json"


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
    hsv = cv2.cvtColor(torso, cv2.COLOR_BGR2HSV)
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
    THRESHOLD_NO_FACE = 0.62
    FRESH_TTL = 45.0
    _W = {"face": 0.50, "body": 0.30, "gait": 0.15, "cloth": 0.05}

    def __init__(self):
        self._store: dict = {}
        self._lock = threading.Lock()

    def deposit(self, name, cam_id, location, score, face_emb, body_emb, gait_emb, cloth_hist):
        if name == "Unknown" or name.startswith("Person_"):
            return
        with self._lock:
            existing = self._store.get(name)
            if existing is None or score > existing["score"]:
                self._store[name] = {
                    "face_emb": face_emb,
                    "body_emb": body_emb,
                    "gait_emb": gait_emb,
                    "cloth_hist": cloth_hist,
                    "score": score,
                    "cam_id": cam_id,
                    "location": location,
                    "timestamp": time.time(),
                }
                stored = [
                    m
                    for m, v in [("face", face_emb), ("body", body_emb), ("gait", gait_emb), ("cloth", cloth_hist)]
                    if v is not None
                ]
                print(f"[CACHE] 💾  '{name}' updated from {location} " f"(score={score:.3f}, stored={stored})")

    def query(self, face_emb, body_emb, gait_emb, cloth_hist):
        with self._lock:
            store_copy = dict(self._store)
        if not store_copy:
            return None, 0.0

        has_face = face_emb is not None
        threshold = self.THRESHOLD_WITH_FACE if has_face else self.THRESHOLD_NO_FACE
        best_name = None
        best_sim = 0.0
        now = time.time()

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
                best_sim = combined
                best_name = name

        if best_sim >= threshold:
            return best_name, best_sim
        return None, best_sim

    def clear(self):
        with self._lock:
            self._store.clear()
        print("[CACHE] 🧹 Cleared")

    def summary(self) -> str:
        with self._lock:
            store_copy = dict(self._store)
        if not store_copy:
            return "cache empty"
        now = time.time()
        parts = []
        for name, e in store_copy.items():
            age = now - e["timestamp"]
            tag = "".join(
                [
                    "F" if e["face_emb"] is not None else "_",
                    "B" if e["body_emb"] is not None else "_",
                    "G" if e["gait_emb"] is not None else "_",
                    "C" if e["cloth_hist"] is not None else "_",
                ]
            )
            parts.append(f"{name}[{tag}]({age:.0f}s)")
        return " | ".join(parts)


# ─────────────────────────────────────────────────────────────────────────────
# Auto Enroller
# ─────────────────────────────────────────────────────────────────────────────


class AutoEnroller:
    """
    Automatically registers unknown persons into embeddings_db/.

    KEY FIX — ghost candidate prevention
    ─────────────────────────────────────
    Old behaviour: _load_names() created empty candidates for every Person_N
    in unknown_persons.json. These ghost candidates had no embeddings, so
    _best_candidate_match() skipped them — but they still took up slots and
    confused the counter. Worse, a new track arriving would not match any
    ghost (no embeddings to compare), creating yet another Person_N.

    New behaviour: _load_names() only restores the counter and display names.
    It does NOT create candidate objects. Candidates are only created when
    a track actually arrives and needs one. This means:
      - No ghost candidates on startup
      - Counter continues from where it left off (no duplicate IDs)
      - Display names from the JSON are applied when a candidate IS created

    KEY FIX — track-stable buffering
    ──────────────────────────────────
    When detector assigns a new track_id to the same person (they moved,
    re-entered frame), the new track is first compared against all existing
    candidates by face+body similarity. If a match is found above threshold,
    the new track is linked to the existing candidate — embeddings keep
    accumulating on the same person rather than restarting.
    """

    MATCH_WITH_FACE = 0.68
    MATCH_BODY_ONLY = 0.78
    FACE_TARGET = 12
    BODY_TARGET = 12
    NAMES_FILE = "unknown_persons.json"

    def __init__(self, matcher: "Matcher"):
        self._matcher = matcher
        self._counter = 0
        self._lock = threading.Lock()
        self._candidates: dict = {}  # label → candidate dict
        self._track_map: dict = {}  # (cam_id, track_id) → label
        self._name_overrides: dict = {}  # label → display_name from JSON

        self._load_names()

    # ── Name persistence ──────────────────────────────────────────────────────

    def _load_names(self):
        """
        Load display-name overrides and restore counter.
        Does NOT create ghost candidates — only records the name mapping
        so it can be applied when a candidate is actually created.
        """
        if not os.path.exists(self.NAMES_FILE):
            return
        try:
            with open(self.NAMES_FILE) as f:
                data = json.load(f)

            for label, display_name in data.items():
                # Restore counter so new IDs don't collide
                try:
                    num = int(label.split("_")[1]) if "_" in label else 0
                    self._counter = max(self._counter, num)
                except (ValueError, IndexError):
                    pass

                # Store name override — applied when candidate is created
                self._name_overrides[label] = display_name

                # If candidate already exists in memory, update its name
                if label in self._candidates:
                    self._candidates[label]["display_name"] = display_name

            if data:
                print(
                    f"[AutoEnroller] Loaded {len(data)} name overrides "
                    f"from {self.NAMES_FILE} (counter={self._counter})"
                )
        except Exception as e:
            print(f"[AutoEnroller] Could not load names: {e}")

    def _save_names(self):
        try:
            data = {}
            # Save all candidates
            for label, c in self._candidates.items():
                data[label] = c["display_name"]
            # Also save overrides for labels not yet active as candidates
            for label, name in self._name_overrides.items():
                if label not in data:
                    data[label] = name
            with open(self.NAMES_FILE, "w") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            print(f"[AutoEnroller] Could not save names: {e}")

    # ── Public: rename ────────────────────────────────────────────────────────

    def rename(self, label: str, new_name: str) -> bool:
        """Rename a Person_N live. Also renames embedding files if promoted."""
        with self._lock:
            # Update name override regardless
            self._name_overrides[label] = new_name

            if label not in self._candidates:
                # Candidate not in memory yet — save override for when it appears
                self._save_names()
                print(f"[AutoEnroller] ✏️  Name override saved: " f"'{label}' → '{new_name}'")
                return True

            old_name = self._candidates[label]["display_name"]
            self._candidates[label]["display_name"] = new_name

            if self._candidates[label]["promoted"]:
                self._rename_embedding_files(old_name, new_name)

        self._save_names()
        print(f"[AutoEnroller] ✏️  '{label}' renamed: '{old_name}' → '{new_name}'")
        return True

    def _rename_embedding_files(self, old_name: str, new_name: str):
        for kind in ("face", "body", "gait"):
            old_path = os.path.join("embeddings_db", f"{old_name}_{kind}.npy")
            new_path = os.path.join("embeddings_db", f"{new_name}_{kind}.npy")
            if os.path.exists(old_path):
                try:
                    os.rename(old_path, new_path)
                    print(f"[AutoEnroller] 📁 {old_path} → {new_path}")
                except Exception as e:
                    print(f"[AutoEnroller] Rename failed for {kind}: {e}")
        try:
            self._matcher.reload()
            print(f"[AutoEnroller] 🔄 Matcher reloaded after rename")
        except Exception as e:
            print(f"[AutoEnroller] Matcher reload failed: {e}")

    def list_persons(self) -> list:
        with self._lock:
            now = time.time()
            result = []
            for label, c in self._candidates.items():
                age = (now - c["last_seen"]) if c["last_seen"] else None
                result.append(
                    {
                        "label": label,
                        "display_name": c["display_name"],
                        "sighting_count": c["sighting_count"],
                        "face_collected": len(c["face_embs"]),
                        "body_collected": len(c["body_embs"]),
                        "face_target": self.FACE_TARGET,
                        "body_target": self.BODY_TARGET,
                        "promoted": c["promoted"],
                        "age_s": round(age, 1) if age else None,
                        "cam_id": c["cam_id"],
                    }
                )
        return result

    def reset(self):
        """Clear all candidates and track mappings. Called by admin controls."""
        with self._lock:
            self._candidates.clear()
            self._track_map.clear()
            self._counter = 0
            self._name_overrides.clear()
        if os.path.exists(self.NAMES_FILE):
            os.remove(self.NAMES_FILE)
        print("[AutoEnroller] 🧹 Reset — all candidates cleared")

    # ── Internal: find best candidate match ───────────────────────────────────

    def _best_candidate_match(
        self,
        face_emb: np.ndarray | None,
        body_emb: np.ndarray | None,
    ) -> tuple[str | None, float]:
        """
        Find the best matching existing candidate.
        Only candidates with at least one embedding are considered —
        ghost candidates (empty stacks) are automatically excluded.
        """
        has_face = face_emb is not None
        threshold = self.MATCH_WITH_FACE if has_face else self.MATCH_BODY_ONLY

        best_label = None
        best_sim = 0.0

        for label, c in self._candidates.items():
            rep_face = None
            rep_body = None

            if c["face_embs"]:
                rep_face = np.mean(c["face_embs"], axis=0)
                n = np.linalg.norm(rep_face)
                rep_face = rep_face / n if n > 1e-8 else rep_face

            if c["body_embs"]:
                rep_body = np.mean(c["body_embs"], axis=0)
                n = np.linalg.norm(rep_body)
                rep_body = rep_body / n if n > 1e-8 else rep_body

            # Skip candidates with NO embeddings — they cannot be matched
            if rep_face is None and rep_body is None:
                continue

            sims, weights = [], []
            if face_emb is not None and rep_face is not None:
                sims.append(_cosine_sim(face_emb, rep_face))
                weights.append(0.65)
            if body_emb is not None and rep_body is not None:
                sims.append(_cosine_sim(body_emb, rep_body))
                weights.append(0.35)
            if not sims:
                continue

            combined = sum(s * w for s, w in zip(sims, weights)) / sum(weights)
            if combined > best_sim:
                best_sim = combined
                best_label = label

        if best_sim >= threshold:
            return best_label, best_sim
        return None, best_sim

    # ── Main entry ────────────────────────────────────────────────────────────

    def query_or_enroll(
        self,
        cam_id: int,
        track_id: int,
        face_emb: np.ndarray | None,
        body_emb: np.ndarray | None,
    ) -> tuple[str, float]:
        if face_emb is None and body_emb is None:
            return "Unknown", 0.0

        track_key = (cam_id, track_id)
        now = time.time()

        with self._lock:
            # ── Step 1: look up existing track assignment ─────────────────
            label = self._track_map.get(track_key)

            if label is None or label not in self._candidates:
                # ── Step 2: search existing candidates by similarity ──────
                label, sim = self._best_candidate_match(face_emb, body_emb)

                if label is not None:
                    self._track_map[track_key] = label
                    print(f"[AutoEnroller] 🔗 Cam{cam_id} track={track_id} → " f"'{label}' re-linked (sim={sim:.3f})")
                else:
                    # ── Step 3: create new candidate ──────────────────────
                    self._counter += 1
                    label = f"Person_{self._counter}"

                    # Apply saved name override if one exists
                    display_name = self._name_overrides.get(label, label)

                    self._candidates[label] = {
                        "face_embs": [],
                        "body_embs": [],
                        "display_name": display_name,
                        "promoted": False,
                        "first_seen": now,
                        "last_seen": now,
                        "sighting_count": 0,
                        "cam_id": cam_id,
                    }
                    self._track_map[track_key] = label
                    self._save_names()
                    print(
                        f"[AutoEnroller] 🆕 New candidate '{label}' "
                        f"({display_name}) from Cam{cam_id} track={track_id}"
                    )

            c = self._candidates[label]
            c["last_seen"] = now
            c["sighting_count"] += 1
            c["cam_id"] = cam_id

            # ── Step 4: collect embeddings ────────────────────────────────
            if not c["promoted"]:
                if face_emb is not None and len(c["face_embs"]) < self.FACE_TARGET:
                    c["face_embs"].append(face_emb.copy())
                if body_emb is not None and len(c["body_embs"]) < self.BODY_TARGET:
                    c["body_embs"].append(body_emb.copy())

                f_n = len(c["face_embs"])
                b_n = len(c["body_embs"])

                print(
                    f"[AutoEnroller] ⏳ '{label}' ({c['display_name']}) "
                    f"face={f_n}/{self.FACE_TARGET} "
                    f"body={b_n}/{self.BODY_TARGET}"
                )

                # ── Step 5: promote when both stacks full ─────────────────
                if f_n >= self.FACE_TARGET and b_n >= self.BODY_TARGET:
                    self._promote(label, c)

            display_name = c["display_name"]
            score = 0.6 if c["promoted"] else 0.5

        return display_name, score

    # ── Promote ───────────────────────────────────────────────────────────────

    def _promote(self, label: str, c: dict):
        display_name = c["display_name"]
        face_stack = np.stack(c["face_embs"])
        body_stack = np.stack(c["body_embs"])
        try:
            save_embedding(display_name, face_stack, "face")
            save_embedding(display_name, body_stack, "body")
            c["promoted"] = True
            self._save_names()
            self._matcher.reload()
            print(
                f"[AutoEnroller] ✅ '{label}' ({display_name}) PROMOTED → "
                f"embeddings_db/  face={face_stack.shape} body={body_stack.shape}"
            )
            print(f"[AutoEnroller]    '{display_name}' now shows as green box")
            print(f"[AutoEnroller]    To rename: " f"python -m utils.admin_controls rename {label} NewName")
        except Exception as e:
            print(f"[AutoEnroller] ❌ Promotion failed for '{label}': {e}")

    def force_promote(self, label: str, min_samples: int = 3) -> bool:
        """Force promote a candidate early (admin command)."""
        with self._lock:
            c = self._candidates.get(label)
            if not c:
                print(f"[AutoEnroller] '{label}' not found")
                return False
            f_n = len(c["face_embs"])
            b_n = len(c["body_embs"])
            if f_n < min_samples or b_n < min_samples:
                print(
                    f"[AutoEnroller] Not enough samples to force promote "
                    f"'{label}' (face={f_n}/{min_samples}, body={b_n}/{min_samples})"
                )
                return False
            self._promote(label, c)
        return True

    def summary(self) -> str:
        with self._lock:
            if not self._candidates:
                return "none"
            parts = []
            for label, c in self._candidates.items():
                dn = c["display_name"]
                prom = (
                    "✓"
                    if c["promoted"]
                    else f"F{len(c['face_embs'])}/{self.FACE_TARGET}" f"·B{len(c['body_embs'])}/{self.BODY_TARGET}"
                )
                parts.append(f"{dn}[{prom}]")
            return " | ".join(parts)


# ─────────────────────────────────────────────────────────────────────────────
# Global Identity Manager
# ─────────────────────────────────────────────────────────────────────────────


class GlobalIdentityManager:
    RETENTION_SECONDS = 300
    SIGHTING_DEBOUNCE_S = 2.0

    def __init__(self, cam_locations: dict):
        self.cam_locations = cam_locations
        self.registry: dict = {}
        self.event_log: list = []  # In-memory fallback
        self._last_sighting: dict = {}
        self._lock = threading.Lock()
        self._use_db = USE_DATABASE
        self._cache = get_cache() if USE_DATABASE else None

    def _loc(self, cam_id) -> str:
        return self.cam_locations.get(cam_id, f"Cam{cam_id}")

    def update(self, person_name, cam_id, track_id, confidence: float = 0.0):
        if person_name == "Unknown":
            return
        location = self._loc(cam_id)
        now = time.time()
        with self._lock:
            prev = self.registry.get(person_name)
            if prev and prev["cam_id"] != cam_id:
                elapsed = now - prev["last_seen"]
                if elapsed > 1.5:
                    prev_loc = self._loc(prev["cam_id"])
                    print(f"[HANDOFF] {person_name}  " f"{prev_loc} → {location}  ({elapsed:.1f}s gap)")
                    event_data = {
                        "type": "handoff",
                        "person": person_name,
                        "from_loc": prev_loc,
                        "to_loc": location,
                        "from_cam": prev["cam_id"],
                        "to_cam": cam_id,
                        "elapsed_s": round(elapsed, 1),
                        "timestamp": now,
                    }
                    if self._use_db:
                        try:
                            DBEvent.log(**event_data)
                        except Exception as e:
                            print(f"[IDENTITY] DB log failed: {e}")
                            self.event_log.append(event_data)
                    else:
                        self.event_log.append(event_data)
            self.registry[person_name] = {
                "cam_id": cam_id,
                "location": location,
                "track_id": track_id,
                "last_seen": now,
                "confidence": round(confidence, 3),
            }
            last = self._last_sighting.get(person_name, 0)
            if now - last >= self.SIGHTING_DEBOUNCE_S:
                self._last_sighting[person_name] = now
                event_data = {
                    "type": "sighting",
                    "person": person_name,
                    "location": location,
                    "cam_id": cam_id,
                    "confidence": round(confidence, 3),
                    "timestamp": now,
                }
                if self._use_db:
                    try:
                        DBEvent.log(**event_data)
                        # Also save detection
                        DBDetection.save(person_name, cam_id, track_id, confidence, [], location, now)
                    except Exception as e:
                        print(f"[IDENTITY] DB log failed: {e}")
                        self.event_log.append(event_data)
                else:
                    self.event_log.append(event_data)
            # Cleanup in-memory events if not using DB
            if not self._use_db:
                cutoff = now - self.RETENTION_SECONDS
                self.event_log = [e for e in self.event_log if e["timestamp"] >= cutoff]

    def evict_stale_people(self):
        cutoff = time.time() - self.RETENTION_SECONDS
        with self._lock:
            stale = [n for n, info in self.registry.items() if info["last_seen"] < cutoff]
            for n in stale:
                print(f"[IDENTITY] 🗑️  '{n}' evicted")
                del self.registry[n]

        # Also prune old events from DB
        if self._use_db:
            try:
                DBEvent.prune_old(cutoff)
                DBDetection.prune_old(cutoff)
            except Exception as e:
                print(f"[IDENTITY] DB prune failed: {e}")

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
        if self._use_db:
            try:
                since = time.time() - self.RETENTION_SECONDS
                return DBEvent.get_recent(limit=n, since=since)
            except Exception as e:
                print(f"[IDENTITY] DB query failed: {e}")
        # Fallback to in-memory
        cutoff = time.time() - self.RETENTION_SECONDS
        with self._lock:
            valid = [e for e in self.event_log if e["timestamp"] >= cutoff]
        return valid[-n:]

    def active_people(self) -> list:
        cutoff = time.time() - self.RETENTION_SECONDS
        with self._lock:
            return [{"name": name, **info} for name, info in self.registry.items() if info["last_seen"] >= cutoff]


# ─────────────────────────────────────────────────────────────────────────────
# Per-camera frame capture thread
# ─────────────────────────────────────────────────────────────────────────────


class CameraReader(threading.Thread):
    def __init__(self, cam_id, cap, loc_name):
        super().__init__(daemon=True, name=f"CamReader-{cam_id}")
        self.cam_id = cam_id
        self.cap = cap
        self.loc_name = loc_name
        self._frame: np.ndarray | None = None
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self.online = cap.isOpened()

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
                self._frame = frame

    def get_latest(self) -> np.ndarray | None:
        with self._lock:
            return self._frame.copy() if self._frame is not None else None

    def replace_cap(self, new_cap):
        with self._lock:
            self.cap = new_cap
            self._frame = None
        self.online = new_cap.isOpened()

    def stop(self):
        self._stop.set()


# ─────────────────────────────────────────────────────────────────────────────
# Multi-Camera Tracker
# ─────────────────────────────────────────────────────────────────────────────


class MultiCameraTracker:

    GAIT_BUFFER_SIZE = 10
    PRED_BUFFER_SIZE = 7
    RETRY_INTERVAL = 3.0
    CACHE_DEPOSIT_THRESHOLD = 0.55
    PROCESS_EVERY_N_FRAMES = 2

    def __init__(self, cam_sources: dict, cam_locations: dict = None):
        print("\n[MULTI] Loading shared models...")
        self.detector = PersonDetector()
        self.face_model = FaceRecognizer()
        self.reid_model = ReIDModel()
        self.gait_model = GaitModel()
        self.matcher = Matcher()
        print("[MULTI] Models ready.\n")

        self.cam_sources = cam_sources
        self.cam_locations = cam_locations or {cid: f"Cam{cid}" for cid in cam_sources}

        print("[MULTI] Camera → Location mapping:")
        for cid, loc in self.cam_locations.items():
            src = cam_sources.get(cid, "?")
            print(f"         Cam{cid} → '{loc}'  (source={src})")
        print()

        self.identity_manager = GlobalIdentityManager(self.cam_locations)
        self.id_cache = SharedIdentityCache()
        self.auto_enroller = AutoEnroller(self.matcher)

        self._last_retry: dict = {}
        self.cameras: dict = {}
        self._readers: dict = {}
        self.gait_buffers: dict = {}
        self.pred_buffers: dict = {}
        self._last_results: dict = {}
        self._frame_counters: dict = {}
        self._frame_buffer: dict = {}
        self._frame_buffer_lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

        for cam_id, source in cam_sources.items():
            cap = self._open_camera(cam_id, source)
            self.cameras[cam_id] = cap
            self._readers[cam_id] = CameraReader(cam_id, cap, self._loc(cam_id))
            self._last_results[cam_id] = []
            self._frame_counters[cam_id] = 0
            self._readers[cam_id].start()

    def _loc(self, cam_id) -> str:
        return self.cam_locations.get(cam_id, f"Cam{cam_id}")

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
                print(f"[MULTI] ❌ '{loc}' unavailable — will retry every " f"{self.RETRY_INTERVAL:.0f}s")
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
        if cam_id in self._readers:
            self._readers[cam_id].replace_cap(cap)
        return cap.isOpened()

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

    # ── Admin controls ────────────────────────────────────────────────────────

    def _check_admin_commands(self):
        """
        Poll ADMIN_CONTROL_FILE for runtime commands sent from a second terminal.
        Commands are written by: python -m utils.admin_controls <command>

        Supported commands:
          reset_enroller   — clear all Person_N candidates, start fresh
          clear_cache      — clear the session identity cache
          rename           — rename a Person_N to a real name
          force_promote    — promote a candidate early (needs ≥3 samples)
          list             — print all current candidates to terminal
        """
        if not os.path.exists(ADMIN_CONTROL_FILE):
            return
        try:
            with open(ADMIN_CONTROL_FILE) as f:
                cmd = json.load(f)
            os.remove(ADMIN_CONTROL_FILE)

            action = cmd.get("action")
            print(f"[ADMIN] 📨 Command received: {action}")

            if action == "reset_enroller":
                self.auto_enroller.reset()

            elif action == "clear_cache":
                self.id_cache.clear()

            elif action == "rename":
                label = cmd.get("label", "")
                new_name = cmd.get("new_name", "")
                if label and new_name:
                    self.auto_enroller.rename(label, new_name)
                else:
                    print("[ADMIN] ❌ rename requires 'label' and 'new_name'")

            elif action == "force_promote":
                label = cmd.get("label", "")
                if label:
                    ok = self.auto_enroller.force_promote(label, min_samples=3)
                    if not ok:
                        print(f"[ADMIN] ❌ Could not force promote '{label}'")
                else:
                    print("[ADMIN] ❌ force_promote requires 'label'")

            elif action == "list":
                persons = self.auto_enroller.list_persons()
                if not persons:
                    print("[ADMIN] No candidates currently enrolling")
                else:
                    print("\n[ADMIN] Current enrolling persons:")
                    for p in persons:
                        status = (
                            "PROMOTED ✅"
                            if p["promoted"]
                            else f"face={p['face_collected']}/{p['face_target']} "
                            f"body={p['body_collected']}/{p['body_target']}"
                        )
                        print(
                            f"  {p['label']:12s} → '{p['display_name']:15s}' "
                            f"[{status}] "
                            f"seen={p['sighting_count']} "
                            f"age={p['age_s']}s "
                            f"cam={p['cam_id']}"
                        )
                    print()

            else:
                print(f"[ADMIN] ❌ Unknown action: '{action}'")

        except Exception as e:
            print(f"[ADMIN] Error: {e}")
            try:
                os.remove(ADMIN_CONTROL_FILE)
            except Exception:
                pass

    # ── Core: process one frame ───────────────────────────────────────────────

    def process_frame(self, frame, cam_id) -> list:
        detections = self.detector.detect(frame)
        results = []
        loc = self._loc(cam_id)

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

            face_emb = None
            try:
                face_emb = self.face_model.get_embedding(person_crop)
                if face_emb is not None:
                    face_emb = face_emb / (np.linalg.norm(face_emb) + 1e-8)
            except Exception as e:
                print(f"[{loc}] face error: {e}")

            body_emb = None
            try:
                body_emb = self.reid_model.get_embedding(person_crop)
                if body_emb is not None:
                    body_emb = body_emb / (np.linalg.norm(body_emb) + 1e-8)
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

            cloth_hist = _clothing_histogram(person_crop)
            face_visible = face_emb is not None

            print(
                f"[{loc}] track={track_id} | "
                f"face={'✅' if face_visible else '❌'} | "
                f"body={'✅' if body_emb  is not None else '❌'} | "
                f"gait={'✅' if gait_emb  is not None else f'buf({len(gait_buf)}/5)'} | "
                f"cloth={'✅' if cloth_hist is not None else '❌'}"
            )

            # ── Step 1: Matcher ───────────────────────────────────────────
            matcher_name, matcher_score = self.matcher.identify(
                face_emb=face_emb,
                body_emb=body_emb,
                gait_emb=gait_emb,
            )

            # ── Step 2: Cache ─────────────────────────────────────────────
            cache_name, cache_sim = self.id_cache.query(face_emb, body_emb, gait_emb, cloth_hist)

            # ── Step 3: Resolve identity ──────────────────────────────────
            if matcher_name != "Unknown":
                name, score, source_tag = matcher_name, matcher_score, "matcher"

            elif cache_name is not None:
                name, score, source_tag = cache_name, cache_sim, "cache"
                view_note = " (no face)" if not face_visible else ""
                print(f"[CACHE] 🎯 {loc} track={track_id} → '{name}' " f"(sim={cache_sim:.3f}){view_note}")

            else:
                # ── Step 4: Auto-enroller ─────────────────────────────────
                enroll_name, enroll_score = self.auto_enroller.query_or_enroll(cam_id, track_id, face_emb, body_emb)
                if enroll_name != "Unknown":
                    name, score, source_tag = enroll_name, enroll_score, "enrolling"
                else:
                    name, score, source_tag = "Unknown", 0.0, "none"

            # ── Step 5: Cache deposit (known persons only) ────────────────
            if name != "Unknown" and source_tag in ("matcher", "cache") and score >= self.CACHE_DEPOSIT_THRESHOLD:
                self.id_cache.deposit(name, cam_id, loc, score, face_emb, body_emb, gait_emb, cloth_hist)

            # ── Smooth predictions ────────────────────────────────────────
            pred_buf = self._pred_buf(cam_id, track_id)
            pred_buf.append(name)
            final_name = max(set(pred_buf), key=pred_buf.count)

            self.identity_manager.update(final_name, cam_id, track_id, confidence=score)

            results.append(
                {
                    "bbox": (x1, y1, x2, y2),
                    "name": final_name,
                    "score": score,
                    "track_id": track_id,
                    "cam_id": cam_id,
                    "location": loc,
                    "source_tag": source_tag,
                    "face_visible": face_visible,
                }
            )

        return results

    # ── Draw results ──────────────────────────────────────────────────────────

    def draw_results(self, frame, results, cam_id) -> np.ndarray:
        loc = self._loc(cam_id)
        cv2.putText(frame, loc, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 0), 2)

        for res in results:
            x1, y1, x2, y2 = res["bbox"]
            name = res["name"]
            score = res["score"]
            tid = res["track_id"]
            src = res.get("source_tag", "")
            no_face = not res.get("face_visible", True)

            if name == "Unknown":
                color = (0, 0, 255)
            elif src == "enrolling":
                color = (255, 255, 0)
            elif src == "cache":
                color = (0, 140, 255)
            else:
                color = (0, 255, 0)

            label = f"[{tid}] {name} ({score:.2f})"
            if src == "cache":
                label += " [C]"
            if src == "enrolling":
                label += " [~]"
            if no_face and name != "Unknown":
                label += " [rear/side]"

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, max(y1 - 10, 15)), cv2.FONT_HERSHEY_SIMPLEX, 0.52, color, 2)

        return frame

    def _no_signal_panel(self, cam_id, w, h) -> np.ndarray:
        loc = self._loc(cam_id)
        panel = np.zeros((h, w, 3), dtype=np.uint8)
        for i in range(0, h, 40):
            cv2.line(panel, (0, i), (w, i), (30, 30, 30), 1)
        for j in range(0, w, 40):
            cv2.line(panel, (j, 0), (j, h), (30, 30, 30), 1)
        cv2.putText(panel, loc, (w // 2 - 80, h // 2 - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (80, 80, 80), 2)
        cv2.putText(
            panel,
            "NO SIGNAL — retrying...",
            (w // 2 - 145, h // 2 + 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 80, 200),
            2,
        )
        return panel

    # ── Dashboard API ─────────────────────────────────────────────────────────

    def get_structured_data(self) -> dict:
        def _safe_opened(cap) -> bool:
            try:
                return cap is not None and cap.isOpened()
            except Exception:
                return False

        return {
            "active_people": self.identity_manager.active_people(),
            "events": self.identity_manager.recent_events(50),
            "enrolling_persons": self.auto_enroller.list_persons(),
            "cameras": [
                {
                    "cam_id": cid,
                    "location": self._loc(cid),
                    "online": (
                        self._readers[cid].online if cid in self._readers else _safe_opened(self.cameras.get(cid))
                    ),
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

    # ── Background thread ─────────────────────────────────────────────────────

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

    # ── Core inference loop ───────────────────────────────────────────────────

    def _run_loop(self, display: bool = False):
        DISPLAY_W = 640
        DISPLAY_H = 480
        evict_counter = 0
        admin_counter = 0

        while not self._stop_event.is_set():
            frames: dict = {}
            all_results: dict = {}

            # Check admin commands every ~10 iterations
            admin_counter += 1
            if admin_counter >= 10:
                self._check_admin_commands()
                admin_counter = 0

            for cam_id in sorted(self.cameras.keys()):
                reader = self._readers.get(cam_id)

                if reader and not reader.online:
                    self._maybe_retry(cam_id)

                frame = reader.get_latest() if reader else None

                if frame is None:
                    frames[cam_id] = self._no_signal_panel(cam_id, DISPLAY_W, DISPLAY_H)
                    all_results[cam_id] = self._last_results.get(cam_id, [])
                    continue

                self._frame_counters[cam_id] = self._frame_counters.get(cam_id, 0) + 1
                run_inference = self._frame_counters[cam_id] % self.PROCESS_EVERY_N_FRAMES == 0

                if run_inference:
                    results = self.process_frame(frame, cam_id)
                    self._last_results[cam_id] = results
                else:
                    results = self._last_results.get(cam_id, [])

                frame = self.draw_results(frame.copy(), results, cam_id)
                frames[cam_id] = frame
                all_results[cam_id] = results

            with self._frame_buffer_lock:
                for cid, frm in frames.items():
                    self._frame_buffer[cid] = frm.copy()

            evict_counter += 1
            if evict_counter >= 150:
                self.identity_manager.evict_stale_people()
                evict_counter = 0

            if not display:
                continue

            panels = [cv2.resize(frames[cid], (DISPLAY_W, DISPLAY_H)) for cid in sorted(frames.keys())]
            if len(panels) > 1:
                div = np.zeros((DISPLAY_H, 4, 3), dtype=np.uint8)
                div[:] = (255, 255, 0)
                combined = np.hstack([panels[0], div] + panels[1:])
            else:
                combined = panels[0]

            bar_h = 64
            bar = np.zeros((bar_h, combined.shape[1], 3), dtype=np.uint8)
            cv2.putText(
                bar, self.identity_manager.status_str(), (8, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 200, 200), 1
            )
            cv2.putText(
                bar, f"Cache: {self.id_cache.summary()}", (8, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (80, 200, 255), 1
            )
            cv2.putText(
                bar,
                f"Enrolling: {self.auto_enroller.summary()}",
                (8, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.38,
                (255, 255, 0),
                1,
            )
            combined = np.vstack([combined, bar])

            cv2.imshow("Multi-Camera Biometric Tracker", combined)
            if cv2.waitKey(1) & 0xFF == 27:
                break

    def run(self):
        print("✅ Multi-camera tracker running — press ESC to quit\n")
        self._stop_event.clear()
        try:
            self._run_loop(display=True)
        finally:
            self._cleanup()

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
