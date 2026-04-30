# core/matcher.py

import os
import numpy as np
from core.fusion_engine import FusionEngine
from db.models import Embedding as DBEmbedding, Person as DBPerson
from cache import get_cache

# Flag to use database or file-based storage
USE_DATABASE = os.getenv("USE_DATABASE", "true").lower() == "true"


def cosine_similarity(a, b):
    """
    Cosine similarity between query vector a and reference b.
    b can be shape (512,)  — single exemplar
    b can be shape (N, 512) — multiple exemplars (Fix 1)
    In the multi-exemplar case, returns the MAX similarity across all stored samples.
    """
    if a is None or b is None:
        return None

    a = np.array(a, dtype=np.float32).flatten()
    na = np.linalg.norm(a)
    if na == 0:
        return None
    a = a / na

    b = np.array(b, dtype=np.float32)

    # Multi-exemplar: shape (N, 512)
    if b.ndim == 2:
        norms = np.linalg.norm(b, axis=1, keepdims=True)
        valid = norms.flatten() > 0
        if not np.any(valid):
            return None
        b_norm = b[valid] / norms[valid]
        sims = b_norm @ a  # (N,) dot products
        return float(np.max(sims))

    # Single exemplar: shape (512,)
    nb = np.linalg.norm(b)
    if nb == 0:
        return None
    return float(np.dot(a, b / nb))


class Matcher:
    def __init__(self, db_path="embeddings_db"):
        self.db_path = db_path
        self.database = {}
        self.fusion = FusionEngine()
        self.cache = get_cache()

        # Dynamic threshold scales with gallery size
        self.BASE_THRESHOLD = 0.45
        self.THRESHOLD_WITHOUT_FACE = 0.99  # effectively disabled
        self.MARGIN = 0.15

        self.load_database()

    def _dynamic_threshold(self):
        """Threshold scales with gallery size.
        3 people → 0.45 (base), 5 → 0.49, 8 → 0.55, 10 → 0.59
        """
        gallery_size = len(self.database)
        extra = max(0, gallery_size - 3) * 0.02
        return self.BASE_THRESHOLD + extra

    def load_database(self):
        """Load embeddings from PostgreSQL (primary) or .npy files (fallback)."""
        self.database = {}

        if USE_DATABASE:
            try:
                # Try loading from PostgreSQL
                for modality in ["face", "body", "gait"]:
                    db = DBEmbedding.load_all(modality)
                    for name, emb in db.items():
                        if name not in self.database:
                            self.database[name] = {}
                        self.database[name][modality] = emb

                if self.database:
                    print(f"\n[DB LOAD] ✅ Loaded from PostgreSQL: {list(self.database.keys())}")
                    # Cache in Redis for faster lookups
                    if self.cache.is_available:
                        for modality in ["face", "body", "gait"]:
                            modality_db = {
                                name: data.get(modality)
                                for name, data in self.database.items()
                                if data.get(modality) is not None
                            }
                            if modality_db:
                                self.cache.cache_all_embeddings(modality, modality_db)
                    print(
                        f"[DB LOAD] Dynamic threshold for {len(self.database)} people: "
                        f"{self._dynamic_threshold():.2f}\n"
                    )
                    return
                else:
                    print("[DB LOAD] ⚠️  No data in database, falling back to .npy files")
            except Exception as e:
                print(f"[DB LOAD] ⚠️  Database error ({e}), falling back to .npy files")

        # Fallback: load from .npy files (original behavior)
        if not os.path.exists(self.db_path):
            print(f"[DB LOAD] ⚠️  Folder not found: {self.db_path}")
            return

        files = os.listdir(self.db_path)
        print(f"\n[DB LOAD] Files (file-based): {files}\n")

        for file in files:
            if not file.endswith(".npy"):
                continue
            parts = file.replace(".npy", "").split("_")
            if len(parts) < 2:
                continue
            name = parts[0]
            modality = parts[1].lower()
            if modality not in ("face", "body", "gait"):
                continue
            try:
                emb = np.load(os.path.join(self.db_path, file))
                if name not in self.database:
                    self.database[name] = {}
                self.database[name][modality] = emb
                print(f"[DB LOAD] ✅  {name} → {modality}  shape={emb.shape}")
            except Exception as e:
                print(f"[DB LOAD] ❌  {file}: {e}")

        print(f"\n[DB LOAD] Loaded: {list(self.database.keys())}")
        print(f"[DB LOAD] Dynamic threshold for {len(self.database)} people: " f"{self._dynamic_threshold():.2f}\n")

    def reload(self):
        """Reload embeddings from database/files without restarting."""
        print(f"[Matcher] 🔄 Reloading database...")
        # Invalidate Redis cache
        if self.cache.is_available:
            self.cache.invalidate_embedding()
        self.load_database()
        print(
            f"[Matcher] ✅ Reload complete — "
            f"{len(self.database)} persons now in DB: "
            f"{list(self.database.keys())}"
        )

    def identify(self, face_emb=None, body_emb=None, gait_emb=None):
        if not self.database:
            return "Unknown", 0.0

        scores = []
        threshold = self._dynamic_threshold()

        for person, data in self.database.items():
            face_sim = cosine_similarity(face_emb, data.get("face"))
            body_sim = cosine_similarity(body_emb, data.get("body"))
            gait_sim = cosine_similarity(gait_emb, data.get("gait"))

            final_score, trusted = self.fusion.compute_final_score(
                face_score=face_sim, body_score=body_sim, gait_score=gait_sim, verbose=True
            )

            parts = []
            if face_sim is not None:
                parts.append(f"face={face_sim:.3f}")
            if body_sim is not None:
                parts.append(f"body={body_sim:.3f}")
            if gait_sim is not None:
                parts.append(f"gait={gait_sim:.3f}")
            print(f"[MATCHER] {person:15s} | {' | '.join(parts)} | final={final_score:.3f}")

            scores.append((person, final_score, trusted))

        scores.sort(key=lambda x: x[1], reverse=True)

        best_person, best_score, best_trusted = scores[0]
        second_score = scores[1][1] if len(scores) > 1 else 0.0
        margin = best_score - second_score

        active_threshold = threshold if best_trusted else self.THRESHOLD_WITHOUT_FACE

        print(
            f"[MATCHER] Best={best_person} ({best_score:.3f}) | 2nd={second_score:.3f} | "
            f"margin={margin:.3f} | threshold={active_threshold:.2f} | trusted={best_trusted}"
        )

        if not best_trusted:
            print(f"[MATCHER] ❌  → Unknown (no face — refusing to guess)")
            return "Unknown", best_score

        if best_score < active_threshold:
            print(f"[MATCHER] ❌  → Unknown (score too low: {best_score:.3f} < {active_threshold:.2f})")
            return "Unknown", best_score

        if margin < self.MARGIN:
            print(f"[MATCHER] ❌  → Unknown (margin too small: {margin:.3f} < {self.MARGIN})")
            return "Unknown", best_score

        print(f"[MATCHER] ✅  → {best_person}")
        return best_person, best_score
