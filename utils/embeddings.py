# utils/embeddings.py

import numpy as np
import os

DB_PATH = "embeddings_db"
os.makedirs(DB_PATH, exist_ok=True)

# Check if database is available
try:
    from database.db import DB_AVAILABLE
except ImportError:
    DB_AVAILABLE = False


def _db_save(person_id, embedding, modality="face"):
    from database.db import save_embedding as db_save
    db_save(person_id, modality, embedding)


def _db_load(modality="face"):
    from database.db import load_all_embeddings as db_load
    result = db_load(modality)
    return result


def save_embedding(person_id, embedding, modality="face"):
    """
    Save embedding(s) for a person.

    embedding can be:
      - shape (512,)    — single averaged vector (legacy, still supported)
      - shape (N, 512)  — stack of N exemplars (Fix 1 — multi-exemplar matching)

    Tries database first, falls back to .npy file storage.
    """
    shape = np.array(embedding).shape

    # Try database first
    if DB_AVAILABLE:
        try:
            _db_save(person_id, embedding, modality)
            print(f"[EMBEDDINGS] DB saved {person_id}_{modality}  shape={shape}")
            return
        except Exception as e:
            print(f"[EMBEDDINGS] DB save failed ({e}), falling back to .npy")

    # Fallback to .npy file
    file_path = os.path.join(DB_PATH, f"{person_id}_{modality}.npy")
    np.save(file_path, embedding)
    print(f"[EMBEDDINGS] Saved {person_id}_{modality}.npy  shape={shape}")


def load_all_embeddings(modality="face"):
    """
    Load all embeddings for a given modality.
    Returns dict of person_id → np.array (shape (512,) or (N, 512))

    Tries database first, falls back to .npy file storage.
    """
    # Try database first
    if DB_AVAILABLE:
        try:
            result = _db_load(modality)
            if result is not None:
                print(f"[EMBEDDINGS] Loaded {len(result)} embeddings from DB ({modality})")
                return result
        except Exception as e:
            print(f"[EMBEDDINGS] DB load failed ({e}), falling back to .npy")

    # Fallback to .npy files
    db = {}
    if not os.path.exists(DB_PATH):
        return db

    for file in os.listdir(DB_PATH):
        if file.endswith(f"_{modality}.npy"):
            person_id = file.split("_")[0]
            emb = np.load(os.path.join(DB_PATH, file))
            db[person_id] = emb

    print(f"[EMBEDDINGS] Loaded {len(db)} embeddings from .npy ({modality})")
    return db
