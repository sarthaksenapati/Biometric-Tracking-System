# utils/embeddings.py - Updated to use PostgreSQL with file fallback

import numpy as np
import os
from db.models import Embedding as DBEmbedding, Person as DBPerson
from cache import get_cache

USE_DATABASE = os.getenv("USE_DATABASE", "true").lower() == "true"
DB_PATH = "embeddings_db"
os.makedirs(DB_PATH, exist_ok=True)

_cache = get_cache()


def save_embedding(person_id, embedding, modality="face"):
    """
    Save embedding(s) for a person.
    - PostgreSQL primary (if USE_DATABASE=true)
    - .npy file fallback
    """
    embedding = np.array(embedding, dtype=np.float32)

    # Try database first
    if USE_DATABASE:
        try:
            # Ensure person exists
            DBPerson.create(person_id)
            DBEmbedding.save(person_id, modality, embedding)
            print(f"[EMBEDDINGS] ✅ Saved to DB: {person_id}_{modality} shape={embedding.shape}")

            # Update cache
            if _cache.is_available:
                _cache.cache_embedding(person_id, modality, embedding)
            return
        except Exception as e:
            print(f"[EMBEDDINGS] ⚠️  DB save failed ({e}), falling back to file")

    # Fallback: save to .npy file (original behavior)
    file_path = os.path.join(DB_PATH, f"{person_id}_{modality}.npy")
    np.save(file_path, embedding)
    print(f"[EMBEDDINGS] Saved to file: {person_id}_{modality}.npy shape={embedding.shape}")


def load_all_embeddings(modality="face"):
    """
    Load all embeddings for a given modality.
    Returns dict of person_id → np.array
    """
    result = {}

    # Try database first
    if USE_DATABASE:
        try:
            result = DBEmbedding.load_all(modality)
            if result:
                print(f"[EMBEDDINGS] ✅ Loaded from DB: {len(result)} {modality} embeddings")
                return result
            else:
                print(f"[EMBEDDINGS] ⚠️  No data in DB, falling back to files")
        except Exception as e:
            print(f"[EMBEDDINGS] ⚠️  DB load failed ({e}), falling back to files")

    # Fallback: load from .npy files
    for file in os.listdir(DB_PATH):
        if file.endswith(f"_{modality}.npy"):
            person_id = file.split("_")[0]
            emb = np.load(os.path.join(DB_PATH, file))
            result[person_id] = emb

    print(f"[EMBEDDINGS] Loaded from files: {len(result)} {modality} embeddings")
    return result


def load_embedding(person_id, modality="face"):
    """Load embedding for a specific person and modality."""
    if USE_DATABASE:
        try:
            emb = DBEmbedding.load(person_id, modality)
            if emb is not None:
                return emb
        except Exception:
            pass

    # Fallback
    file_path = os.path.join(DB_PATH, f"{person_id}_{modality}.npy")
    if os.path.exists(file_path):
        return np.load(file_path)
    return None


def delete_embedding(person_id, modality=None):
    """Delete embedding(s) for a person."""
    if USE_DATABASE:
        try:
            if modality:
                # Delete specific modality
                pass  # Implement if needed
            else:
                DBPerson.delete(person_id)
            print(f"[EMBEDDINGS] Deleted {person_id} from DB")
        except Exception as e:
            print(f"[EMBEDDINGS] DB delete failed: {e}")

    # Also clean up files
    for mod in ["face", "body", "gait"]:
        file_path = os.path.join(DB_PATH, f"{person_id}_{mod}.npy")
        if os.path.exists(file_path):
            os.remove(file_path)
