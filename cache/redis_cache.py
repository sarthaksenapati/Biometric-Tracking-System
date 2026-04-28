import os
import json
import pickle
import time
import numpy as np

# ── Connection ─────────────────────────────────────────────────────────────

def _get_redis_url():
    return os.environ.get("REDIS_URL", "redis://localhost:6379")


def get_redis_client():
    import redis
    url = _get_redis_url()
    client = redis.from_url(url, decode_responses=False)
    client.ping()  # Test connection
    return client


def _redis_available():
    try:
        client = get_redis_client()
        client.close()
        return True
    except Exception:
        return False


REDIS_AVAILABLE = _redis_available()

# ── Embeddings Cache ─────────────────────────────────────────────────────

EMBEDDING_TTL = 3600  # 1 hour


def set_cached_embeddings(modality, data, ttl=EMBEDDING_TTL):
    """Cache embedding dict (person_name → np.array) for a modality."""
    if not REDIS_AVAILABLE:
        return False
    try:
        client = get_redis_client()
        serialized = pickle.dumps(data)
        client.setex(f"embeddings:{modality}", ttl, serialized)
        client.close()
        return True
    except Exception as e:
        print(f"[REDIS] set_cached_embeddings failed: {e}")
        return False


def get_cached_embeddings(modality):
    """Get cached embedding dict. Returns None if not cached."""
    if not REDIS_AVAILABLE:
        return None
    try:
        client = get_redis_client()
        data = client.get(f"embeddings:{modality}")
        client.close()
        if data:
            return pickle.loads(data)
        return None
    except Exception as e:
        print(f"[REDIS] get_cached_embeddings failed: {e}")
        return None


def invalidate_embeddings_cache(modality=None):
    """Clear cached embeddings. If modality is None, clear all modalities."""
    if not REDIS_AVAILABLE:
        return
    try:
        client = get_redis_client()
        if modality:
            client.delete(f"embeddings:{modality}")
        else:
            for mod in ("face", "body", "gait"):
                client.delete(f"embeddings:{mod}")
        client.close()
        print("[REDIS] Embeddings cache invalidated")
    except Exception as e:
        print(f"[REDIS] invalidate_embeddings_cache failed: {e}")

# ── Identity Cache (SharedIdentityCache backend) ────────────────────────

IDENTITY_TTL = 300  # 5 minutes
CACHE_KEY_PREFIX = "identity_cache:"


def set_cached_identity(track_key, name, score, ttl=IDENTITY_TTL):
    """Cache identity for a track (cam_id, track_id)."""
    if not REDIS_AVAILABLE:
        return False
    try:
        client = get_redis_client()
        key = f"{CACHE_KEY_PREFIX}{track_key}"
        data = json.dumps({"name": name, "score": score, "timestamp": time.time()})
        client.setex(key, ttl, data)
        client.close()
        return True
    except Exception as e:
        print(f"[REDIS] set_cached_identity failed: {e}")
        return False


def get_cached_identity(track_key):
    """Get cached identity. Returns (name, score) or None."""
    if not REDIS_AVAILABLE:
        return None
    try:
        client = get_redis_client()
        key = f"{CACHE_KEY_PREFIX}{track_key}"
        data = client.get(key)
        client.close()
        if data:
            parsed = json.loads(data)
            return parsed["name"], parsed["score"]
        return None
    except Exception as e:
        print(f"[REDIS] get_cached_identity failed: {e}")
        return None


def clear_identity_cache():
    """Clear all cached identities."""
    if not REDIS_AVAILABLE:
        return
    try:
        client = get_redis_client()
        keys = client.keys(f"{CACHE_KEY_PREFIX}*")
        if keys:
            client.delete(*keys)
        client.close()
        print("[REDIS] Identity cache cleared")
    except Exception as e:
        print(f"[REDIS] clear_identity_cache failed: {e}")

# ── Detection History ────────────────────────────────────────────────────

HISTORY_KEY = "detection_history"
HISTORY_MAX = 1000


def log_detection(data):
    """Log a detection event. data should be a dict."""
    if not REDIS_AVAILABLE:
        return
    try:
        client = get_redis_client()
        client.lpush(HISTORY_KEY, json.dumps(data))
        client.ltrim(HISTORY_KEY, 0, HISTORY_MAX - 1)
        client.close()
    except Exception as e:
        print(f"[REDIS] log_detection failed: {e}")


def get_detection_history(person_name=None, limit=50):
    """Get recent detection events. Optionally filter by person_name."""
    if not REDIS_AVAILABLE:
        return []
    try:
        client = get_redis_client()
        raw = client.lrange(HISTORY_KEY, 0, limit - 1)
        client.close()
        events = [json.loads(r) for r in raw]
        if person_name:
            events = [e for e in events if e.get("person") == person_name]
        return events[:limit]
    except Exception as e:
        print(f"[REDIS] get_detection_history failed: {e}")
        return []


# ── SharedIdentityCache Backend ─────────────────────────────────────

SHARED_CACHE_KEY = "shared_identity_cache"
SHARED_CACHE_TTL = 300  # 5 minutes


def set_shared_identity_cache(store_dict):
    """Cache the entire SharedIdentityCache store."""
    if not REDIS_AVAILABLE:
        return False
    try:
        client = get_redis_client()
        serialized = pickle.dumps(store_dict)
        client.setex(SHARED_CACHE_KEY, SHARED_CACHE_TTL, serialized)
        client.close()
        return True
    except Exception as e:
        print(f"[REDIS] set_shared_identity_cache failed: {e}")
        return False


def get_shared_identity_cache():
    """Get the entire SharedIdentityCache store. Returns None if not cached."""
    if not REDIS_AVAILABLE:
        return None
    try:
        client = get_redis_client()
        data = client.get(SHARED_CACHE_KEY)
        client.close()
        if data:
            return pickle.loads(data)
        return None
    except Exception as e:
        print(f"[REDIS] get_shared_identity_cache failed: {e}")
        return None


def clear_shared_identity_cache():
    """Clear the shared identity cache."""
    if not REDIS_AVAILABLE:
        return
    try:
        client = get_redis_client()
        client.delete(SHARED_CACHE_KEY)
        client.close()
        print("[REDIS] Shared identity cache cleared")
    except Exception as e:
        print(f"[REDIS] clear_shared_identity_cache failed: {e}")
