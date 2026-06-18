import os
import io
import json
import base64
import time
import numpy as np

# ── Safe serialization (replaces pickle — pickle.loads on untrusted Redis
#    data is a remote-code-execution risk) ─────────────────────────────────
#
# We encode arbitrary nested dict/list structures as JSON. numpy arrays are
# stored as base64-encoded .npy blobs and reloaded with allow_pickle=False,
# so a malicious cache value can never execute code on load.

_NDARRAY_TAG = "__ndarray_b64__"


def _to_jsonable(obj):
    """Recursively convert obj (dicts, lists, ndarrays, scalars) to a
    JSON-serializable structure. No pickle, no executable payloads."""
    if isinstance(obj, np.ndarray):
        buf = io.BytesIO()
        np.save(buf, obj, allow_pickle=False)
        return {_NDARRAY_TAG: base64.b64encode(buf.getvalue()).decode("ascii")}
    if isinstance(obj, dict):
        return {str(k): _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    return obj  # str, int, float, bool, None


def _from_jsonable(obj):
    """Inverse of _to_jsonable. Rebuilds ndarrays from tagged base64 blobs."""
    if isinstance(obj, dict):
        if len(obj) == 1 and _NDARRAY_TAG in obj:
            raw = base64.b64decode(obj[_NDARRAY_TAG])
            return np.load(io.BytesIO(raw), allow_pickle=False)
        return {k: _from_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_from_jsonable(v) for v in obj]
    return obj


def _safe_dumps(obj) -> bytes:
    return json.dumps(_to_jsonable(obj)).encode("utf-8")


def _safe_loads(data):
    return _from_jsonable(json.loads(data))


# ── Connection ─────────────────────────────────────────────────────────────
#
# A single shared ConnectionPool is created lazily and reused for every client.
# Previously each call did redis.from_url(...) which built a NEW TCP connection
# (plus an extra ping round-trip) every time — set_shared_identity_cache runs on
# every cache deposit in the inference loop, so that was a real per-frame cost.
# With an external pool, client.close() just returns the connection to the pool
# instead of tearing it down, so the existing close() calls stay correct.

def _get_redis_url():
    return os.environ.get("REDIS_URL", "redis://localhost:6379")


_redis_pool = None


def _get_pool():
    global _redis_pool
    if _redis_pool is None:
        import redis
        _redis_pool = redis.ConnectionPool.from_url(
            _get_redis_url(), decode_responses=False
        )
    return _redis_pool


def get_redis_client():
    import redis
    # Lightweight client over the shared pool. No ping here — callers either
    # ping explicitly (_redis_available, health checks) or immediately issue a
    # command that will surface a connection error, so the extra round-trip is
    # unnecessary on the hot path.
    return redis.Redis(connection_pool=_get_pool())


def _redis_available():
    try:
        client = get_redis_client()
        client.ping()   # explicit liveness check (get_redis_client no longer pings)
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
        serialized = _safe_dumps(data)
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
            return _safe_loads(data)
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
        serialized = _safe_dumps(store_dict)
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
            return _safe_loads(data)
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
