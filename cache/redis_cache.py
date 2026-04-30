# cache/redis_cache.py - Redis caching layer

import os
import json
import pickle
import numpy as np
from typing import Optional, Any, List, Dict

try:
    import redis

    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    print("[CACHE] ⚠️  redis package not installed. Caching disabled.")

# Redis connection from environment
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")

# Global cache instance
_cache_instance = None


def get_cache() -> "RedisCache":
    """Get or create the global cache instance."""
    global _cache_instance
    if _cache_instance is None:
        _cache_instance = RedisCache()
    return _cache_instance


class RedisCache:
    """Redis caching for embeddings, detections, and session data."""

    def __init__(self, url: Optional[str] = None):
        self.url = url or REDIS_URL
        self._client = None
        if REDIS_AVAILABLE:
            try:
                self._client = redis.from_url(
                    self.url,
                    decode_responses=False,  # We need binary for numpy arrays
                    socket_connect_timeout=5,
                    socket_timeout=5,
                )
                # Test connection
                self._client.ping()
                print(f"[CACHE] ✅ Connected to Redis at {self.url}")
            except Exception as e:
                print(f"[CACHE] ⚠️  Redis unavailable: {e}. Running without cache.")
                self._client = None
        else:
            print("[CACHE] ⚠️  Running without Redis cache")

    @property
    def is_available(self) -> bool:
        if not self._client:
            return False
        try:
            return self._client.ping()
        except Exception:
            return False

    # ── Embedding Cache ─────────────────────────────────────────

    def cache_embedding(self, person_name: str, modality: str, embedding: np.ndarray):
        """Cache an embedding for fast lookup."""
        if not self.is_available:
            return
        try:
            key = f"emb:{person_name}:{modality}"
            data = {
                "embedding": embedding.tobytes(),
                "shape": list(embedding.shape),
                "dtype": str(embedding.dtype),
            }
            self._client.setex(key, 3600, pickle.dumps(data))  # 1 hour TTL
        except Exception as e:
            print(f"[CACHE] Failed to cache embedding: {e}")

    def get_cached_embedding(self, person_name: str, modality: str) -> Optional[np.ndarray]:
        """Get cached embedding."""
        if not self.is_available:
            return None
        try:
            key = f"emb:{person_name}:{modality}"
            data = self._client.get(key)
            if data:
                parsed = pickle.loads(data)
                return np.frombuffer(parsed["embedding"], dtype=parsed["dtype"]).reshape(parsed["shape"])
        except Exception as e:
            print(f"[CACHE] Failed to get cached embedding: {e}")
        return None

    def cache_all_embeddings(self, modality: str, embeddings_dict: Dict[str, np.ndarray]):
        """Cache all embeddings for a modality."""
        if not self.is_available:
            return
        try:
            pipe = self._client.pipeline()
            for person_name, emb in embeddings_dict.items():
                key = f"emb:{person_name}:{modality}"
                data = {
                    "embedding": emb.tobytes(),
                    "shape": list(emb.shape),
                    "dtype": str(emb.dtype),
                }
                pipe.setex(key, 3600, pickle.dumps(data))
            pipe.execute()
        except Exception as e:
            print(f"[CACHE] Failed to cache all embeddings: {e}")

    def get_all_cached_embeddings(self, modality: str) -> Dict[str, np.ndarray]:
        """Get all cached embeddings for a modality (from pattern match)."""
        if not self.is_available:
            return {}
        try:
            pattern = f"emb:*:{modality}"
            keys = self._client.keys(pattern)
            result = {}
            if keys:
                # Use pipeline for efficiency
                pipe = self._client.pipeline()
                for key in keys:
                    pipe.get(key)
                values = pipe.execute()

                for key, data in zip(keys, values):
                    if data:
                        parsed = pickle.loads(data)
                        person_name = key.decode().split(":")[1]
                        result[person_name] = np.frombuffer(parsed["embedding"], dtype=parsed["dtype"]).reshape(
                            parsed["shape"]
                        )
            return result
        except Exception as e:
            print(f"[CACHE] Failed to get all cached embeddings: {e}")
            return {}

    def invalidate_embedding(self, person_name: Optional[str] = None, modality: Optional[str] = None):
        """Invalidate cached embeddings."""
        if not self.is_available:
            return
        try:
            if person_name and modality:
                self._client.delete(f"emb:{person_name}:{modality}")
            elif modality:
                keys = self._client.keys(f"emb:*:{modality}")
                if keys:
                    self._client.delete(*keys)
            else:
                keys = self._client.keys("emb:*")
                if keys:
                    self._client.delete(*keys)
        except Exception as e:
            print(f"[CACHE] Failed to invalidate cache: {e}")

    # ── Recent Detections Cache ─────────────────────────────────

    def cache_detection(self, cam_id: int, detection: Dict):
        """Cache a recent detection."""
        if not self.is_available:
            return
        try:
            key = f"det:{cam_id}"
            # Add timestamp
            detection["cached_at"] = __import__("time").time()
            self._client.lpush(key, pickle.dumps(detection))
            self._client.ltrim(key, 0, 99)  # Keep last 100
            self._client.expire(key, 300)  # 5 min TTL
        except Exception as e:
            print(f"[CACHE] Failed to cache detection: {e}")

    def get_recent_detections(self, cam_id: int, count: int = 20) -> List[Dict]:
        """Get recent detections for a camera."""
        if not self.is_available:
            return []
        try:
            key = f"det:{cam_id}"
            data = self._client.lrange(key, 0, count - 1)
            return [pickle.loads(d) for d in data if d]
        except Exception as e:
            print(f"[CACHE] Failed to get recent detections: {e}")
            return []

    # ── Identity Cache (for SharedIdentityCache) ───────────────────

    def cache_identity(self, name: str, identity_data: Dict, ttl: int = 300):
        """Cache an identity for cross-camera lookup."""
        if not self.is_available:
            return
        try:
            key = f"identity:{name}"
            self._client.setex(key, ttl, pickle.dumps(identity_data))
        except Exception as e:
            print(f"[CACHE] Failed to cache identity: {e}")

    def get_cached_identity(self, name: str) -> Optional[Dict]:
        """Get cached identity."""
        if not self.is_available:
            return None
        try:
            key = f"identity:{name}"
            data = self._client.get(key)
            if data:
                return pickle.loads(data)
        except Exception as e:
            print(f"[CACHE] Failed to get cached identity: {e}")
        return None

    # ── Session State Cache ──────────────────────────────────────

    def cache_tracker_state(self, state: Dict):
        """Cache tracker state for dashboard."""
        if not self.is_available:
            return
        try:
            self._client.setex("tracker:state", 10, json.dumps(state))  # 10s TTL
        except Exception as e:
            print(f"[CACHE] Failed to cache tracker state: {e}")

    def get_cached_tracker_state(self) -> Optional[Dict]:
        """Get cached tracker state."""
        if not self.is_available:
            return None
        try:
            data = self._client.get("tracker:state")
            if data:
                return json.loads(data)
        except Exception as e:
            print(f"[CACHE] Failed to get cached tracker state: {e}")
        return None

    # ── Utility ─────────────────────────────────────────────────

    def ping(self) -> bool:
        """Check Redis connectivity."""
        if not self._client:
            return False
        try:
            return self._client.ping()
        except Exception:
            return False

    def flush_all(self):
        """Flush all cached data (use with caution)."""
        if not self.is_available:
            return
        try:
            self._client.flushdb()
            print("[CACHE] ✅ Flushed all cache")
        except Exception as e:
            print(f"[CACHE] Failed to flush cache: {e}")

    def close(self):
        """Close Redis connection."""
        if self._client:
            self._client.close()
            print("[CACHE] Redis connection closed")
