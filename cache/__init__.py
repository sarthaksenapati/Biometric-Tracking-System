# cache/__init__.py - Redis caching package
from .redis_cache import RedisCache, get_cache

__all__ = ['RedisCache', 'get_cache']
