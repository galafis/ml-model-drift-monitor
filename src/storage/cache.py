"""
Cache management module.

Provides Redis-based caching for reference distributions, recent
predictions, and computed metrics. Falls back to in-memory LRU
cache when Redis is unavailable.
"""

import hashlib
import json
import time
from collections import OrderedDict
from typing import Any, Dict, List, Optional

import numpy as np

from src.config.settings import StorageConfig
from src.utils.logger import get_logger

logger = get_logger("storage.cache")


class InMemoryLRU:
    """Simple in-memory LRU cache with TTL support."""

    def __init__(self, max_size: int = 256, default_ttl: int = 3600):
        self._cache: OrderedDict[str, Dict[str, Any]] = OrderedDict()
        self._max_size = max_size
        self._default_ttl = default_ttl

    def get(self, key: str) -> Optional[Any]:
        """Get value by key, returns None if expired or missing."""
        if key not in self._cache:
            return None
        entry = self._cache[key]
        if time.time() > entry["expires"]:
            del self._cache[key]
            return None
        self._cache.move_to_end(key)
        return entry["value"]

    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Set a key-value pair with optional TTL."""
        if key in self._cache:
            self._cache.move_to_end(key)
        elif len(self._cache) >= self._max_size:
            self._cache.popitem(last=False)

        self._cache[key] = {
            "value": value,
            "expires": time.time() + (ttl or self._default_ttl),
        }

    def delete(self, key: str) -> bool:
        """Delete a key. Returns True if it existed."""
        if key in self._cache:
            del self._cache[key]
            return True
        return False

    def clear(self) -> None:
        """Clear all entries."""
        self._cache.clear()

    @property
    def size(self) -> int:
        """Current number of entries."""
        return len(self._cache)


class CacheManager:
    """
    Redis-based cache with in-memory fallback.

    Caches reference distributions, recent prediction batches,
    and computed drift metrics to reduce redundant computation.

    Attributes:
        config: Storage configuration.
    """

    def __init__(self, config: Optional[StorageConfig] = None):
        """
        Initialize the cache manager.

        Args:
            config: Storage configuration with Redis connection details.
        """
        self.config = config or StorageConfig()
        self._redis = None
        self._use_redis = False
        self._memory_cache = InMemoryLRU(
            max_size=512, default_ttl=self.config.cache_ttl_seconds
        )

        self._try_redis()

    def _try_redis(self) -> None:
        """Attempt to connect to Redis."""
        try:
            import redis

            self._redis = redis.Redis(
                host=self.config.redis_host,
                port=self.config.redis_port,
                db=self.config.redis_db,
                password=self.config.redis_password,
                socket_timeout=3,
                decode_responses=True,
            )
            self._redis.ping()
            self._use_redis = True
            logger.info(
                "Connected to Redis at %s:%d",
                self.config.redis_host,
                self.config.redis_port,
            )
        except Exception as exc:
            logger.warning(
                "Redis not available (%s). Using in-memory cache.", exc
            )
            self._use_redis = False
            self._redis = None

    def get(self, key: str) -> Optional[Any]:
        """
        Retrieve a cached value.

        Args:
            key: Cache key.

        Returns:
            Cached value or None.
        """
        if self._use_redis:
            try:
                raw = self._redis.get(key)
                if raw is not None:
                    return json.loads(raw)
                return None
            except Exception as exc:
                logger.warning("Redis get failed: %s", exc)

        return self._memory_cache.get(key)

    def set(
        self, key: str, value: Any, ttl: Optional[int] = None
    ) -> None:
        """
        Store a value in cache.

        Args:
            key: Cache key.
            value: Value to cache (must be JSON-serializable).
            ttl: Time-to-live in seconds. Uses default if None.
        """
        ttl = ttl or self.config.cache_ttl_seconds

        if self._use_redis:
            try:
                self._redis.setex(key, ttl, json.dumps(value, default=str))
                return
            except Exception as exc:
                logger.warning("Redis set failed: %s", exc)

        self._memory_cache.set(key, value, ttl)

    def delete(self, key: str) -> bool:
        """
        Delete a cached value.

        Args:
            key: Cache key.

        Returns:
            True if the key existed.
        """
        if self._use_redis:
            try:
                return bool(self._redis.delete(key))
            except Exception as exc:
                logger.warning("Redis delete failed: %s", exc)

        return self._memory_cache.delete(key)

    def cache_reference_distribution(
        self, model_name: str, feature_name: str, distribution: np.ndarray
    ) -> None:
        """
        Cache a reference feature distribution.

        Args:
            model_name: Model identifier.
            feature_name: Feature name.
            distribution: Distribution data as numpy array.
        """
        key = f"ref_dist:{model_name}:{feature_name}"
        self.set(key, distribution.tolist(), ttl=self.config.cache_ttl_seconds * 2)

    def get_reference_distribution(
        self, model_name: str, feature_name: str
    ) -> Optional[np.ndarray]:
        """
        Retrieve a cached reference distribution.

        Args:
            model_name: Model identifier.
            feature_name: Feature name.

        Returns:
            Numpy array or None.
        """
        key = f"ref_dist:{model_name}:{feature_name}"
        data = self.get(key)
        if data is not None:
            return np.array(data)
        return None

    def cache_predictions(
        self, model_name: str, predictions: np.ndarray, batch_id: str
    ) -> None:
        """
        Cache a batch of predictions.

        Args:
            model_name: Model identifier.
            predictions: Prediction array.
            batch_id: Batch identifier.
        """
        key = f"predictions:{model_name}:{batch_id}"
        self.set(key, predictions.tolist())

    def get_predictions(
        self, model_name: str, batch_id: str
    ) -> Optional[np.ndarray]:
        """
        Retrieve cached predictions.

        Args:
            model_name: Model identifier.
            batch_id: Batch identifier.

        Returns:
            Numpy array or None.
        """
        key = f"predictions:{model_name}:{batch_id}"
        data = self.get(key)
        if data is not None:
            return np.array(data)
        return None

    def clear_model_cache(self, model_name: str) -> None:
        """Clear all cached data for a specific model."""
        if self._use_redis:
            try:
                pattern = f"*:{model_name}:*"
                keys = self._redis.keys(pattern)
                if keys:
                    self._redis.delete(*keys)
                logger.info("Cleared Redis cache for model '%s'.", model_name)
                return
            except Exception as exc:
                logger.warning("Redis clear failed: %s", exc)

        self._memory_cache.clear()
        logger.info("Cleared in-memory cache.")

    @staticmethod
    def compute_cache_key(data: np.ndarray) -> str:
        """Compute a hash-based cache key for a data array."""
        data_bytes = data.tobytes()
        return hashlib.sha256(data_bytes).hexdigest()[:16]

    def close(self) -> None:
        """Close connections."""
        if self._redis:
            try:
                self._redis.close()
            except Exception:
                pass
