from __future__ import annotations

import json
from typing import Any

import redis


class RedisCacheError(RuntimeError):
    """Raised when a Redis cache operation fails."""


class RedisCache:
    """Small Redis cache abstraction for RelevanceFlow."""

    def __init__(
        self,
        redis_url: str,
        *,
        default_ttl_seconds: int = 300,
    ) -> None:
        if not redis_url:
            raise RedisCacheError("Redis URL must not be empty.")

        if default_ttl_seconds <= 0:
            raise RedisCacheError("Redis TTL must be greater than zero.")

        self.default_ttl_seconds = default_ttl_seconds

        try:
            self.client = redis.Redis.from_url(
                redis_url,
                decode_responses=True,
            )
        except Exception as exc:
            raise RedisCacheError(f"Failed to initialize Redis client: {exc}") from exc

    def ping(self) -> bool:
        """Check whether Redis is reachable."""

        try:
            return bool(self.client.ping())
        except Exception as exc:
            raise RedisCacheError(f"Redis ping failed: {exc}") from exc

    def get(self, key: str) -> Any | None:
        """Retrieve and deserialize a cached value."""

        if not key:
            raise RedisCacheError("Redis cache key must not be empty.")

        try:
            value = self.client.get(key)

            if value is None:
                return None

            return json.loads(value)

        except RedisCacheError:
            raise
        except Exception as exc:
            raise RedisCacheError(f"Redis get failed: {exc}") from exc

    def set(
        self,
        key: str,
        value: Any,
        *,
        ttl_seconds: int | None = None,
    ) -> None:
        """Serialize and store a value with a TTL."""

        if not key:
            raise RedisCacheError("Redis cache key must not be empty.")

        ttl = ttl_seconds if ttl_seconds is not None else self.default_ttl_seconds

        if ttl <= 0:
            raise RedisCacheError("Redis TTL must be greater than zero.")

        try:
            serialized = json.dumps(value)

            self.client.set(
                key,
                serialized,
                ex=ttl,
            )

        except RedisCacheError:
            raise
        except Exception as exc:
            raise RedisCacheError(f"Redis set failed: {exc}") from exc

    def delete(self, key: str) -> None:
        """Delete a cached value."""

        if not key:
            raise RedisCacheError("Redis cache key must not be empty.")

        try:
            self.client.delete(key)
        except Exception as exc:
            raise RedisCacheError(f"Redis delete failed: {exc}") from exc
