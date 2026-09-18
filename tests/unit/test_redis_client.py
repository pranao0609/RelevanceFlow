import pytest

from relevanceflow.cache.redis_client import (
    RedisCache,
    RedisCacheError,
)


class FakeRedis:
    def __init__(self):
        self.data = {}

    def ping(self):
        return True

    def get(self, key):
        return self.data.get(key)

    def set(self, key, value, ex=None):
        self.data[key] = value

    def delete(self, key):
        self.data.pop(key, None)


def test_redis_cache_rejects_empty_url():
    with pytest.raises(RedisCacheError):
        RedisCache("")


def test_redis_cache_ping(monkeypatch):
    cache = RedisCache("redis://localhost:6379/0")

    fake = FakeRedis()
    monkeypatch.setattr(cache, "client", fake)

    assert cache.ping() is True


def test_redis_cache_set_and_get(monkeypatch):
    cache = RedisCache("redis://localhost:6379/0")

    fake = FakeRedis()
    monkeypatch.setattr(cache, "client", fake)

    value = {
        "query": "wireless mouse",
        "results": [
            {
                "product_id": 123,
                "score": 0.91,
                "rank": 1,
            }
        ],
    }

    cache.set("test-key", value)

    assert cache.get("test-key") == value


def test_redis_cache_miss(monkeypatch):
    cache = RedisCache("redis://localhost:6379/0")

    fake = FakeRedis()
    monkeypatch.setattr(cache, "client", fake)

    assert cache.get("missing-key") is None


def test_redis_cache_delete(monkeypatch):
    cache = RedisCache("redis://localhost:6379/0")

    fake = FakeRedis()
    monkeypatch.setattr(cache, "client", fake)

    cache.set("test-key", {"value": 1})

    assert cache.get("test-key") == {"value": 1}

    cache.delete("test-key")

    assert cache.get("test-key") is None


def test_redis_cache_rejects_empty_key(monkeypatch):
    cache = RedisCache("redis://localhost:6379/0")

    fake = FakeRedis()
    monkeypatch.setattr(cache, "client", fake)

    with pytest.raises(RedisCacheError):
        cache.get("")


def test_redis_cache_rejects_invalid_ttl():
    with pytest.raises(RedisCacheError):
        RedisCache(
            "redis://localhost:6379/0",
            default_ttl_seconds=0,
        )
