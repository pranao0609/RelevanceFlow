from fastapi.testclient import TestClient

from relevanceflow.serving.app import (
    app,
    get_cache,
    get_inference_service,
)


class FakeCache:
    def __init__(self):
        self.data = {}

    def get(self, key):
        return self.data.get(key)

    def set(self, key, value):
        self.data[key] = value


class FakeService:
    def __init__(self):
        self.calls = 0

    def rank(self, query, products, top_k):
        self.calls += 1

        return [
            {
                "product_id": 1,
                "product_name": "Test",
                "score": 1.0,
                "rank": 1,
            }
        ]


def test_cache_hit_skips_second_inference():
    cache = FakeCache()
    service = FakeService()

    app.dependency_overrides[get_cache] = lambda: cache
    app.dependency_overrides[get_inference_service] = lambda: service

    client = TestClient(app)

    payload = {
        "query": "wireless headphones",
        "products": [
            {
                "product_id": 1,
                "product_name": "Test",
            }
        ],
    }

    client.post("/rank", json=payload)

    client.post("/rank", json=payload)

    assert service.calls == 1

    app.dependency_overrides.clear()
