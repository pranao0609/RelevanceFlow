from __future__ import annotations

from fastapi.testclient import TestClient

from relevanceflow.serving.app import app, get_inference_service


class FakeInferenceService:
    def rank(self, query, products, top_k):
        return [
            {
                "product_id": product["product_id"],
                "product_name": product["product_name"],
                "score": float(len(products) - index),
                "rank": index + 1,
            }
            for index, product in enumerate(products[:top_k])
        ]

    def _initialize(self):
        return None


def override_inference_service():
    return FakeInferenceService()


def setup_function():
    app.dependency_overrides.clear()
    get_inference_service.cache_clear()


def teardown_function():
    app.dependency_overrides.clear()
    get_inference_service.cache_clear()


def test_health_endpoint():
    app.dependency_overrides[get_inference_service] = override_inference_service

    client = TestClient(app)

    response = client.get("/health")

    assert response.status_code == 200

    body = response.json()

    assert body["status"] == "ok"
    assert body["model"] == "RelevanceFlowRanker"
    assert body["alias"] == "champion"


def test_rank_endpoint():
    app.dependency_overrides[get_inference_service] = override_inference_service

    client = TestClient(app)

    payload = {
        "query": "wireless headphones",
        "products": [
            {
                "product_id": 1,
                "product_name": "Wireless Bluetooth Headphones",
                "product_class": "Electronics",
                "category_hierarchy": ("Electronics > Audio > Headphones"),
                "product_description": ("Wireless Bluetooth headphones"),
                "product_features": "Bluetooth, wireless",
                "rating_count": 100,
                "average_rating": 4.5,
                "review_count": 80,
            },
            {
                "product_id": 2,
                "product_name": "Cotton T Shirt",
                "product_class": "Clothing",
                "category_hierarchy": "Clothing > Shirts",
                "product_description": "Cotton casual shirt",
                "product_features": "Cotton",
                "rating_count": 50,
                "average_rating": 4.2,
                "review_count": 40,
            },
        ],
    }

    response = client.post("/rank", json=payload)

    assert response.status_code == 200

    body = response.json()

    assert body["query"] == "wireless headphones"
    assert len(body["results"]) == 2

    assert body["results"][0]["product_id"] == 1
    assert body["results"][0]["rank"] == 1

    assert body["results"][1]["product_id"] == 2
    assert body["results"][1]["rank"] == 2


def test_rank_rejects_empty_query():
    app.dependency_overrides[get_inference_service] = override_inference_service

    client = TestClient(app)

    payload = {
        "query": "",
        "products": [
            {
                "product_id": 1,
                "product_name": "Test Product",
            }
        ],
    }

    response = client.post("/rank", json=payload)

    assert response.status_code == 422


def test_rank_rejects_empty_products():
    app.dependency_overrides[get_inference_service] = override_inference_service

    client = TestClient(app)

    payload = {
        "query": "headphones",
        "products": [],
    }

    response = client.post("/rank", json=payload)

    assert response.status_code == 422
