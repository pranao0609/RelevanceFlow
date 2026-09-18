from __future__ import annotations

from fastapi.testclient import TestClient
from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from relevanceflow.database.models import RankingRequest
from relevanceflow.database.session import Base, get_session
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


def override_inference_service():
    return FakeInferenceService()


def create_test_database():
    """Create a thread-safe shared in-memory SQLite database."""
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )

    Base.metadata.create_all(engine)

    session_factory = sessionmaker(
        bind=engine,
        autoflush=False,
        autocommit=False,
    )

    return engine, session_factory


def test_rank_persists_request_metadata():
    engine, session_factory = create_test_database()

    def override_session():
        session = session_factory()

        try:
            yield session
        finally:
            session.close()

    app.dependency_overrides[get_inference_service] = override_inference_service
    app.dependency_overrides[get_session] = override_session

    try:
        client = TestClient(app)

        payload = {
            "query": "wireless headphones",
            "products": [
                {
                    "product_id": 1,
                    "product_name": "Wireless Bluetooth Headphones",
                },
                {
                    "product_id": 2,
                    "product_name": "Cotton T Shirt",
                },
            ],
        }

        response = client.post("/rank", json=payload)

        assert response.status_code == 200
        assert "X-Request-ID" in response.headers

        request_id = response.headers["X-Request-ID"]

        with session_factory() as session:
            record = session.execute(
                select(RankingRequest).where(RankingRequest.request_id == request_id)
            ).scalar_one()

            assert record.query == "wireless headphones"
            assert record.candidate_count == 2
            assert record.top_k == 2
            assert record.model_name == "RelevanceFlowRanker"
            assert record.model_alias == "champion"
            assert record.latency_ms is not None

    finally:
        app.dependency_overrides.clear()
        get_inference_service.cache_clear()
        engine.dispose()
