from __future__ import annotations

from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker

from relevanceflow.database.models import RankingRequest
from relevanceflow.database.repository import create_ranking_request
from relevanceflow.database.session import Base, get_database_url


def create_test_session():
    """Create an isolated in-memory SQLite database."""
    engine = create_engine("sqlite:///:memory:")

    Base.metadata.create_all(engine)

    session_factory = sessionmaker(
        bind=engine,
        autoflush=False,
        autocommit=False,
    )

    return engine, session_factory()


def test_database_url_has_postgresql_default():
    url = get_database_url()

    assert url.startswith("postgresql+psycopg://")


def test_ranking_request_table_can_be_created():
    engine, session = create_test_session()

    try:
        result = session.execute(select(RankingRequest)).scalars().all()

        assert result == []
    finally:
        session.close()
        engine.dispose()


def test_create_ranking_request():
    engine, session = create_test_session()

    try:
        record = create_ranking_request(
            session,
            request_id="test-request-001",
            query="wireless headphones",
            candidate_count=25,
            top_k=10,
            model_name="RelevanceFlowRanker",
            model_alias="champion",
            latency_ms=42.5,
        )

        assert record.id is not None
        assert record.request_id == "test-request-001"
        assert record.query == "wireless headphones"
        assert record.candidate_count == 25
        assert record.top_k == 10
        assert record.model_name == "RelevanceFlowRanker"
        assert record.model_alias == "champion"
        assert record.latency_ms == 42.5
        assert record.created_at is not None
    finally:
        session.close()
        engine.dispose()


def test_ranking_request_can_be_retrieved():
    engine, session = create_test_session()

    try:
        create_ranking_request(
            session,
            request_id="test-request-002",
            query="bluetooth speaker",
            candidate_count=15,
            top_k=5,
            model_name="RelevanceFlowRanker",
            model_alias="champion",
        )

        record = session.execute(
            select(RankingRequest).where(
                RankingRequest.request_id == "test-request-002"
            )
        ).scalar_one()

        assert record.query == "bluetooth speaker"
        assert record.candidate_count == 15
        assert record.top_k == 5
    finally:
        session.close()
        engine.dispose()
