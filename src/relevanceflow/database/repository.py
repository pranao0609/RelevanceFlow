from __future__ import annotations

from sqlalchemy.orm import Session

from relevanceflow.database.models import RankingRequest


def create_ranking_request(
    session: Session,
    *,
    request_id: str,
    query: str,
    candidate_count: int,
    top_k: int,
    model_name: str,
    model_alias: str,
    latency_ms: float | None = None,
) -> RankingRequest:
    """Persist ranking request metadata."""

    record = RankingRequest(
        request_id=request_id,
        query=query,
        candidate_count=candidate_count,
        top_k=top_k,
        model_name=model_name,
        model_alias=model_alias,
        latency_ms=latency_ms,
    )

    session.add(record)
    session.commit()
    session.refresh(record)

    return record
