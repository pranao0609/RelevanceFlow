from __future__ import annotations

from functools import lru_cache

from fastapi import Depends, FastAPI, HTTPException

from relevanceflow.serving.inference import (
    InferenceConfig,
    RankingInferenceService,
)
from relevanceflow.serving.model_loader import ModelLoadingError
from relevanceflow.serving.schemas import (
    HealthResponse,
    RankingRequest,
    RankingResponse,
)


@lru_cache(maxsize=1)
def get_inference_service() -> RankingInferenceService:
    """Create and cache the ranking inference service."""
    return RankingInferenceService(InferenceConfig())


app = FastAPI(
    title="RelevanceFlow Ranking API",
    description="E-commerce product search relevance and ranking API.",
    version="0.1.0",
)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Return API configuration health."""
    config = InferenceConfig()

    return HealthResponse(
        status="ok",
        model=config.model_name,
        alias=config.model_alias,
    )


@app.post("/rank", response_model=RankingResponse)
def rank(
    request: RankingRequest,
    service: RankingInferenceService = Depends(get_inference_service),
) -> RankingResponse:
    """Rank candidate products for a search query."""
    try:
        results = service.rank(
            query=request.query,
            products=[product.model_dump() for product in request.products],
            top_k=len(request.products),
        )

    except ModelLoadingError as exc:
        raise HTTPException(
            status_code=503,
            detail=f"Ranking model unavailable: {exc}",
        ) from exc

    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Ranking inference failed: {exc}",
        ) from exc

    return RankingResponse(
        query=request.query,
        results=results,
    )
