from __future__ import annotations

import time
import uuid
from contextlib import asynccontextmanager
from functools import lru_cache

from fastapi import Depends, FastAPI, HTTPException, Request

from relevanceflow.serving.inference import (
    InferenceConfig,
    RankingInferenceService,
)
from relevanceflow.serving.logging import (
    configure_logging,
    get_logger,
)
from relevanceflow.serving.model_loader import ModelLoadingError
from relevanceflow.serving.schemas import (
    HealthResponse,
    RankingRequest,
    RankingResponse,
)

configure_logging()

logger = get_logger(__name__)


@lru_cache(maxsize=1)
def get_inference_service() -> RankingInferenceService:
    """Create and cache the ranking inference service."""
    return RankingInferenceService(InferenceConfig())


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage FastAPI application lifecycle."""
    logger.info("Starting RelevanceFlow Ranking API")

    yield

    logger.info("Stopping RelevanceFlow Ranking API")

    get_inference_service.cache_clear()


app = FastAPI(
    title="RelevanceFlow Ranking API",
    description="E-commerce product search relevance and ranking API.",
    version="0.1.0",
    lifespan=lifespan,
)


@app.middleware("http")
async def request_logging_middleware(request: Request, call_next):
    """Add request IDs and record request latency."""
    request_id = str(uuid.uuid4())

    request.state.request_id = request_id

    start_time = time.perf_counter()

    response = None

    try:
        response = await call_next(request)

        return response

    finally:
        elapsed_ms = (time.perf_counter() - start_time) * 1000

        if response is not None:
            response.headers["X-Request-ID"] = request_id

            logger.info(
                "request_id=%s method=%s path=%s " "status=%s latency_ms=%.2f",
                request_id,
                request.method,
                request.url.path,
                response.status_code,
                elapsed_ms,
            )
        else:
            logger.exception(
                "request_id=%s method=%s path=%s " "request_failed latency_ms=%.2f",
                request_id,
                request.method,
                request.url.path,
                elapsed_ms,
            )


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Return API configuration health."""
    config = InferenceConfig()

    logger.info(
        "health_check model=%s alias=%s",
        config.model_name,
        config.model_alias,
    )

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
    getattr(request, "state", None)

    try:
        results = service.rank(
            query=request.query,
            products=[product.model_dump() for product in request.products],
            top_k=len(request.products),
        )

        logger.info(
            "ranking_success query_length=%d " "candidate_count=%d result_count=%d",
            len(request.query),
            len(request.products),
            len(results),
        )

        return RankingResponse(
            query=request.query,
            results=results,
        )

    except ModelLoadingError as exc:
        logger.error(
            "ranking_model_unavailable error=%s",
            exc,
        )

        raise HTTPException(
            status_code=503,
            detail=f"Ranking model unavailable: {exc}",
        ) from exc

    except ValueError as exc:
        logger.warning(
            "ranking_invalid_request error=%s",
            exc,
        )

        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc

    except Exception as exc:
        logger.exception("ranking_inference_failed error")

        raise HTTPException(
            status_code=500,
            detail="Ranking inference failed.",
        ) from exc
