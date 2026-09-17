from pydantic import BaseModel, Field


class ProductRequest(BaseModel):
    product_id: int
    product_name: str

    product_class: str | None = None
    category_hierarchy: str | None = None
    product_description: str | None = None
    product_features: str | None = None

    rating_count: float | None = None
    average_rating: float | None = None
    review_count: float | None = None


class RankingRequest(BaseModel):
    query: str = Field(..., min_length=1)
    products: list[ProductRequest] = Field(..., min_length=1)


class RankedProduct(BaseModel):
    product_id: int
    product_name: str
    score: float
    rank: int


class RankingResponse(BaseModel):
    query: str
    results: list[RankedProduct]


class HealthResponse(BaseModel):
    status: str
    model: str
    alias: str
