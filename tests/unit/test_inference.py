import pytest

from relevanceflow.serving.inference import (
    InferenceConfig,
    RankingInferenceService,
)


def test_normalize_product_maps_category_field():
    product = {
        "product_id": 123,
        "product_name": "Wireless Headphones",
        "category_hierarchy": "Electronics > Audio",
    }

    normalized = RankingInferenceService._normalize_product(product)

    assert normalized["product_id"] == 123
    assert normalized["product_name"] == ("Wireless Headphones")
    assert normalized["category hierarchy"] == ("Electronics > Audio")


def test_normalize_product_defaults_missing_values():
    product = {
        "product_id": 123,
        "product_name": "Headphones",
    }

    normalized = RankingInferenceService._normalize_product(product)

    assert normalized["product_class"] == ""
    assert normalized["category hierarchy"] == ""
    assert normalized["product_description"] == ""
    assert normalized["product_features"] == ""
    assert normalized["rating_count"] == 0.0
    assert normalized["average_rating"] == 0.0
    assert normalized["review_count"] == 0.0


def test_empty_query_is_rejected():
    service = RankingInferenceService(InferenceConfig())

    with pytest.raises(
        ValueError,
        match="Query",
    ):
        service.rank(
            query="",
            products=[
                {
                    "product_id": 1,
                    "product_name": "Headphones",
                }
            ],
        )


def test_whitespace_query_is_rejected():
    service = RankingInferenceService(InferenceConfig())

    with pytest.raises(
        ValueError,
        match="Query",
    ):
        service.rank(
            query="   ",
            products=[
                {
                    "product_id": 1,
                    "product_name": "Headphones",
                }
            ],
        )


def test_empty_products_are_rejected():
    service = RankingInferenceService(InferenceConfig())

    with pytest.raises(
        ValueError,
        match="At least one product",
    ):
        service.rank(
            query="headphones",
            products=[],
        )


def test_non_positive_top_k_is_rejected():
    service = RankingInferenceService(InferenceConfig())

    # top_k validation must happen before expensive
    # model/retriever initialization.
    with pytest.raises(
        ValueError,
        match="top_k",
    ):
        service.rank(
            query="headphones",
            products=[
                {
                    "product_id": 1,
                    "product_name": "Headphones",
                }
            ],
            top_k=0,
        )


def test_negative_top_k_is_rejected():
    service = RankingInferenceService(InferenceConfig())

    with pytest.raises(
        ValueError,
        match="top_k",
    ):
        service.rank(
            query="headphones",
            products=[
                {
                    "product_id": 1,
                    "product_name": "Headphones",
                }
            ],
            top_k=-1,
        )
