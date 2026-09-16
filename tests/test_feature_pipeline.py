import pandas as pd

from relevanceflow.features import (
    FeatureConfig,
    build_query_product_features,
)


def test_feature_pipeline():
    pairs = pd.DataFrame(
        [
            {
                "query_id": "q1",
                "product_id": "p1",
                "query": "wooden chair",
                "product_name": "Modern Wooden Chair",
                "product_description": ("Comfortable wooden dining chair"),
                "product_features": ("wood\ncomfortable\nindoor"),
                "category hierarchy": ("Furniture > Chairs > Dining Chairs"),
                "product_class": "Chair",
                "average_rating": 4.5,
                "rating_count": 100,
                "review_count": 50,
                "bm25_score": 2.5,
                "tfidf_similarity": 0.7,
            }
        ]
    )

    result = build_query_product_features(
        pairs,
        FeatureConfig(),
    )

    assert len(result) == 1

    expected_features = {
        "bm25_score",
        "tfidf_similarity",
        "exact_match",
        "token_overlap",
        "character_overlap",
        "query_title_overlap",
        "title_length",
        "description_length",
        "feature_count",
        "category_depth",
        "average_rating",
        "rating_count",
        "review_count",
        "category_match",
        "product_class_match",
        "phrase_match",
        "semantic_similarity",
        "query_embedding_norm",
        "product_embedding_norm",
    }

    assert expected_features.issubset(set(result.columns))


def test_semantic_features_are_zero_initially():
    pairs = pd.DataFrame(
        [
            {
                "query_id": "q1",
                "product_id": "p1",
                "query": "chair",
                "product_name": "wooden chair",
            }
        ]
    )

    result = build_query_product_features(pairs)

    assert result["semantic_similarity"].iloc[0] == 0.0
    assert result["query_embedding_norm"].iloc[0] == 0.0
    assert result["product_embedding_norm"].iloc[0] == 0.0
