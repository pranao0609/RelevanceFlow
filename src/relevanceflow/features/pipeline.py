from __future__ import annotations

import pandas as pd

from relevanceflow.features.compatibility import (
    compatibility_features,
)
from relevanceflow.features.config import FeatureConfig
from relevanceflow.features.lexical import lexical_features
from relevanceflow.features.metadata import metadata_features
from relevanceflow.features.text import (
    category_depth,
    description_length,
    feature_count,
    title_length,
)

SEMANTIC_FEATURES = [
    "semantic_similarity",
    "query_embedding_norm",
    "product_embedding_norm",
]


def build_query_product_features(
    pairs: pd.DataFrame,
    config: FeatureConfig | None = None,
) -> pd.DataFrame:
    """
    Build reusable query-product feature matrix.

    Required columns:
        query_id
        product_id
        query
        product_name

    Optional product columns:
        product_description
        product_features
        category hierarchy
        product_class
        average_rating
        rating_count
        review_count
        bm25_score
        tfidf_similarity
    """
    config = config or FeatureConfig()

    required_columns = {
        "query_id",
        "product_id",
        "query",
        "product_name",
    }

    missing = required_columns - set(pairs.columns)

    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    features = pd.DataFrame(
        {
            "query_id": pairs["query_id"].values,
            "product_id": pairs["product_id"].values,
        }
    )

    for index, row in pairs.reset_index(drop=True).iterrows():
        query = row["query"]
        product_name = row["product_name"]

        row_features: dict[str, float] = {}

        if config.lexical_enabled:
            row_features.update(
                lexical_features(
                    query=query,
                    product_name=product_name,
                    bm25_score=row.get(
                        "bm25_score",
                        0.0,
                    ),
                    tfidf_similarity=row.get(
                        "tfidf_similarity",
                        0.0,
                    ),
                )
            )

        if config.product_text_enabled:
            row_features.update(
                {
                    "title_length": title_length(product_name),
                    "description_length": description_length(
                        row.get(
                            "product_description",
                            "",
                        )
                    ),
                    "feature_count": feature_count(
                        row.get(
                            "product_features",
                            "",
                        )
                    ),
                    "category_depth": category_depth(
                        row.get(
                            "category hierarchy",
                            "",
                        )
                    ),
                }
            )

        if config.metadata_enabled:
            row_features.update(
                metadata_features(
                    average_rating=row.get(
                        "average_rating",
                        0.0,
                    ),
                    rating_count=row.get(
                        "rating_count",
                        0.0,
                    ),
                    review_count=row.get(
                        "review_count",
                        0.0,
                    ),
                )
            )

        if config.compatibility_enabled:
            row_features.update(
                compatibility_features(
                    query=query,
                    product_name=product_name,
                    category_hierarchy=row.get(
                        "category hierarchy",
                        "",
                    ),
                    product_class=row.get(
                        "product_class",
                        "",
                    ),
                )
            )

        if config.semantic_enabled:
            for feature_name in SEMANTIC_FEATURES:
                row_features[feature_name] = 0.0
        else:
            for feature_name in SEMANTIC_FEATURES:
                row_features[feature_name] = 0.0

        for feature_name, value in row_features.items():
            if feature_name not in features:
                features[feature_name] = 0.0

            features.loc[index, feature_name] = value

    return features
