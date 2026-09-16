import numpy as np
import pandas as pd

from scripts.optimize_ltr import (
    evaluate_ndcg_at_10,
    prepare_features,
    prepare_groups,
)


def test_prepare_groups():
    dataframe = pd.DataFrame(
        {
            "query_id": [
                1,
                1,
                1,
                2,
                2,
                3,
            ]
        }
    )

    groups = prepare_groups(dataframe)

    assert groups.tolist() == [3, 2, 1]
    assert groups.sum() == len(dataframe)


def test_prepare_features():
    dataframe = pd.DataFrame(
        {
            "query_id": [1],
            "product_id": [10],
            "bm25_score": [2.0],
            "tfidf_similarity": [0.5],
            "relevance_score": [2],
            "label": ["Exact"],
        }
    )

    features = prepare_features(dataframe)

    assert list(features.columns) == [
        "bm25_score",
        "tfidf_similarity",
    ]


def test_evaluate_ndcg_at_10():
    dataframe = pd.DataFrame(
        {
            "query_id": [
                1,
                1,
                1,
            ],
            "product_id": [
                101,
                102,
                103,
            ],
            "relevance_score": [
                2,
                1,
                0,
            ],
        }
    )

    predictions = pd.Series(
        np.array(
            [
                0.9,
                0.5,
                0.1,
            ]
        )
    )

    score = evaluate_ndcg_at_10(
        predictions,
        dataframe,
    )

    assert score == 1.0
