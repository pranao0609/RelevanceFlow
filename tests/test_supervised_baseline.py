import numpy as np
import pandas as pd
import pytest

from relevanceflow.models.supervised_baseline import (
    SupervisedBaseline,
    SupervisedBaselineConfig,
)


@pytest.fixture
def training_data() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "query_id": [1, 1, 1, 2, 2, 2, 3, 3, 3],
            "product_id": [101, 102, 103, 201, 202, 203, 301, 302, 303],
            "bm25_score": [
                5.0,
                1.0,
                0.2,
                4.5,
                1.2,
                0.1,
                4.8,
                1.0,
                0.2,
            ],
            "tfidf_similarity": [
                0.9,
                0.3,
                0.05,
                0.8,
                0.25,
                0.03,
                0.85,
                0.2,
                0.04,
            ],
            "title_length": [
                3,
                5,
                8,
                3,
                6,
                9,
                3,
                5,
                8,
            ],
            "category_match": [
                1,
                0,
                0,
                1,
                0,
                0,
                1,
                0,
                0,
            ],
            "relevance_score": [2, 1, 0, 2, 1, 0, 2, 1, 0],
            "label": [
                "Exact",
                "Partial",
                "Irrelevant",
                "Exact",
                "Partial",
                "Irrelevant",
                "Exact",
                "Partial",
                "Irrelevant",
            ],
        }
    )


def test_model_trains(training_data):
    model = SupervisedBaseline(
        SupervisedBaselineConfig(
            n_estimators=20,
            random_state=42,
        )
    )

    model.fit(training_data)

    assert model.model is not None
    assert len(model.feature_columns) == 4


def test_prediction_shape(training_data):
    model = SupervisedBaseline(
        SupervisedBaselineConfig(
            n_estimators=20,
            random_state=42,
        )
    )

    model.fit(training_data)

    scores = model.predict_scores(training_data)

    assert len(scores) == len(training_data)
    assert np.isfinite(scores).all()


def test_scores_are_probabilities(training_data):
    model = SupervisedBaseline(
        SupervisedBaselineConfig(
            n_estimators=20,
            random_state=42,
        )
    )

    model.fit(training_data)

    scores = model.predict_scores(training_data)

    assert (scores >= 0).all()
    assert (scores <= 1).all()


def test_rank_candidates(training_data):
    model = SupervisedBaseline(
        SupervisedBaselineConfig(
            n_estimators=20,
            random_state=42,
        )
    )

    model.fit(training_data)

    ranked = model.rank_candidates(training_data)

    assert {"query_id", "product_id", "score", "rank"} <= set(ranked.columns)

    assert ranked["rank"].min() == 1

    for _, group in ranked.groupby("query_id"):
        assert group["score"].is_monotonic_decreasing


def test_model_rejects_missing_target():
    dataframe = pd.DataFrame(
        {
            "query_id": [1],
            "product_id": [10],
            "bm25_score": [1.0],
        }
    )

    model = SupervisedBaseline()

    with pytest.raises(ValueError, match="Missing target"):
        model.fit(dataframe)
