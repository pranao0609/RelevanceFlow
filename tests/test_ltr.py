import numpy as np
import pandas as pd
import pytest

from relevanceflow.models.ltr import (
    LightGBMLTR,
    LTRConfig,
)


@pytest.fixture
def training_data() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "query_id": [
                1,
                1,
                1,
                2,
                2,
                2,
                3,
                3,
                3,
            ],
            "product_id": [
                101,
                102,
                103,
                201,
                202,
                203,
                301,
                302,
                303,
            ],
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
            "relevance_score": [
                2,
                1,
                0,
                2,
                1,
                0,
                2,
                1,
                0,
            ],
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
    model = LightGBMLTR(
        LTRConfig(
            n_estimators=20,
            random_state=42,
        )
    )

    model.fit(training_data)

    assert model.model is not None
    assert len(model.feature_columns) == 4


def test_predictions_are_finite(training_data):
    model = LightGBMLTR(
        LTRConfig(
            n_estimators=20,
            random_state=42,
        )
    )

    model.fit(training_data)

    scores = model.predict_scores(training_data)

    assert len(scores) == len(training_data)
    assert np.isfinite(scores).all()


def test_group_sizes(training_data):
    model = LightGBMLTR()

    group = model._prepare_group(training_data)

    assert group.tolist() == [3, 3, 3]
    assert group.sum() == len(training_data)


def test_rank_candidates(training_data):
    model = LightGBMLTR(
        LTRConfig(
            n_estimators=20,
            random_state=42,
        )
    )

    model.fit(training_data)

    ranked = model.rank_candidates(training_data)

    assert {
        "query_id",
        "product_id",
        "score",
        "rank",
    }.issubset(ranked.columns)

    assert len(ranked) == len(training_data)

    for _, group in ranked.groupby("query_id"):
        assert group["rank"].tolist() == [1, 2, 3]


def test_missing_query_id_is_rejected(training_data):
    invalid = training_data.drop(columns=["query_id"])

    model = LightGBMLTR()

    with pytest.raises(ValueError, match="query_id"):
        model.fit(invalid)
