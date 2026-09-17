from __future__ import annotations

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from relevanceflow.serving.model_loader import (
    ModelLoadingError,
    get_model_feature_columns,
    load_registered_model,
    prepare_model_features,
)


def test_load_registered_model():
    mock_model = MagicMock()

    with patch(
        "relevanceflow.serving.model_loader.mlflow.lightgbm.load_model",
        return_value=mock_model,
    ) as mock_load:
        result = load_registered_model(
            "RelevanceFlowRanker",
            "champion",
        )

    mock_load.assert_called_once_with("models:/RelevanceFlowRanker@champion")

    assert result is mock_model


def test_load_registered_model_error():
    with patch(
        "relevanceflow.serving.model_loader.mlflow.lightgbm.load_model",
        side_effect=RuntimeError("load failed"),
    ), pytest.raises(ModelLoadingError):
        load_registered_model(
            "RelevanceFlowRanker",
            "champion",
        )


def test_get_model_feature_columns():
    model = MagicMock()
    model.feature_name_ = [
        "bm25_score",
        "semantic_similarity",
    ]

    result = get_model_feature_columns(model)

    assert result == [
        "bm25_score",
        "semantic_similarity",
    ]


def test_get_model_feature_columns_missing():
    model = MagicMock()
    model.feature_name_ = None

    with pytest.raises(ModelLoadingError):
        get_model_feature_columns(model)


def test_prepare_model_features_preserves_order():
    model = MagicMock()
    model.feature_name_ = [
        "semantic_similarity",
        "bm25_score",
    ]

    dataframe = pd.DataFrame(
        {
            "bm25_score": [1.0, 2.0],
            "semantic_similarity": [0.5, 0.6],
            "extra_feature": [10.0, 20.0],
        }
    )

    result = prepare_model_features(
        dataframe,
        model,
    )

    assert result.columns.tolist() == [
        "semantic_similarity",
        "bm25_score",
    ]

    assert result.shape == (2, 2)


def test_prepare_model_features_missing_column():
    model = MagicMock()
    model.feature_name_ = [
        "bm25_score",
        "semantic_similarity",
    ]

    dataframe = pd.DataFrame(
        {
            "bm25_score": [1.0],
        }
    )

    with pytest.raises(ModelLoadingError):
        prepare_model_features(
            dataframe,
            model,
        )
