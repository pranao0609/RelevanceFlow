import pytest

from relevanceflow.serving.model_loader import (
    ModelLoadingError,
    get_model_feature_columns,
    load_registered_model,
)


def test_load_registered_champion_model():
    model = load_registered_model(
        model_name="RelevanceFlowRanker",
        alias="champion",
        tracking_uri="sqlite:///mlflow.db",
    )

    assert model is not None


def test_loaded_model_has_expected_features():
    model = load_registered_model(
        model_name="RelevanceFlowRanker",
        alias="champion",
        tracking_uri="sqlite:///mlflow.db",
    )

    feature_columns = get_model_feature_columns(model)

    assert feature_columns
    assert len(feature_columns) == 17


def test_missing_model_name():
    with pytest.raises(ModelLoadingError):
        load_registered_model(
            model_name="",
            alias="champion",
            tracking_uri="sqlite:///mlflow.db",
        )


def test_missing_alias():
    with pytest.raises(ModelLoadingError):
        load_registered_model(
            model_name="RelevanceFlowRanker",
            alias="",
            tracking_uri="sqlite:///mlflow.db",
        )
