from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import pytest
from lightgbm import LGBMRanker

from relevanceflow.serving.model_loader import (
    ModelLoadingError,
    get_model_feature_columns,
    load_registered_model,
)

MODEL_NAME = "RelevanceFlowRanker"
MODEL_ALIAS = "champion"

FEATURE_COLUMNS = [
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
]


@pytest.fixture()
def mlflow_test_registry(tmp_path: Path):
    """Create an isolated MLflow registry containing a tiny test model."""

    db_path = tmp_path / "mlflow.db"
    tracking_uri = f"sqlite:///{db_path.as_posix()}"

    mlflow.set_tracking_uri(tracking_uri)

    X = pd.DataFrame(
        [
            [
                1.0,
                0.8,
                1,
                0.5,
                0.4,
                1,
                5,
                20,
                3,
                2,
                4.5,
                100,
                80,
                1,
                1,
                1,
                0.9,
            ],
            [
                0.2,
                0.1,
                0,
                0.1,
                0.2,
                0,
                10,
                30,
                2,
                1,
                3.5,
                20,
                10,
                0,
                0,
                0,
                0.2,
            ],
            [
                0.8,
                0.6,
                1,
                0.4,
                0.5,
                1,
                7,
                25,
                4,
                2,
                4.2,
                80,
                60,
                1,
                1,
                1,
                0.8,
            ],
            [
                0.1,
                0.2,
                0,
                0.2,
                0.1,
                0,
                12,
                40,
                1,
                1,
                3.0,
                10,
                5,
                0,
                0,
                0,
                0.1,
            ],
        ],
        columns=FEATURE_COLUMNS,
    )

    y = np.array([2.0, 0.0, 2.0, 0.0])
    groups = [2, 2]

    model = LGBMRanker(
        objective="lambdarank",
        n_estimators=10,
        learning_rate=0.1,
        num_leaves=7,
        random_state=42,
        verbosity=-1,
    )

    model.fit(X, y, group=groups)

    experiment_name = "model-loader-tests"
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run():
        mlflow.lightgbm.log_model(
            lgb_model=model,
            name="model",
            registered_model_name=MODEL_NAME,
            skops_trusted_types=[
                "lightgbm.sklearn.LGBMRanker",
                "lightgbm.sklearn.LGBMClassifier",
                "lightgbm.basic.Booster",
                "lightgbm.sklearn.LGBMRegressor",
                "collections.OrderedDict",
            ],
        )

    client = mlflow.MlflowClient()

    versions = client.search_model_versions(f"name='{MODEL_NAME}'")

    assert versions

    latest_version = max(
        versions,
        key=lambda version: int(version.version),
    )

    client.set_registered_model_alias(
        MODEL_NAME,
        MODEL_ALIAS,
        latest_version.version,
    )

    yield tracking_uri

    mlflow.set_tracking_uri("sqlite:///mlflow.db")


def test_load_registered_champion_model(mlflow_test_registry):
    model = load_registered_model(
        model_name=MODEL_NAME,
        alias=MODEL_ALIAS,
        tracking_uri=mlflow_test_registry,
    )

    assert model is not None


def test_loaded_model_has_expected_features(mlflow_test_registry):
    model = load_registered_model(
        model_name=MODEL_NAME,
        alias=MODEL_ALIAS,
        tracking_uri=mlflow_test_registry,
    )

    feature_columns = get_model_feature_columns(model)

    assert feature_columns
    assert len(feature_columns) == 17


def test_missing_model_name():
    with pytest.raises(ModelLoadingError):
        load_registered_model(
            model_name="",
            alias=MODEL_ALIAS,
            tracking_uri="sqlite:///mlflow.db",
        )


def test_missing_alias():
    with pytest.raises(ModelLoadingError):
        load_registered_model(
            model_name=MODEL_NAME,
            alias="",
            tracking_uri="sqlite:///mlflow.db",
        )
