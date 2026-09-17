"""MLflow registered-model loading utilities."""

from __future__ import annotations

import mlflow.lightgbm
import pandas as pd


class ModelLoadingError(RuntimeError):
    """Raised when a registered model cannot be loaded correctly."""


def load_registered_model(
    model_name: str,
    alias: str,
):
    """Load a registered LightGBM model through an MLflow alias."""

    model_uri = f"models:/{model_name}@{alias}"

    try:
        return mlflow.lightgbm.load_model(model_uri)
    except Exception as exc:
        raise ModelLoadingError(
            f"Unable to load model '{model_name}@{alias}'."
        ) from exc


def get_model_feature_columns(
    model,
) -> list[str]:
    """Return the exact feature columns used by the trained model."""

    feature_columns = getattr(
        model,
        "feature_name_",
        None,
    )

    if not feature_columns:
        raise ModelLoadingError("Registered model does not expose feature_name_.")

    return list(feature_columns)


def prepare_model_features(
    dataframe: pd.DataFrame,
    model,
) -> pd.DataFrame:
    """Prepare inference features using the trained model schema."""

    feature_columns = get_model_feature_columns(model)

    missing_columns = [
        column for column in feature_columns if column not in dataframe.columns
    ]

    if missing_columns:
        raise ModelLoadingError(
            "Input data is missing required model features: "
            + ", ".join(missing_columns)
        )

    return dataframe.loc[
        :,
        feature_columns,
    ].copy()
