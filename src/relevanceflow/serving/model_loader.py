from __future__ import annotations

from typing import Any

import mlflow


class ModelLoadingError(RuntimeError):
    """Raised when the production ranking model cannot be loaded."""


def configure_mlflow(tracking_uri: str) -> None:
    """Configure MLflow tracking for the serving process."""
    if not tracking_uri:
        raise ModelLoadingError("MLflow tracking URI is empty.")

    mlflow.set_tracking_uri(tracking_uri)


def load_registered_model(
    model_name: str,
    alias: str,
    tracking_uri: str | None = None,
) -> Any:
    """Load a registered MLflow model by alias."""

    if tracking_uri:
        configure_mlflow(tracking_uri)

    if not model_name:
        raise ModelLoadingError("Model name is empty.")

    if not alias:
        raise ModelLoadingError("Model alias is empty.")

    model_uri = f"models:/{model_name}@{alias}"

    try:
        return mlflow.pyfunc.load_model(model_uri)
    except Exception as exc:
        raise ModelLoadingError(
            f"Failed to load registered model '{model_uri}': {exc}"
        ) from exc


def get_model_feature_columns(model: Any) -> list[str]:
    """Extract feature names expected by the underlying LightGBM model."""

    try:
        booster = model._model_impl.lgb_model.booster_
        feature_names = booster.feature_name()
    except Exception as exc:
        raise ModelLoadingError(
            f"Unable to read LightGBM feature names: {exc}"
        ) from exc

    if not feature_names:
        raise ModelLoadingError("Loaded LightGBM model has no feature names.")

    return list(feature_names)


def prepare_model_features(
    features,
    feature_columns: list[str],
):
    """Select model features in the exact training order."""

    missing = [column for column in feature_columns if column not in features.columns]

    if missing:
        raise ModelLoadingError(f"Missing model features: {missing}")

    return features.loc[:, feature_columns]
