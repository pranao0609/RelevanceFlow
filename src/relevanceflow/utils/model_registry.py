"""MLflow Model Registry utilities for RelevanceFlow."""

from __future__ import annotations

from typing import Any

import mlflow
from mlflow import MlflowClient

from relevanceflow.utils.config import (
    ConfigurationError,
    get_config_value,
)


class ModelRegistryError(RuntimeError):
    """Raised when model registry operations fail."""


def get_registry_model_name(config: dict[str, Any]) -> str:
    """Return the configured registered model name."""

    name = get_config_value(
        config,
        "production",
        "mlflow",
        "registered_model_name",
    )

    if not name or not str(name).strip():
        raise ConfigurationError(
            "registered_model_name is required when model registry is enabled."
        )

    return str(name).strip()


def get_candidate_alias(config: dict[str, Any]) -> str:
    """Return the configured candidate alias."""

    alias = get_config_value(
        config,
        "production",
        "mlflow",
        "candidate_alias",
        default="candidate",
    )

    return str(alias).strip()


def get_champion_alias(config: dict[str, Any]) -> str:
    """Return the configured champion alias."""

    alias = get_config_value(
        config,
        "production",
        "mlflow",
        "champion_alias",
        default="champion",
    )

    return str(alias).strip()


def get_mlflow_client() -> MlflowClient:
    """Return an MLflow client using the configured tracking URI."""

    return MlflowClient()


def register_run_model(
    run_id: str,
    artifact_path: str,
    model_name: str,
):
    """Register an MLflow model artifact as a new model version."""

    model_uri = f"runs:/{run_id}/{artifact_path}"

    try:
        return mlflow.register_model(
            model_uri=model_uri,
            name=model_name,
        )
    except Exception as exc:
        raise ModelRegistryError(f"Unable to register model '{model_name}'.") from exc


def set_model_version_metadata(
    client: MlflowClient,
    model_name: str,
    version: str,
    tags: dict[str, str],
    description: str | None = None,
) -> None:
    """Set tags and optional description on a model version."""

    for key, value in tags.items():
        client.set_model_version_tag(
            name=model_name,
            version=version,
            key=str(key),
            value=str(value),
        )

    if description:
        client.update_model_version(
            name=model_name,
            version=version,
            description=description,
        )


def set_alias(
    client: MlflowClient,
    model_name: str,
    alias: str,
    version: str,
) -> None:
    """Assign an MLflow alias to a model version."""

    try:
        client.set_registered_model_alias(
            name=model_name,
            alias=alias,
            version=version,
        )
    except Exception as exc:
        raise ModelRegistryError(
            f"Unable to assign alias '{alias}' "
            f"to model '{model_name}' version '{version}'."
        ) from exc


def get_model_version_by_alias(
    client: MlflowClient,
    model_name: str,
    alias: str,
):
    """Resolve a registered model version through an alias."""

    try:
        return client.get_model_version_by_alias(
            name=model_name,
            alias=alias,
        )
    except Exception as exc:
        raise ModelRegistryError(
            f"Unable to resolve alias '{alias}' " f"for model '{model_name}'."
        ) from exc
