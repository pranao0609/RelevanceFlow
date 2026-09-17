"""MLflow experiment tracking utilities for RelevanceFlow."""

from __future__ import annotations

import subprocess
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import mlflow

from relevanceflow.utils.config import (
    ConfigurationError,
    get_config_value,
)


class MLflowTrackingError(RuntimeError):
    """Raised when MLflow tracking configuration is invalid."""


def get_git_commit(project_root: str | Path = ".") -> str:
    """Return the current Git commit hash.

    Returns
    -------
    str
        Full Git commit hash.

    Raises
    ------
    MLflowTrackingError
        If the Git commit cannot be determined.
    """
    root = Path(project_root)

    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise MLflowTrackingError(
            "Unable to determine the current Git commit."
        ) from exc

    commit = result.stdout.strip()

    if not commit:
        raise MLflowTrackingError("Git returned an empty commit hash.")

    return commit


def configure_mlflow(
    config: dict[str, Any],
    project_root: str | Path = ".",
) -> bool:
    """Configure MLflow from RelevanceFlow configuration."""
    del project_root

    enabled = bool(
        get_config_value(
            config,
            "production",
            "mlflow",
            "enabled",
            default=False,
        )
    )

    if not enabled:
        return False

    tracking_uri = get_config_value(
        config,
        "production",
        "mlflow",
        "tracking_uri",
    )

    if not tracking_uri:
        raise ConfigurationError(
            "MLflow tracking URI is required when MLflow is enabled."
        )

    mlflow.set_tracking_uri(str(tracking_uri))

    return True


def get_experiment_name(config: dict[str, Any]) -> str:
    """Return the configured MLflow experiment name."""

    experiment_name = get_config_value(
        config,
        "production",
        "mlflow",
        "experiment_name",
    )

    if not experiment_name or not str(experiment_name).strip():
        raise ConfigurationError(
            "MLflow experiment_name is required when MLflow is enabled."
        )

    return str(experiment_name).strip()


def setup_experiment(
    config: dict[str, Any],
    project_root: str | Path = ".",
) -> str | None:
    """Configure MLflow and select the RelevanceFlow experiment."""

    enabled = configure_mlflow(
        config=config,
        project_root=project_root,
    )

    if not enabled:
        return None

    experiment_name = get_experiment_name(config)

    experiment = mlflow.get_experiment_by_name(experiment_name)

    if experiment is None:
        experiment_id = mlflow.create_experiment(name=experiment_name)
    else:
        experiment_id = experiment.experiment_id

    mlflow.set_experiment(experiment_name=experiment_name)

    return str(experiment_id)


def log_params(params: dict[str, Any]) -> None:
    """Log model or experiment parameters."""
    if not params:
        return

    normalized = {
        str(key): str(value) for key, value in params.items() if value is not None
    }

    mlflow.log_params(normalized)


def log_metrics(metrics: dict[str, float]) -> None:
    """Log numeric experiment metrics."""
    if not metrics:
        return

    normalized = {
        str(key): float(value) for key, value in metrics.items() if value is not None
    }

    mlflow.log_metrics(normalized)


def log_git_metadata(project_root: str | Path = ".") -> str:
    """Log the current Git commit as an MLflow tag."""
    commit = get_git_commit(project_root)

    mlflow.set_tag(
        "git_commit",
        commit,
    )

    return commit


@contextmanager
def start_run(
    config: dict[str, Any],
    run_name: str,
    project_root: str | Path = ".",
) -> Iterator[Any]:
    """Start an MLflow run when tracking is enabled."""
    experiment_name = setup_experiment(
        config=config,
        project_root=project_root,
    )

    if experiment_name is None:
        yield None
        return

    with mlflow.start_run(run_name=run_name) as run:
        yield run


def log_artifact_if_exists(path: str | Path) -> None:
    """Log a local artifact when the file exists."""
    artifact_path = Path(path)

    if artifact_path.exists():
        mlflow.log_artifact(str(artifact_path))
