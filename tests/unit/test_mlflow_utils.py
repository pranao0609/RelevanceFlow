from pathlib import Path
from unittest.mock import patch

import mlflow
import pytest

from relevanceflow.utils.mlflow_utils import (
    MLflowTrackingError,
    configure_mlflow,
    get_experiment_name,
    get_git_commit,
    log_git_metadata,
    log_metrics,
    log_params,
    setup_experiment,
    start_run,
)


def test_get_git_commit() -> None:
    commit = get_git_commit(".")

    assert len(commit) == 40
    assert all(character in "0123456789abcdef" for character in commit)


def test_get_git_commit_invalid_repository(
    tmp_path: Path,
) -> None:
    with pytest.raises(MLflowTrackingError):
        get_git_commit(tmp_path)


def test_configure_mlflow_disabled() -> None:
    config = {
        "production": {
            "mlflow": {
                "enabled": False,
            }
        }
    }

    assert configure_mlflow(config) is False


def test_configure_mlflow_sqlite(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "mlflow.db"

    config = {
        "production": {
            "mlflow": {
                "enabled": True,
                "tracking_uri": (f"sqlite:///{database_path.as_posix()}"),
            }
        }
    }

    assert (
        configure_mlflow(
            config,
            tmp_path,
        )
        is True
    )


def test_configure_mlflow_requires_tracking_uri() -> None:
    config = {
        "production": {
            "mlflow": {
                "enabled": True,
            }
        }
    }

    with pytest.raises(RuntimeError):
        configure_mlflow(config)


def test_get_experiment_name() -> None:
    config = {
        "production": {
            "mlflow": {
                "experiment_name": ("relevanceflow-ranking"),
            }
        }
    }

    assert get_experiment_name(config) == "relevanceflow-ranking"


def test_get_experiment_name_requires_value() -> None:
    config = {"production": {"mlflow": {}}}

    with pytest.raises(RuntimeError):
        get_experiment_name(config)


def test_setup_experiment(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "mlflow.db"

    config = {
        "production": {
            "mlflow": {
                "enabled": True,
                "tracking_uri": (f"sqlite:///{database_path.as_posix()}"),
                "experiment_name": ("test-relevanceflow-ranking"),
            }
        }
    }

    experiment_id = setup_experiment(
        config,
        tmp_path,
    )

    assert experiment_id is not None

    experiment = mlflow.get_experiment(experiment_id)

    assert experiment is not None
    assert experiment.name == "test-relevanceflow-ranking"


@patch("relevanceflow.utils.mlflow_utils.mlflow.log_params")
def test_log_params(mock_log_params) -> None:
    log_params(
        {
            "learning_rate": 0.05,
            "num_leaves": 31,
        }
    )

    mock_log_params.assert_called_once_with(
        {
            "learning_rate": "0.05",
            "num_leaves": "31",
        }
    )


@patch("relevanceflow.utils.mlflow_utils.mlflow.log_metrics")
def test_log_metrics(mock_log_metrics) -> None:
    log_metrics(
        {
            "ndcg@10": 0.742351,
            "mrr@10": 0.933697,
        }
    )

    mock_log_metrics.assert_called_once_with(
        {
            "ndcg@10": 0.742351,
            "mrr@10": 0.933697,
        }
    )


@patch("relevanceflow.utils.mlflow_utils.mlflow.set_tag")
@patch("relevanceflow.utils.mlflow_utils.get_git_commit")
def test_log_git_metadata(
    mock_get_git_commit,
    mock_set_tag,
) -> None:
    mock_get_git_commit.return_value = "abc123"

    commit = log_git_metadata(".")

    assert commit == "abc123"

    mock_set_tag.assert_called_once_with(
        "git_commit",
        "abc123",
    )


def test_start_run_creates_run(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "mlflow.db"

    config = {
        "production": {
            "mlflow": {
                "enabled": True,
                "tracking_uri": (f"sqlite:///{database_path.as_posix()}"),
                "experiment_name": ("test-relevanceflow-ranking"),
            }
        }
    }

    with start_run(
        config=config,
        run_name="test-run",
        project_root=tmp_path,
    ) as run:
        assert run is not None
        assert run.info.run_id is not None

        mlflow.log_param(
            "model_type",
            "test",
        )


def test_start_run_disabled(
    tmp_path: Path,
) -> None:
    config = {
        "production": {
            "mlflow": {
                "enabled": False,
            }
        }
    }

    with start_run(
        config=config,
        run_name="disabled-run",
        project_root=tmp_path,
    ) as run:
        assert run is None
