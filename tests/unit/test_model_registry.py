from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from relevanceflow.utils.model_registry import (
    ModelRegistryError,
    get_candidate_alias,
    get_champion_alias,
    get_registry_model_name,
    register_run_model,
    set_alias,
    set_model_version_metadata,
)


@pytest.fixture
def config() -> dict:
    """Return a minimal MLflow registry configuration."""
    return {
        "production": {
            "mlflow": {
                "enabled": True,
                "tracking_uri": "sqlite:///mlflow.db",
                "experiment_name": "relevanceflow-ranking",
                "registered_model_name": "RelevanceFlowRanker",
                "candidate_alias": "candidate",
                "champion_alias": "champion",
            }
        }
    }


def test_get_registry_model_name(config):
    assert get_registry_model_name(config) == "RelevanceFlowRanker"


def test_get_candidate_alias(config):
    assert get_candidate_alias(config) == "candidate"


def test_get_champion_alias(config):
    assert get_champion_alias(config) == "champion"


def test_register_run_model():
    registered_version = MagicMock()
    registered_version.version = "1"

    with patch(
        "relevanceflow.utils.model_registry.mlflow.register_model",
        return_value=registered_version,
    ) as mock_register:
        result = register_run_model(
            run_id="run123",
            artifact_path="ranker",
            model_name="RelevanceFlowRanker",
        )

    mock_register.assert_called_once_with(
        model_uri="runs:/run123/ranker",
        name="RelevanceFlowRanker",
    )

    assert result.version == "1"


def test_register_run_model_error():
    with (
        patch(
            "relevanceflow.utils.model_registry.mlflow.register_model",
            side_effect=RuntimeError("registration failed"),
        ),
        pytest.raises(ModelRegistryError),
    ):
        register_run_model(
            run_id="run123",
            artifact_path="ranker",
            model_name="RelevanceFlowRanker",
        )


def test_set_alias():
    client = MagicMock()

    set_alias(
        client=client,
        model_name="RelevanceFlowRanker",
        alias="candidate",
        version="1",
    )

    client.set_registered_model_alias.assert_called_once_with(
        name="RelevanceFlowRanker",
        alias="candidate",
        version="1",
    )


def test_set_alias_error():
    client = MagicMock()

    client.set_registered_model_alias.side_effect = RuntimeError("alias failed")

    with pytest.raises(ModelRegistryError):
        set_alias(
            client=client,
            model_name="RelevanceFlowRanker",
            alias="candidate",
            version="1",
        )


def test_set_model_version_metadata():
    client = MagicMock()

    set_model_version_metadata(
        client=client,
        model_name="RelevanceFlowRanker",
        version="1",
        tags={
            "model_family": "LightGBM LambdaRank",
            "dataset": "WANDS",
        },
        description="Test model",
    )

    assert client.set_model_version_tag.call_count == 2

    client.update_model_version.assert_called_once_with(
        name="RelevanceFlowRanker",
        version="1",
        description="Test model",
    )
