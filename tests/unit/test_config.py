from pathlib import Path

import pytest

from relevanceflow.utils.config import (
    ConfigurationError,
    get_config_value,
    load_configs,
    load_yaml_config,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = PROJECT_ROOT / "configs"


def test_all_expected_configs_exist() -> None:
    expected = {
        "base",
        "data",
        "retrieval",
        "ranking",
        "transformer",
        "training",
        "production",
    }

    actual = {path.stem for path in CONFIG_DIR.glob("*.yaml")}

    assert expected.issubset(actual)


def test_load_base_config() -> None:
    config = load_yaml_config(CONFIG_DIR / "base.yaml")

    assert config["project"]["name"] == "RelevanceFlow"
    assert config["random_seed"] == 42


def test_load_all_configs() -> None:
    configs = load_configs(CONFIG_DIR)

    assert "base" in configs
    assert "data" in configs
    assert "retrieval" in configs
    assert "ranking" in configs
    assert "transformer" in configs
    assert "training" in configs
    assert "production" in configs


def test_nested_config_lookup() -> None:
    config = load_yaml_config(CONFIG_DIR / "training.yaml")

    seed = get_config_value(
        config,
        "training",
        "random_seed",
    )

    assert seed == 42


def test_missing_config_raises_error(tmp_path: Path) -> None:
    missing_file = tmp_path / "missing.yaml"

    with pytest.raises(ConfigurationError):
        load_yaml_config(missing_file)


def test_relevance_labels_are_ordered() -> None:
    config = load_yaml_config(CONFIG_DIR / "ranking.yaml")

    labels = config["ranking"]["relevance_labels"]

    assert labels["Irrelevant"] == 0
    assert labels["Partial"] == 1
    assert labels["Exact"] == 2


def test_primary_metric() -> None:
    config = load_yaml_config(CONFIG_DIR / "ranking.yaml")

    assert config["ranking"]["primary_metric"] == "ndcg@10"


def test_transformers_are_disabled_initially() -> None:
    config = load_yaml_config(CONFIG_DIR / "transformer.yaml")

    assert config["transformer"]["enabled"] is False


def test_redis_is_disabled_initially() -> None:
    config = load_yaml_config(CONFIG_DIR / "production.yaml")

    assert config["redis"]["enabled"] is False


def test_aws_is_disabled_initially() -> None:
    config = load_yaml_config(CONFIG_DIR / "production.yaml")

    assert config["aws"]["enabled"] is False
