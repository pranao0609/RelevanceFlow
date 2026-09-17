from pathlib import Path

from relevanceflow.utils.provenance import (
    build_provenance,
    fingerprint_artifacts,
    get_platform_info,
    get_python_version,
    save_provenance,
)


def test_python_version_available():
    assert get_python_version()


def test_platform_info_available():
    info = get_platform_info()

    assert "system" in info
    assert "machine" in info


def test_fingerprint_artifacts(tmp_path: Path):
    artifact = tmp_path / "artifact.txt"
    artifact.write_text("hello", encoding="utf-8")

    result = fingerprint_artifacts(
        tmp_path,
        {"sample": artifact},
    )

    assert result["sample"]["size_bytes"] == 5
    assert len(result["sample"]["sha256"]) == 64


def test_build_provenance(tmp_path: Path):
    dataset = tmp_path / "dataset.txt"
    split = tmp_path / "split.txt"
    feature = tmp_path / "feature.parquet"

    dataset.write_text("dataset", encoding="utf-8")
    split.write_text("split", encoding="utf-8")
    feature.write_text("features", encoding="utf-8")

    provenance = build_provenance(
        project_root=Path.cwd(),
        pipeline_name="test_pipeline",
        pipeline_version="1.0",
        random_seed=42,
        feature_fingerprint="abc123",
        dataset_artifacts={"dataset": dataset},
        split_artifacts={"train": split},
        feature_artifacts={"train": feature},
        model_info={
            "registered_model": "TestModel",
            "version": "1",
        },
        evaluation_metrics={
            "ndcg@10": 0.5,
        },
    )

    assert provenance["pipeline"]["name"] == "test_pipeline"
    assert provenance["reproducibility"]["random_seed"] == 42
    assert provenance["features"]["fingerprint"] == "abc123"
    assert provenance["model"]["version"] == "1"
    assert provenance["evaluation"]["ndcg@10"] == 0.5


def test_save_provenance(tmp_path: Path):
    path = tmp_path / "provenance.json"

    save_provenance(
        path,
        {"test": True},
    )

    assert path.exists()
    assert '"test": true' in path.read_text(encoding="utf-8")
