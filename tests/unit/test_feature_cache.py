from pathlib import Path

from relevanceflow.features.cache import (
    build_feature_fingerprint,
    hash_json,
    is_cache_valid,
    save_manifest,
    sha256_file,
)


def test_sha256_file(tmp_path: Path):
    file_path = tmp_path / "sample.txt"
    file_path.write_text("hello", encoding="utf-8")

    digest = sha256_file(file_path)

    assert len(digest) == 64
    assert digest == (
        "2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824"
    )


def test_hash_json_is_deterministic():
    first = hash_json({"b": 2, "a": 1})
    second = hash_json({"a": 1, "b": 2})

    assert first == second


def test_build_feature_fingerprint_changes_with_config(tmp_path: Path):
    data_file = tmp_path / "data.parquet"
    split_file = tmp_path / "split.parquet"
    source_file = tmp_path / "source.py"

    data_file.write_text("data", encoding="utf-8")
    split_file.write_text("split", encoding="utf-8")
    source_file.write_text("source", encoding="utf-8")

    fingerprint_one, _ = build_feature_fingerprint(
        project_root=tmp_path,
        processed_data_paths={"judgments": data_file},
        split_paths={"train": split_file},
        feature_config={"ngram": [1, 2]},
        retrieval_config={"bm25_k1": 1.5},
        source_paths=[source_file],
    )

    fingerprint_two, _ = build_feature_fingerprint(
        project_root=tmp_path,
        processed_data_paths={"judgments": data_file},
        split_paths={"train": split_file},
        feature_config={"ngram": [1, 3]},
        retrieval_config={"bm25_k1": 1.5},
        source_paths=[source_file],
    )

    assert fingerprint_one != fingerprint_two


def test_cache_hit(tmp_path: Path):
    artifact = tmp_path / "train.parquet"
    artifact.write_text("feature-data", encoding="utf-8")

    manifest_path = tmp_path / "feature_manifest.json"

    fingerprint = "abc123"

    save_manifest(
        manifest_path,
        {
            "fingerprint": fingerprint,
            "artifacts": {
                "train": {
                    "sha256": sha256_file(artifact),
                    "exists": True,
                    "size_bytes": artifact.stat().st_size,
                }
            },
        },
    )

    valid, reason = is_cache_valid(
        project_root=tmp_path,
        manifest_path=manifest_path,
        expected_fingerprint=fingerprint,
        artifact_paths={"train": artifact},
    )

    assert valid is True
    assert reason == "cache_hit"


def test_cache_miss_when_artifact_changes(tmp_path: Path):
    artifact = tmp_path / "train.parquet"
    artifact.write_text("original", encoding="utf-8")

    manifest_path = tmp_path / "feature_manifest.json"

    fingerprint = "abc123"

    save_manifest(
        manifest_path,
        {
            "fingerprint": fingerprint,
            "artifacts": {
                "train": {
                    "sha256": sha256_file(artifact),
                    "exists": True,
                    "size_bytes": artifact.stat().st_size,
                }
            },
        },
    )

    artifact.write_text("changed", encoding="utf-8")

    valid, reason = is_cache_valid(
        project_root=tmp_path,
        manifest_path=manifest_path,
        expected_fingerprint=fingerprint,
        artifact_paths={"train": artifact},
    )

    assert valid is False
    assert reason == "artifact_changed:train"
