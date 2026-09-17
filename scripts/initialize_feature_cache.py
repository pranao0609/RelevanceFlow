"""Initialize the feature cache from existing feature artifacts."""

from __future__ import annotations

from pathlib import Path

from relevanceflow.features.cache import (
    adopt_existing_artifacts,
    build_feature_fingerprint,
)
from relevanceflow.utils.config import load_configs

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    config = load_configs()

    cache_config = config["features_cache"]["features_cache"]

    manifest_path = PROJECT_ROOT / cache_config["manifest_path"]

    artifact_paths = {
        name: PROJECT_ROOT / path for name, path in cache_config["artifacts"].items()
    }

    processed_data_paths = {
        "products": PROJECT_ROOT / "data/processed/wands/products.parquet",
        "queries": PROJECT_ROOT / "data/processed/wands/queries.parquet",
        "judgments": PROJECT_ROOT / "data/processed/wands/judgments.parquet",
    }

    split_paths = {
        "train": PROJECT_ROOT / "data/processed/wands/train.parquet",
        "validation": PROJECT_ROOT / "data/processed/wands/validation.parquet",
        "test": PROJECT_ROOT / "data/processed/wands/test.parquet",
    }

    feature_config = config.get("features", {})
    retrieval_config = config.get("retrieval", {})

    source_paths = [
        PROJECT_ROOT / "scripts/generate_features.py",
        PROJECT_ROOT / "src/relevanceflow/features/pipeline.py",
        PROJECT_ROOT / "src/relevanceflow/features/lexical.py",
        PROJECT_ROOT / "src/relevanceflow/features/text.py",
        PROJECT_ROOT / "src/relevanceflow/features/metadata.py",
        PROJECT_ROOT / "src/relevanceflow/features/compatibility.py",
        PROJECT_ROOT / "src/relevanceflow/features/semantic.py",
    ]

    fingerprint, fingerprint_payload = build_feature_fingerprint(
        project_root=PROJECT_ROOT,
        processed_data_paths=processed_data_paths,
        split_paths=split_paths,
        feature_config=feature_config,
        retrieval_config=retrieval_config,
        source_paths=source_paths,
    )

    manifest = adopt_existing_artifacts(
        project_root=PROJECT_ROOT,
        manifest_path=manifest_path,
        fingerprint=fingerprint,
        fingerprint_payload=fingerprint_payload,
        artifact_paths=artifact_paths,
    )

    print("=" * 70)
    print("RelevanceFlow — Feature Cache Initialization")
    print("=" * 70)
    print()
    print("Existing feature artifacts adopted successfully.")
    print()
    print(f"Fingerprint: {manifest['fingerprint']}")
    print(f"Manifest:    {manifest_path}")
    print()
    print("Artifacts:")

    for name, metadata in manifest["artifacts"].items():
        print(
            f"  {name:12s} "
            f"{metadata['size_bytes']:,} bytes "
            f"{metadata['sha256'][:16]}..."
        )

    print()
    print("Future pipeline runs can reuse these artifacts when the")
    print("fingerprint remains unchanged.")
    print("=" * 70)


if __name__ == "__main__":
    main()
