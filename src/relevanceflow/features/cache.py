"""Feature artifact caching and fingerprinting utilities."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class FeatureCacheError(RuntimeError):
    """Raised when feature-cache operations fail."""


@dataclass(frozen=True)
class FeatureCacheConfig:
    """Configuration for feature artifact caching."""

    manifest_path: Path
    artifact_paths: dict[str, Path]


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    """Calculate SHA256 for a file."""
    if not path.exists():
        raise FeatureCacheError(f"File does not exist: {path}")

    digest = hashlib.sha256()

    with path.open("rb") as file:
        while chunk := file.read(chunk_size):
            digest.update(chunk)

    return digest.hexdigest()


def sha256_text(value: str) -> str:
    """Calculate SHA256 for text."""
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def hash_json(value: Any) -> str:
    """Create a deterministic SHA256 fingerprint for JSON-compatible data."""
    serialized = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )

    return sha256_text(serialized)


def get_file_metadata(path: Path) -> dict[str, Any]:
    """Return metadata and SHA256 fingerprint for a file."""
    if not path.exists():
        return {
            "path": str(path),
            "exists": False,
        }

    stat = path.stat()

    return {
        "path": str(path),
        "exists": True,
        "size_bytes": stat.st_size,
        "sha256": sha256_file(path),
    }


def build_feature_fingerprint(
    project_root: Path,
    processed_data_paths: dict[str, Path],
    split_paths: dict[str, Path],
    feature_config: dict[str, Any],
    retrieval_config: dict[str, Any],
    source_paths: list[Path],
) -> tuple[str, dict[str, Any]]:
    """
    Build a deterministic fingerprint describing feature generation inputs.

    The fingerprint changes when:
    - processed data changes
    - split data changes
    - feature configuration changes
    - retrieval configuration changes
    - feature-generation source code changes
    """

    def resolve(path: Path) -> Path:
        return path if path.is_absolute() else project_root / path

    processed_metadata = {
        name: get_file_metadata(resolve(path))
        for name, path in sorted(processed_data_paths.items())
    }

    split_metadata = {
        name: get_file_metadata(resolve(path))
        for name, path in sorted(split_paths.items())
    }

    source_metadata = {
        str(path): get_file_metadata(resolve(path))
        for path in sorted(source_paths, key=str)
    }

    fingerprint_payload = {
        "fingerprint_version": "1",
        "processed_data": processed_metadata,
        "splits": split_metadata,
        "feature_config": feature_config,
        "retrieval_config": retrieval_config,
        "source_files": source_metadata,
    }

    fingerprint = hash_json(fingerprint_payload)

    return fingerprint, fingerprint_payload


def load_manifest(path: Path) -> dict[str, Any] | None:
    """Load the feature cache manifest if it exists."""
    if not path.exists():
        return None

    try:
        with path.open("r", encoding="utf-8") as file:
            return json.load(file)
    except (OSError, json.JSONDecodeError) as exc:
        raise FeatureCacheError(
            f"Unable to read feature cache manifest: {path}"
        ) from exc


def save_manifest(path: Path, manifest: dict[str, Any]) -> None:
    """Write the feature cache manifest."""
    path.parent.mkdir(parents=True, exist_ok=True)

    temporary_path = path.with_suffix(".tmp")

    try:
        with temporary_path.open("w", encoding="utf-8") as file:
            json.dump(
                manifest,
                file,
                indent=2,
                sort_keys=True,
            )
            file.write("\n")

        temporary_path.replace(path)

    except OSError as exc:
        raise FeatureCacheError(
            f"Unable to write feature cache manifest: {path}"
        ) from exc


def validate_artifacts(
    project_root: Path,
    artifact_paths: dict[str, Path],
) -> tuple[bool, dict[str, Any]]:
    """
    Check whether all expected feature artifacts exist and record metadata.
    """
    metadata: dict[str, Any] = {}

    for name, path in sorted(artifact_paths.items()):
        resolved = path if path.is_absolute() else project_root / path

        metadata[name] = get_file_metadata(resolved)

        if not metadata[name]["exists"]:
            return False, metadata

        if metadata[name]["size_bytes"] == 0:
            return False, metadata

    return True, metadata


def is_cache_valid(
    project_root: Path,
    manifest_path: Path,
    expected_fingerprint: str,
    artifact_paths: dict[str, Path],
) -> tuple[bool, str]:
    """
    Determine whether cached feature artifacts can safely be reused.
    """
    manifest = load_manifest(manifest_path)

    if manifest is None:
        return False, "feature_manifest_missing"

    stored_fingerprint = manifest.get("fingerprint")

    if stored_fingerprint != expected_fingerprint:
        return False, "fingerprint_mismatch"

    artifacts_valid, _ = validate_artifacts(
        project_root,
        artifact_paths,
    )

    if not artifacts_valid:
        return False, "feature_artifact_missing_or_empty"

    stored_artifacts = manifest.get("artifacts", {})

    for name, path in artifact_paths.items():
        resolved = path if path.is_absolute() else project_root / path

        current_sha = sha256_file(resolved)

        stored_sha = stored_artifacts.get(name, {}).get("sha256")

        if stored_sha != current_sha:
            return False, f"artifact_changed:{name}"

    return True, "cache_hit"


def build_manifest(
    fingerprint: str,
    fingerprint_payload: dict[str, Any],
    artifact_metadata: dict[str, Any],
    source: str,
) -> dict[str, Any]:
    """Build a persisted feature-cache manifest."""
    return {
        "cache_type": "feature_artifacts",
        "manifest_version": "1",
        "fingerprint": fingerprint,
        "fingerprint_inputs": fingerprint_payload,
        "artifacts": artifact_metadata,
        "source": source,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }


def adopt_existing_artifacts(
    project_root: Path,
    manifest_path: Path,
    fingerprint: str,
    fingerprint_payload: dict[str, Any],
    artifact_paths: dict[str, Path],
) -> dict[str, Any]:
    """
    Create a cache manifest for already-generated feature artifacts.

    This is intended for artifacts that have already been successfully
    generated and verified.
    """
    artifacts_valid, artifact_metadata = validate_artifacts(
        project_root,
        artifact_paths,
    )

    if not artifacts_valid:
        raise FeatureCacheError(
            "Cannot adopt feature artifacts because one or more "
            "artifacts are missing or empty."
        )

    manifest = build_manifest(
        fingerprint=fingerprint,
        fingerprint_payload=fingerprint_payload,
        artifact_metadata=artifact_metadata,
        source="adopted_existing_artifacts",
    )

    save_manifest(manifest_path, manifest)

    return manifest
