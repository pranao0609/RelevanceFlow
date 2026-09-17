"""Pipeline artifact provenance utilities."""

from __future__ import annotations

import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from relevanceflow.features.cache import sha256_file
from relevanceflow.utils.pipeline import get_git_branch, get_git_commit


class ProvenanceError(RuntimeError):
    """Raised when provenance information cannot be collected."""


def get_python_version() -> str:
    """Return the running Python version."""
    return sys.version


def get_platform_info() -> dict[str, str]:
    """Return basic runtime platform information."""
    return {
        "system": platform.system(),
        "release": platform.release(),
        "version": platform.version(),
        "machine": platform.machine(),
        "processor": platform.processor(),
    }


def fingerprint_artifacts(
    project_root: Path,
    artifact_paths: dict[str, Path],
) -> dict[str, dict[str, Any]]:
    """
    Collect SHA256 and size information for pipeline artifacts.
    """
    result: dict[str, dict[str, Any]] = {}

    for name, path in sorted(artifact_paths.items()):
        resolved = path if path.is_absolute() else project_root / path

        if not resolved.exists():
            raise ProvenanceError(
                f"Required provenance artifact does not exist: {resolved}"
            )

        result[name] = {
            "path": str(resolved),
            "sha256": sha256_file(resolved),
            "size_bytes": resolved.stat().st_size,
        }

    return result


def build_provenance(
    project_root: Path,
    pipeline_name: str,
    pipeline_version: str,
    random_seed: int,
    feature_fingerprint: str | None = None,
    dataset_artifacts: dict[str, Path] | None = None,
    split_artifacts: dict[str, Path] | None = None,
    feature_artifacts: dict[str, Path] | None = None,
    model_info: dict[str, Any] | None = None,
    evaluation_metrics: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a complete provenance record for a pipeline run."""
    dataset_artifacts = dataset_artifacts or {}
    split_artifacts = split_artifacts or {}
    feature_artifacts = feature_artifacts or {}

    return {
        "provenance_version": "1",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "pipeline": {
            "name": pipeline_name,
            "version": pipeline_version,
        },
        "source": {
            "git_commit": get_git_commit(project_root),
            "git_branch": get_git_branch(project_root),
        },
        "runtime": {
            "python_version": get_python_version(),
            "platform": get_platform_info(),
        },
        "reproducibility": {
            "random_seed": random_seed,
        },
        "dataset": fingerprint_artifacts(
            project_root,
            dataset_artifacts,
        ),
        "splits": fingerprint_artifacts(
            project_root,
            split_artifacts,
        ),
        "features": {
            "fingerprint": feature_fingerprint,
            "artifacts": fingerprint_artifacts(
                project_root,
                feature_artifacts,
            ),
        },
        "model": model_info or {},
        "evaluation": evaluation_metrics or {},
    }


def save_provenance(
    path: Path,
    provenance: dict[str, Any],
) -> None:
    """Atomically save provenance information as JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)

    temporary_path = path.with_suffix(".tmp")

    with temporary_path.open("w", encoding="utf-8") as file:
        json.dump(
            provenance,
            file,
            indent=2,
            sort_keys=True,
        )
        file.write("\n")

    temporary_path.replace(path)
