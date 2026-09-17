from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


class PipelineError(RuntimeError):
    """Raised when a pipeline stage fails."""


@dataclass(frozen=True)
class PipelineStageResult:
    """Result of one pipeline stage."""

    name: str
    success: bool
    started_at: str
    finished_at: str


def get_git_commit(project_root: Path) -> str | None:
    """Return the current Git commit hash."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=project_root,
            check=True,
            capture_output=True,
            text=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None

    commit = result.stdout.strip()

    return commit or None


def get_git_branch(project_root: Path) -> str | None:
    """Return the current Git branch."""
    try:
        result = subprocess.run(
            ["git", "branch", "--show-current"],
            cwd=project_root,
            check=True,
            capture_output=True,
            text=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None

    branch = result.stdout.strip()

    return branch or None


def get_file_sha256(path: Path) -> str:
    """Calculate SHA-256 checksum for a file."""
    digest = hashlib.sha256()

    with path.open("rb") as file:
        while chunk := file.read(1024 * 1024):
            digest.update(chunk)

    return digest.hexdigest()


def utc_now() -> str:
    """Return the current UTC timestamp."""
    return datetime.now(UTC).isoformat()


def build_pipeline_manifest(
    project_root: Path,
    config: dict[str, Any],
) -> dict[str, Any]:
    """Build reproducibility metadata for a pipeline execution."""

    manifest: dict[str, Any] = {
        "project": config["base"]["project"]["name"],
        "project_version": config["base"]["project"]["version"],
        "pipeline": config.get("pipeline", {}),
        "dataset": config["data"]["dataset"],
        "random_seed": config["base"]["random_seed"],
        "git": {
            "commit": get_git_commit(project_root),
            "branch": get_git_branch(project_root),
        },
        "runtime": {
            "python_version": platform.python_version(),
            "platform": platform.platform(),
            "executable": sys.executable,
        },
        "created_at": utc_now(),
    }

    return manifest


def write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write a JSON artifact atomically enough for pipeline usage."""
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as file:
        json.dump(
            payload,
            file,
            indent=2,
            sort_keys=True,
        )
        file.write("\n")


def run_stage(
    name: str,
    command: list[str],
    project_root: Path,
) -> PipelineStageResult:
    """Run a pipeline stage as a subprocess."""

    started_at = utc_now()

    print()
    print("=" * 72)
    print(f"PIPELINE STAGE: {name}")
    print("=" * 72)
    print("Command:")
    print(" ".join(command))
    print()

    result = subprocess.run(
        command,
        cwd=project_root,
        check=False,
    )

    finished_at = utc_now()

    if result.returncode != 0:
        raise PipelineError(
            f"Pipeline stage '{name}' failed " f"with exit code {result.returncode}."
        )

    print()
    print(f"Stage '{name}' completed successfully.")

    return PipelineStageResult(
        name=name,
        success=True,
        started_at=started_at,
        finished_at=finished_at,
    )
