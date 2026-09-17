from pathlib import Path

from relevanceflow.utils.pipeline import (
    build_pipeline_manifest,
    get_file_sha256,
    get_git_branch,
    get_git_commit,
    write_json,
)


def test_git_commit_is_available():
    project_root = Path(__file__).resolve().parents[2]

    commit = get_git_commit(project_root)

    assert commit is not None
    assert len(commit) >= 7


def test_git_branch_is_available():
    project_root = Path(__file__).resolve().parents[2]

    branch = get_git_branch(project_root)

    assert branch


def test_file_sha256(tmp_path):
    path = tmp_path / "example.txt"

    path.write_text(
        "relevanceflow",
        encoding="utf-8",
    )

    checksum = get_file_sha256(path)

    assert len(checksum) == 64
    assert checksum.isalnum()


def test_write_json(tmp_path):
    path = tmp_path / "output.json"

    payload = {
        "project": "RelevanceFlow",
        "seed": 42,
    }

    write_json(path, payload)

    assert path.exists()

    content = path.read_text(encoding="utf-8")

    assert "RelevanceFlow" in content
    assert "42" in content


def test_build_pipeline_manifest():
    project_root = Path(__file__).resolve().parents[2]

    config = {
        "base": {
            "project": {
                "name": "RelevanceFlow",
                "version": "0.1.0",
            },
            "random_seed": 42,
        },
        "data": {
            "dataset": {
                "name": "WANDS",
                "version": "1.0",
            },
        },
        "pipeline": {
            "name": "relevanceflow_training",
            "version": "1.0",
        },
    }

    manifest = build_pipeline_manifest(
        project_root=project_root,
        config=config,
    )

    assert manifest["project"] == "RelevanceFlow"
    assert manifest["project_version"] == "0.1.0"
    assert manifest["dataset"]["name"] == "WANDS"
    assert manifest["random_seed"] == 42
    assert manifest["git"]["commit"]
    assert manifest["runtime"]["python_version"]
