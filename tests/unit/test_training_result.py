import json
from pathlib import Path


def test_training_result_schema(tmp_path: Path):
    result = {
        "mlflow_run_id": "test-run",
        "registered_model": "RelevanceFlowRanker",
        "model_version": "1",
        "candidate_alias": "candidate",
        "champion_alias": "champion",
        "champion_initialized": True,
        "validation": {
            "ndcg@10": 0.5,
        },
        "test": {
            "ndcg@10": 0.4,
        },
    }

    path = tmp_path / "training_result.json"

    path.write_text(
        json.dumps(result),
        encoding="utf-8",
    )

    loaded = json.loads(path.read_text(encoding="utf-8"))

    required_fields = {
        "mlflow_run_id",
        "registered_model",
        "model_version",
        "candidate_alias",
        "champion_alias",
        "validation",
        "test",
    }

    assert required_fields.issubset(loaded)
    assert loaded["registered_model"] == "RelevanceFlowRanker"
    assert loaded["model_version"] == "1"
