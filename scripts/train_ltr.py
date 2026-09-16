from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from relevanceflow.evaluation.ranking_metrics import (
    average_precision_at_k,
    ndcg_at_k,
    recall_at_k,
    reciprocal_rank_at_k,
)
from relevanceflow.models.ltr import (
    LightGBMLTR,
    LTRConfig,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]

FEATURE_DIR = PROJECT_ROOT / "data" / "features"
OUTPUT_DIR = PROJECT_ROOT / "experiments" / "ltr"

TRAIN_PATH = FEATURE_DIR / "train_features.parquet"
VALIDATION_PATH = FEATURE_DIR / "validation_features.parquet"
TEST_PATH = FEATURE_DIR / "test_features.parquet"


def evaluate_ranked(
    ranked: pd.DataFrame,
    k_values: tuple[int, ...] = (1, 5, 10, 20),
) -> dict[str, float]:
    metrics: dict[str, list[float]] = {}

    for k in k_values:
        metrics[f"recall@{k}"] = []
        metrics[f"ndcg@{k}"] = []

    metrics["mrr@10"] = []
    metrics["map@10"] = []

    for _, group in ranked.groupby("query_id", sort=False):
        relevance = group["relevance_score"].to_numpy()

        for k in k_values:
            metrics[f"recall@{k}"].append(recall_at_k(relevance, k))

            metrics[f"ndcg@{k}"].append(ndcg_at_k(relevance, k))

        metrics["mrr@10"].append(reciprocal_rank_at_k(relevance, 10))

        metrics["map@10"].append(average_precision_at_k(relevance, 10))

    return {
        metric: float(sum(values) / len(values)) for metric, values in metrics.items()
    }


def attach_relevance(
    ranked: pd.DataFrame,
    source: pd.DataFrame,
) -> pd.DataFrame:
    return ranked.merge(
        source[
            [
                "query_id",
                "product_id",
                "relevance_score",
            ]
        ],
        on=["query_id", "product_id"],
        how="left",
        validate="one_to_one",
    )


def main() -> None:
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("Loading feature matrices...")

    train = pd.read_parquet(TRAIN_PATH)
    validation = pd.read_parquet(VALIDATION_PATH)
    test = pd.read_parquet(TEST_PATH)

    print(f"Train:      {train.shape}")
    print(f"Validation: {validation.shape}")
    print(f"Test:       {test.shape}")

    print("\nTraining LightGBM LambdaRank...")

    model = LightGBMLTR(
        LTRConfig(
            n_estimators=500,
            learning_rate=0.05,
            num_leaves=31,
            random_state=42,
        )
    )

    model.fit(
        train,
        validation=validation,
    )

    print(f"\nFeatures used: {len(model.feature_columns)}")

    for feature in model.feature_columns:
        print(f"  - {feature}")

    print("\nGenerating validation rankings...")

    validation_ranked = model.rank_candidates(validation)

    validation_ranked = attach_relevance(
        validation_ranked,
        validation,
    )

    validation_metrics = evaluate_ranked(validation_ranked)

    print("\nValidation metrics:")

    for metric, value in validation_metrics.items():
        print(f"{metric}: {value:.6f}")

    print("\nGenerating test rankings...")

    test_ranked = model.rank_candidates(test)

    test_ranked = attach_relevance(
        test_ranked,
        test,
    )

    test_metrics = evaluate_ranked(test_ranked)

    print("\nTest metrics:")

    for metric, value in test_metrics.items():
        print(f"{metric}: {value:.6f}")

    metrics = {
        "model": "LightGBM LGBMRanker",
        "objective": "lambdarank",
        "target": "relevance_score",
        "label_mapping": {
            "Irrelevant": 0,
            "Partial": 1,
            "Exact": 2,
        },
        "group_column": "query_id",
        "features": model.feature_columns,
        "validation": validation_metrics,
        "test": test_metrics,
    }

    with open(
        OUTPUT_DIR / "metrics.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            metrics,
            file,
            indent=2,
        )

    config = {
        "n_estimators": 500,
        "learning_rate": 0.05,
        "num_leaves": 31,
        "max_depth": -1,
        "min_child_samples": 20,
        "random_state": 42,
        "objective": "lambdarank",
        "metric": "ndcg",
        "eval_at": [5, 10, 20],
    }

    with open(
        OUTPUT_DIR / "model_config.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            config,
            file,
            indent=2,
        )

    print(f"\nSaved metrics: {OUTPUT_DIR / 'metrics.json'}")


if __name__ == "__main__":
    main()
