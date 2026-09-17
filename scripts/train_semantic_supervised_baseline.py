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
from relevanceflow.models.supervised_baseline import (
    SupervisedBaseline,
    SupervisedBaselineConfig,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]

FEATURE_DIR = PROJECT_ROOT / "data" / "features"
OUTPUT_DIR = PROJECT_ROOT / "experiments" / "semantic_supervised_baseline"

TRAIN_PATH = FEATURE_DIR / "train_features.parquet"
VALIDATION_PATH = FEATURE_DIR / "validation_features.parquet"
TEST_PATH = FEATURE_DIR / "test_features.parquet"


def evaluate_ranked(
    ranked: pd.DataFrame,
    k_values: tuple[int, ...] = (1, 5, 10, 20),
) -> dict[str, float]:
    """Evaluate ranked candidate lists query-by-query."""

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


def evaluate_split(
    model: SupervisedBaseline,
    dataframe: pd.DataFrame,
) -> dict[str, float]:
    ranked = model.rank_candidates(dataframe)

    ranked = ranked.merge(
        dataframe[["query_id", "product_id", "relevance_score"]],
        on=["query_id", "product_id"],
        how="left",
        validate="one_to_one",
    )

    return evaluate_ranked(ranked)


def main() -> None:
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("Loading semantic-enhanced feature matrices...")

    train = pd.read_parquet(TRAIN_PATH)
    validation = pd.read_parquet(VALIDATION_PATH)
    test = pd.read_parquet(TEST_PATH)

    print(f"Train:      {train.shape}")
    print(f"Validation: {validation.shape}")
    print(f"Test:       {test.shape}")

    semantic_columns = [
        "semantic_similarity",
        "query_embedding_norm",
        "product_embedding_norm",
    ]

    print("\nChecking semantic features...")

    for column in semantic_columns:
        if column not in train.columns:
            raise ValueError(f"Missing semantic feature: {column}")

        print(
            f"{column}: "
            f"mean={train[column].mean():.6f}, "
            f"std={train[column].std():.6f}, "
            f"min={train[column].min():.6f}, "
            f"max={train[column].max():.6f}"
        )

    print("\nTraining semantic-enhanced LightGBM...")

    model = SupervisedBaseline(
        SupervisedBaselineConfig(
            n_estimators=300,
            learning_rate=0.05,
            num_leaves=31,
            random_state=42,
        )
    )

    model.fit(train)

    print(f"\nFeatures used: {len(model.feature_columns)}")

    for feature in model.feature_columns:
        print(f"  - {feature}")

    print("\nEvaluating validation set...")

    validation_metrics = evaluate_split(
        model,
        validation,
    )

    print("\nValidation metrics:")

    for metric, value in validation_metrics.items():
        print(f"{metric}: {value:.6f}")

    print("\nEvaluating test set...")

    test_metrics = evaluate_split(
        model,
        test,
    )

    print("\nTest metrics:")

    for metric, value in test_metrics.items():
        print(f"{metric}: {value:.6f}")

    metrics = {
        "model": "LightGBMClassifier",
        "experiment": "semantic_enhanced_supervised_baseline",
        "objective": "multiclass",
        "target": "relevance_score",
        "label_mapping": {
            "Irrelevant": 0,
            "Partial": 1,
            "Exact": 2,
        },
        "semantic_model": ("sentence-transformers/all-MiniLM-L6-v2"),
        "semantic_features": semantic_columns,
        "num_features": len(model.feature_columns),
        "features": model.feature_columns,
        "validation": validation_metrics,
        "test": test_metrics,
    }

    metrics_path = OUTPUT_DIR / "metrics.json"

    with open(
        metrics_path,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            metrics,
            file,
            indent=2,
        )

    config = {
        "n_estimators": 300,
        "learning_rate": 0.05,
        "num_leaves": 31,
        "max_depth": -1,
        "min_child_samples": 20,
        "random_state": 42,
        "semantic_model": ("sentence-transformers/all-MiniLM-L6-v2"),
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

    print(f"\nSaved metrics: {metrics_path}")


if __name__ == "__main__":
    main()
