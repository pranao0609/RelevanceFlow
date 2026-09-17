from __future__ import annotations

import json
import time
from pathlib import Path

import pandas as pd

from relevanceflow.evaluation.ranking_metrics import (
    average_precision_at_k,
    ndcg_at_k,
    recall_at_k,
    reciprocal_rank_at_k,
)
from relevanceflow.models.ltr import LightGBMLTR, LTRConfig

PROJECT_ROOT = Path(__file__).resolve().parents[1]

FEATURE_DIR = PROJECT_ROOT / "data" / "features"
EXPERIMENT_DIR = PROJECT_ROOT / "experiments" / "hybrid_ltr"

TRAIN_PATH = FEATURE_DIR / "train_features.parquet"
VALIDATION_PATH = FEATURE_DIR / "validation_features.parquet"
TEST_PATH = FEATURE_DIR / "test_features.parquet"

METRICS_PATH = EXPERIMENT_DIR / "metrics.json"
CONFIG_PATH = EXPERIMENT_DIR / "model_config.json"
FEATURE_IMPORTANCE_PATH = EXPERIMENT_DIR / "feature_importance.csv"


def load_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    print("Loading feature matrices...")

    train = pd.read_parquet(TRAIN_PATH)
    validation = pd.read_parquet(VALIDATION_PATH)
    test = pd.read_parquet(TEST_PATH)

    print(f"Train: {train.shape}")
    print(f"Validation: {validation.shape}")
    print(f"Test: {test.shape}")

    return train, validation, test


def validate_splits(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    test: pd.DataFrame,
) -> None:
    for name, dataframe in [
        ("train", train),
        ("validation", validation),
        ("test", test),
    ]:
        if dataframe["query_id"].isna().any():
            raise ValueError(f"{name} contains missing query_id.")

        if dataframe["product_id"].isna().any():
            raise ValueError(f"{name} contains missing product_id.")

        if dataframe["relevance_score"].isna().any():
            raise ValueError(f"{name} contains missing relevance_score.")

        if dataframe["query_id"].duplicated().all():
            raise ValueError(f"{name} contains invalid query groups.")

    train_queries = set(train["query_id"].unique())
    validation_queries = set(validation["query_id"].unique())
    test_queries = set(test["query_id"].unique())

    if train_queries & validation_queries:
        raise ValueError("Train and validation query groups overlap.")

    if train_queries & test_queries:
        raise ValueError("Train and test query groups overlap.")

    if validation_queries & test_queries:
        raise ValueError("Validation and test query groups overlap.")


def evaluate_predictions(
    predictions: pd.DataFrame,
    ground_truth: pd.DataFrame,
) -> dict[str, float]:
    metrics: dict[str, list[float]] = {
        "recall@1": [],
        "recall@5": [],
        "recall@10": [],
        "recall@20": [],
        "mrr@10": [],
        "ndcg@5": [],
        "ndcg@10": [],
        "ndcg@20": [],
        "map@10": [],
    }

    relevance_lookup = ground_truth.set_index(["query_id", "product_id"])[
        "relevance_score"
    ].to_dict()

    for query_id, group in predictions.groupby("query_id", sort=False):
        ranked_products = group.sort_values("rank")["product_id"].tolist()

        relevance = [
            int(relevance_lookup.get((query_id, product_id), 0))
            for product_id in ranked_products
        ]

        metrics["recall@1"].append(recall_at_k(relevance, 1))
        metrics["recall@5"].append(recall_at_k(relevance, 5))
        metrics["recall@10"].append(recall_at_k(relevance, 10))
        metrics["recall@20"].append(recall_at_k(relevance, 20))

        metrics["mrr@10"].append(reciprocal_rank_at_k(relevance, 10))

        metrics["ndcg@5"].append(ndcg_at_k(relevance, 5))
        metrics["ndcg@10"].append(ndcg_at_k(relevance, 10))
        metrics["ndcg@20"].append(ndcg_at_k(relevance, 20))

        metrics["map@10"].append(average_precision_at_k(relevance, 10))

    return {
        metric: float(sum(values) / len(values)) for metric, values in metrics.items()
    }


def main() -> None:
    start_time = time.perf_counter()

    EXPERIMENT_DIR.mkdir(parents=True, exist_ok=True)

    train, validation, test = load_data()

    validate_splits(train, validation, test)

    print("\nTraining hybrid LightGBM LambdaRank model...")

    config = LTRConfig(
        n_estimators=500,
        learning_rate=0.05,
        num_leaves=31,
        max_depth=-1,
        min_child_samples=20,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.0,
        reg_lambda=0.0,
        random_state=42,
        eval_at=(5, 10, 20),
        early_stopping_rounds=50,
    )

    model = LightGBMLTR(config)

    training_start = time.perf_counter()

    model.fit(
        train=train,
        validation=validation,
    )

    training_time = time.perf_counter() - training_start

    print(f"Training time: {training_time:.2f} seconds")

    print("\nGenerating validation predictions...")
    validation_predictions = model.rank_candidates(validation)

    print("Generating test predictions...")
    test_predictions = model.rank_candidates(test)
    test_predictions.to_csv(
        EXPERIMENT_DIR / "test_predictions.csv",
        index=False,
    )
    validation_metrics = evaluate_predictions(
        validation_predictions,
        validation,
    )

    test_metrics = evaluate_predictions(
        test_predictions,
        test,
    )

    print("\n" + "=" * 55)
    print("Hybrid LightGBM LambdaRank")
    print("=" * 55)

    print("\nValidation:")
    for metric, value in validation_metrics.items():
        print(f"{metric}: {value:.6f}")

    print("\nTest:")
    for metric, value in test_metrics.items():
        print(f"{metric}: {value:.6f}")

    feature_importance = pd.DataFrame(
        {
            "feature": model.feature_columns,
            "importance": model.model.feature_importances_,
        }
    ).sort_values(
        "importance",
        ascending=False,
    )

    feature_importance.to_csv(
        FEATURE_IMPORTANCE_PATH,
        index=False,
    )

    best_iteration = getattr(
        model.model,
        "best_iteration_",
        None,
    )

    metrics_output = {
        "model": "LightGBM LambdaRank",
        "objective": "lambdarank",
        "features": len(model.feature_columns),
        "train_queries": int(train["query_id"].nunique()),
        "validation_queries": int(validation["query_id"].nunique()),
        "test_queries": int(test["query_id"].nunique()),
        "best_iteration": (int(best_iteration) if best_iteration is not None else None),
        "training_time_seconds": training_time,
        "validation": validation_metrics,
        "test": test_metrics,
        "total_runtime_seconds": time.perf_counter() - start_time,
    }

    with METRICS_PATH.open("w", encoding="utf-8") as file:
        json.dump(metrics_output, file, indent=2)

    config_output = {
        "n_estimators": config.n_estimators,
        "learning_rate": config.learning_rate,
        "num_leaves": config.num_leaves,
        "max_depth": config.max_depth,
        "min_child_samples": config.min_child_samples,
        "subsample": config.subsample,
        "colsample_bytree": config.colsample_bytree,
        "reg_alpha": config.reg_alpha,
        "reg_lambda": config.reg_lambda,
        "random_state": config.random_state,
        "eval_at": list(config.eval_at),
        "early_stopping_rounds": config.early_stopping_rounds,
        "feature_columns": model.feature_columns,
    }

    with CONFIG_PATH.open("w", encoding="utf-8") as file:
        json.dump(config_output, file, indent=2)

    print("\nArtifacts saved:")
    print(METRICS_PATH)
    print(CONFIG_PATH)
    print(FEATURE_IMPORTANCE_PATH)


if __name__ == "__main__":
    main()
