from __future__ import annotations

import json
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from relevanceflow.evaluation.ranking_metrics import (
    average_precision_at_k,
    ndcg_at_k,
    recall_at_k,
    reciprocal_rank_at_k,
)

# ============================================================
# Paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

FEATURE_DIR = PROJECT_ROOT / "data" / "features"
OPT_DIR = PROJECT_ROOT / "experiments" / "ltr_optimization"

TRAIN_PATH = FEATURE_DIR / "train_features.parquet"
VALIDATION_PATH = FEATURE_DIR / "validation_features.parquet"
TEST_PATH = FEATURE_DIR / "test_features.parquet"

BEST_PARAMS_PATH = OPT_DIR / "best_params.json"

FINAL_METRICS_PATH = OPT_DIR / "final_metrics.json"
FINAL_CONFIG_PATH = OPT_DIR / "final_model_config.json"


# ============================================================
# Configuration
# ============================================================

TARGET_COLUMN = "relevance_score"

EXCLUDED_COLUMNS = {
    "query_id",
    "product_id",
    "relevance_score",
    "label",
}


# ============================================================
# Data Loading
# ============================================================


def load_feature_data():
    print("Loading feature matrices...")

    train = pd.read_parquet(TRAIN_PATH)
    validation = pd.read_parquet(VALIDATION_PATH)
    test = pd.read_parquet(TEST_PATH)

    print(f"Train:       {train.shape}")
    print(f"Validation:  {validation.shape}")
    print(f"Test:        {test.shape}")

    return train, validation, test


# ============================================================
# Optuna Parameters
# ============================================================


def load_best_params():
    print("\nLoading frozen Optuna parameters...")

    with BEST_PARAMS_PATH.open(
        "r",
        encoding="utf-8",
    ) as file:
        data = json.load(file)

    if "parameters" in data:
        params = data["parameters"]
    elif "best_params" in data:
        params = data["best_params"]
    else:
        params = data

    print("Best parameters:")

    for key, value in params.items():
        print(f"  {key}: {value}")

    return params


# ============================================================
# Feature Preparation
# ============================================================


def prepare_features(df: pd.DataFrame):
    feature_columns = [
        column for column in df.columns if column not in EXCLUDED_COLUMNS
    ]

    if not feature_columns:
        raise ValueError("No model feature columns were found.")

    X = df[feature_columns].copy()

    for column in feature_columns:
        X[column] = pd.to_numeric(
            X[column],
            errors="raise",
        )

    if X.isnull().any().any():
        raise ValueError("Feature matrix contains missing values.")

    return X, feature_columns


# ============================================================
# Query Groups
# ============================================================


def build_groups(df: pd.DataFrame):
    if "query_id" not in df.columns:
        raise ValueError("query_id column is required.")

    return (
        df.groupby(
            "query_id",
            sort=False,
        )
        .size()
        .to_numpy(dtype=np.int32)
    )


# ============================================================
# Ranking
# ============================================================


def build_rankings(
    df: pd.DataFrame,
    scores: np.ndarray,
):
    """
    Create deterministic query-wise rankings.

    Ranking:
        query_id ASC
        score DESC
        product_id ASC
    """

    predictions = df[
        [
            "query_id",
            "product_id",
            TARGET_COLUMN,
        ]
    ].copy()

    predictions["score"] = scores

    predictions = predictions.sort_values(
        [
            "query_id",
            "score",
            "product_id",
        ],
        ascending=[
            True,
            False,
            True,
        ],
    ).reset_index(drop=True)

    return predictions


# ============================================================
# Ranking Evaluation
# ============================================================


def evaluate_predictions(
    predictions: pd.DataFrame,
):
    """
    Evaluate ranking metrics query-by-query.

    The existing ranking_metrics API expects:

        metric(relevance, k)

    Therefore relevance values are extracted after sorting
    candidates according to the model's predicted score.
    """

    metric_values = {
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

    for _, group in predictions.groupby(
        "query_id",
        sort=False,
    ):
        relevance = group[TARGET_COLUMN].to_numpy()

        metric_values["recall@1"].append(
            recall_at_k(
                relevance,
                1,
            )
        )

        metric_values["recall@5"].append(
            recall_at_k(
                relevance,
                5,
            )
        )

        metric_values["recall@10"].append(
            recall_at_k(
                relevance,
                10,
            )
        )

        metric_values["recall@20"].append(
            recall_at_k(
                relevance,
                20,
            )
        )

        metric_values["mrr@10"].append(
            reciprocal_rank_at_k(
                relevance,
                10,
            )
        )

        metric_values["ndcg@5"].append(
            ndcg_at_k(
                relevance,
                5,
            )
        )

        metric_values["ndcg@10"].append(
            ndcg_at_k(
                relevance,
                10,
            )
        )

        metric_values["ndcg@20"].append(
            ndcg_at_k(
                relevance,
                20,
            )
        )

        metric_values["map@10"].append(
            average_precision_at_k(
                relevance,
                10,
            )
        )

    return {metric: float(np.mean(values)) for metric, values in metric_values.items()}


# ============================================================
# Inference Benchmark
# ============================================================


def benchmark_inference(
    model,
    df: pd.DataFrame,
    X: pd.DataFrame,
):
    """
    Measure prediction latency and calculate ranking metrics.
    """

    start = time.perf_counter()

    scores = model.predict(X)

    inference_seconds = time.perf_counter() - start

    predictions = build_rankings(
        df,
        scores,
    )

    metrics = evaluate_predictions(predictions)

    query_count = int(df["query_id"].nunique())

    candidate_count = len(df)

    metrics["queries_evaluated"] = query_count

    metrics["candidates_evaluated"] = candidate_count

    metrics["inference_time_seconds"] = float(inference_seconds)

    metrics["average_inference_ms_per_query"] = float(
        inference_seconds / query_count * 1000
    )

    metrics["average_inference_ms_per_candidate"] = float(
        inference_seconds / candidate_count * 1000
    )

    return metrics


# ============================================================
# Main
# ============================================================


def main():
    OPT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ========================================================
    # 1. Load data
    # ========================================================

    train, validation, test = load_feature_data()

    # ========================================================
    # 2. Load frozen Optuna parameters
    # ========================================================

    best_params = load_best_params()

    # ========================================================
    # 3. Prepare features
    # ========================================================

    X_train, feature_columns = prepare_features(train)

    X_validation, validation_features = prepare_features(validation)

    X_test, test_features = prepare_features(test)

    # ========================================================
    # 4. Verify feature consistency
    # ========================================================

    if feature_columns != validation_features:
        raise ValueError("Training and validation feature columns do not match.")

    if feature_columns != test_features:
        raise ValueError("Training and test feature columns do not match.")

    # ========================================================
    # 5. Prepare training target/groups
    # ========================================================

    y_train = train[TARGET_COLUMN].astype(np.float32).to_numpy()

    train_groups = build_groups(train)

    print("\nTraining information:")

    print(f"Features:           {len(feature_columns)}")

    print(f"Training queries:   {len(train_groups)}")

    print(f"Validation queries: " f"{validation['query_id'].nunique()}")

    print(f"Test queries:       " f"{test['query_id'].nunique()}")

    # ========================================================
    # 6. Build final frozen model
    # ========================================================

    model_params = {
        "objective": "lambdarank",
        "metric": "ndcg",
        "boosting_type": "gbdt",
        "random_state": 42,
        "verbosity": -1,
        "n_jobs": -1,
        "n_estimators": int(best_params["n_estimators"]),
        "learning_rate": float(best_params["learning_rate"]),
        "num_leaves": int(best_params["num_leaves"]),
        "max_depth": int(best_params["max_depth"]),
        "min_child_samples": int(best_params["min_child_samples"]),
        "subsample": float(best_params["subsample"]),
        "colsample_bytree": float(best_params["colsample_bytree"]),
        "reg_alpha": float(best_params["reg_alpha"]),
        "reg_lambda": float(best_params["reg_lambda"]),
    }

    model = lgb.LGBMRanker(**model_params)

    # ========================================================
    # 7. Final model training
    # ========================================================

    print("\nTraining final optimized LTR model...")

    print("Training on TRAIN split only.")

    train_start = time.perf_counter()

    model.fit(
        X_train,
        y_train,
        group=train_groups,
    )

    training_seconds = time.perf_counter() - train_start

    print(f"Training time: " f"{training_seconds:.4f} sec")

    # ========================================================
    # 8. Validation evaluation
    # ========================================================

    print("\nEvaluating validation set...")

    validation_metrics = benchmark_inference(
        model,
        validation,
        X_validation,
    )

    # ========================================================
    # 9. Final test evaluation
    # ========================================================

    print("\nEvaluating FINAL held-out test set...")

    test_metrics = benchmark_inference(
        model,
        test,
        X_test,
    )

    # ========================================================
    # 10. Print benchmark
    # ========================================================

    print("\n" + "=" * 70)

    print("RelevanceFlow — " "Final Optimized LTR Benchmark")

    print("=" * 70)

    # --------------------------------------------------------
    # Training
    # --------------------------------------------------------

    print("\nTraining:")

    print(f"  Training time: " f"{training_seconds:.4f} sec")

    # --------------------------------------------------------
    # Validation
    # --------------------------------------------------------

    print("\nValidation:")

    for metric, value in validation_metrics.items():
        print(f"  {metric}: " f"{value:.6f}")

    # --------------------------------------------------------
    # Test
    # --------------------------------------------------------

    print("\nFINAL TEST:")

    for metric, value in test_metrics.items():
        print(f"  {metric}: " f"{value:.6f}")

    # ========================================================
    # 11. Save final metrics
    # ========================================================

    final_metrics = {
        "model": "LightGBM LambdaRank",
        "optimization": {
            "method": "Optuna",
            "trials": 30,
            "objective": ("validation_ndcg@10"),
            "best_trial": 28,
            "best_validation_ndcg@10": (0.93112094217387),
        },
        "data": {
            "train_rows": len(train),
            "validation_rows": len(validation),
            "test_rows": len(test),
            "train_queries": int(train["query_id"].nunique()),
            "validation_queries": int(validation["query_id"].nunique()),
            "test_queries": int(test["query_id"].nunique()),
        },
        "training": {"training_time_seconds": float(training_seconds)},
        "validation": validation_metrics,
        "test": test_metrics,
    }

    with FINAL_METRICS_PATH.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            final_metrics,
            file,
            indent=2,
        )

    # ========================================================
    # 12. Save model configuration
    # ========================================================

    final_config = {
        "model": ("LightGBM LambdaRank"),
        "objective": ("lambdarank"),
        "metric": "ndcg",
        "feature_count": len(feature_columns),
        "feature_columns": (feature_columns),
        "hyperparameters": (model_params),
        "training_protocol": (
            "Optuna selected hyperparameters "
            "using the validation split. "
            "Hyperparameters were frozen before "
            "final test evaluation. The final "
            "model was fitted on the training "
            "split only. Validation and test "
            "were used only for evaluation."
        ),
    }

    with FINAL_CONFIG_PATH.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            final_config,
            file,
            indent=2,
        )

    # ========================================================
    # 13. Completion
    # ========================================================

    print("\nArtifacts saved:")

    print(f"  {FINAL_METRICS_PATH}")

    print(f"  {FINAL_CONFIG_PATH}")

    print("\nStage 17 final benchmark completed.")


if __name__ == "__main__":
    main()
