from __future__ import annotations

import json
import time
from pathlib import Path

import mlflow
import pandas as pd

from relevanceflow.evaluation.common_ranking import evaluate_rankings
from relevanceflow.models.ltr import LightGBMLTR, LTRConfig
from relevanceflow.utils.config import load_configs
from relevanceflow.utils.mlflow_utils import (
    log_artifact_if_exists,
    log_git_metadata,
    log_metrics,
    log_params,
    start_run,
)

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
    """Load train, validation, and test feature matrices."""
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
    """Validate query-grouped train/validation/test splits."""

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


def prepare_ranking_inputs(
    predictions: pd.DataFrame,
    ground_truth: pd.DataFrame,
) -> tuple[dict[int, list[int]], dict[int, dict[int, int]]]:
    """Convert ranking DataFrames to common evaluator inputs."""

    ranked_predictions: dict[int, list[int]] = {}

    for query_id, group in predictions.groupby(
        "query_id",
        sort=False,
    ):
        ranked_predictions[int(query_id)] = (
            group.sort_values("rank")["product_id"].astype(int).tolist()
        )

    ground_truth_mapping: dict[int, dict[int, int]] = {}

    for query_id, group in ground_truth.groupby(
        "query_id",
        sort=False,
    ):
        ground_truth_mapping[int(query_id)] = {
            int(product_id): int(relevance)
            for product_id, relevance in zip(
                group["product_id"],
                group["relevance_score"],
            )
        }

    return ranked_predictions, ground_truth_mapping


def main() -> None:
    """Train, evaluate, and track the hybrid LightGBM ranking model."""

    mlflow_config = load_configs()

    with start_run(
        config=mlflow_config,
        run_name="Hybrid-LightGBM-LTR",
        project_root=PROJECT_ROOT,
    ):
        start_time = time.perf_counter()

        EXPERIMENT_DIR.mkdir(
            parents=True,
            exist_ok=True,
        )

        # ---------------------------------------------------------
        # Load data
        # ---------------------------------------------------------

        train, validation, test = load_data()

        validate_splits(
            train,
            validation,
            test,
        )

        # ---------------------------------------------------------
        # Model configuration
        # ---------------------------------------------------------

        ltr_config = LTRConfig(
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

        model = LightGBMLTR(ltr_config)

        print("\nTraining hybrid LightGBM LambdaRank model...")

        # ---------------------------------------------------------
        # Train
        # ---------------------------------------------------------

        training_start = time.perf_counter()

        model.fit(
            train=train,
            validation=validation,
        )

        training_time = time.perf_counter() - training_start

        print(f"Training time: {training_time:.2f} seconds")

        # ---------------------------------------------------------
        # MLflow parameters
        # ---------------------------------------------------------

        log_params(
            {
                "model_type": "LightGBM LambdaRank",
                "objective": "lambdarank",
                "feature_count": len(model.feature_columns),
                "features": ",".join(model.feature_columns),
                "n_estimators": ltr_config.n_estimators,
                "learning_rate": ltr_config.learning_rate,
                "num_leaves": ltr_config.num_leaves,
                "max_depth": ltr_config.max_depth,
                "min_child_samples": ltr_config.min_child_samples,
                "subsample": ltr_config.subsample,
                "colsample_bytree": ltr_config.colsample_bytree,
                "reg_alpha": ltr_config.reg_alpha,
                "reg_lambda": ltr_config.reg_lambda,
                "random_state": ltr_config.random_state,
                "eval_at": ",".join(str(value) for value in ltr_config.eval_at),
                "early_stopping_rounds": (ltr_config.early_stopping_rounds),
                "dataset": "WANDS",
                "split_strategy": "query_grouped",
                "train_queries": train["query_id"].nunique(),
                "validation_queries": validation["query_id"].nunique(),
                "test_queries": test["query_id"].nunique(),
                "train_judgments": len(train),
                "validation_judgments": len(validation),
                "test_judgments": len(test),
            }
        )

        # ---------------------------------------------------------
        # MLflow tags
        # ---------------------------------------------------------

        mlflow.set_tag(
            "task",
            "product_search_relevance_ranking",
        )

        mlflow.set_tag(
            "model_family",
            "gradient_boosted_learning_to_rank",
        )

        mlflow.set_tag(
            "feature_set",
            "lexical_text_metadata_compatibility_semantic",
        )

        mlflow.set_tag(
            "split_strategy",
            "query_grouped",
        )

        mlflow.set_tag(
            "primary_metric",
            "NDCG@10",
        )

        mlflow.set_tag(
            "dataset",
            "WANDS",
        )

        git_commit = log_git_metadata(PROJECT_ROOT)

        print(f"Git commit: {git_commit}")

        # ---------------------------------------------------------
        # Validation prediction
        # ---------------------------------------------------------

        print("\nGenerating validation predictions...")

        validation_inference_start = time.perf_counter()

        validation_predictions = model.rank_candidates(validation)

        validation_inference_time = time.perf_counter() - validation_inference_start

        # ---------------------------------------------------------
        # Test prediction
        # ---------------------------------------------------------

        print("Generating test predictions...")

        test_inference_start = time.perf_counter()

        test_predictions = model.rank_candidates(test)

        test_inference_time = time.perf_counter() - test_inference_start

        test_predictions.to_csv(
            EXPERIMENT_DIR / "test_predictions.csv",
            index=False,
        )

        # ---------------------------------------------------------
        # Authoritative common ranking evaluation
        # ---------------------------------------------------------

        validation_ranked, validation_ground_truth = prepare_ranking_inputs(
            validation_predictions,
            validation,
        )

        test_ranked, test_ground_truth = prepare_ranking_inputs(
            test_predictions,
            test,
        )

        print(
            "\nEvaluator input types:",
            type(validation_ranked),
            type(validation_ground_truth),
        )

        validation_metrics = evaluate_rankings(
            predictions=validation_ranked,
            ground_truth=validation_ground_truth,
            k=10,
        )

        test_metrics = evaluate_rankings(
            predictions=test_ranked,
            ground_truth=test_ground_truth,
            k=10,
        )

        # ---------------------------------------------------------
        # Console output
        # ---------------------------------------------------------

        print("\n" + "=" * 55)
        print("Hybrid LightGBM LambdaRank")
        print("=" * 55)

        print("\nValidation:")

        for metric, value in validation_metrics.items():
            print(f"{metric}: {value:.6f}")

        print("\nTest:")

        for metric, value in test_metrics.items():
            print(f"{metric}: {value:.6f}")

        # ---------------------------------------------------------
        # Feature importance
        # ---------------------------------------------------------

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

        # ---------------------------------------------------------
        # Best iteration
        # ---------------------------------------------------------

        best_iteration = getattr(
            model.model,
            "best_iteration_",
            None,
        )

        # ---------------------------------------------------------
        # Save metrics artifact
        # ---------------------------------------------------------

        metrics_output = {
            "model": "LightGBM LambdaRank",
            "objective": "lambdarank",
            "features": len(model.feature_columns),
            "train_queries": int(train["query_id"].nunique()),
            "validation_queries": int(validation["query_id"].nunique()),
            "test_queries": int(test["query_id"].nunique()),
            "best_iteration": (
                int(best_iteration) if best_iteration is not None else None
            ),
            "training_time_seconds": training_time,
            "validation": validation_metrics,
            "test": test_metrics,
            "validation_inference_time_ms": (validation_inference_time * 1000),
            "test_inference_time_ms": (test_inference_time * 1000),
            "total_runtime_seconds": (time.perf_counter() - start_time),
        }

        with METRICS_PATH.open(
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(
                metrics_output,
                file,
                indent=2,
            )

        # ---------------------------------------------------------
        # Save model configuration
        # ---------------------------------------------------------

        config_output = {
            "n_estimators": ltr_config.n_estimators,
            "learning_rate": ltr_config.learning_rate,
            "num_leaves": ltr_config.num_leaves,
            "max_depth": ltr_config.max_depth,
            "min_child_samples": ltr_config.min_child_samples,
            "subsample": ltr_config.subsample,
            "colsample_bytree": ltr_config.colsample_bytree,
            "reg_alpha": ltr_config.reg_alpha,
            "reg_lambda": ltr_config.reg_lambda,
            "random_state": ltr_config.random_state,
            "eval_at": list(ltr_config.eval_at),
            "early_stopping_rounds": (ltr_config.early_stopping_rounds),
            "feature_columns": model.feature_columns,
        }

        with CONFIG_PATH.open(
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(
                config_output,
                file,
                indent=2,
            )

        # ---------------------------------------------------------
        # MLflow metrics
        # ---------------------------------------------------------

        log_metrics(
            {
                "validation_recall_at_10": validation_metrics["recall@10"],
                "validation_mrr_at_10": validation_metrics["mrr@10"],
                "validation_ndcg_at_10": validation_metrics["ndcg@10"],
                "validation_map_at_10": validation_metrics["map@10"],
                "test_recall_at_10": test_metrics["recall@10"],
                "test_mrr_at_10": test_metrics["mrr@10"],
                "test_ndcg_at_10": test_metrics["ndcg@10"],
                "test_map_at_10": test_metrics["map@10"],
                "training_time_seconds": training_time,
                "validation_inference_time_ms": (validation_inference_time * 1000),
                "test_inference_time_ms": (test_inference_time * 1000),
            }
        )

        # ---------------------------------------------------------
        # MLflow artifacts
        # ---------------------------------------------------------

        log_artifact_if_exists(METRICS_PATH)
        log_artifact_if_exists(CONFIG_PATH)
        log_artifact_if_exists(FEATURE_IMPORTANCE_PATH)

        # ---------------------------------------------------------
        # Final output
        # ---------------------------------------------------------

        print("\nArtifacts saved:")
        print(METRICS_PATH)
        print(CONFIG_PATH)
        print(FEATURE_IMPORTANCE_PATH)

        print("\nMLflow tracking completed.")


if __name__ == "__main__":
    main()
