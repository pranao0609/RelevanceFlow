from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from relevanceflow.evaluation.common_ranking import (
    evaluate_rankings,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]

TEST_PATH = PROJECT_ROOT / "data" / "processed" / "wands" / "test.parquet"

OUTPUT_DIR = PROJECT_ROOT / "experiments" / "model_comparison"

COMPARISON_CSV = OUTPUT_DIR / "comparison.csv"

COMPARISON_JSON = OUTPUT_DIR / "comparison_metrics.json"

PREDICTION_FILES = {
    "TF-IDF": (
        PROJECT_ROOT / "experiments" / "tfidf_baseline" / "test_predictions.csv"
    ),
    "BM25": (PROJECT_ROOT / "experiments" / "bm25_baseline" / "test_predictions.csv"),
    "Transformer": (
        PROJECT_ROOT / "experiments" / "semantic_vs_bm25" / "test_predictions.csv"
    ),
    "Hybrid LightGBM LTR": (
        PROJECT_ROOT / "experiments" / "hybrid_ltr" / "test_predictions.csv"
    ),
}


def load_ground_truth() -> dict[int, dict[int, int]]:
    """Load complete WANDS test candidate relevance."""

    test = pd.read_parquet(TEST_PATH)

    required_columns = {
        "query_id",
        "product_id",
        "relevance_score",
    }

    missing = required_columns - set(test.columns)

    if missing:
        raise ValueError("Test data is missing columns: " f"{sorted(missing)}")

    ground_truth = {}

    for query_id, group in test.groupby(
        "query_id",
        sort=True,
    ):
        query_id = int(query_id)

        ground_truth[query_id] = {
            int(product_id): int(relevance_score)
            for product_id, relevance_score in zip(
                group["product_id"],
                group["relevance_score"],
            )
        }

    return ground_truth


def load_predictions(
    path: Path,
) -> dict[int, list[int]]:
    """
    Load saved rankings.

    Expected columns:

        query_id
        product_id
        score
        rank
    """

    if not path.exists():
        raise FileNotFoundError(f"Prediction artifact not found: {path}")

    predictions_df = pd.read_csv(path)

    required_columns = {
        "query_id",
        "product_id",
        "score",
        "rank",
    }

    missing = required_columns - set(predictions_df.columns)

    if missing:
        raise ValueError(f"{path} is missing columns: " f"{sorted(missing)}")

    predictions_df = predictions_df.sort_values(
        ["query_id", "rank"],
        ascending=[True, True],
    )

    predictions = {}

    for query_id, group in predictions_df.groupby(
        "query_id",
        sort=True,
    ):
        query_id = int(query_id)

        predictions[query_id] = group["product_id"].astype(int).tolist()

    return predictions


def validate_predictions(
    model_name: str,
    predictions: dict[int, list[int]],
    ground_truth: dict[int, dict[int, int]],
) -> None:
    """Validate prediction/ground-truth compatibility."""

    prediction_queries = set(predictions)

    ground_truth_queries = set(ground_truth)

    if prediction_queries != ground_truth_queries:
        missing_predictions = ground_truth_queries - prediction_queries

        extra_predictions = prediction_queries - ground_truth_queries

        raise ValueError(
            f"{model_name}: query mismatch. "
            f"Missing predictions: "
            f"{sorted(missing_predictions)[:10]}, "
            f"extra predictions: "
            f"{sorted(extra_predictions)[:10]}"
        )

    for query_id, ranked_products in predictions.items():
        if len(ranked_products) == 0:
            raise ValueError(f"{model_name}: query " f"{query_id} has no predictions.")

        if len(ranked_products) != len(set(ranked_products)):
            raise ValueError(
                f"{model_name}: duplicate " f"product IDs for query " f"{query_id}."
            )

        candidate_products = set(ground_truth[query_id])

        unknown_products = set(ranked_products) - candidate_products

        if unknown_products:
            raise ValueError(
                f"{model_name}: predictions "
                f"contain products not present "
                f"in WANDS candidates for query "
                f"{query_id}: "
                f"{sorted(unknown_products)[:10]}"
            )


def main() -> None:
    print("=" * 70)
    print("RelevanceFlow — " "Common Model Comparison")
    print("=" * 70)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # -------------------------------------------------------------
    # Ground truth
    # -------------------------------------------------------------

    print("\nLoading complete test ground truth...")

    ground_truth = load_ground_truth()

    print(f"Test queries: " f"{len(ground_truth):,}")

    total_judgments = sum(len(products) for products in ground_truth.values())

    print(f"Test judgments: " f"{total_judgments:,}")

    # -------------------------------------------------------------
    # Evaluate models
    # -------------------------------------------------------------

    results = []

    for model_name, prediction_path in PREDICTION_FILES.items():
        print("\n" + "-" * 70)
        print(model_name)
        print("-" * 70)

        print(f"Loading: {prediction_path}")

        predictions = load_predictions(prediction_path)

        print(f"Prediction queries: " f"{len(predictions):,}")

        total_predictions = sum(len(products) for products in predictions.values())

        print(f"Prediction rows: " f"{total_predictions:,}")

        validate_predictions(
            model_name=model_name,
            predictions=predictions,
            ground_truth=ground_truth,
        )

        metrics = evaluate_rankings(
            predictions=predictions,
            ground_truth=ground_truth,
            k=10,
        )

        result = {
            "model": model_name,
            **metrics,
        }

        results.append(result)

        print(f"Recall@10: " f"{metrics['recall@10']:.6f}")

        print(f"MRR@10: " f"{metrics['mrr@10']:.6f}")

        print(f"NDCG@10: " f"{metrics['ndcg@10']:.6f}")

        print(f"MAP@10: " f"{metrics['map@10']:.6f}")

    # -------------------------------------------------------------
    # Save comparison
    # -------------------------------------------------------------

    comparison_df = pd.DataFrame(results)

    comparison_df = comparison_df[
        [
            "model",
            "queries_evaluated",
            "recall@10",
            "mrr@10",
            "ndcg@10",
            "map@10",
        ]
    ]

    comparison_df.to_csv(
        COMPARISON_CSV,
        index=False,
    )

    comparison_json = {
        "evaluation_protocol": {
            "ground_truth": ("complete WANDS test " "candidate sets"),
            "ranking_cutoff": 10,
            "relevance_definition": (
                "relevance_score > 0 " "for Recall/MRR/MAP; " "graded 0/1/2 for NDCG"
            ),
            "evaluation_method": ("common ranking evaluator"),
            "prediction_source": ("saved model rankings"),
        },
        "models": results,
    }

    with COMPARISON_JSON.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            comparison_json,
            file,
            indent=2,
        )

    # -------------------------------------------------------------
    # Display
    # -------------------------------------------------------------

    print("\n" + "=" * 70)
    print("FINAL COMMON COMPARISON")
    print("=" * 70)

    print(
        comparison_df.to_string(
            index=False,
            float_format=lambda value: (f"{value:.6f}"),
        )
    )

    print("\nArtifacts:")
    print(f"CSV:  {COMPARISON_CSV}")
    print(f"JSON: {COMPARISON_JSON}")

    print("\nCommon model comparison " "completed successfully.")


if __name__ == "__main__":
    main()
