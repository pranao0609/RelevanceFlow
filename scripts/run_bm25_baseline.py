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
from relevanceflow.retrieval.bm25 import (
    BM25Config,
    BM25Retriever,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]

PRODUCT_PATH = PROJECT_ROOT / "data" / "processed" / "wands" / "products.parquet"

QUERY_PATH = PROJECT_ROOT / "data" / "processed" / "wands" / "queries.parquet"

TRAIN_PATH = PROJECT_ROOT / "data" / "processed" / "wands" / "train.parquet"

VALIDATION_PATH = PROJECT_ROOT / "data" / "processed" / "wands" / "validation.parquet"

TEST_PATH = PROJECT_ROOT / "data" / "processed" / "wands" / "test.parquet"

OUTPUT_DIR = PROJECT_ROOT / "experiments" / "bm25_baseline"

METRICS_PATH = OUTPUT_DIR / "metrics.json"

TEST_PREDICTIONS_PATH = OUTPUT_DIR / "test_predictions.csv"

TOP_K = 10


def load_data():
    """Load WANDS products, queries, and grouped splits."""

    products = pd.read_parquet(PRODUCT_PATH)
    queries = pd.read_parquet(QUERY_PATH)

    train = pd.read_parquet(TRAIN_PATH)
    validation = pd.read_parquet(VALIDATION_PATH)
    test = pd.read_parquet(TEST_PATH)

    required_product_columns = {
        "product_id",
        "product_name",
    }

    required_query_columns = {
        "query_id",
        "query",
    }

    required_judgment_columns = {
        "query_id",
        "product_id",
        "relevance_score",
    }

    if not required_product_columns.issubset(products.columns):
        missing = required_product_columns - set(products.columns)

        raise ValueError("Products missing columns: " f"{sorted(missing)}")

    if not required_query_columns.issubset(queries.columns):
        missing = required_query_columns - set(queries.columns)

        raise ValueError("Queries missing columns: " f"{sorted(missing)}")

    for name, df in {
        "train": train,
        "validation": validation,
        "test": test,
    }.items():
        if not required_judgment_columns.issubset(df.columns):
            missing = required_judgment_columns - set(df.columns)

            raise ValueError(f"{name} missing columns: " f"{sorted(missing)}")

    return (
        products,
        queries,
        train,
        validation,
        test,
    )


def evaluate(
    retriever: BM25Retriever,
    split: pd.DataFrame,
    queries: pd.DataFrame,
    products: pd.DataFrame,
    prediction_output_path: Path | None = None,
) -> dict[str, float]:
    """
    Evaluate BM25 on a WANDS split.

    The candidate products are the complete judged
    candidate set for each query.

    The retriever returns the top-k ranking.

    If prediction_output_path is provided, actual
    rankings are saved as:

        query_id, product_id, score, rank
    """

    query_lookup = queries.set_index("query_id")["query"].to_dict()

    metric_values = {
        "recall@10": [],
        "mrr@10": [],
        "ndcg@10": [],
        "map@10": [],
        "query_latency_ms": [],
    }

    prediction_rows = []

    for query_id, group in split.groupby(
        "query_id",
        sort=True,
    ):
        query_id = int(query_id)

        if query_id not in query_lookup:
            raise ValueError(f"Query ID {query_id} " "not found in queries.")

        query_text = str(query_lookup[query_id])

        # ---------------------------------------------------------
        # Join product names to the judgment candidates.
        # ---------------------------------------------------------

        candidates = group[["product_id"]].merge(
            products[["product_id", "product_name"]],
            on="product_id",
            how="left",
            validate="many_to_one",
        )

        if candidates["product_name"].isna().any():
            missing_products = (
                candidates.loc[
                    candidates["product_name"].isna(),
                    "product_id",
                ]
                .astype(int)
                .tolist()
            )

            raise ValueError(
                "Missing product_name for " f"product IDs: " f"{missing_products[:10]}"
            )

        # ---------------------------------------------------------
        # BM25 ranking
        # ---------------------------------------------------------

        start_time = time.perf_counter()

        ranked = retriever.rank_candidates(
            query=query_text,
            candidate_products=candidates.to_dict("records"),
            top_k=TOP_K,
        )

        latency_ms = (time.perf_counter() - start_time) * 1000.0

        if not ranked:
            raise ValueError(f"BM25 returned no candidates " f"for query_id={query_id}")

        # ---------------------------------------------------------
        # Extract ranked product IDs.
        # ---------------------------------------------------------

        ranked_product_ids = [int(item["product_id"]) for item in ranked]

        relevance_lookup = {
            int(product_id): int(relevance_score)
            for product_id, relevance_score in zip(
                group["product_id"],
                group["relevance_score"],
            )
        }

        relevance = [relevance_lookup[product_id] for product_id in ranked_product_ids]

        binary_relevance = [1 if value > 0 else 0 for value in relevance]

        # ---------------------------------------------------------
        # Metrics
        #
        # These are retained for the baseline's historical
        # metrics.json. The authoritative cross-model comparison
        # will use common_ranking.py.
        # ---------------------------------------------------------

        metric_values["recall@10"].append(
            recall_at_k(
                relevance=relevance,
                k=TOP_K,
            )
        )

        metric_values["mrr@10"].append(
            reciprocal_rank_at_k(
                relevance=relevance,
                k=TOP_K,
            )
        )

        metric_values["ndcg@10"].append(
            ndcg_at_k(
                relevance=relevance,
                k=TOP_K,
            )
        )

        metric_values["map@10"].append(
            average_precision_at_k(
                relevance=binary_relevance,
                k=TOP_K,
            )
        )

        metric_values["query_latency_ms"].append(latency_ms)

        # ---------------------------------------------------------
        # Save actual predictions.
        # ---------------------------------------------------------

        if prediction_output_path is not None:
            for rank_position, item in enumerate(
                ranked,
                start=1,
            ):
                prediction_rows.append(
                    {
                        "query_id": query_id,
                        "product_id": int(item["product_id"]),
                        "score": float(item["score"]),
                        "rank": rank_position,
                    }
                )

    number_of_queries = len(metric_values["recall@10"])

    if number_of_queries == 0:
        raise ValueError("No queries were evaluated.")

    metrics = {
        "queries_evaluated": int(number_of_queries),
        "recall@10": float(sum(metric_values["recall@10"]) / number_of_queries),
        "mrr@10": float(sum(metric_values["mrr@10"]) / number_of_queries),
        "ndcg@10": float(sum(metric_values["ndcg@10"]) / number_of_queries),
        "map@10": float(sum(metric_values["map@10"]) / number_of_queries),
        "average_query_latency_ms": float(
            sum(metric_values["query_latency_ms"]) / number_of_queries
        ),
    }

    # -------------------------------------------------------------
    # Save predictions.
    # -------------------------------------------------------------

    if prediction_output_path is not None:
        prediction_output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        predictions_df = pd.DataFrame(prediction_rows)

        predictions_df.to_csv(
            prediction_output_path,
            index=False,
        )

        print("\nSaved predictions: " f"{prediction_output_path}")

        print("Prediction rows: " f"{len(predictions_df):,}")

    return metrics


def main() -> None:
    print("=" * 70)
    print("RelevanceFlow — " "BM25 Retrieval Baseline")
    print("=" * 70)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # -------------------------------------------------------------
    # Load data
    # -------------------------------------------------------------

    print("\nLoading data...")

    (
        products,
        queries,
        train,
        validation,
        test,
    ) = load_data()

    print(f"Products:       {len(products):,}")

    print(f"Queries:        {len(queries):,}")

    print(f"Train rows:     {len(train):,}")

    print(f"Validation:     {len(validation):,}")

    print(f"Test rows:      {len(test):,}")

    print("Train queries: " f"{train['query_id'].nunique():,}")

    print("Validation queries: " f"{validation['query_id'].nunique():,}")

    print("Test queries: " f"{test['query_id'].nunique():,}")

    # -------------------------------------------------------------
    # TRAIN-ONLY BM25 CORPUS
    # -------------------------------------------------------------

    train_product_ids = train["product_id"].unique()

    train_products = products[products["product_id"].isin(train_product_ids)].copy()

    train_products = train_products.sort_values("product_id").reset_index(drop=True)

    print("\nBM25 training corpus")
    print("-" * 70)

    print("Training-associated products: " f"{len(train_products):,}")

    print("Total catalog products: " f"{len(products):,}")

    print(
        "Products excluded from "
        "BM25 fitting: "
        f"{len(products) - len(train_products):,}"
    )

    # -------------------------------------------------------------
    # Configuration
    # -------------------------------------------------------------

    config = BM25Config(
        k1=1.5,
        b=0.75,
    )

    retriever = BM25Retriever(config)

    # -------------------------------------------------------------
    # Fit
    # -------------------------------------------------------------

    print("\nFitting BM25...")

    fit_start = time.perf_counter()

    retriever.fit(train_products["product_name"])

    fit_time = time.perf_counter() - fit_start

    print(f"BM25 fit time: " f"{fit_time:.3f} seconds")

    # -------------------------------------------------------------
    # Validation
    # -------------------------------------------------------------

    print("\n" + "=" * 70)
    print("Validation Evaluation")
    print("=" * 70)

    validation_metrics = evaluate(
        retriever=retriever,
        split=validation,
        queries=queries,
        products=products,
    )

    for metric, value in validation_metrics.items():
        print(f"{metric}: " f"{value:.6f}")

    # -------------------------------------------------------------
    # Test
    # -------------------------------------------------------------

    print("\n" + "=" * 70)
    print("Test Evaluation")
    print("=" * 70)

    test_metrics = evaluate(
        retriever=retriever,
        split=test,
        queries=queries,
        products=products,
        prediction_output_path=(TEST_PREDICTIONS_PATH),
    )

    for metric, value in test_metrics.items():
        print(f"{metric}: " f"{value:.6f}")

    # -------------------------------------------------------------
    # Save metrics
    # -------------------------------------------------------------

    metrics = {
        "model": "BM25",
        "evaluation_protocol": {
            "candidate_source": ("WANDS judged " "query-product candidates"),
            "ranking_cutoff": TOP_K,
            "split_strategy": ("query_grouped"),
            "bm25_fit_scope": ("training-associated " "products only"),
        },
        "config": {
            "k1": config.k1,
            "b": config.b,
        },
        "training": {
            "total_products": len(products),
            "training_products": len(train_products),
            "training_queries": int(train["query_id"].nunique()),
            "fit_time_seconds": float(fit_time),
        },
        "validation": validation_metrics,
        "test": test_metrics,
        "artifacts": {
            "test_predictions": str(TEST_PREDICTIONS_PATH.relative_to(PROJECT_ROOT)),
        },
    }

    with METRICS_PATH.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            metrics,
            file,
            indent=2,
        )

    # -------------------------------------------------------------
    # Final output
    # -------------------------------------------------------------

    print("\n" + "=" * 70)
    print("Artifacts")
    print("=" * 70)

    print(f"Metrics:      {METRICS_PATH}")

    print("Predictions:  " f"{TEST_PREDICTIONS_PATH}")

    print("\nBM25 baseline " "completed successfully.")


if __name__ == "__main__":
    main()
