from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from relevanceflow.evaluation.ranking_metrics import (
    average_precision_at_k,
    ndcg_at_k,
    recall_at_k,
    reciprocal_rank_at_k,
)
from relevanceflow.retrieval.tfidf import (
    TfidfConfig,
    TfidfRetriever,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data" / "processed" / "wands"

OUTPUT_DIR = PROJECT_ROOT / "experiments" / "tfidf_baseline"

TOP_K = 10


def load_data() -> (
    tuple[
        pd.DataFrame,
        pd.DataFrame,
        pd.DataFrame,
    ]
):
    products = pd.read_parquet(DATA_DIR / "products.parquet")

    queries = pd.read_parquet(DATA_DIR / "queries.parquet")

    train = pd.read_parquet(DATA_DIR / "train.parquet")

    validation = pd.read_parquet(DATA_DIR / "validation.parquet")

    test = pd.read_parquet(DATA_DIR / "test.parquet")

    return (
        products,
        queries,
        train,
        validation,
        test,
    )


def evaluate_split(
    retriever: TfidfRetriever,
    products: pd.DataFrame,
    queries: pd.DataFrame,
    judgments: pd.DataFrame,
) -> dict[str, float]:
    """Evaluate TF-IDF against one query-grouped split."""

    query_lookup = queries.set_index("query_id")["query"]

    product_lookup = products.set_index("product_id")["product_name"]

    metric_rows = []

    for query_id, group in judgments.groupby(
        "query_id",
        sort=True,
    ):
        if query_id not in query_lookup.index:
            continue

        query_text = query_lookup.loc[query_id]

        candidate_product_ids = group["product_id"].tolist()

        candidate_products = pd.DataFrame(
            {
                "product_id": candidate_product_ids,
                "product_name": [
                    product_lookup.get(
                        product_id,
                        "",
                    )
                    for product_id in candidate_product_ids
                ],
            }
        )

        ranked = retriever.rank_candidates(
            query_text=query_text,
            candidate_products=candidate_products,
            top_k=TOP_K,
        )

        relevance_lookup = group.set_index("product_id")["relevance_score"]

        ranked_relevance = [
            relevance_lookup.get(
                product_id,
                0,
            )
            for product_id in ranked["product_id"]
        ]

        relevance_array = np.asarray(
            ranked_relevance,
            dtype=float,
        )

        metric_rows.append(
            {
                "query_id": query_id,
                "recall@10": recall_at_k(
                    relevance_array,
                    TOP_K,
                ),
                "mrr@10": reciprocal_rank_at_k(
                    relevance_array,
                    TOP_K,
                ),
                "ndcg@10": ndcg_at_k(
                    relevance_array,
                    TOP_K,
                ),
                "map@10": average_precision_at_k(
                    relevance_array,
                    TOP_K,
                ),
            }
        )

    metrics_df = pd.DataFrame(metric_rows)

    if metrics_df.empty:
        raise RuntimeError("No queries were evaluated.")

    return {
        "queries_evaluated": len(metrics_df),
        "recall@10": float(metrics_df["recall@10"].mean()),
        "mrr@10": float(metrics_df["mrr@10"].mean()),
        "ndcg@10": float(metrics_df["ndcg@10"].mean()),
        "map@10": float(metrics_df["map@10"].mean()),
    }


def main() -> None:
    products, queries, train, validation, test = load_data()

    # IMPORTANT:
    # Fit TF-IDF ONLY on training products.
    retriever = TfidfRetriever(
        TfidfConfig(
            ngram_range=(1, 2),
            min_df=1,
            max_df=0.95,
            sublinear_tf=True,
            lowercase=True,
            max_features=200_000,
        )
    )

    retriever.fit(
        product_ids=products["product_id"],
        product_text=products["product_name"],
    )

    print("=" * 70)
    print("RelevanceFlow — TF-IDF Baseline")
    print("=" * 70)

    print(f"Vocabulary size: {len(retriever.vectorizer.vocabulary_):,}")

    print(f"Train queries: {train['query_id'].nunique()}")

    print(f"Validation queries: {validation['query_id'].nunique()}")

    print(f"Test queries: {test['query_id'].nunique()}")

    print("\nEvaluating validation...")

    validation_metrics = evaluate_split(
        retriever=retriever,
        products=products,
        queries=queries,
        judgments=validation,
    )

    print("\nValidation metrics:")

    for key, value in validation_metrics.items():
        if key == "queries_evaluated":
            print(f"{key:20s}: {value}")
        else:
            print(f"{key:20s}: {value:.6f}")

    print("\nEvaluating test...")

    test_metrics = evaluate_split(
        retriever=retriever,
        products=products,
        queries=queries,
        judgments=test,
    )

    print("\nTest metrics:")

    for key, value in test_metrics.items():
        if key == "queries_evaluated":
            print(f"{key:20s}: {value}")
        else:
            print(f"{key:20s}: {value:.6f}")

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    results = {
        "experiment": "tfidf_baseline",
        "model": "TF-IDF",
        "fields": [
            "product_name",
        ],
        "ngram_range": [1, 2],
        "min_df": 1,
        "max_df": 0.95,
        "sublinear_tf": True,
        "lowercase": True,
        "max_features": 200_000,
        "top_k": TOP_K,
        "fit_corpus": "training_product_names",
        "validation": validation_metrics,
        "test": test_metrics,
    }

    with open(
        OUTPUT_DIR / "metrics.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            results,
            file,
            indent=2,
        )

    print(f"\nMetrics saved to: {OUTPUT_DIR / 'metrics.json'}")


if __name__ == "__main__":
    main()
