from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from relevanceflow.evaluation.ranking_metrics import (
    average_precision_at_k,
    ndcg_at_k,
    recall_at_k,
    reciprocal_rank_at_k,
)
from relevanceflow.retrieval.bm25 import BM25Retriever
from relevanceflow.retrieval.semantic import (
    SemanticRetriever,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed" / "wands"

SEMANTIC_DIR = PROJECT_ROOT / "data" / "interim" / "semantic"

OUTPUT_DIR = PROJECT_ROOT / "experiments" / "semantic_vs_bm25"

PRODUCT_PATH = PROCESSED_DIR / "products.parquet"

QUERY_PATH = PROCESSED_DIR / "queries.parquet"

TRAIN_PATH = PROCESSED_DIR / "train.parquet"

TEST_PATH = PROCESSED_DIR / "test.parquet"

PRODUCT_EMBEDDINGS_PATH = SEMANTIC_DIR / "product_embeddings.npy"

QUERY_EMBEDDINGS_PATH = SEMANTIC_DIR / "query_embeddings.npy"


def evaluate_predictions(
    predictions: dict[int, list[int]],
    ground_truth: dict[int, dict[int, int]],
) -> dict[str, float]:
    values = {
        "recall@10": [],
        "mrr@10": [],
        "ndcg@10": [],
        "map@10": [],
    }

    for query_id, ranked_products in predictions.items():
        if query_id not in ground_truth:
            continue

        relevance_lookup = ground_truth[query_id]

        relevance = [
            relevance_lookup.get(
                product_id,
                0,
            )
            for product_id in ranked_products[:10]
        ]

        values["recall@10"].append(recall_at_k(relevance, 10))

        values["mrr@10"].append(
            reciprocal_rank_at_k(
                relevance,
                10,
            )
        )

        values["ndcg@10"].append(ndcg_at_k(relevance, 10))

        values["map@10"].append(
            average_precision_at_k(
                relevance,
                10,
            )
        )

    return {
        metric: float(np.mean(scores)) if scores else 0.0
        for metric, scores in values.items()
    }


def build_ground_truth(
    dataframe: pd.DataFrame,
) -> dict[int, dict[int, int]]:
    return {
        int(query_id): {
            int(row.product_id): int(row.relevance_score) for row in group.itertuples()
        }
        for query_id, group in dataframe.groupby("query_id")
    }


def main() -> None:
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("Loading data...")

    products = pd.read_parquet(PRODUCT_PATH)

    queries = pd.read_parquet(QUERY_PATH)

    train = pd.read_parquet(TRAIN_PATH)

    test = pd.read_parquet(TEST_PATH)

    test = test.merge(
        queries[["query_id", "query"]],
        on="query_id",
        how="left",
        validate="many_to_one",
    )
    test = test.merge(
        products[["product_id", "product_name"]],
        on="product_id",
        how="left",
        validate="many_to_one",
    )
    print(f"Products: {len(products):,}")

    print(f"Queries: {len(queries):,}")

    print(f"Test queries: " f"{test['query_id'].nunique():,}")

    print(f"Test judgments: {len(test):,}")

    # ---------------------------------------------------------
    # Load embeddings
    # ---------------------------------------------------------

    print("\nLoading embeddings...")

    product_embeddings = np.load(
        PRODUCT_EMBEDDINGS_PATH,
        mmap_mode="r",
    )

    query_embeddings = np.load(QUERY_EMBEDDINGS_PATH)

    print(f"Product embeddings: " f"{product_embeddings.shape}")

    print(f"Query embeddings: " f"{query_embeddings.shape}")

    product_ids = products["product_id"].to_numpy()

    query_ids = queries["query_id"].to_numpy()

    query_embedding_lookup = dict(
        zip(
            query_ids,
            query_embeddings,
        )
    )

    semantic_retriever = SemanticRetriever(
        product_ids=product_ids,
        product_embeddings=product_embeddings,
    )

    ground_truth = build_ground_truth(test)

    # ---------------------------------------------------------
    # Transformer similarity
    # ---------------------------------------------------------

    print("\nEvaluating Transformer similarity...")

    semantic_predictions = {}
    semantic_latencies = []

    for query_id, group in test.groupby(
        "query_id",
        sort=False,
    ):
        query_embedding = query_embedding_lookup[query_id]

        candidate_ids = group["product_id"].to_numpy()

        start = time.perf_counter()

        results = semantic_retriever.rank_candidates(
            query_embedding=query_embedding,
            candidate_product_ids=candidate_ids,
            top_k=10,
        )

        latency = time.perf_counter() - start

        semantic_latencies.append(latency)

        semantic_predictions[int(query_id)] = [
            result["product_id"] for result in results
        ]

    semantic_metrics = evaluate_predictions(
        semantic_predictions,
        ground_truth,
    )

    # ---------------------------------------------------------
    # BM25
    # ---------------------------------------------------------

    print("\nBuilding BM25...")

    train_product_ids = set(train["product_id"])

    train_products = products[products["product_id"].isin(train_product_ids)]

    bm25 = BM25Retriever()

    bm25.fit(train_products["product_name"].tolist())

    print(f"BM25 documents: " f"{len(train_products):,}")

    print(f"BM25 vocabulary: " f"{len(bm25.vocabulary_):,}")

    print("\nEvaluating BM25...")

    bm25_predictions = {}
    bm25_latencies = []

    for query_id, group in test.groupby(
        "query_id",
        sort=False,
    ):
        query = group["query"].iloc[0]

        candidates = group[["product_id", "product_name"]].to_dict(orient="records")

        start = time.perf_counter()

        results = bm25.rank_candidates(
            query=query,
            candidate_products=candidates,
            top_k=10,
        )

        latency = time.perf_counter() - start

        bm25_latencies.append(latency)

        bm25_predictions[int(query_id)] = [result["product_id"] for result in results]

    bm25_metrics = evaluate_predictions(
        bm25_predictions,
        ground_truth,
    )

    # ---------------------------------------------------------
    # Comparison
    # ---------------------------------------------------------

    comparison = pd.DataFrame(
        [
            {
                "model": "BM25",
                **bm25_metrics,
                "avg_query_latency_ms": (np.mean(bm25_latencies) * 1000),
            },
            {
                "model": "Transformer",
                **semantic_metrics,
                "avg_query_latency_ms": (np.mean(semantic_latencies) * 1000),
            },
        ]
    )

    print("\n========================================")
    print("Transformer vs BM25")
    print("========================================")

    print(comparison.to_string(index=False))

    comparison.to_csv(
        OUTPUT_DIR / "comparison.csv",
        index=False,
    )

    benchmark = {
        "dataset": {
            "products": len(products),
            "queries": len(queries),
            "test_queries": int(test["query_id"].nunique()),
            "test_judgments": len(test),
        },
        "transformer": {
            "model": ("sentence-transformers/" "all-MiniLM-L6-v2"),
            "embedding_dimension": int(product_embeddings.shape[1]),
            "query_embeddings": len(query_embeddings),
            "product_embeddings": len(product_embeddings),
            "avg_query_ranking_latency_ms": (float(np.mean(semantic_latencies) * 1000)),
        },
        "bm25": {
            "documents": len(train_products),
            "vocabulary_size": len(bm25.vocabulary_),
            "index_creation_time_seconds": (bm25.index_creation_time_seconds_),
            "estimated_memory_bytes": (bm25.estimated_memory_bytes_),
            "avg_query_latency_ms": (float(np.mean(bm25_latencies) * 1000)),
        },
    }

    with open(
        OUTPUT_DIR / "benchmark.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            benchmark,
            file,
            indent=2,
        )

    print("\nSaved:")

    print(OUTPUT_DIR / "comparison.csv")

    print(OUTPUT_DIR / "benchmark.json")


if __name__ == "__main__":
    main()
