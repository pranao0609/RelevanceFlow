from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import pandas as pd
import yaml

from relevanceflow.evaluation.ranking_metrics import (
    average_precision_at_k,
    ndcg_at_k,
    recall_at_k,
    reciprocal_rank_at_k,
)
from relevanceflow.retrieval.bm25 import BM25Config, BM25Retriever

PROJECT_ROOT = Path(__file__).resolve().parents[1]

sys.path.insert(0, str(PROJECT_ROOT / "src"))


DATA_DIR = PROJECT_ROOT / "data" / "processed" / "wands"
CONFIG_PATH = PROJECT_ROOT / "configs" / "retrieval.yaml"
OUTPUT_DIR = PROJECT_ROOT / "experiments" / "bm25_baseline"

TRAIN_PATH = DATA_DIR / "train.parquet"
VALIDATION_PATH = DATA_DIR / "validation.parquet"
TEST_PATH = DATA_DIR / "test.parquet"
PRODUCT_PATH = DATA_DIR / "products.parquet"


def load_config() -> dict:
    with open(CONFIG_PATH, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def load_data():
    train = pd.read_parquet(TRAIN_PATH)
    validation = pd.read_parquet(VALIDATION_PATH)
    test = pd.read_parquet(TEST_PATH)
    products = pd.read_parquet(PRODUCT_PATH)

    queries_path = DATA_DIR / "queries.parquet"
    queries = pd.read_parquet(queries_path)

    required_query_columns = {"query_id", "query"}

    missing_columns = required_query_columns - set(queries.columns)

    if missing_columns:
        raise ValueError(
            f"queries.parquet is missing required columns: {sorted(missing_columns)}"
        )

    # Attach query text to each judgment split.
    train = train.merge(
        queries[["query_id", "query"]],
        on="query_id",
        how="left",
        validate="many_to_one",
    )

    validation = validation.merge(
        queries[["query_id", "query"]],
        on="query_id",
        how="left",
        validate="many_to_one",
    )

    test = test.merge(
        queries[["query_id", "query"]],
        on="query_id",
        how="left",
        validate="many_to_one",
    )

    for name, dataframe in [
        ("train", train),
        ("validation", validation),
        ("test", test),
    ]:
        missing_queries = dataframe["query"].isna().sum()

        if missing_queries > 0:
            raise ValueError(
                f"{name} contains {missing_queries} judgments "
                f"without matching query text."
            )

    return train, validation, test, products


def evaluate(
    retriever: BM25Retriever,
    judgments: pd.DataFrame,
    products: pd.DataFrame,
    top_k: int = 10,
):
    product_lookup = products.set_index("product_id")["product_name"].to_dict()

    metrics = {
        "queries_evaluated": 0,
        "recall@10": [],
        "mrr@10": [],
        "ndcg@10": [],
        "map@10": [],
    }

    total_latency = 0.0

    for query_id, group in judgments.groupby("query_id", sort=True):
        query_text = str(group["query"].iloc[0])

        candidate_products = []

        relevance_lookup = {}

        for _, row in group.iterrows():
            product_id = row["product_id"]

            candidate_products.append(
                {
                    "product_id": product_id,
                    "product_name": product_lookup.get(
                        product_id,
                        "",
                    ),
                }
            )

            relevance_lookup[product_id] = int(row["relevance_score"])

        ranked, latency = retriever.score_query_latency(
            query=query_text,
            candidate_products=candidate_products,
            top_k=top_k,
        )

        total_latency += latency
        metrics["queries_evaluated"] += 1

        ranked_relevance = [
            relevance_lookup.get(item["product_id"], 0) for item in ranked
        ]

        metrics["recall@10"].append(
            recall_at_k(
                ranked_relevance,
                k=top_k,
            )
        )

        metrics["mrr@10"].append(
            reciprocal_rank_at_k(
                ranked_relevance,
                k=top_k,
            )
        )

        metrics["ndcg@10"].append(
            ndcg_at_k(
                ranked_relevance,
                k=top_k,
            )
        )

        metrics["map@10"].append(
            average_precision_at_k(
                ranked_relevance,
                k=top_k,
            )
        )

    query_count = metrics["queries_evaluated"]

    return {
        "queries_evaluated": query_count,
        "recall@10": sum(metrics["recall@10"]) / query_count,
        "mrr@10": sum(metrics["mrr@10"]) / query_count,
        "ndcg@10": sum(metrics["ndcg@10"]) / query_count,
        "map@10": sum(metrics["map@10"]) / query_count,
        "average_query_latency_ms": (total_latency / query_count * 1000),
    }


def main():
    print("=" * 70)
    print("RelevanceFlow — BM25 Retrieval Baseline")
    print("=" * 70)

    config = load_config()
    train, validation, test, products = load_data()

    bm25_config = config["bm25"]

    retriever = BM25Retriever(
        BM25Config(
            k1=float(bm25_config["k1"]),
            b=float(bm25_config["b"]),
        )
    )

    # ---------------------------------------------------------------
    # Build training corpus
    # ---------------------------------------------------------------

    train_product_ids = train["product_id"].unique()

    train_products = products[products["product_id"].isin(train_product_ids)].copy()

    train_product_names = train_products["product_name"].fillna("").astype(str).tolist()

    print()
    print("Building BM25 index...")
    print(f"Training-associated products: {len(train_products):,}")

    index_start = time.perf_counter()

    retriever.fit(train_product_names)

    index_creation_time = time.perf_counter() - index_start

    print(f"Index creation time: {index_creation_time:.4f} seconds")

    print(f"Vocabulary size: {len(retriever.vocabulary_):,}")

    print(f"Average document length: {retriever.avg_doc_length_:.2f}")

    print()
    print("Evaluating validation...")

    validation_metrics = evaluate(
        retriever=retriever,
        judgments=validation,
        products=products,
        top_k=10,
    )

    print()
    print("Validation metrics:")

    for key, value in validation_metrics.items():
        print(f"{key:28}: {value:.6f}")

    print()
    print("Evaluating test...")

    test_metrics = evaluate(
        retriever=retriever,
        judgments=test,
        products=products,
        top_k=10,
    )

    print()
    print("Test metrics:")

    for key, value in test_metrics.items():
        print(f"{key:28}: {value:.6f}")

    # ---------------------------------------------------------------
    # Save experiment
    # ---------------------------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    metrics = {
        "experiment": "bm25_baseline",
        "model": "BM25",
        "fields": ["product_name"],
        "k1": float(bm25_config["k1"]),
        "b": float(bm25_config["b"]),
        "top_k": 10,
        "fit_corpus": "training_associated_product_names",
        "train_products": len(train_products),
        "vocabulary_size": len(retriever.vocabulary_),
        "avg_document_length": float(retriever.avg_doc_length_),
        "index_creation_time_seconds": float(index_creation_time),
        "estimated_index_memory_bytes": (retriever.estimated_memory_bytes_),
        "validation": validation_metrics,
        "test": test_metrics,
    }

    output_path = OUTPUT_DIR / "metrics.json"

    with open(
        output_path,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            metrics,
            file,
            indent=2,
        )

    print()
    print(f"Metrics saved to: {output_path}")


if __name__ == "__main__":
    main()
