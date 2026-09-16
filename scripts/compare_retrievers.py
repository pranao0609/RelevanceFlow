from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

from relevanceflow.evaluation import evaluate_retriever
from relevanceflow.retrieval.bm25 import BM25Config, BM25Retriever
from relevanceflow.retrieval.tfidf import TfidfConfig, TfidfRetriever

PROJECT_ROOT = Path(__file__).resolve().parents[1]

sys.path.insert(
    0,
    str(PROJECT_ROOT / "src"),
)


DATA_DIR = PROJECT_ROOT / "data" / "processed" / "wands"
OUTPUT_DIR = PROJECT_ROOT / "experiments" / "retrieval_comparison"

TRAIN_PATH = DATA_DIR / "train.parquet"
TEST_PATH = DATA_DIR / "test.parquet"
PRODUCT_PATH = DATA_DIR / "products.parquet"
QUERY_PATH = DATA_DIR / "queries.parquet"


def load_data():
    train = pd.read_parquet(TRAIN_PATH)
    test = pd.read_parquet(TEST_PATH)
    products = pd.read_parquet(PRODUCT_PATH)
    queries = pd.read_parquet(QUERY_PATH)

    test = test.merge(
        queries[["query_id", "query"]],
        on="query_id",
        how="left",
        validate="many_to_one",
    )

    return train, test, products


def build_ground_truth(test: pd.DataFrame):
    ground_truth = {}

    for query_id, group in test.groupby(
        "query_id",
        sort=True,
    ):
        ground_truth[query_id] = {
            row["product_id"]: int(row["relevance_score"])
            for _, row in group.iterrows()
        }

    return ground_truth


def build_candidates(
    group: pd.DataFrame,
    product_lookup: dict,
):
    return [
        {
            "product_id": product_id,
            "product_name": product_lookup.get(
                product_id,
                "",
            ),
        }
        for product_id in group["product_id"]
    ]


def generate_predictions(
    retriever,
    test: pd.DataFrame,
    products: pd.DataFrame,
):
    product_lookup = (
        products.set_index("product_id")["product_name"]
        .fillna("")
        .astype(str)
        .to_dict()
    )

    predictions = {}

    for query_id, group in test.groupby(
        "query_id",
        sort=True,
    ):
        query_text = str(group["query"].iloc[0])

        candidate_dataframe = group[["product_id"]].copy()

        candidate_dataframe["product_name"] = (
            candidate_dataframe["product_id"].map(product_lookup).fillna("")
        )

        # TF-IDF expects a pandas DataFrame.
        if isinstance(retriever, TfidfRetriever):
            ranked = retriever.rank_candidates(
                query_text=query_text,
                candidate_products=candidate_dataframe,
                top_k=20,
            )

            predictions[query_id] = ranked["product_id"].tolist()

        # BM25 expects a list of dictionaries.
        elif isinstance(retriever, BM25Retriever):
            candidate_products = [
                {
                    "product_id": row["product_id"],
                    "product_name": row["product_name"],
                }
                for _, row in candidate_dataframe.iterrows()
            ]

            ranked = retriever.rank_candidates(
                query=query_text,
                candidate_products=candidate_products,
                top_k=20,
            )

            predictions[query_id] = [result["product_id"] for result in ranked]

        else:
            raise TypeError(f"Unsupported retriever type: {type(retriever).__name__}")

    return predictions


def create_tfidf(train: pd.DataFrame, products: pd.DataFrame):
    train_product_ids = train["product_id"].unique()

    train_products = products[products["product_id"].isin(train_product_ids)].copy()

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
        product_ids=train_products["product_id"],
        product_text=train_products["product_name"],
    )

    return retriever


def create_bm25(train: pd.DataFrame, products: pd.DataFrame):
    train_product_ids = train["product_id"].unique()

    train_products = products[products["product_id"].isin(train_product_ids)].copy()

    retriever = BM25Retriever(
        BM25Config(
            k1=1.5,
            b=0.75,
        )
    )

    retriever.fit(train_products["product_name"].fillna("").astype(str).tolist())

    return retriever


def main():
    print("=" * 70)
    print("RelevanceFlow — Retrieval Comparison")
    print("=" * 70)

    train, test, products = load_data()

    ground_truth = build_ground_truth(test)

    print()
    print("Building TF-IDF...")
    tfidf = create_tfidf(
        train=train,
        products=products,
    )

    print("Generating TF-IDF predictions...")
    tfidf_predictions = generate_predictions(
        retriever=tfidf,
        test=test,
        products=products,
    )

    print("Evaluating TF-IDF...")
    tfidf_metrics = evaluate_retriever(
        predictions=tfidf_predictions,
        ground_truth=ground_truth,
        k=[1, 5, 10, 20],
    )

    print()
    print("Building BM25...")
    bm25 = create_bm25(
        train=train,
        products=products,
    )

    print("Generating BM25 predictions...")
    bm25_predictions = generate_predictions(
        retriever=bm25,
        test=test,
        products=products,
    )

    print("Evaluating BM25...")
    bm25_metrics = evaluate_retriever(
        predictions=bm25_predictions,
        ground_truth=ground_truth,
        k=[1, 5, 10, 20],
    )

    # ---------------------------------------------------------------
    # Comparison table
    # ---------------------------------------------------------------

    rows = []

    for model_name, metrics in [
        ("TF-IDF", tfidf_metrics),
        ("BM25", bm25_metrics),
    ]:
        rows.append(
            {
                "Model": model_name,
                "Recall@1": metrics["recall@1"],
                "Recall@5": metrics["recall@5"],
                "Recall@10": metrics["recall@10"],
                "Recall@20": metrics["recall@20"],
                "MRR@10": metrics["mrr@10"],
                "NDCG@5": metrics["ndcg@5"],
                "NDCG@10": metrics["ndcg@10"],
                "NDCG@20": metrics["ndcg@20"],
                "MAP@10": metrics["map@10"],
            }
        )

    comparison = pd.DataFrame(rows)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = OUTPUT_DIR / "comparison.csv"

    comparison.to_csv(
        output_path,
        index=False,
    )

    print()
    print("=" * 70)
    print("RETRIEVAL COMPARISON")
    print("=" * 70)
    print()

    print(
        comparison.to_string(
            index=False,
            float_format=lambda value: f"{value:.6f}",
        )
    )

    print()
    print(f"Saved to: {output_path}")

    # Save complete metrics as JSON.
    json_path = OUTPUT_DIR / "comparison_metrics.json"

    with open(
        json_path,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            {
                "TF-IDF": tfidf_metrics,
                "BM25": bm25_metrics,
            },
            file,
            indent=2,
        )

    print(f"Saved to: {json_path}")


if __name__ == "__main__":
    main()
