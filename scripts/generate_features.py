"""
RelevanceFlow — Feature Generation Script

Loads processed WANDS splits, computes TF-IDF and BM25 lexical scores
using training-fitted retrievers, generates feature matrices via the
feature pipeline, attaches target labels, and exports feature files.
"""

from __future__ import annotations

import time
from pathlib import Path

import pandas as pd

from relevanceflow.features import (
    FeatureConfig,
    build_query_product_features,
)
from relevanceflow.retrieval.bm25 import BM25Config, BM25Retriever
from relevanceflow.retrieval.tfidf import TfidfConfig, TfidfRetriever

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data" / "processed" / "wands"
OUTPUT_DIR = PROJECT_ROOT / "data" / "features"


def load_data() -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    """Load processed WANDS datasets."""
    products = pd.read_parquet(DATA_DIR / "products.parquet")
    queries = pd.read_parquet(DATA_DIR / "queries.parquet")
    train = pd.read_parquet(DATA_DIR / "train.parquet")
    validation = pd.read_parquet(DATA_DIR / "validation.parquet")
    test = pd.read_parquet(DATA_DIR / "test.parquet")

    return products, queries, train, validation, test


def build_pairs(
    judgments: pd.DataFrame,
    queries: pd.DataFrame,
    products: pd.DataFrame,
) -> pd.DataFrame:
    """
    Merge judgment rows with query text and product metadata
    to create the input pairs for the feature pipeline.
    """
    pairs = judgments.merge(
        queries[["query_id", "query"]],
        on="query_id",
        how="left",
        validate="many_to_one",
    )

    pairs = pairs.merge(
        products,
        on="product_id",
        how="left",
        validate="many_to_one",
    )

    return pairs


def attach_lexical_scores(
    pairs: pd.DataFrame,
    tfidf_retriever: TfidfRetriever,
    bm25_retriever: BM25Retriever,
) -> pd.DataFrame:
    """Compute TF-IDF and BM25 scores and attach them to the pairs."""
    pairs = pairs.copy()

    # TF-IDF cosine similarity
    query_vectors = tfidf_retriever.transform_queries(pairs["query"])
    product_vectors = tfidf_retriever.transform_products(
        pairs["product_name"].fillna("").astype(str)
    )

    import numpy as np
    from sklearn.metrics.pairwise import cosine_similarity

    tfidf_scores = np.array(
        [
            cosine_similarity(
                query_vectors[i],
                product_vectors[i],
            )[0, 0]
            for i in range(query_vectors.shape[0])
        ]
    )

    pairs["tfidf_similarity"] = tfidf_scores

    # BM25 scores
    bm25_scores = []

    for _, row in pairs.iterrows():
        query_tokens = bm25_retriever.tokenize(str(row["query"]))
        doc_tokens = bm25_retriever.tokenize(
            str(row["product_name"]) if row["product_name"] else ""
        )

        score = bm25_retriever._score_document(
            query_tokens=query_tokens,
            document_tokens=doc_tokens,
        )

        bm25_scores.append(score)

    pairs["bm25_score"] = bm25_scores

    return pairs


def generate_features_for_split(
    split_name: str,
    judgments: pd.DataFrame,
    queries: pd.DataFrame,
    products: pd.DataFrame,
    tfidf_retriever: TfidfRetriever,
    bm25_retriever: BM25Retriever,
    config: FeatureConfig,
) -> pd.DataFrame:
    """Generate feature matrix for a single split."""
    print(f"\n  Building pairs for {split_name}...")

    pairs = build_pairs(
        judgments=judgments,
        queries=queries,
        products=products,
    )

    print(f"  Attaching lexical scores for {split_name}...")

    pairs = attach_lexical_scores(
        pairs=pairs,
        tfidf_retriever=tfidf_retriever,
        bm25_retriever=bm25_retriever,
    )

    print(f"  Generating features for {split_name}...")

    features = build_query_product_features(
        pairs=pairs,
        config=config,
    )

    # Attach target labels from the original judgments.
    # These are NOT part of the feature matrix but are needed
    # for model training and evaluation.
    if "relevance_score" in judgments.columns:
        features["relevance_score"] = judgments["relevance_score"].values

    if "label" in judgments.columns:
        features["label"] = judgments["label"].values

    return features


def main() -> None:
    print("=" * 70)
    print("RelevanceFlow — Feature Generation")
    print("=" * 70)

    products, queries, train, validation, test = load_data()

    print(f"Products:    {len(products):,}")
    print(f"Queries:     {len(queries):,}")
    print(f"Train:       {len(train):,}")
    print(f"Validation:  {len(validation):,}")
    print(f"Test:        {len(test):,}")

    # ---------------------------------------------------------------
    # Fit retrievers on training data ONLY
    # ---------------------------------------------------------------

    train_product_ids = train["product_id"].unique()
    train_products = products[products["product_id"].isin(train_product_ids)].copy()

    print(f"\nFitting TF-IDF on {len(train_products):,} training products...")

    tfidf_retriever = TfidfRetriever(
        TfidfConfig(
            ngram_range=(1, 2),
            min_df=1,
            max_df=0.95,
            sublinear_tf=True,
            lowercase=True,
            max_features=200_000,
        )
    )

    tfidf_retriever.fit(
        product_ids=train_products["product_id"],
        product_text=train_products["product_name"],
    )

    print(f"TF-IDF vocabulary size: {len(tfidf_retriever.vectorizer.vocabulary_):,}")

    print(f"\nFitting BM25 on {len(train_products):,} training products...")

    bm25_retriever = BM25Retriever(BM25Config(k1=1.5, b=0.75))

    bm25_retriever.fit(train_products["product_name"].fillna("").astype(str).tolist())

    print(f"BM25 vocabulary size: {len(bm25_retriever.vocabulary_):,}")

    # ---------------------------------------------------------------
    # Generate features for each split
    # ---------------------------------------------------------------

    config = FeatureConfig()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for split_name, judgments in [
        ("train", train),
        ("validation", validation),
        ("test", test),
    ]:
        start = time.perf_counter()

        features = generate_features_for_split(
            split_name=split_name,
            judgments=judgments,
            queries=queries,
            products=products,
            tfidf_retriever=tfidf_retriever,
            bm25_retriever=bm25_retriever,
            config=config,
        )

        elapsed = time.perf_counter() - start

        output_path = OUTPUT_DIR / f"{split_name}_features.parquet"

        features.to_parquet(output_path, index=False)

        print(f"\n  {split_name}:")
        print(f"    Rows:     {len(features):,}")
        print(f"    Columns:  {len(features.columns)}")
        print(f"    Time:     {elapsed:.2f}s")
        print(f"    Saved:    {output_path}")

    print("\n" + "=" * 70)
    print("Feature generation complete.")
    print(f"Output directory: {OUTPUT_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    main()
