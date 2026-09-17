from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from relevanceflow.features.semantic import (
    SemanticEncoder,
    SemanticEncoderConfig,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed" / "wands"
FEATURE_DIR = DATA_DIR / "features"
SEMANTIC_DIR = DATA_DIR / "interim" / "semantic"

PRODUCT_PATH = PROCESSED_DIR / "products.parquet"
QUERY_PATH = PROCESSED_DIR / "queries.parquet"

TRAIN_FEATURE_PATH = FEATURE_DIR / "train_features.parquet"
VALIDATION_FEATURE_PATH = FEATURE_DIR / "validation_features.parquet"
TEST_FEATURE_PATH = FEATURE_DIR / "test_features.parquet"


def main() -> None:
    SEMANTIC_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("Loading products and queries...")

    products = pd.read_parquet(PRODUCT_PATH)
    queries = pd.read_parquet(QUERY_PATH)

    print(f"Products: {len(products):,}")
    print(f"Queries:  {len(queries):,}")

    config = SemanticEncoderConfig(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        device="cpu",
        batch_size=32,
        normalize_embeddings=True,
    )

    print("\nLoading Sentence Transformer...")
    print(f"Model: {config.model_name}")
    print(f"Device: {config.device}")

    encoder = SemanticEncoder(config)

    print("\nEncoding queries...")

    query_texts = queries["query"].fillna("").astype(str).tolist()

    query_embeddings = encoder.encode(query_texts)

    print(f"Query embeddings: {query_embeddings.shape}")

    print("\nEncoding products...")

    product_texts = products["product_name"].fillna("").astype(str).tolist()

    product_embeddings = encoder.encode(product_texts)

    print(f"Product embeddings: {product_embeddings.shape}")

    query_embedding_path = SEMANTIC_DIR / "query_embeddings.npy"

    product_embedding_path = SEMANTIC_DIR / "product_embeddings.npy"

    np.save(
        query_embedding_path,
        query_embeddings,
    )

    np.save(
        product_embedding_path,
        product_embeddings,
    )

    print(f"\nSaved query embeddings: " f"{query_embedding_path}")

    print(f"Saved product embeddings: " f"{product_embedding_path}")

    query_norms = np.linalg.norm(
        query_embeddings,
        axis=1,
    )

    product_norms = np.linalg.norm(
        product_embeddings,
        axis=1,
    )

    query_embedding_lookup = dict(
        zip(
            queries["query_id"],
            query_embeddings,
        )
    )

    product_embedding_lookup = dict(
        zip(
            products["product_id"],
            product_embeddings,
        )
    )

    query_norm_lookup = dict(
        zip(
            queries["query_id"],
            query_norms,
        )
    )

    product_norm_lookup = dict(
        zip(
            products["product_id"],
            product_norms,
        )
    )

    feature_paths = {
        "train": TRAIN_FEATURE_PATH,
        "validation": VALIDATION_FEATURE_PATH,
        "test": TEST_FEATURE_PATH,
    }

    for split_name, feature_path in feature_paths.items():
        print(f"\nProcessing {split_name}...")

        features = pd.read_parquet(feature_path)

        similarities = []

        for row in features.itertuples(index=False):
            query_embedding = query_embedding_lookup[row.query_id]

            product_embedding = product_embedding_lookup[row.product_id]

            similarity = encoder.cosine_similarity(
                query_embedding,
                product_embedding,
            )

            similarities.append(similarity)

        features["semantic_similarity"] = np.asarray(
            similarities,
            dtype=np.float32,
        )

        features["query_embedding_norm"] = (
            features["query_id"].map(query_norm_lookup).astype(np.float32)
        )

        features["product_embedding_norm"] = (
            features["product_id"].map(product_norm_lookup).astype(np.float32)
        )

        features.to_parquet(
            feature_path,
            index=False,
        )

        print(f"Updated: {feature_path}")

        print(
            "Semantic similarity range: "
            f"{features['semantic_similarity'].min():.6f} "
            f"to "
            f"{features['semantic_similarity'].max():.6f}"
        )

    metadata = {
        "model_name": config.model_name,
        "device": config.device,
        "batch_size": config.batch_size,
        "normalize_embeddings": (config.normalize_embeddings),
        "embedding_dimension": int(query_embeddings.shape[1]),
        "num_queries": len(query_embeddings),
        "num_products": len(product_embeddings),
    }

    metadata_path = SEMANTIC_DIR / "metadata.json"

    with open(
        metadata_path,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            metadata,
            file,
            indent=2,
        )

    print(f"\nSaved metadata: {metadata_path}")

    print("\nStage 19 semantic feature generation complete.")


if __name__ == "__main__":
    main()
