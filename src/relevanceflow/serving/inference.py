from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

from relevanceflow.features.config import FeatureConfig
from relevanceflow.features.pipeline import build_query_product_features
from relevanceflow.features.semantic import SemanticEncoder
from relevanceflow.retrieval.bm25 import BM25Retriever
from relevanceflow.retrieval.tfidf import TfidfRetriever
from relevanceflow.serving.model_loader import (
    ModelLoadingError,
    get_model_feature_columns,
    load_registered_model,
    prepare_model_features,
)


@dataclass(frozen=True)
class InferenceConfig:
    """Configuration for ranking inference."""

    tracking_uri: str = "sqlite:///mlflow.db"
    model_name: str = "RelevanceFlowRanker"
    model_alias: str = "champion"

    products_path: Path = Path("data/processed/wands/products.parquet")
    train_path: Path = Path("data/processed/wands/train.parquet")

    top_k: int = 10


class RankingInferenceService:
    """Production inference service for RelevanceFlow ranking."""

    def __init__(
        self,
        config: InferenceConfig | None = None,
    ) -> None:
        self.config = config or InferenceConfig()

        self.model = None
        self.model_feature_columns: list[str] = []

        self.products: pd.DataFrame | None = None
        self.training_product_ids: set[int] = set()

        self.tfidf: TfidfRetriever | None = None
        self.bm25: BM25Retriever | None = None
        self.semantic_encoder: SemanticEncoder | None = None

        self.feature_config = FeatureConfig(
            lexical_enabled=True,
            product_text_enabled=True,
            metadata_enabled=True,
            compatibility_enabled=True,
            semantic_enabled=True,
        )

        self._initialized = False

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------

    def initialize(self) -> None:
        """Initialize all artifacts required for inference."""

        if self._initialized:
            return

        self.model = load_registered_model(
            model_name=self.config.model_name,
            alias=self.config.model_alias,
            tracking_uri=self.config.tracking_uri,
        )

        self.model_feature_columns = get_model_feature_columns(self.model)

        self._validate_model_schema()

        self._load_catalog()
        self._load_retrievers()
        self._load_semantic_encoder()

        self._initialized = True

    def _validate_model_schema(self) -> None:
        """Validate the feature schema of the registered model."""

        if not self.model_feature_columns:
            raise ModelLoadingError("Registered model exposes no feature names.")

        expected_features = {
            "bm25_score",
            "tfidf_similarity",
            "exact_match",
            "token_overlap",
            "character_overlap",
            "query_title_overlap",
            "title_length",
            "description_length",
            "feature_count",
            "category_depth",
            "average_rating",
            "rating_count",
            "review_count",
            "category_match",
            "product_class_match",
            "phrase_match",
            "semantic_similarity",
        }

        actual_features = set(self.model_feature_columns)

        missing = expected_features - actual_features

        if missing:
            raise ModelLoadingError(
                "Registered model is missing expected features: " f"{sorted(missing)}"
            )

    # ------------------------------------------------------------------
    # Catalog
    # ------------------------------------------------------------------

    def _load_catalog(self) -> None:
        """Load products and identify training-associated products."""

        if not self.config.products_path.exists():
            raise FileNotFoundError(
                "Product catalog not found: " f"{self.config.products_path}"
            )

        if not self.config.train_path.exists():
            raise FileNotFoundError(
                "Training split not found: " f"{self.config.train_path}"
            )

        self.products = pd.read_parquet(self.config.products_path)

        required_product_columns = {
            "product_id",
            "product_name",
            "product_class",
            "category hierarchy",
            "product_description",
            "product_features",
            "rating_count",
            "average_rating",
            "review_count",
        }

        missing_product_columns = required_product_columns - set(self.products.columns)

        if missing_product_columns:
            raise ValueError(
                "Product catalog is missing columns: "
                f"{sorted(missing_product_columns)}"
            )

        train = pd.read_parquet(self.config.train_path)

        if "product_id" not in train.columns:
            raise ValueError("Training split is missing 'product_id'.")

        self.training_product_ids = set(train["product_id"].astype(int).unique())

        if not self.training_product_ids:
            raise ValueError("No training-associated product IDs were found.")

    # ------------------------------------------------------------------
    # Retrieval models
    # ------------------------------------------------------------------

    def _load_retrievers(self) -> None:
        """
        Fit TF-IDF and BM25 on training-associated products only.
        """

        if self.products is None:
            raise RuntimeError("Product catalog has not been loaded.")

        training_products = self.products[
            self.products["product_id"].isin(self.training_product_ids)
        ].copy()

        if training_products.empty:
            raise RuntimeError("No training-associated products are available.")

        # TF-IDF
        self.tfidf = TfidfRetriever()

        self.tfidf.fit(
            product_ids=training_products["product_id"],
            product_text=(training_products["product_name"].fillna("").astype(str)),
        )

        # BM25
        self.bm25 = BM25Retriever()

        self.bm25.fit(training_products["product_name"].fillna("").astype(str).tolist())

    def _load_semantic_encoder(self) -> None:
        """Load the pretrained MiniLM semantic encoder."""

        self.semantic_encoder = SemanticEncoder()

    # ------------------------------------------------------------------
    # Product normalization
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize_product(product: dict) -> dict:
        """
        Convert API product fields to the WANDS feature schema.
        """

        return {
            "product_id": int(product["product_id"]),
            "product_name": (product.get("product_name") or ""),
            "product_class": (product.get("product_class") or ""),
            "category hierarchy": (product.get("category_hierarchy") or ""),
            "product_description": (product.get("product_description") or ""),
            "product_features": (product.get("product_features") or ""),
            "rating_count": (
                product["rating_count"]
                if product.get("rating_count") is not None
                else 0.0
            ),
            "average_rating": (
                product["average_rating"]
                if product.get("average_rating") is not None
                else 0.0
            ),
            "review_count": (
                product["review_count"]
                if product.get("review_count") is not None
                else 0.0
            ),
        }

    # ------------------------------------------------------------------
    # Feature generation
    # ------------------------------------------------------------------

    def _build_features(
        self,
        query: str,
        products: pd.DataFrame,
    ) -> pd.DataFrame:
        """Build the 17 model features for inference."""

        if self.tfidf is None:
            raise RuntimeError("TF-IDF retriever is not initialized.")

        if self.bm25 is None:
            raise RuntimeError("BM25 retriever is not initialized.")

        if self.semantic_encoder is None:
            raise RuntimeError("Semantic encoder is not initialized.")

        pairs = products.copy()

        # Required columns for build_query_product_features().
        pairs.insert(
            0,
            "query_id",
            -1,
        )

        pairs.insert(
            3,
            "query",
            query,
        )

        # --------------------------------------------------------------
        # TF-IDF
        # --------------------------------------------------------------

        query_vector = self.tfidf.transform_queries(pd.Series([query]))

        product_vectors = self.tfidf.transform_products(pairs["product_name"])

        tfidf_scores = cosine_similarity(
            query_vector,
            product_vectors,
        ).ravel()

        pairs["tfidf_similarity"] = tfidf_scores

        # --------------------------------------------------------------
        # BM25
        # --------------------------------------------------------------

        bm25_candidates = pairs[
            [
                "product_id",
                "product_name",
            ]
        ].to_dict("records")

        bm25_results = self.bm25.score_candidates(
            query=query,
            candidate_products=bm25_candidates,
        )

        bm25_scores = {
            int(item["product_id"]): float(item["score"]) for item in bm25_results
        }

        pairs["bm25_score"] = pairs["product_id"].map(bm25_scores).fillna(0.0)

        # --------------------------------------------------------------
        # Standard feature pipeline
        # --------------------------------------------------------------

        features = build_query_product_features(
            pairs=pairs,
            config=self.feature_config,
        )

        # --------------------------------------------------------------
        # Semantic similarity
        # --------------------------------------------------------------

        query_embedding = self.semantic_encoder.encode([query])[0]

        product_embeddings = self.semantic_encoder.encode(
            pairs["product_name"].fillna("").astype(str).tolist()
        )

        semantic_scores = cosine_similarity(
            query_embedding.reshape(1, -1),
            product_embeddings,
        ).ravel()

        features["semantic_similarity"] = semantic_scores

        # These are generated by the feature pipeline but are not
        # among the 17 features used by the LightGBM ranker.
        if "query_embedding_norm" in features.columns:
            features["query_embedding_norm"] = 0.0

        if "product_embedding_norm" in features.columns:
            features["product_embedding_norm"] = 0.0

        return features

    # ------------------------------------------------------------------
    # Ranking
    # ------------------------------------------------------------------

    def rank(
        self,
        query: str,
        products: list[dict],
        top_k: int | None = None,
    ) -> list[dict]:
        """Rank candidate products for a query."""

        # Validate before expensive initialization.
        if not query or not query.strip():
            raise ValueError("Query must not be empty.")

        if not products:
            raise ValueError("At least one product is required.")

        limit = self.config.top_k if top_k is None else top_k

        if limit <= 0:
            raise ValueError("top_k must be greater than zero.")

        self.initialize()

        normalized_products = [self._normalize_product(product) for product in products]

        product_df = pd.DataFrame(normalized_products)

        features = self._build_features(
            query=query,
            products=product_df,
        )

        model_features = prepare_model_features(
            features=features,
            feature_columns=self.model_feature_columns,
        )

        predictions = self.model.predict(model_features)

        scores = np.asarray(
            predictions,
            dtype=float,
        )

        if len(scores) != len(product_df):
            raise ModelLoadingError(
                "Model prediction count does not match " "candidate product count."
            )

        if not np.all(np.isfinite(scores)):
            raise ModelLoadingError("Model produced non-finite ranking scores.")

        results = []

        for product, score in zip(
            normalized_products,
            scores,
        ):
            results.append(
                {
                    "product_id": int(product["product_id"]),
                    "product_name": (product["product_name"]),
                    "score": float(score),
                }
            )

        # Deterministic ranking:
        # score descending, product_id ascending.
        results.sort(
            key=lambda item: (
                -item["score"],
                item["product_id"],
            )
        )

        results = results[: min(limit, len(results))]

        for rank, result in enumerate(
            results,
            start=1,
        ):
            result["rank"] = rank

        return results
