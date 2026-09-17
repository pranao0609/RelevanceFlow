from __future__ import annotations

import numpy as np


class SemanticRetrievalError(ValueError):
    """Raised when semantic retrieval receives invalid input."""


class SemanticRetriever:
    """Semantic retriever using precomputed transformer embeddings."""

    def __init__(
        self,
        product_ids: np.ndarray,
        product_embeddings: np.ndarray,
    ) -> None:
        product_ids = np.asarray(product_ids)

        product_embeddings = np.asarray(
            product_embeddings,
            dtype=np.float32,
        )

        if product_embeddings.ndim != 2:
            raise SemanticRetrievalError("product_embeddings must be a 2D array.")

        if len(product_ids) != len(product_embeddings):
            raise SemanticRetrievalError(
                "product_ids and product_embeddings must " "have the same length."
            )

        self.product_ids = product_ids
        self.product_embeddings = product_embeddings

    def rank_candidates(
        self,
        query_embedding: np.ndarray,
        candidate_product_ids: np.ndarray,
        top_k: int = 10,
    ) -> list[dict]:
        """Rank only the supplied candidate products."""

        if top_k <= 0:
            raise SemanticRetrievalError("top_k must be greater than zero.")

        query_embedding = np.asarray(
            query_embedding,
            dtype=np.float32,
        )

        if query_embedding.ndim != 1:
            raise SemanticRetrievalError("query_embedding must be one-dimensional.")

        if query_embedding.shape[0] != self.product_embeddings.shape[1]:
            raise SemanticRetrievalError(
                "Query embedding dimension does not match " "product embeddings."
            )

        candidate_product_ids = np.asarray(candidate_product_ids)

        product_index = {
            int(product_id): index for index, product_id in enumerate(self.product_ids)
        }

        indices = []

        for product_id in candidate_product_ids:
            product_id = int(product_id)

            if product_id not in product_index:
                raise SemanticRetrievalError(f"Unknown product_id: {product_id}")

            indices.append(product_index[product_id])

        candidate_embeddings = self.product_embeddings[indices]

        query_norm = np.linalg.norm(query_embedding)

        candidate_norms = np.linalg.norm(
            candidate_embeddings,
            axis=1,
        )

        if query_norm == 0.0:
            scores = np.zeros(
                len(candidate_embeddings),
                dtype=np.float32,
            )
        else:
            denominator = candidate_norms * query_norm

            scores = np.divide(
                candidate_embeddings @ query_embedding,
                denominator,
                out=np.zeros_like(candidate_norms),
                where=denominator != 0,
            )

        results = [
            {
                "product_id": int(product_id),
                "score": float(score),
            }
            for product_id, score in zip(
                candidate_product_ids,
                scores,
            )
        ]

        results.sort(
            key=lambda item: (
                -item["score"],
                str(item["product_id"]),
            )
        )

        for rank, result in enumerate(
            results,
            start=1,
        ):
            result["rank"] = rank

        return results[:top_k]
