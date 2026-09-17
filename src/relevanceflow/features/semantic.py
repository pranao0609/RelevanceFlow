from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sentence_transformers import SentenceTransformer


@dataclass(frozen=True)
class SemanticEncoderConfig:
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    device: str = "cpu"
    batch_size: int = 32
    normalize_embeddings: bool = True


class SemanticEncoder:
    """Generate pretrained transformer embeddings for queries/products."""

    def __init__(
        self,
        config: SemanticEncoderConfig | None = None,
    ) -> None:
        self.config = config or SemanticEncoderConfig()

        self.model = SentenceTransformer(
            self.config.model_name,
            device=self.config.device,
        )

    def encode(
        self,
        texts: list[str],
    ) -> np.ndarray:
        if not texts:
            return np.empty((0, 384), dtype=np.float32)

        embeddings = self.model.encode(
            texts,
            batch_size=self.config.batch_size,
            normalize_embeddings=self.config.normalize_embeddings,
            show_progress_bar=True,
            convert_to_numpy=True,
        )

        return np.asarray(embeddings, dtype=np.float32)

    @staticmethod
    def cosine_similarity(
        query_embedding: np.ndarray,
        product_embedding: np.ndarray,
    ) -> float:
        query_norm = np.linalg.norm(query_embedding)
        product_norm = np.linalg.norm(product_embedding)

        if query_norm == 0.0 or product_norm == 0.0:
            return 0.0

        return float(
            np.dot(query_embedding, product_embedding) / (query_norm * product_norm)
        )
