from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class TfidfRetrievalError(ValueError):
    """Raised when TF-IDF retrieval cannot be performed."""


@dataclass(frozen=True)
class TfidfConfig:
    """Configuration for the TF-IDF retrieval baseline."""

    ngram_range: tuple[int, int] = (1, 2)
    min_df: int = 1
    max_df: float = 0.95
    sublinear_tf: bool = True
    lowercase: bool = True
    max_features: int | None = 200_000


class TfidfRetriever:
    """
    TF-IDF lexical retrieval model.

    The vectorizer is fitted only on training product text.
    Queries and candidate products from validation/test are transformed
    using the already-fitted vectorizer.
    """

    def __init__(self, config: TfidfConfig | None = None) -> None:
        self.config = config or TfidfConfig()

        self.vectorizer = TfidfVectorizer(
            analyzer="word",
            ngram_range=self.config.ngram_range,
            min_df=self.config.min_df,
            max_df=self.config.max_df,
            sublinear_tf=self.config.sublinear_tf,
            lowercase=self.config.lowercase,
            max_features=self.config.max_features,
        )

        self.product_matrix: csr_matrix | None = None
        self.product_ids: np.ndarray | None = None
        self.is_fitted = False

    def fit(
        self,
        product_ids: pd.Series,
        product_text: pd.Series,
    ) -> TfidfRetriever:
        """Fit TF-IDF on training product text."""

        if len(product_ids) != len(product_text):
            raise TfidfRetrievalError(
                "product_ids and product_text must have equal length."
            )

        if len(product_text) == 0:
            raise TfidfRetrievalError("Cannot fit TF-IDF on an empty product corpus.")

        text = product_text.fillna("").astype(str)

        self.product_matrix = self.vectorizer.fit_transform(text)
        self.product_ids = product_ids.to_numpy()

        self.is_fitted = True

        return self

    def transform_queries(
        self,
        queries: pd.Series,
    ) -> csr_matrix:
        """Transform queries using the fitted TF-IDF vocabulary."""

        self._check_fitted()

        return self.vectorizer.transform(queries.fillna("").astype(str))

    def transform_products(
        self,
        product_text: pd.Series,
    ) -> csr_matrix:
        """Transform candidate product text."""

        self._check_fitted()

        return self.vectorizer.transform(product_text.fillna("").astype(str))

    def rank_candidates(
        self,
        query_text: str,
        candidate_products: pd.DataFrame,
        top_k: int | None = None,
    ) -> pd.DataFrame:
        """
        Rank candidate products for one query.

        Required candidate column:
            product_id

        Text column:
            product_name
        """

        self._check_fitted()

        if "product_id" not in candidate_products.columns:
            raise TfidfRetrievalError("candidate_products must contain 'product_id'.")

        if "product_name" not in candidate_products.columns:
            raise TfidfRetrievalError("candidate_products must contain 'product_name'.")

        if candidate_products.empty:
            return pd.DataFrame(
                columns=[
                    "product_id",
                    "score",
                    "rank",
                ]
            )

        query_vector = self.vectorizer.transform([str(query_text)])

        candidate_matrix = self.vectorizer.transform(
            candidate_products["product_name"].fillna("").astype(str)
        )

        scores = cosine_similarity(
            query_vector,
            candidate_matrix,
        ).ravel()

        result = pd.DataFrame(
            {
                "product_id": candidate_products["product_id"].to_numpy(),
                "score": scores,
            }
        )

        result = result.sort_values(
            by=["score", "product_id"],
            ascending=[False, True],
        ).reset_index(drop=True)

        result["rank"] = np.arange(1, len(result) + 1)

        if top_k is not None:
            result = result.head(top_k).copy()

        return result

    def _check_fitted(self) -> None:
        if not self.is_fitted:
            raise TfidfRetrievalError("TfidfRetriever must be fitted before retrieval.")
