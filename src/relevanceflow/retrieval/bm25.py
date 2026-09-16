from __future__ import annotations

import math
import time
from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np


class BM25RetrievalError(ValueError):
    """Raised when BM25 retrieval receives invalid input."""


@dataclass(frozen=True)
class BM25Config:
    """Configuration for BM25 retrieval."""

    k1: float = 1.5
    b: float = 0.75


class BM25Retriever:
    """
    BM25 lexical retriever.

    The BM25 corpus statistics are learned only from the supplied
    training corpus. Candidate documents can then be scored against
    those fixed corpus statistics.
    """

    def __init__(self, config: BM25Config | None = None) -> None:
        self.config = config or BM25Config()

        if self.config.k1 < 0:
            raise BM25RetrievalError("k1 must be >= 0.")

        if not 0 <= self.config.b <= 1:
            raise BM25RetrievalError("b must be between 0 and 1.")

        self.vocabulary_: dict[str, int] = {}
        self.idf_: dict[str, float] = {}
        self.avg_doc_length_: float = 0.0
        self.num_documents_: int = 0
        self.is_fitted_: bool = False

        self.index_creation_time_seconds_: float | None = None
        self.estimated_memory_bytes_: int | None = None

    @staticmethod
    def tokenize(text: str) -> list[str]:
        """
        Simple deterministic whitespace tokenizer.

        BM25 is intentionally kept lexical at this stage.
        """
        if not isinstance(text, str):
            text = "" if text is None else str(text)

        return text.lower().split()

    def fit(self, documents: Iterable[str]) -> BM25Retriever:
        """
        Build BM25 corpus statistics from training documents.
        """
        start_time = time.perf_counter()

        documents = list(documents)

        if not documents:
            raise BM25RetrievalError("Cannot fit BM25 on an empty corpus.")

        tokenized_documents = [self.tokenize(document) for document in documents]

        self.num_documents_ = len(tokenized_documents)

        document_lengths = [len(tokens) for tokens in tokenized_documents]

        self.avg_doc_length_ = float(np.mean(document_lengths))

        # Document frequency:
        # count each term at most once per document.
        document_frequency: dict[str, int] = {}

        for tokens in tokenized_documents:
            unique_tokens = set(tokens)

            for token in unique_tokens:
                document_frequency[token] = document_frequency.get(token, 0) + 1

        self.vocabulary_ = {
            token: index for index, token in enumerate(sorted(document_frequency))
        }

        # BM25 IDF with the standard smoothed formulation.
        self.idf_ = {
            token: math.log(
                1.0 + (self.num_documents_ - frequency + 0.5) / (frequency + 0.5)
            )
            for token, frequency in document_frequency.items()
        }

        self.is_fitted_ = True

        self.index_creation_time_seconds_ = time.perf_counter() - start_time

        # Approximate memory footprint of the learned dictionaries.
        # This is intentionally an estimate, not process RSS.
        self.estimated_memory_bytes_ = (
            sum(len(token.encode("utf-8")) + 8 for token in self.vocabulary_)
            + len(self.idf_) * 8
            + len(self.vocabulary_) * 8
        )

        return self

    def _score_document(
        self,
        query_tokens: list[str],
        document_tokens: list[str],
    ) -> float:
        """
        Compute BM25 score for one query-document pair.
        """
        if not self.is_fitted_:
            raise BM25RetrievalError("BM25Retriever must be fitted before scoring.")

        if not document_tokens:
            return 0.0

        document_length = len(document_tokens)

        term_frequency: dict[str, int] = {}

        for token in document_tokens:
            term_frequency[token] = term_frequency.get(token, 0) + 1

        score = 0.0

        for term in query_tokens:
            if term not in self.idf_:
                continue

            frequency = term_frequency.get(term, 0)

            if frequency == 0:
                continue

            idf = self.idf_[term]

            numerator = frequency * (self.config.k1 + 1)

            denominator = frequency + self.config.k1 * (
                1
                - self.config.b
                + self.config.b * document_length / self.avg_doc_length_
            )

            score += idf * numerator / denominator

        return float(score)

    def score_candidates(
        self,
        query: str,
        candidate_products: Iterable[dict],
    ) -> list[dict]:
        """
        Score and rank candidate products.

        Each candidate must contain:
        - product_id
        - product_name
        """
        if not self.is_fitted_:
            raise BM25RetrievalError("BM25Retriever must be fitted before ranking.")

        query_tokens = self.tokenize(query)

        results = []

        for candidate in candidate_products:
            if "product_id" not in candidate:
                raise BM25RetrievalError("Candidate is missing 'product_id'.")

            if "product_name" not in candidate:
                raise BM25RetrievalError("Candidate is missing 'product_name'.")

            document_tokens = self.tokenize(candidate["product_name"])

            score = self._score_document(
                query_tokens=query_tokens,
                document_tokens=document_tokens,
            )

            results.append(
                {
                    "product_id": candidate["product_id"],
                    "score": score,
                }
            )

        # Deterministic ordering:
        # 1. score descending
        # 2. product_id ascending
        results.sort(
            key=lambda item: (
                -item["score"],
                str(item["product_id"]),
            )
        )

        for rank, result in enumerate(results, start=1):
            result["rank"] = rank

        return results

    def rank_candidates(
        self,
        query: str,
        candidate_products,
        top_k: int = 10,
    ) -> list[dict]:
        """
        Return top-k BM25-ranked candidates.
        """
        if top_k <= 0:
            raise BM25RetrievalError("top_k must be greater than zero.")

        ranked = self.score_candidates(
            query=query,
            candidate_products=candidate_products,
        )

        return ranked[:top_k]

    def score_query_latency(
        self,
        query: str,
        candidate_products,
        top_k: int = 10,
    ) -> tuple[list[dict], float]:
        """
        Rank candidates and return latency in seconds.
        """
        start_time = time.perf_counter()

        ranked = self.rank_candidates(
            query=query,
            candidate_products=candidate_products,
            top_k=top_k,
        )

        latency = time.perf_counter() - start_time

        return ranked, latency
