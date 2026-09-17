import numpy as np
import pytest

from relevanceflow.retrieval.semantic import (
    SemanticRetriever,
)


def test_ranks_by_cosine_similarity():
    product_ids = np.array([101, 102, 103])

    embeddings = np.array(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [-1.0, 0.0],
        ],
        dtype=np.float32,
    )

    retriever = SemanticRetriever(
        product_ids,
        embeddings,
    )

    results = retriever.rank_candidates(
        query_embedding=np.array(
            [1.0, 0.0],
            dtype=np.float32,
        ),
        candidate_product_ids=np.array([101, 102, 103]),
    )

    assert [result["product_id"] for result in results] == [101, 102, 103]


def test_top_k():
    product_ids = np.array([101, 102, 103])

    embeddings = np.array(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [-1.0, 0.0],
        ],
        dtype=np.float32,
    )

    retriever = SemanticRetriever(
        product_ids,
        embeddings,
    )

    results = retriever.rank_candidates(
        np.array([1.0, 0.0]),
        np.array([101, 102, 103]),
        top_k=2,
    )

    assert len(results) == 2


def test_dimension_mismatch():
    retriever = SemanticRetriever(
        np.array([101, 102]),
        np.array(
            [
                [1.0, 0.0],
                [0.0, 1.0],
            ],
            dtype=np.float32,
        ),
    )

    with pytest.raises(ValueError):
        retriever.rank_candidates(
            np.array([1.0, 0.0, 0.0]),
            np.array([101, 102]),
        )


def test_unknown_product():
    retriever = SemanticRetriever(
        np.array([101]),
        np.array(
            [[1.0, 0.0]],
            dtype=np.float32,
        ),
    )

    with pytest.raises(ValueError):
        retriever.rank_candidates(
            np.array([1.0, 0.0]),
            np.array([999]),
        )
